import numpy as np
import os
import torch
from typing import Deque # from smg_gym
from collections import deque # from smg_gym
from isaacgym.torch_utils import quat_unit # from smg_gym

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask

class EdgeFollow(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.sim_device = sim_device
        self.headless = headless
        cfg["device_type"] = "cuda"
        cfg["device_id"] = 0
        cfg["rl_device"] = "cuda:0"
        headless = False # Start by rendering and then change once implementation verified
        cfg["physics_engine"] = "physx"
        env = {}
        env["numEnvs"] = 5 # Alter as needed
        env["numObservations"] = 0 #! Requires embodiment in PyBullet (customisable param), Observation space dependant on tactile sensor
        env["numActions"] = 2 # From PyBullet variables dump

        #Optional env params
        env["numAgents"] = 1 # Default
        env["numStates"] = None # Don't think is required
        env["controlFrequencyInv"] = self.calculate_controlFrequencyInv() # Taken from BaeTactileEnv
        env["clipObservations"] = None #! Obs space dependant on tactile sensor setup
        env["clipActions"] = None  #! PyBullet has Min/Max actions at -1.0 and 1.0
        env["enableCameraSensors"] = True # Assumed True
        cfg["env"] = env

        #! Custom env params
        robot_arm_params = {}
        robot_arm_params["type"] = "ur5"
        robot_arm_params["tcp_link_name"] = "tcp_link" # Not associated with mesh file
        robot_arm_params["tactip_tip_link_name"] = "tactip_tip_link" # Associated with mesh file
        robot_arm_params["rest_pose"] = np.array([0.0, 1.374, 0.871,  1.484, -2.758, 4.940, 2.2409, 7.165, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.default_robot_rest_pose = torch.tensor(robot_arm_params["rest_pose"], dtype=torch.float, device=self.device) # Matching data conversion from FrankaCabinet
        cfg["robot_arm_params"] = robot_arm_params

        tactile_sensor_params = {}
        tactile_sensor_params["type"] = "standard_tactip" # Used in PyBullet for loading urdf in TactileArmEmbodiment

        # tcp_link_name code from VisuoTactileArmEmbodiment initialisation
        if "tcp_link_name" in robot_arm_params:
            self.tcp_link_name = robot_arm_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"

        # Contact sensing configuration
        self.enable_dof_force_sensors = cfg["env"]["enable_dof_force_sensors"]
        self.contact_sensor_modality = cfg["env"]["contact_sensor_modality"]

        # Finish config
        self.cfg = cfg
        obs_cfg = self.cfg["enabled_obs"] #! Implement observation config as in smg_gym base_hand, smg_gaiting and smg_reorient.yaml
        self.action_scale = self.cfg["env"]["actionScale"] # Taen from FrankaCabinet

        self.dt = 1/60 # Taken from FrankaCabinet

        # Configure observation space
        self._state_history_len = 2 # From smg_gym
        self._tcp_joint_pos_history: Deque[torch.Tensor] = deque(maxlen=self._state_history_len)
        self._tcp_joint_vel_history: Deque[torch.Tensor] = deque(maxlen=self._state_history_len)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Initialise robot state
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dofs] #.view() reshapes fod dof_state tensor - the list slicing to follow slices it to focus only on the DOFs belonging to the ur5 robot
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        # self.num_envs is dim1, -1 means auto calc size of dim 2, 13 represents the size of ator root body state
        # 13 because Pos = 3; Orn = 4, Lin Vel = 3, Ang vel = 3
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13) 
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13) 

        # Setup tensors for contact forces
        self.n_env_bodies = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.enable_dof_force_sensors:
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        if self.contact_sensor_modality == 'ft_sensor':
            force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # Create Views for contact data
        self.contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.n_env_bodies, 3)

        if self.contact_sensor_modality == 'ft_sensor':
            self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(
                self.num_envs, 1, 6 # 1 because one sensor, 6 because Wrench dims are 3D for force and 3D for torque
            )
        else:
            self.force_sensor_tensor = torch.zeros(
                size=(self.num_envs, 1, 6)
            )

        # Calculate and store metrics and constants that are necessary for the operation
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # Setup success storage
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # Initialise indexing for efficiently handling multiple envs and apply initial setup to all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        
    # From BaseTactileEnv, using the calculation for _velocity_action_repeat
    def calculate_controlFrequencyInv(self):
        sim_time_step = 1.0 / 240.0
        control_rate = 1.0 / 10.0
        return int(np.floor(control_rate / sim_time_step))

    # Required for VecTask
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self.setup_edge()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    # Taken from FrankaCabinet: replaces loading of plane urdf #! Positioning relative to table may be different
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    # From PyBullet
    def setup_edge(self):
        """
        Defines params for loading/resetting the edge object.
        """
        # define an initial position for the objects (world coords)
        self.edge_pos = [0.65, 0.0, 0.0]
        self.edge_height = 0.035
        self.edge_len = 0.175

    def _create_envs(self, num_envs, spacing, num_per_row):
        # Define lower and upper bounds for environment placement
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Get asset file locations
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        robotic_arm_asset_file = self.cfg["env"]["asset"].get("assetFileNameRobot", robotic_arm_asset_file)
        edge_asset_file = self.cfg["env"]["asset"].get("assetFileNameEdge", edge_asset_file)
        goal_indicator_file = self.cfg["env"]["asset"].get("assetFileNameGoalIndicator", goal_indicator_file)
        table_asset_file = self.cfg["env"]["asset"].get("assetFileNameTable", table_asset_file)

        # Set asset options and load assets
        robot_options = gymapi.AssetOptions()
        robot_options.flip_visual_attachments = False # Might need to change
        robot_options.fix_base_link = True
        robot_options.collapse_fixed_joints = True
        robot_options.disable_gravity = False
        robot_options.thickness = 0.001
        robot_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_options.use_mesh_materials = True # might not be relevant
        robotic_arm_asset = self.gym.load_asset(self.sim, asset_root, robotic_arm_asset_file, robot_options) 

        #! Implementing attachement of tactile sensor
        self.setup_mappings(robotic_arm_asset)
        self.tcp_link_id = self.link_name_to_index[self.tcp_link_name]
        
        edge_options = gymapi.AssetOptions()
        edge_options.disable_gravity = True # Object is stationary
        edge_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        edge_options.armature = 0.005 # Not sure if this is needed
        edge_asset = self.gym.load_asset(self.sim, asset_root, edge_asset_file, edge_options)

        goal_indicator_options = gymapi.AssetOptions()
        goal_indicator_options.disable_gravity = True # Object is stationary
        goal_indicator_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        goal_indicator_options.armature = 0.005 # Not sure if this is needed
        goal_indicator_asset = self.gym.load_asset(self.sim, asset_root, goal_indicator_file, goal_indicator_options)

        table_options = gymapi.AssetOptions()
        table_options.disable_gravity = True # Object is stationary
        table_options.fix_base_link = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset = self.gym.load_asset(self.sim, asset_root, table_asset_file, table_options)

        table_pose = gymapi.Transform() # Position and orientation taken from PyBullet
        table_pose.p = gymapi.Vec3(0.50, 0.00, -0.625)  # Position
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Orientation as a quaternion

        dofs = self.cfg["env"]["numActions"]
        default_stiffness = 400 #! Unsure
        default_damping = 80 #! Unsure

        dof_stiffness = to_torch([default_stiffness] * dofs, dtype=torch.float, device=self.device)
        dof_damping = to_torch([default_damping] * dofs, dtype=torch.float, device=self.device)

        self.num_dofs = dofs
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robotic_arm_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robotic_arm_asset)
        self.num_edge_bodies = self.gym.get_asset_rigid_body_count(edge_asset)
        self.num_goal_indicator_bodies = self.gym.get_asset_rigid_body_count(goal_indicator_asset)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)

        print(f'Robot arm count: {self.num_robot_bodies}')
        print(f'Robot arm dofs: {self.num_robot_dofs}')
        print(f'Edge count: {self.num_edge_bodies}')
        print(f'Goal Indicator count: {self.num_goal_indicator_bodies}')
        print(f'Table count: {self.num_table_bodies}')

        robot_dof_props = self.gym.get_asset_dof_properties(robotic_arm_asset)

        # Configure robotic arm DoFs
        self.robot_dof_lower_limits = robot_dof_props['lower']
        self.robot_dof_upper_limits = robot_dof_props['upper']
        robot_dof_props['effort'].fill_(1000.0) # From PyBullet UR5 class
        for i in range(self.num_robot_dofs):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            robot_dof_props['stiffness'][i] = dof_stiffness[i]
            robot_dof_props['damping'][i] = dof_damping[i]

        self.robot_dof_lower_limits = to_torch(robot_dof_props['lower'], dtype=torch.float, device=self.device)
        self.robot_dof_upper_limits = to_torch(robot_dof_props['upper'], dtype=torch.float, device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_speed_scales[[7, 8]] = 0.1 #! Unsure, taken from FrankaCabinet
        #! Franka implements some features to do with dof props 'effort'. Not sure if needed here.

        # Define starting poses for assets 
        robot_start_pose = gymapi.Transform()  # Set the position and orientation
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # From PyBullet ArmEmbodiment.load_urdf()
        robot_start_pose.r = gymapi.Quat(0, 0, 0, 1)  # Neutral orientation assumed
        robot_rest_pose = self.cfg["robot_arm_params"].get("rest_pose") # From PyBullet
        
        edge_start_pose = gymapi.Transform()   # Set the position and orientation
        edge_start_pose.p = gymapi.Vec3(self.edge_pos[0], self.edge_pos[1], self.edge_pos[2])
        edge_start_pose.r = gymapi.Quat(0, 0, 0, 1)  # Neutral orientation  
        
        goal_indicator_start_pose = gymapi.Transform()  # Set the position and orientation
        goal_indicator_start_pose.p = gymapi.Vec3(self.edge_pos[0], self.edge_pos[1], self.edge_pos[2])
        goal_indicator_start_pose.r = gymapi.Quat(0, 0, 0, 1)  # Neutral orientation  

        self.envs = []
        self.robots = []
        self.edges = []
        self.goal_indicators = []

        # Create environments and place assets #! Can add randomisation as in franka_cabinet later
        for i in range(num_envs):
            # Create environment
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # Add robotic arm
            robot_actor = self.gym.create_actor(env_ptr, robotic_arm_asset, robot_start_pose, "robot", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            self.robots.append(robot_actor)

            # Set initial joint positions (rest pose) for the robotic arm
            dof_states = self.gym.get_actor_dof_states(env_ptr, robot_actor, gymapi.STATE_ALL)
            dof_states["pos"][:len(robot_rest_pose)] = robot_rest_pose
            self.gym.set_actor_dof_states(env_ptr, robot_actor, dof_states, gymapi.STATE_ALL)

            #! Implement edge and goal pose randomisation as in EdgeFollow.update_edge() for edge_start_pose

            # Add edge
            edge_actor = self.gym.create_actor(env_ptr, edge_asset, edge_start_pose, "edge", i, 0, 0)
            self.edges.append(edge_actor)

            # Add goal indicator
            goal_indicator_actor = self.gym.create_actor(env_ptr, goal_indicator_asset, goal_indicator_start_pose, "goal_indicator", i, 0, 0)
            self.goal_indicators.append(goal_indicator_actor)
        
        # Get handle for the tcp_link, edge and goal indicator
        self.tcp_link_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robotic_arm_asset, self.tcp_link_name)
        self.edge_handle = self.gym.find_actor_rigid_body_handle(env_ptr, edge_asset, "long_edge")
        self.goal_indicator_handle = self.gym.find_actor_rigid_body_handle(env_ptr, goal_indicator_asset, "sphere_indicator")

        #! Setup self variables for _setup_contact_tracking
        self.robotic_arm_asset = robotic_arm_asset
        
        self._setup_tip_tracking()
        self._setup_contact_tracking()
            
        self.init_data()

    # Configure data related to how observations are generated and control commands are applied to the robot env
    #! Location to implement custom tactile sensor behaviour in terms of what relational measuraments are required. 
    def init_data(self):
        # Find handle for tcp

        # Get the current rigid body transformation (position and orientation) of the tcp

        # Find handle for edge

        # Find handle for goal_indicator

        # Define relative locations and orientations between tcp, edge and goal indicator

        # Initialise tensors for global positions and orientations of the robot and edge (updated each sim step to reflect current state)

        return

    # Mimicing functionality in ArmEmbodiment.load_urdf()
    def setup_mappings(self, robotic_arm_asset):
        self.num_joints, self.link_name_to_index, self.joint_name_to_index = self.create_link_joint_mappings(robotic_arm_asset)

    # IsaacGym version of ArmEmbodiment function
    def create_link_joint_mappings(self, asset):

        num_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_dofs = self.gym.get_asset_dof_count(asset)

        # Initialize dictionaries to store mappings
        joint_name_to_index = {}
        link_name_to_index = {}

        # Iterate over all DOFs (joints) to create joint name to index mapping
        for i in range(num_dofs):
            joint_name = self.gym.get_asset_dof_name(asset, i)
            joint_name_to_index[joint_name] = i

        # Iterate over all bodies (links) to create link name to index mapping
        for i in range(num_bodies):
            link_name = self.gym.get_asset_rigid_body_name(asset, i)
            link_name_to_index[link_name] = i

        return num_dofs, link_name_to_index, joint_name_to_index
    
    # Reset the environments
    def reset_idx(self, env_ids):
        # Reset the robot's DOF positions and velocities while introducing noise
        pos = tensor_clamp(
            self.default_robot_rest_pose.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self.device) - 0.5), 
            self.robot_dof_lower_limits, self.robot_dof_upper_limits
            )
        self.robot_dof_pos[env_ids, :] = pos
        self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])
        self.robot_dof_targets[env_ids, :self.num_robot_dofs] = pos

        # Single id per env as there is only one actor in each env
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.robot_dof_targets),
                                                        gymtorch.unwrap_tensor(env_ids_int32),
                                                        len(env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
    

    #Required for Eureka
    def compute_reward(self):
        return
    
    # Required for VecTask
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.robot_dof_targets[:, :self.num_robot_dofs] + self.robot_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.robot_dof_targets[:, :self.num_robot_dofs] = tensor_clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.robot_dof_targets))
    
    # Required for VecTask
    def post_physics_step(self):
        self.progress_buf += 1
        return
    
    # From smg_gym: The tcp's are tcp links and are not associated with mesh files (in smg_gym)
    def _setup_tip_tracking(self):
        robot_body_names = self.gym.get_asset_rigid_body_names(self.robotic_arm_asset)
        tcp_body_names = [name for name in robot_body_names if self.cfg["robot_arm_params"]["tcp_link_name"] in name]
        self.tcp_body_idxs = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in tcp_body_names
        ]
        return
    
    def _setup_contact_tracking(self):
        robot_body_names = self.gym.get_asset_rigid_body_names(self.robotic_arm_asset)
        tip_body_names = [name for name in robot_body_names if self.cfg["robot_arm_params"]["tactip_tip_link_name"] in name]
        non_tip_body_names = [name for name in robot_body_names if self.cfg["robot_arm_params"]["tactip_tip_link_name"] not in name]
        self.tip_body_idxs = [
            self.gym.find_asset_rigid_body_index(self.robotic_arm_asset, name) for name in tip_body_names
        ]
        self.non_tip_body_idxs = [
            self.gym.find_asset_rigid_body_index(self.robotic_arm_asset, name) for name in non_tip_body_names
        ]
        self.n_tips = 1
        self.n_non_tip_links = len(self.non_tip_body_idxs)

        # Add ft sensor to tactip
        if self.contact_sensor_modality == 'ft_sensor':
            sensor_pose = gymapi.Transform()
            for idx in self.tip_body_idxs:
                sensor_options = gymapi.ForceSensorProperties()
                sensor_options.enable_forward_dynamics_forces = False
                sensor_options.enable_constraint_solver_forces = True
                sensor_options.use_world_frame = True
                self.gym.create_asset_force_sensor(self.robotic_arm_asset, idx, sensor_pose, sensor_options)
        # No rich contacts
        return
    
    # From smg_gym
    def canonicalise_quat(self, quat):
        canon_quat = quat_unit(quat)
        canon_quat[torch.where(canon_quat[..., 3] < 0)] *= -1
        return canon_quat
    
    # Using info from smg_gym
    def compute_observations(self):
        # Obtain tactip contact information
        self.net_tip_contact_forces = self.contact_force_tensor[:, self.tip_body_idxs, :]
        self.n_tip_contacts = torch.where(
            torch.count_nonzero(self.net_tip_contact_forces, dim=2) > 0,
            torch.ones(size=(self.num_envs, self.n_tips), device=self.sim_device),
            torch.zeros(size=(self.num_envs, self.n_tips), device=self.sim_device)
        )
        self.n_tip_contacts = torch.sum(self.n_tip_contacts, dim=1)
        
        # Obtain tactip state
        tip_states = self.rigid_body_states[:, self.tcp_body_idxs, :]
        self.tip_pos = tip_states[..., 0:3].reshape(self.num_envs, 3) # 3 represents x,y,z coordinates of tip
        self.tip_orn = self.canonicalise_quat(tip_states[..., 3:7])
        self.tip_linvel = tip_states[..., 7:10]
        self.tip_angvel = tip_states[..., 10:13]

        # Get start position?

        # Get goal position
        #! In order to understand whether the observations are being obtained correctly, I will \
        #! need to run the simulation and print out variables like the contact_force_tensor to \
        #! assess their contents. Not worth progressing until using appropriate machine.

        # Get edge position?

        # Append observations to history stack
        self._tcp_joint_pos_history.appendleft(self.tcp_joint_pos.clone())
        self._tcp_joint_vel_history.appendleft(self.tcp_joint_vel.clone())

# Required for Eureka
@torch.jit.script
def compute_success():
    return