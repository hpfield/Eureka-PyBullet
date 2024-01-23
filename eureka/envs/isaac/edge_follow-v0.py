import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask

class EdgeFollow(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        #! All of these will need to be organised into the self.cfg object as in the other VecTask classes
        env_params = {}
        # add environment specific env parameters
        env_params["workframe"] = np.array([0.65, 0.0, 0.035, -np.pi, 0.0, np.pi/2])

        tcp_lims = np.zeros(shape=(6, 2))
        tcp_lims[0, 0], tcp_lims[0, 1] = -0.175, +0.175  # x lims
        tcp_lims[1, 0], tcp_lims[1, 1] = -0.175, +0.175  # y lims
        tcp_lims[2, 0], tcp_lims[2, 1] = -0.1, +0.1  # z lims
        tcp_lims[3, 0], tcp_lims[3, 1] = 0.0, 0.0  # roll lims
        tcp_lims[4, 0], tcp_lims[4, 1] = 0.0, 0.0  # pitch lims
        tcp_lims[5, 0], tcp_lims[5, 1] = -np.pi, np.pi  # yaw lims
        env_params["tcp_lims"] = tcp_lims

        self.env_params = env_params

        robot_arm_params = {}
        # add environment specific robot arm parameters
        robot_arm_params["use_tcp_frame_control"] = False
        robot_arm_params["rest_pose"] = np.array([0.0, 1.374, 0.871,  1.484, -2.758, 4.940, 2.2409, 7.165, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot_arm_params["tcp_link_name"] = "tcp_link"

        self.robot_arm_params = robot_arm_params

        tactile_sensor_params = {}

        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "no_core"
        tactile_sensor_params["dynamics"] = {'stiffness': 50, 'damping': 100, 'friction': 10.0}

        visual_sensor_params = {}

        # add environment specific visual sensor parameters
        visual_sensor_params["dist"] = 0.4
        visual_sensor_params["yaw"] = 90.0
        visual_sensor_params["pitch"] = -25.0
        visual_sensor_params["pos"] = [0.65, 0.0, 0.035]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.1
        visual_sensor_params["far_val"] = 100.0

        #! Below variables are required for VecTask
        # numEnvs in yaml
        self.cfg["env"]["numObservations"] = 128 #? Maybe...
        self.cfg["env"]["numActions"] = len(self.robot_arm_params["control_dofs"]) 
        self.cfg["env"]["numAgents"] = 1 #! Default
        self.cfg["env"]["numStates"] = None
        self.cfg["env"]["controlFrequencyInv"] = None
        # self.cfg["env"]["clipObservations"] in yaml
        # clipActions in yaml
        # enableCameraSensors not required

        #! self.cfg needs to be setup so that VecTask knows how to initialise
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        #! Add code to load embodiment

        # distance from goal to cause termination
        self.termination_dist = 0.01

        # how much penetration of the tip to optimize for
        # randomly vary this on each episode
        self.embed_dist = 0.0035

        #! Add code to load environment and camera
        self.setup_edge()

        #! Add code to setup action and observation space

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self.setup_edge()
        self.load_edge()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def setup_edge(self):
        """
        Defines params for loading/resetting the edge object.
        """
        # define an initial position for the objects (world coords)
        self.edge_pos = [0.65, 0.0, 0.0]
        self.edge_height = 0.035
        self.edge_len = 0.175

    #! Maintains lists (self.envs, self.robots, self.edges, self.goal_indicators) that store references to the created environment instances and their respective assets. 
    def _create_envs(self, num_envs, spacing, num_per_row):
        # Define lower and upper bounds for environment placement
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Get asset file locations
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        robotic_arm_asset_file = self.cfg["env"]["asset"].get("assetFileNameRobot", robotic_arm_asset_file)
        edge_asset_file = self.cfg["env"]["asset"].get("assetFileNameEdge", edge_asset_file)
        goal_indicator_file = self.cfg["env"]["asset"].get("assetFileNameGoalIndicator", goal_indicator_file)

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

        dofs = self.cfg["env"]["numActions"]
        default_stiffness = 400
        default_damping = 80

        dof_stiffness = to_torch([default_stiffness] * dofs, dtype=torch.float, device=self.device)
        dof_damping = to_torch([default_damping] * dofs, dtype=torch.float, device=self.device)

        self.num_dofs = dofs
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robotic_arm_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robotic_arm_asset)
        self.num_edge_bodies = self.gym.get_asset_rigid_body_count(edge_asset)
        self.num_goal_indicator_bodies = self.gym.get_asset_rigid_body_count(goal_indicator_asset)

        print(f'Robot arm count: {self.num_robot_bodies}')
        print(f'Robot arm dofs: {self.num_robot_dofs}')
        print(f'Edge count: {self.num_edge_bodies}')
        print(f'Goal Indicator count: {self.num_goal_indicator_bodies}')

        robot_dof_props = self.gym.get_asset_dof_properties(robotic_arm_asset)

        # Configure robotic arm DoFs
        for i in range(self.num_robot_dofs):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            robot_dof_props['stiffness'][i] = dof_stiffness[i]
            robot_dof_props['damping'][i] = dof_damping[i]

        # Define starting poses for assets #! This might need adjusting
        robot_start_pose = gymapi.Transform()  # Set the position and orientation
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # Adjust as needed #! Not sure starting pos from pybullet, assuming origin
        robot_start_pose.r = gymapi.Quat(1, 0, 0, 0)  # Neutral orientation
        robot_rest_pose = self.robot_arm_params["rest_pose"] #! From tactile
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
        #! Might need to add in the tactile sensor later
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

            # Add edge
            edge_actor = self.gym.create_actor(env_ptr, edge_asset, edge_start_pose, "edge", i, 0, 0)
            self.edges.append(edge_actor)

            # Add goal indicator
            goal_indicator_actor = self.gym.create_actor(env_ptr, goal_indicator_asset, goal_indicator_start_pose, "goal_indicator", i, 0, 0)
            self.goal_indicators.append(goal_indicator_actor)


    def dist_to_goal(self):
        """
        Euclidean distance from the tcp to the goal pose.
        """
        dist = np.linalg.norm(
            np.array(self.cur_tcp_pos_worldframe) - np.array(self.goal_pos_worldframe)
        )
        return dist

    def dist_to_center_edge(self):
        """
        Perpendicular distance from the current tcp to the center edge.
        """
        # use only x/y dont need z
        p1 = self.edge_end_points[0, :2]
        p2 = self.edge_end_points[1, :2]
        p3 = self.cur_tcp_pos_worldframe[:2]

        # calculate perpendicular distance between EE and edge
        dist = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

        return dist

    def get_reward(self):
        """
        Weighted distance between current tool center point and goal pose.
        """
        W_goal = 1.0
        W_edge = 10.0

        goal_dist = self.dist_to_goal()
        edge_dist = self.dist_to_center_edge()

        # sum rewards with multiplicative factors
        reward = -(
            (W_goal * goal_dist)
            + (W_edge * edge_dist)
        )

        return reward
