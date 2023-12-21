import numpy as np
import os
import torch
import pybullet as p
import pybullet_data

class Cartpole:

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500
        self.rl_device = rl_device
        self.sim_device = sim_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless
        self.virtual_screen_capture = virtual_screen_capture
        self.force_render = force_render

        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.rl_device)

        self.create_sim()

    def create_sim(self):
        if self.headless:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(asset_root, self.cfg["env"]["asset"].get("assetRoot", ""))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
        asset_path = os.path.join(asset_root, asset_file)

        self.cartpole_id = p.loadURDF(asset_path, basePosition=[0, 0, 1])

    def compute_observations(self):
        pos, orn = p.getBasePositionAndOrientation(self.cartpole_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.cartpole_id)
        joint_states = p.getJointStates(self.cartpole_id, range(p.getNumJoints(self.cartpole_id)))

        positions = [j[0] for j in joint_states]
        velocities = [j[1] for j in joint_states]

        obs = np.array(positions + velocities + list(linear_vel) + list(angular_vel))
        return torch.tensor(obs, dtype=torch.float32, device=self.rl_device)

    def compute_reward(self, observations):
        # Define your reward function based on observations
        # Example:
        pole_angle = observations[2]
        reward = 1.0 - pole_angle ** 2
        return reward

    def reset(self):
        p.resetBasePositionAndOrientation(self.cartpole_id, [0, 0, 1], [0, 0, 0, 1])
        for i in range(p.getNumJoints(self.cartpole_id)):
            p.resetJointState(self.cartpole_id, i, 0, 0)

    def step(self, action):
        # Apply action
        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=action * self.max_push_effort)
        
        # Step simulation
        p.stepSimulation()

        # Compute observations and reward
        observations = self.compute_observations()
        reward = self.compute_reward(observations)

        return observations, reward

# Example usage
cfg = {
    "env": {
        "resetDist": 1.0,
        "maxEffort": 10.0,
    }
}

cartpole = Cartpole(cfg, "cpu", "cpu", 0, True, False, False)
observations = cartpole.compute_observations()
reward = cartpole.compute_reward(observations)
