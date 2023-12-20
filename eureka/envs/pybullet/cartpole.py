import pybullet as p
import numpy as np
import torch

class CartpolePyBullet:

    def __init__(self):

        # Environment parameters
        #! Some params were not clear in isaacgym, but matched ones that could be read
        self.reset_dist = 2.4 
        self.max_push_force = 10.0  
        self.max_episode_length = 500

        # PyBullet setup
        self.physicsClient = p.connect(p.DIRECT)
        p.setGravity(0,0,-9.8)

        # Create cartpole #! Joint info handles are for later use
        self.cartpole = p.loadURDF("cartpole.urdf")  
        self.pole = p.getJointInfo(self.cartpole, 1)[0]
        self.cart_joint = p.getJointInfo(self.cartpole, 0)[0] 

        # Joint indices 
        self.cart_joint_index = 0
        self.pole_joint_index = 1

        # Set joint control mode
        p.setJointMotorControl2(self.cartpole, self.cart_joint_index, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, self.pole_joint_index, p.VELOCITY_CONTROL, force=0)

        # Additional metrics to match isaacgym
        self.consecutive_successes = 0
        self.progress_buf = 0

    def get_observation(self):
        cart_pos, cart_vel,_ = p.getJointState(self.cartpole, self.cart_joint_index)[:3]
        pole_angle = p.getJointState(self.cartpole, self.pole_joint_index)[0] 
        pole_vel,_ = p.getJointState(self.cartpole, self.pole_joint_index)[1:]
        observation = [cart_pos, cart_vel, pole_angle, pole_vel]
        return np.array(observation)

    def apply_action(self, action): 
        p.setJointMotorControl2(self.cartpole, self.cart_joint_index, p.TORQUE_CONTROL, force=action) 

    def step(self, action):
        
        # Apply action force
        self.apply_action(action[0] * self.max_push_force)

        # Get observation and reward
        obs = self.get_observation()  
        reward, done, self.consecutive_successes, self.progress_buf = self.get_reward(obs)

        # Simulate one step
        p.stepSimulation()

        # Return step info
        return obs, reward, done

    # Reward function from isaacgym reference
    def get_reward(self, obs):

        pole_angle = obs[2] 
        pole_vel = obs[3]
        cart_vel = obs[1]
        cart_pos = obs[0]

        self.gt_rew_buf, self.reset_buf, self.consecutive_successes, self.progress_buf = compute_success(
             pole_angle, pole_vel, cart_vel, cart_pos,
             self.reset_dist, self.reset_buf, self.consecutive_successes, self.progress_buf, self.max_episode_length
        )

        done = bool(abs(cart_pos) > self.reset_dist)
        done = done or bool(abs(pole_angle) > .2) 

        #! Reward is not returned in isaacgym. The important thing is how the metrics are logged in training.
        #! Keeping the process in line with pybullet standards might make the training easier later
        return self.gt_rew_buf, done, self.consecutive_successes, self.progress_buf   

    def reset(self):
        # Reset pybullet simulation
        p.resetSimulation(self.physicsClient)
        p.setGravity(0,0,-9.8)
        return self.get_observation()
    
@torch.jit.script
def compute_success(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, consecutive_successes, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]

    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    if reset.sum() > 0:
        consecutive_successes = (progress_buf.float() * reset).sum() / reset.sum()
    else:
        consecutive_successes = torch.zeros_like(consecutive_successes).mean()
    return reward, reset, consecutive_successes