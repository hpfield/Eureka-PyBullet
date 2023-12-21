import pybullet as p
import numpy as np

class CartpolePyBullet:

    def __init__(self):

        # Environment parameters
        #! Taking param vals from isaacgymenvs/isaacgymenvs/cfg/task/Cartpole.yaml
        self.reset_dist = 3.0
        self.max_push_effort = 400.0
        self.max_episode_length = 500

        #! New timestep param
        self.time_step = 0.01

        # PyBullet setup
        self.client_id = p.connect(p.DIRECT) #! Functional equivalent of create_sim
        p.setGravity(0,0,-9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        # Create cartpole #! Joint info handles are for later use
        #! For now, decided not to load a plane.urdf as don't have one
        self.cartpole = p.loadURDF("cartpole.urdf", [0,0,0], physicsClientId=self.client_id)  #! Copied file from isaac, placed in this dir
        self.pole = p.getJointInfo(self.cartpole, 1)[0]
        self.cart_joint = p.getJointInfo(self.cartpole, 0)[0] 

        # Joint indices 
        self.cart_joint_index = 0
        self.pole_joint_index = 1

        # Set joint control mode
        p.setJointMotorControl2(self.cartpole, self.cart_joint_index, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, self.pole_joint_index, p.VELOCITY_CONTROL, force=0)

        # Additional metrics 
        self.consecutive_successes = np.zeros(1, dtype=float)
        self.progress = 0

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
        # Extract observations
        pole_angle = obs[2]  
        pole_vel = obs[3]   
        cart_vel = obs[1]    
        cart_pos = obs[0]    

        # Increment the progress of the episode
        self.progress += 1 

        # Use compute_success to compute the reward, reset condition, and success status
        reward, reset, success = compute_success(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.progress, self.max_episode_length
        )

        # If the episode is reset, reset the progress counter
        if reset:
            self.progress = 0

        return reward, reset, success

    def reset(self):
        # Reset pybullet simulation
        p.resetSimulation(self.physicsClient)
        p.setGravity(0,0,-9.8)
        return self.get_observation()
    

def compute_success(pole_angle, pole_vel, cart_vel, cart_pos, reset_dist, progress, max_episode_length):
    # Calculate the reward
    reward = 1.0 - pole_angle**2 - 0.01 * abs(cart_vel) - 0.005 * abs(pole_vel)

    # Check for failure condition
    failure = abs(cart_pos) > reset_dist or abs(pole_angle) > np.pi / 2
    if failure:
        reward -= 2.0

    # Determine if the episode should be reset (either failure or max length reached)
    reset = failure or progress >= max_episode_length

    # Determine if the attempt was successful
    # A success could be defined as reaching max_episode_length without failure
    success = progress >= max_episode_length and not failure

    return reward, reset, success