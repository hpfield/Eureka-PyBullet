class Cartpole:
    """Rest of the environment definition omitted."""
    def get_observation(self):
        cart_pos, cart_vel,_ = p.getJointState(self.cartpole, self.cart_joint_index)[:3]
        pole_angle = p.getJointState(self.cartpole, self.pole_joint_index)[0] 
        pole_vel,_ = p.getJointState(self.cartpole, self.pole_joint_index)[1:]
        observation = [cart_pos, cart_vel, pole_angle, pole_vel]
        return np.array(observation)