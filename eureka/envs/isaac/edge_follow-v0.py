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

        rest_poses_dict = {
            "ur5":  np.array([0.0, 0.315, -1.401, -2.401, -0.908, 1.570, 3.461, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "franka_panda": np.array([0.0, 1.374, 0.871,  1.484, -2.758, 4.940, 2.2409, 7.165, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "kuka_iiwa": np.array([0.0, 0.559, 0.777, 2.501, 2.251, -2.012, 0.481, 3.704, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "cr3":  np.array([0.0, -0.3644, 0.047, 1.997, -0.473, -1.571, -0.361, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "mg400":  np.array([0.0, 3.139, 0.772,  0.100, -0.872, -1.571, 0.0, 0.0, 0.0, 0.0, 0.772, -0.772, 0.873, 0.0]),
        }

        robot_arm_params = {}
        # add environment specific robot arm parameters
        robot_arm_params["use_tcp_frame_control"] = False
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"

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

    def setup_edge(self):
        """
        Defines params for loading/resetting the edge object.
        """
        # define an initial position for the objects (world coords)
        self.edge_pos = [0.65, 0.0, 0.0]
        self.edge_height = 0.035
        self.edge_len = 0.175

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
