import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask

class EdgeFollow(VecTask):
    def __init__(self, cfg, sim_device, headless):
        self.sim_device = sim_device
        self.headless = headless
        cfg["device_type"] = "cuda"
        cfg["device_id"] = 0
        cfg["rl_device"] = "cuda:0"
        headless = False # Start by rendering and then change once implementation verified
        cfg["physics_engine"] = "physx"
        env = {}
        env["numEnvs"] = 5 # Alter as needed
        env["numObservations"] = 0 #! Requires embodiment in PyBullet (customisable param)
        env["numActions"] = 2 # From PyBullet variables dump

        #Optional env params
        env["numAgents"] = 1 # Default
        env["numStates"] = None # Don't think is required
        env["controlFrequencyInv"] = self.calculate_controlFrequencyInv() # Taken from BaeTactileEnv
        env["clipObservations"] = None #! Pybullet obs space depends on command line arg
        env["clipActions"] = None  #! PyBullet has Min/Max actions at -1.0 and 1.0
        env["enableCameraSensors"] = True # Assumed True

        cfg["env"] = env
        self.cfg = cfg

        super().__init__(self.cfg)

    # From BaseTactileEnv, using the calculation for _velocity_action_repeat
    def calculate_controlFrequencyInv(self):
        sim_time_step = 1.0 / 240.0
        control_rate = 1.0 / 10.0
        return int(np.floor(control_rate / sim_time_step))

    # Required for VecTask
    def create_sim(self):
        return
    
    # Required for VecTask
    def pre_physics_step(self, actions):
        return
    
    # Required for VecTask
    def post_physics_step(self):
        return

