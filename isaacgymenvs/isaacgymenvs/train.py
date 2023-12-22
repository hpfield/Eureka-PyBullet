# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import os
import datetime

import isaacgym

import hydra
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym
import sys 
import shutil
from pathlib import Path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']
    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict

#! Will look in folder './cfg' for a file named 'config.yaml'
@hydra.main(config_name="config", config_path="./cfg")
#! cfg is an object representation of the config.yaml file, as well as any other config files referenced there
def launch_rlg_hydra(cfg: DictConfig):

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs


    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}" #! wandb configuration kept in hydra

    # ensure checkpoints can be specified as relative paths
    #! No checkpoint specified by default
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting() #! Enhance readability of numpy array outputs during debugging

    # sets seed. if seed is -1 will pick a random one
    #! LOCAL_RANK is an environment variable typically used in multi-GPU setups to identify the ID of each process
    rank = int(os.getenv("LOCAL_RANK", "0")) 
    #! Adjusting the seed based on the rank ensures that each process uses a different random seed
    cfg.seed += rank
    #! if cfg.seed is -1, a random seed will be generated
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    #! If the cfg calls for multi-gpu, this instruction is passed into another part of the cfg manually for training
    cfg.train.params.config.multi_gpu = cfg.multi_gpu


    def create_isaacgym_env(**kwargs): #! **kwargs allows for the acceptance of an arbitrary number of keywork args
        #! This is where the new reward is used, because isaacgym makes the env on the fly
        #! it will also incorporate the new reward fn, looking for the output file from eureka.py
        envs = isaacgymenvs.make( 
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg, #! Passing the whole cfg allows the make fn to access any additional params needed
            **kwargs,
        )
        #! Option to capture video
        if cfg.capture_video:
            envs.is_vector_env = True #! env supports vectorized operations (not for pybullet)
            if cfg.test:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0), #! Configure at which steps video recording should occur
                    video_length=cfg.capture_video_len,
                )
            else:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0) and (step > 0),
                    video_length=cfg.capture_video_len,
                )
        return envs

    #! This is registering a new environment configuration. It does not actually call create_isaacgym_env()
    #! Telling the framework "here's how to create an environment of type 'rlgpu'"
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })
    
    # Save the environment code!
    try:
        #! Searches for a local version of the env
        #! From a first glance, this has minor differences to the other cartpole env
        #! e.g. the gt_reward is not saved
        output_file = f"{ROOT_DIR}/tasks/{cfg.task.env.env_name.lower()}.py"
        #! Copy the contents of the existing file to a new file to standardise the file name for future operations
        shutil.copy(output_file, f"env.py")
    except:
        import re
        def camel_to_snake(name): #! Convert from camelCase to snake_case
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(cfg.task.name)}.py"

        shutil.copy(output_file, f"env.py")

    #! add a new type of env to the system
    #! lambda function serves as a factory for creating RLGPUEnvs
    #! RLGPUEnv is a class representing a type of env optimised for running on GPUs
    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train) #! Convert cfg.train to a standard python dictionary
    #! fn defined at top of file
    #! used to modify the configuration variables before training
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    # register new AMP network builder and agent #! Adaptive Motion Prediction
    #! The resulting runner object is what conducts the training
    #! The runner is filled with factories to make algorithms, players, models and networks
    #! The components needed for AMP based continuous control training
    #! The factories are only used to create instances once runner.run() is called
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        #! Algo factory creates algorithm agents
        #! register_builder() registers a method for creating agents of type 'amp_continuous'
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    observers = [RLGPUAlgoObserver()] #! A specific observer for GPU-based training

    #! Specific weights and biases observer is added for recording training
    if cfg.wandb_activate and rank ==0 :

        import wandb
        
        # initialize wandb only once per horovod run (or always for non-horovod runs)
        wandb_observer = WandbAlgoObserver(cfg)
        observers.append(wandb_observer)

    # dump config dict
    exp_date = cfg.train.params.config.name + '-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    experiment_dir = os.path.join('runs', exp_date)
    print("Network Directory:", Path.cwd() / experiment_dir / "nn")
    print("Tensorboard Directory:", Path.cwd() / experiment_dir / "summaries")

    #! Setup experiment directory for logging and dump configuration into yaml file there
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    rlg_config_dict['params']['config']['log_dir'] = exp_date

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict) #! Runner loads training configuration
    runner.reset() #! Initialise the runner in prep for training

    #! Execute the training
    #! runner.run() executes the factories registered in the build_runner function
    #! Would need to examine the run() fn to find what values are passed in to **kwargs
    statistics = runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

    if cfg.wandb_activate and rank == 0:
        wandb.finish() #! Properly finish the wandb session
        
if __name__ == "__main__":
    launch_rlg_hydra()
