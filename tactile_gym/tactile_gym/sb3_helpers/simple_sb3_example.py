from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import tactile_gym.envs
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor
#! Adding
import os

if __name__ == "__main__":

    algo_name = 'ppo'
    # algo_name = 'sac'

    # show gui can only be enabled for n_envs = 1
    # if using image observation SubprocVecEnv is needed to replace DummyVecEnv
    # as pybullet EGL rendering requires separate processes to avoid silent
    # rendering issues.
    seed = 1
    n_envs = 1
    show_gui = False

    env_id = "edge_follow-v0"
    # env_id = "surface_follow-v0"
    # env_id = "surface_follow-v1"
    # env_id = "object_roll-v0"
    # env_id = "object_push-v0"
    # env_id = "object_balance-v0"

    env_args, rl_params, algo_params = import_parameters(env_id, algo_name)
    env_args['env_params']['show_gui'] = show_gui

    env = make_vec_env(
        env_id,
        env_kwargs=env_args,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )

    algo_params = {
        "policy_kwargs": {
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "cnn_output_dim": 128,
                "mlp_extractor_net_arch": [64, 64],
            },
            "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
        },
    }

    if algo_name == "ppo":
        model = PPO("MultiInputPolicy", env, **algo_params, verbose=1, n_epochs=3)

    elif algo_name == "sac":
        model = SAC("MultiInputPolicy", env, **algo_params, verbose=1)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Log to file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    file_handler = logging.FileHandler(current_file_directory + '/training.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log to stdout too
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Train...

    model.learn(total_timesteps=100000) 

    n_eval_episodes = 10
    for i in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

    env.close()

    # Save the model
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    model.save(current_file_directory + "/../../saved_models/edge_follow-v0/ppo/ppo_edge_follow")
