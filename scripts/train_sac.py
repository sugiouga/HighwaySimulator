"""Train SAC on the merge environment.

Usage:
    python -m scripts.train_sac --config config.yaml --timesteps 200000 --save models/sac_merge
"""
import os
import sys
import argparse
import numpy as np

# ensure project root is on sys.path for imports when executing as script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from env.merging_env import MergingEnv
from utils.config_loader import MasterConfig


def make_env(config_path):
    def _init():
        cfg = MasterConfig.from_yaml(config_path)
        env = MergingEnv(cfg)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--save", type=str, default="models/sac_merge")
    parser.add_argument("--tensorboard", type=str, default="runs/sac_merge")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None

    # Create tensorboard dir only if requested; allow empty string to disable TB logging
    if args.tensorboard:
        os.makedirs(args.tensorboard, exist_ok=True)
        tb_log = args.tensorboard
    else:
        tb_log = None

    env = DummyVecEnv([make_env(args.config)])

    sac_config = MasterConfig.from_yaml(args.config).sac
    sac_kwargs = {
        "learning_rate": sac_config.learning_rate,
        "buffer_size": sac_config.buffer_size,
        "learning_starts": sac_config.learning_starts,
        "batch_size": sac_config.batch_size,
        "tau": sac_config.tau,
        "gamma": sac_config.gamma,
        "train_freq": sac_config.train_freq,
        "gradient_steps": sac_config.gradient_steps,
        "ent_coef": sac_config.ent_coef,
        "target_update_interval": sac_config.target_update_interval,
        "device": sac_config.device,
        "policy_kwargs": sac_config.policy_kwargs,
    }

    model = SAC(sac_config.policy, env, verbose=1, tensorboard_log=tb_log, **sac_kwargs)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)

    # simple evaluation rollout
    vec_env = env
    obs = vec_env.reset()
    total_reward = 0.0
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += np.asarray(rewards).sum()
        if dones.any():
            break

    print(f"Sample rollout reward: {total_reward}")


if __name__ == "__main__":
    main()
