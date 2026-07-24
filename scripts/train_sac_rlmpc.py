"""DRLによる意思決定 + 5次多項式による経路計画 + MPCによる追従制御 を
組み合わせた階層型パイプラインをSACで学習するスクリプト。

行動空間は [target_y, target_v, planning_time, w_pos, w_yaw, w_vel, w_accel, w_steer, w_steer_rate]
の9次元（意思決定＋MPCコスト重みのみ）。
env内部（MergingEnvRLMPC）で毎ステップ、5次多項式で目標軌道を生成し、
MPC（CasADi/ipopt）で追従制御入力 [acceleration, steering_rate] を計算する。
[acceleration, steering_rate] を直接学習する scripts/train_sac.py と比べ、
毎ステップMPCを解く分だけ学習速度は大幅に遅くなる点に注意。

Usage:
    python -m scripts.train_sac_rlmpc --config config.yaml --timesteps 50000 --save models/sac_rlmpc_merge
"""
import os
import sys
import argparse

# stable_baselines3のtensorboardロギング経由でtensorflowが読み込まれる際に出る
# 無害なoneDNN/abslログを抑制する（tensorflow importより前に設定する必要がある）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ensure project root is on sys.path for imports when executing as script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.merging_env_rlmpc import MergingEnvRLMPC
from utils.config_loader import MasterConfig
from scripts.train_sac import MergingEnvMetricsCallback


def make_env(config_path):
    def _init():
        cfg = MasterConfig.from_yaml(config_path)
        env = MergingEnvRLMPC(cfg)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--save", type=str, default="models/sac_rlmpc_merge")
    parser.add_argument("--tensorboard", type=str, default="runs/sac_rlmpc_merge")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None

    if args.tensorboard:
        os.makedirs(args.tensorboard, exist_ok=True)
        tb_log = args.tensorboard
    else:
        tb_log = None

    env = DummyVecEnv([make_env(args.config)])
    metrics_callback = MergingEnvMetricsCallback()

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
    model.learn(total_timesteps=args.timesteps, callback=metrics_callback)
    model.save(args.save)


if __name__ == "__main__":
    main()
