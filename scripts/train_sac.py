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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from env.merging_env import MergingEnv
from utils.config_loader import MasterConfig


def make_env(config_path):
    def _init():
        cfg = MasterConfig.from_yaml(config_path)
        env = MergingEnv(cfg)
        env = Monitor(env)
        return env
    return _init

class MergingEnvMetricsCallback(BaseCallback):
    """
    合流環境（MergingEnv）のカスタム指標（衝突率、成功率、平均速度など）を
    TensorBoard の `episode_metrics/` カテゴリに記録するコールバック。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # エピソードごとの統計を一時保存するバッファ
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_lane_deviations = []
        self.episode_timeouts = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # DummyVecEnv から info 辞書を取得 (環境が複数ある場合は配列になるため [0] を指定)
        info = self.locals["infos"][0]

        # エピソードが終了したタイミング（Monitorラッパー経由で 'episode' が入る）
        if "episode" in info:
            self.episode_count += 1
            success = 0
            collision = 0
            lane_deviation = 0
            timeout = 0

            # 環境側（MergingEnv）の step() 内の info に用意されていると仮定する変数
            if "termination_reason" in info:
                termination_reason = info["termination_reason"]
                if termination_reason == "goal_reached":
                    success = 1

                elif termination_reason == "collision":
                    collision = 1

                elif termination_reason == "lane_deviation":
                    lane_deviation = 1


            elif "truncation_reason" in info:
                truncation_reason = info["truncation_reason"]
                if truncation_reason == "timeout":
                    timeout = 1

            # エピソード統計をバッファに追加
            self.episode_successes.append(success)
            self.episode_collisions.append(collision)
            self.episode_lane_deviations.append(lane_deviation)
            self.episode_timeouts.append(timeout)

        # 一定エピソードごとに TensorBoard に平均値を書き出し
        if self.episode_count % 100 == 0:
            if len(self.episode_successes) > 0:
                success_rate = np.mean(self.episode_successes)
                self.logger.record("episode_metrics/success_rate", success_rate)
                self.episode_successes.clear()
            if len(self.episode_collisions) > 0:
                collision_rate = np.mean(self.episode_collisions)
                self.logger.record("episode_metrics/collision_rate", collision_rate)
                self.episode_collisions.clear()
            if len(self.episode_lane_deviations) > 0:
                lane_deviation_rate = np.mean(self.episode_lane_deviations)
                self.logger.record("episode_metrics/lane_deviation_rate", lane_deviation_rate)
                self.episode_lane_deviations.clear()
            if len(self.episode_timeouts) > 0:
                timeout_rate = np.mean(self.episode_timeouts)
                self.logger.record("episode_metrics/timeout_rate", timeout_rate)
                self.episode_timeouts.clear()

        self.logger.dump(step=self.num_timesteps)

        return True

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

