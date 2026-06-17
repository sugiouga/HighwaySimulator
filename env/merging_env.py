import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.config_loader import MasterConfig
from utils.safety_checker import SafetyChecker
from manager.traffic_manager import TrafficManager
from manager.road_network import RoadNetwork
from observer.visualizer import Visualizer
from observer.metrics_observer import MetricsObserver
from observer.termination_observer import TerminationObserver
from observer.truncation_observer import TruncationObserver
from observer.jerk_observer import JerkObserver

class MergingEnv(gym.Env):
    """高速道路シミュレーションのGym環境クラス"""

    def __init__(self, config: MasterConfig):
        super(MergingEnv, self).__init__()
        self.config = config

        # 状態空間と行動空間の定義
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(36,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([self.config.vehicle.min_acceleration, -self.config.vehicle.max_steering_rate], dtype=np.float32),
            high=np.array([self.config.vehicle.max_acceleration, self.config.vehicle.max_steering_rate], dtype=np.float32),
            dtype=np.float32,
        )

        self.ego_vehicle_id = "ego_vehicle"
        self.road_network = RoadNetwork(config)
        self.safety_checker = SafetyChecker()
        self.visualizer = Visualizer(config) if config.visualization.enable else None
        self.metrics_observer = MetricsObserver(config)
        self.termination_observer = TerminationObserver(config)
        self.truncation_observer = TruncationObserver(config)
        self.jerk_observer = JerkObserver(config)

        # 観測正規化のための範囲を初期化
        all_waypoints = [wp for lane in self.config.road_network.lanes for wp in lane["waypoints"]]
        xs = [wp[0] for wp in all_waypoints]
        ys = [wp[1] for wp in all_waypoints]
        self._x_min = float(min(xs))
        self._x_max = float(max(xs))
        self._y_min = float(min(ys) - self.config.road_network.lane_width)
        self._y_max = float(max(ys) + self.config.road_network.lane_width)

        self._v_min = float(self.config.vehicle.min_velocity)
        self._v_max = float(self.config.vehicle.max_velocity)
        self._a_min = float(self.config.vehicle.min_acceleration)
        self._a_max = float(self.config.vehicle.max_acceleration)
        self._steer_abs_max = float(np.deg2rad(self.config.vehicle.max_steering_angle))

        sensor_front_max = max(float(p.sensor_range[0]) for p in self.config.policies.values())
        sensor_side_max = max(float(p.sensor_range[1]) for p in self.config.policies.values())
        self._rel_x_abs_max = sensor_front_max
        self._rel_y_abs_max = sensor_side_max
        self._rel_v_abs_max = 2.0 * self._v_max

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Accepts `seed` and `options` for Gymnasium compatibility and returns (observation, info).
        """
        if seed is not None:
            try:
                np.random.seed(seed)
            except Exception:
                pass

        self.road_network.reset()
        self.traffic_manager = TrafficManager(self.road_network, self.config, dt=self.config.simulation.time_step)

        # ウォームアップステップを実行して、初期状態を安定させる
        warmup_steps = int(self.config.simulation.warmup_time / self.config.simulation.time_step)
        for _ in range(warmup_steps):
            self.traffic_manager.step()

        # ego車両の初期化
        lane_id = "merge_1"
        lane = self.road_network.get_lane(lane_id)
        x, y = lane.get_cartesian(5.0, 0.0)
        self.ego_vehicle = self.traffic_manager.vehicle_manager.factory.create_vehicle(
            vehicle_id=self.ego_vehicle_id,
            lane_id=lane_id,
            init_state=[x, y, 0.0, 6.0, 0.0],
            policy_id="DRL_Agent",
            is_ego=True
        )
        self.traffic_manager.add_vehicle(self.ego_vehicle)

        # add observers only if present
        if self.visualizer is not None:
            self.traffic_manager.add_observer(self.visualizer)
        self.traffic_manager.add_observer(self.metrics_observer)
        self.traffic_manager.add_observer(self.termination_observer)
        self.traffic_manager.add_observer(self.truncation_observer)
        self.traffic_manager.add_observer(self.jerk_observer)

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # ego車両の行動を更新
        self.ego_vehicle.set_action(action)

        # シミュレーションを1ステップ進める
        self.traffic_manager.step()

        # 観測、報酬、終了条件の計算
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self._check_truncation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """観測の取得（全要素を[-1, 1]に正規化）"""
        # ego車両の状態を正規化して観測ベクトルへ追加
        ego_state = self.ego_vehicle.state
        ego_action_raw = getattr(self.ego_vehicle, "current_action", 0.0)

        if np.isscalar(ego_action_raw):
            ego_action = self._normalize_discrete_action(float(ego_action_raw))
        else:
            action_array = np.asarray(ego_action_raw, dtype=np.float32).reshape(-1)
            action_scalar = float(action_array[0]) if action_array.size > 0 else 0.0
            ego_action = self._normalize(action_scalar, self._a_min, self._a_max)

        observation = [
            self._normalize(float(ego_state[0]), self._x_min, self._x_max),
            self._normalize(float(ego_state[1]), self._y_min, self._y_max),
            self._normalize(float(ego_state[2]), -np.pi, np.pi),
            self._normalize(float(ego_state[3]), self._v_min, self._v_max),
            self._normalize(float(ego_state[4]), -self._steer_abs_max, self._steer_abs_max),
            ego_action,
        ]

        # 周囲車両の相対状態を正規化して追加（6方向 × 5特徴量 = 30次元）
        perception = self.traffic_manager.perceptions[self.ego_vehicle_id]
        surrounding_vehicles = perception.observe(self.ego_vehicle, self.traffic_manager.vehicles, self.road_network)
        for key in ['left_front', 'left_rear', 'right_front', 'right_rear', 'front', 'rear']:
            vehicle = surrounding_vehicles[key]
            if vehicle is not None:
                relative_position = perception.calculate_relative_position(self.ego_vehicle, vehicle)
                relative_velocity = perception.calculate_relative_velocity(self.ego_vehicle, vehicle)
                acceleration = vehicle.current_action[0] if hasattr(vehicle, 'current_action') else 0.0
                observation.extend([
                    self._normalize(float(relative_position[0]), -self._rel_x_abs_max, self._rel_x_abs_max),
                    self._normalize(float(relative_position[1]), -self._rel_y_abs_max, self._rel_y_abs_max),
                    self._normalize(float(relative_velocity[0]), -self._rel_v_abs_max, self._rel_v_abs_max),
                    self._normalize(float(relative_velocity[1]), -self._rel_v_abs_max, self._rel_v_abs_max),
                    self._normalize(float(acceleration), self._a_min, self._a_max),
                ])
            else:
                # 車両が存在しない場合は0埋め（正規化空間の中心）
                observation.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        observation = np.asarray(observation, dtype=np.float32)

        # 念のため固定次元に整える
        expected_dim = int(self.observation_space.shape[0])
        if observation.size < expected_dim:
            observation = np.pad(observation, (0, expected_dim - observation.size), mode="constant")
        elif observation.size > expected_dim:
            observation = observation[:expected_dim]

        return np.clip(observation, -1.0, 1.0)

    @staticmethod
    def _normalize(value, min_value, max_value):
        """[min, max]を[-1, 1]に線形正規化"""
        value = float(np.nan_to_num(value, nan=0.0, posinf=max_value, neginf=min_value))
        min_value = float(min_value)
        max_value = float(max_value)
        if max_value <= min_value:
            return 0.0
        normalized = 2.0 * (value - min_value) / (max_value - min_value) - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    def _normalize_discrete_action(self, action):
        """離散行動を[-1, 1]へ正規化"""
        n = int(getattr(self.action_space, "n", 0))
        if n <= 1:
            return 0.0
        action = float(np.clip(action, 0, n - 1))
        return float(2.0 * action / (n - 1) - 1.0)

    def _calculate_reward(self):
        """報酬関数の実装（例: 安全性、快適性、効率性などを考慮）"""
        reward = 0.0

        # 合流の成功とするための報酬
        if self.termination_observer.get_info().get("termination_reason") == "goal_reached":
            reward += self.config.reward.success_reward

        # 衝突ペナルティ
        if self.termination_observer.get_info().get("termination_reason") == "collision":
            reward +=  self.config.reward.collision_penalty

        # 車線離脱ペナルティ
        if self.termination_observer.get_info().get("termination_reason") == "lane_deviation":
            reward +=  self.config.reward.lane_deviation_penalty

        # 時間切れペナルティ
        if self.truncation_observer.get_info().get("termination_reason") == "timeout":
            reward +=  self.config.reward.timeout_penalty

        # y座標に基づく報酬（目標位置に近いほど高い報酬）
        if self.config.reward.y_position_reward.enabled:
            weight = self.config.reward.y_position_reward.weight
            target = self.config.reward.y_position_reward.target
            reward += weight / (1 + (self.ego_vehicle.state[1] - target) ** 2)

        # 目標速度への近さに基づく報酬
        if self.config.reward.target_velocity_reward.enabled:
            weight = self.config.reward.target_velocity_reward.weight
            target = self.config.reward.target_velocity_reward.target
            std = self.config.reward.target_velocity_reward.std
            # ガウス分布に基づく報酬
            reward += weight * np.exp(-0.5 * ((self.ego_vehicle.state[3] - target) / std) ** 2)

        # 追従車両の減速度に対するペナルティ
        if self.config.reward.following_vehicle_deceleration_penalty.enabled:
            weight = self.config.reward.following_vehicle_deceleration_penalty.weight
            for vehicle in self.traffic_manager.vehicles:
                if vehicle.lane_id == self.ego_vehicle.lane_id and vehicle.state[0] < self.ego_vehicle.state[0]:  # 追従車両
                    vehicle_deceleration = -min(vehicle.current_action[0], 0)  # 追従車両の減速量
                    reward += weight * vehicle_deceleration**2

        # ジャーク（加速度の変化率）に対するペナルティ
        if self.config.reward.jerk_penalty.enabled:
            weight = self.config.reward.jerk_penalty.weight
            jerk = self.jerk_observer.get_current_jerk(self.ego_vehicle_id)
            if jerk is not None:
                reward += weight * jerk**2

        return reward

    def _check_termination(self):
        # 衝突と車線離脱、目標位置への到達のチェック
        if self.termination_observer.get_info().get("termination_reason") in ["collision", "lane_deviation", "goal_reached"]:
            return True
        return False

    def _check_truncation(self):
        # タイムアウトのチェック
        if self.truncation_observer.get_info().get("termination_reason") == "timeout":
            return True
        return False

    def _get_info(self):
        # 追加の情報を返す（例: 合流の成功、衝突の有無など）
        info = {}
        # termination info
        term_info = self.termination_observer.get_info()
        if term_info:
            info.update(term_info)

        # truncation info
        trunc_info = self.truncation_observer.get_info()
        if trunc_info:
            info.update(trunc_info)

        # metrics
        metrics = self.metrics_observer.get_current_metrics(self.ego_vehicle_id)
        if metrics:
            info.setdefault('metrics', metrics)

        return info