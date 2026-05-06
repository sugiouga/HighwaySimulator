from abc import ABC, abstractmethod
import numpy as np

# 車両の基底クラス
class BaseVehicle(ABC):
    def __init__(self,
                 vehicle_id: str,
                 lane_id: str,
                 init_state,
                 policy,
                 vehicle_config,
                 policy_config
                 ):
        """
        Args
        - vehicle_id (str): 車両ID
        - lane_id (str): 車両が現在いる車線のID
        - init_state (list): 初期状態 [x, y, yaw, velocity, steering_angle]
        - policy: 車両のポリシー
        - vehicle_config: 車両の物理パラメータ
        - policy_config: 車両のポリシーに関する設定情報
        """

        self.vehicle_id = vehicle_id
        self.lane_id = lane_id
        self.state = np.array(init_state, dtype=np.float64)
        self.policy = policy

        # 車両の物理パラメータ
        self.mass = vehicle_config.mass
        self.length = vehicle_config.length
        self.width = vehicle_config.width
        self.color = policy_config.color
        self.min_velocity = vehicle_config.min_velocity
        self.max_velocity = vehicle_config.max_velocity
        self.min_acceleration = vehicle_config.min_acceleration
        self.max_acceleration = vehicle_config.max_acceleration
        self.max_steering_angle = vehicle_config.max_steering_angle
        self.max_steering_rate = vehicle_config.max_steering_rate

    @property
    def x(self):
        """車両のx座標を返すプロパティ"""
        return self.state[0]
    @property
    def y(self):
        """車両のy座標を返すプロパティ"""
        return self.state[1]
    @property
    def yaw(self):
        """車両のyaw角を返すプロパティ"""
        return self.state[2]
    @property
    def velocity(self):
        """車両の速度を返すプロパティ"""
        return self.state[3]
    @property
    def steering_angle(self):
        """車両のステアリング角を返すプロパティ"""
        return self.state[4]
    @property
    def acceleration(self):
        """車両の加速度を返すプロパティ"""
        return self.current_action[0]
    @property
    def steering_rate(self):
        """車両のステアリングレートを返すプロパティ"""
        return self.current_action[1]

    @abstractmethod
    def get_dynamics(self, state, action):
        """車両の運動方程式を定義する抽象メソッド
        Args
        - state: 車両の状態 [x, y, yaw, velocity, steering_angle]
        - action: 行動入力 [acceleration, steering_rate]
        """
        pass

    def plan(self, environment_info):
        """車両の制御入力を計算するメソッド
        Args
        - environment_info: 環境情報（例: 他の車両の状態、道路情報など）
        """
        self.current_action = self.policy.action(self.state, environment_info)

    def update_state(self, dt, integrator_fn = None):
        """車両の状態を更新するメソッド
        Args
        - dt: タイムステップ
        - integrator_fn: 状態更新のための数値積分関数
        """
        if integrator_fn is None:
            integrator_fn = self.rk4_integrator
        self.state = integrator_fn(self.get_dynamics, self.state, self.current_action, dt)

    def update_lane_id(self, lane):
        """車両のlane_idを更新するメソッド
        Args
        - lane: 車両が現在いるLaneオブジェクト
        """
        self.lane_id = lane.lane_id

    def get_corners(self):
        """車両の四隅の座標を計算するメソッド
        Returns
        - corners: 車両の四隅の座標 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        x, y, yaw, _, _ = self.state
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 車両の四隅のオフセットを計算
        half_length = self.length / 2
        half_width = self.width / 2

        corners = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ])

        # 四隅の座標を計算
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_corners = corners @ rotation_matrix.T
        corners_coordinates = rotated_corners + np.array([x, y])

        return corners_coordinates

    @staticmethod
    def euler_integrator(dynamics_fn, state, action, dt):
        """単純なオイラー積分器
        Args
        - dynamics_fn: 車両の運動方程式を定義する関数
        - state: 現在の状態
        - action: 行動入力
        - dt: タイムステップ
        Returns
        - new_state: 更新された状態
        """
        derivatives = dynamics_fn(state, action)
        new_state = state + derivatives * dt
        return new_state

    @staticmethod
    def rk4_integrator(dynamics_fn, state, action, dt):
        """4次のルンゲクッタ積分器
        Args
        - dynamics_fn: 車両の運動方程式を定義する関数
        - state: 現在の状態
        - action: 行動入力
        - dt: タイムステップ
        Returns
        - new_state: 更新された状態
        """
        k1 = dynamics_fn(state, action)
        k2 = dynamics_fn(state + 0.5 * dt * k1, action)
        k3 = dynamics_fn(state + 0.5 * dt * k2, action)
        k4 = dynamics_fn(state + dt * k3, action)

        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state