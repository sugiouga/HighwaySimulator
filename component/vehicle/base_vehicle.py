from abc import ABC, abstractmethod
import numpy as np

# 車両の基底クラス
class BaseVehicle(ABC):
    def __init__(self,
                 vehicle_id: int,
                 init_state,
                 controller,
                 vehicle_config):
        """
        Args
        - vehicle_id (int): 車両ID
        - init_state (list): 初期状態 [s, d, yaw, velocity, steering_angle]
        - controller: 車両の制御器
        - vehicle_config: 車両の物理パラメータ
        """

        self.vehicle_id = vehicle_id
        self.state = np.array(init_state, dtype=np.float64)
        self.controller = controller

        # 車両の物理パラメータ
        self.mass = vehicle_config.mass
        self.length = vehicle_config.length
        self.width = vehicle_config.width
        self.min_velocity = vehicle_config.min_velocity
        self.max_velocity = vehicle_config.max_velocity
        self.min_acceleration = vehicle_config.min_acceleration
        self.max_acceleration = vehicle_config.max_acceleration
        self.max_steering_angle = vehicle_config.max_steering_angle
        self.max_steering_rate = vehicle_config.max_steering_rate

    @property
    def s(self): return self.state
    @property
    def d(self): return self.state
    @property
    def yaw(self): return self.state
    @property
    def velocity(self): return self.state
    @property
    def steering_angle(self): return self.state

    @abstractmethod
    def get_dynamics(self, state, control_input):
        """車両の運動方程式を定義する抽象メソッド
        Args
        - state: 車両の状態 [s, d, yaw, velocity, steering_angle]
        - control_input: 制御入力 [acceleration, steering_rate]
        """
        pass

    def plan(self, environment_info):
        """車両の制御入力を計算するメソッド
        Args
        - environment_info: 環境情報（例: 他の車両の状態、道路情報など）
        """
        self.current_action = self.controller.compute_control(self.state, environment_info)

    def update_state(self, dt, integrator_fn = None):
        """車両の状態を更新するメソッド
        Args
        - dt: タイムステップ
        - integrator_fn: 状態更新のための数値積分関数
        """
        if integrator_fn is None:
            integrator_fn = self.rk4_integrator
        self.state = integrator_fn(self.get_dynamics, self.state, self.current_action, dt)

    def get_corners(self):
        """車両の四隅の座標を計算するメソッド
        Returns
        - corners: 車両の四隅の座標 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        s, d, yaw, _, _ = self.state
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 車両の中心位置を計算
        center_x = s * cos_yaw - d * sin_yaw
        center_y = s * sin_yaw + d * cos_yaw

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
        corners_coordinates = rotated_corners + np.array([center_x, center_y])

        return corners_coordinates

    def euler_integrator(dynamics_fn, state, control_input, dt):
        """単純なオイラー積分器
        Args
        - dynamics_fn: 車両の運動方程式を定義する関数
        - state: 現在の状態
        - control_input: 制御入力
        - dt: タイムステップ
        Returns
        - new_state: 更新された状態
        """
        derivatives = dynamics_fn(state, control_input)
        new_state = state + derivatives * dt
        return new_state

    def rk4_integrator(dynamics_fn, state, control_input, dt):
        """4次のルンゲクッタ積分器
        Args
        - dynamics_fn: 車両の運動方程式を定義する関数
        - state: 現在の状態
        - control_input: 制御入力
        - dt: タイムステップ
        Returns
        - new_state: 更新された状態
        """
        k1 = dynamics_fn(state, control_input)
        k2 = dynamics_fn(state + 0.5 * dt * k1, control_input)
        k3 = dynamics_fn(state + 0.5 * dt * k2, control_input)
        k4 = dynamics_fn(state + dt * k3, control_input)

        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state