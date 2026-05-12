from .base_policy import BasePolicy
from typing import Dict, Any, List
import numpy as np

class IDMPolicy(BasePolicy):
    """
    Intelligent Driver Model (IDM) に基づく意思決定モデル
    IDMは、車両の加減速を決定するためのモデルで、前方車両との距離や速度差に基づいて制御入力を計算する。
    """

    def __init__(self,
                 vehicle_config,
                 idm_config
                 ):

        self.vehicle_config = vehicle_config
        self.idm_config = idm_config

        self.desired_velocity = idm_config.desired_velocity
        self.desired_time_headway = idm_config.desired_time_headway
        self.min_spacing = idm_config.min_spacing
        self.max_acceleration = vehicle_config.max_acceleration
        self.comfortable_deceleration = idm_config.comfortable_deceleration

    def action(self, state: list, environment_info: Dict[str, Any]) -> List[float]:
        """IDMに基づいて制御入力を計算するメソッド
        Args:
        - state: 車両の状態 [x, y, yaw, velocity, steering_angle]
        - environment_info: 周囲の環境情報を表す辞書（例: 他の車両の状態、道路情報など）
        Returns:
        - action: 行動入力を表すリスト（例: [acceleration, steering_rate]）
        """
        velocity = float(np.nan_to_num(state[3], nan=0.0, posinf=0.0, neginf=0.0))
        desired_velocity = max(float(self.desired_velocity), 1e-3)
        max_acceleration = max(float(self.max_acceleration), 1e-6)
        comfortable_deceleration = max(float(self.comfortable_deceleration), 1e-6)

        # 前方車両の情報を取得
        lead_vehicle = environment_info.get('front', None)

        # IDMの加速度計算
        if lead_vehicle is not None:
            # support both dict and object representations
            if isinstance(lead_vehicle, dict):
                lead_velocity = lead_vehicle.get('velocity', 0.0)
                lead_position = lead_vehicle.get('position', 0.0)
            else:
                lead_velocity = getattr(lead_vehicle, 'velocity', 0.0)
                lead_position = getattr(lead_vehicle, 'x', 0.0)

            lead_velocity = float(np.nan_to_num(lead_velocity, nan=0.0, posinf=0.0, neginf=0.0))
            lead_position = float(np.nan_to_num(lead_position, nan=0.0, posinf=0.0, neginf=0.0))

            delta_v = velocity - lead_velocity  # 速度差
            denom = 2.0 * np.sqrt(max_acceleration * comfortable_deceleration)
            s_alpha = self.min_spacing + max(0.0, velocity * self.desired_time_headway + (velocity * delta_v) / max(denom, 1e-6))  # 安全距離
            gap = max(abs(state[0] - lead_position), 0.1)

            v_ratio = np.clip(velocity / desired_velocity, -20.0, 20.0)
            interaction = np.clip((s_alpha / gap) ** 2, 0.0, 1e6)
            acceleration = max_acceleration * (1.0 - v_ratio ** 4 - interaction)
        else:
            v_ratio = np.clip(velocity / desired_velocity, -20.0, 20.0)
            acceleration = max_acceleration * (1.0 - v_ratio ** 4)

        acceleration = float(np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0))
        min_acc = getattr(self.vehicle_config, 'min_acceleration', -10.0)
        max_acc = getattr(self.vehicle_config, 'max_acceleration', 10.0)
        acceleration = float(np.clip(acceleration, min_acc, max_acc))

        return [acceleration, 0.0]  # 加速度とステアリングレート（ここではステアリングは考慮しない） 

