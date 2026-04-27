from .base_behavior_model import BaseBehaviorModel
from typing import Dict, Any, List
import numpy as np

class IDMController(BaseBehaviorModel):
    """
    Intelligent Driver Model (IDM) に基づく意思決定モデル
    IDMは、車両の加減速を決定するためのモデルで、前方車両との距離や速度差に基づいて制御入力を計算する。
    """

    def __init__(self,
                 vehicle_config,
                 idm_config):

        self.vehicle_config = vehicle_config
        self.idm_config = idm_config

        self.desired_velocity = idm_config.desired_velocity
        self.desired_time_headway = idm_config.desired_time_headway
        self.min_spacing = idm_config.min_spacing
        self.max_acceleration = vehicle_config.max_acceleration
        self.comfortable_deceleration = idm_config.comfortable_deceleration

    def compute_control(self, state: list, environment_info: Dict[str, Any]) -> List[float]:
        """IDMに基づいて制御入力を計算するメソッド
        Args:
        - state: 車両の状態 [s, d, yaw, velocity, steering_angle]
        - environment_info: 周囲の環境情報を表す辞書（例: 他の車両の状態、道路情報など）
        Returns:
        - control_input: 車両の制御入力を表すリスト（例: [acceleration, steering_rate]）
        """
        # 前方車両の情報を取得
        lead_vehicle = environment_info.get('lead_vehicle', None)

        # IDMの加速度計算
        if lead_vehicle is not None:
            delta_v = state[3] - lead_vehicle['velocity']  # 速度差
            s_alpha = self.min_spacing + max(0, state[3] * self.desired_time_headway + (state[3] * delta_v) / (2 * np.sqrt(self.max_acceleration * self.comfortable_deceleration)))  # 安全距離
            acceleration = self.max_acceleration * (1 - (state[3] / self.desired_velocity) ** 4 - (s_alpha / max(state[0] - lead_vehicle['position'], 0.1)) ** 2)
        else:
            acceleration = self.max_acceleration * (1 - (state[3] / self.desired_velocity) ** 4)

        return [acceleration, 0.0]  # 加速度とステアリングレート（ここではステアリングは考慮しない） 

