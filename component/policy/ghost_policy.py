from .base_policy import BasePolicy
from typing import Dict, Any, List

class GhostPolicy(BasePolicy):
    """ゴースト車両のポリシー。常に前方に一定速度で移動するだけの単純なポリシー。"""

    def __init__(self, vehicle_config, policy_config):
        self.vehicle_config = vehicle_config
        self.policy_config = policy_config

    def action(self, state: list, environment_info: Dict[str, Any]) -> List[float]:
        """ゴースト車両は常に前方に一定速度で移動するだけの単純なポリシー"""
        return [0.0, 0.0]  # [加速度, ステアリング角]