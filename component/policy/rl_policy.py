from .base_policy import BasePolicy

class RLPolicy(BasePolicy):
    """
    Deep Reinforcement Learning (DRL) に基づく制御器
    DRLを用いて車両の制御入力を学習する。
    """

    def __init__(self,
                 vehicle_config,
                 policy_config
                 ):

        self.vehicle_config = vehicle_config
        self.policy_config = policy_config
        # Placeholder for RL-based policy. For training the ego vehicle this
        # policy object is instantiated but `is_ego` vehicles will have actions
        # provided by the external agent; provide a safe default action here.

    def action(self, state: list, environment_info: dict):
        """Return a safe zero-action by default: [acceleration, steering_rate]."""
        return [0.0, 0.0]