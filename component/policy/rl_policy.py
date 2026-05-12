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