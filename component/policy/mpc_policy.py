from .base_policy import BasePolicy

class MPCPolicy(BasePolicy):
    """
    Model Predictive Control (MPC) に基づく制御器
    MPCを用いて最適な制御入力を計算する。
    """

    def __init__(self,
                 vehicle_config,
                 policy_config
                 ):

        self.vehicle_config = vehicle_config
        self.policy_config = policy_config

        # MPCのパラメータ
        self.mpc_horizon = policy_config.parameters.mpc_horizon
        self.mpc_time_step = policy_config.parameters.mpc_time_step