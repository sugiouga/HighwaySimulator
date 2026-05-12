from component.policy.idm_policy import IDMPolicy

class PolicyFactory:
    def __init__(self, config):
        self.config = config

    def create_policy(self, policy_id):
        """
        ポリシーを生成するメソッド
        Args:
        - policy_id: ポリシーID
        """
        policy_config = self.config.policies[policy_id]

        if policy_config.type == "IDM":
            policy = IDMPolicy(vehicle_config=self.config.vehicle, idm_config=policy_config.parameters)
            # attach sensor_range and color from policy_config for runtime use
            setattr(policy, 'sensor_range', policy_config.sensor_range)
            setattr(policy, 'color', policy_config.color)
            return policy
        elif policy_config.type == "MPC":
            from component.policy.mpc_policy import MPCPolicy
            policy = MPCPolicy(vehicle_config=self.config.vehicle, policy_config=policy_config)
            setattr(policy, 'sensor_range', policy_config.sensor_range)
            setattr(policy, 'color', policy_config.color)
            return policy
        elif policy_config.type == "RL":
            from component.policy.rl_policy import RLPolicy
            policy = RLPolicy(vehicle_config=self.config.vehicle, policy_config=policy_config)
            setattr(policy, 'sensor_range', policy_config.sensor_range)
            setattr(policy, 'color', policy_config.color)
            return policy
        elif policy_config.type == "RLMPC":
            from component.policy.rl_mpc_policy import RLMPCPolicy
            policy = RLMPCPolicy(vehicle_config=self.config.vehicle, policy_config=policy_config)
            setattr(policy, 'sensor_range', policy_config.sensor_range)
            setattr(policy, 'color', policy_config.color)
            return policy
        else:
            raise ValueError(f"Unsupported policy type: {policy_config.type}")