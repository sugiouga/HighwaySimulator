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
        else:
            raise ValueError(f"Unknown policy type: {policy_config.type}")