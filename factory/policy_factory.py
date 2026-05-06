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

        if controller_config.type == "IDM":
            return IDMPolicy(policy_config.parameters, policy_config.sensor_range)
        else:
            raise ValueError(f"Unknown policy type: {policy_config.type}")