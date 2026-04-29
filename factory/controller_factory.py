from component.controller.idm_controller import IDMController

class ControllerFactory:
    def __init__(self, config):
        self.config = config

    def create_controller(self, controller_id):
        """
        コントローラーを生成するメソッド
        Args:
        - controller_id: コントローラーID
        """
        controller_config = self.config.controllers[controller_id]

        if controller_config.type == "IDM":
            return IDMController(controller_config.parameters, controller_config.sensor_range)
        else:
            raise ValueError(f"Unknown controller type: {controller_config.type}")