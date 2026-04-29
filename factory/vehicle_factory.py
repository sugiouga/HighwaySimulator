from .controller_factory import ControllerFactory
from component.vehicle.kinematic_bicycle_model import KinematicBicycleModel

class VehicleFactory:
    """
    車両のファクトリークラス
    車両のファクトリーは、車両の生成を担当するクラスで、車両の種類や数を管理する役割を担う。
    """
    def __init__(self, config):
        self.config = config

    def create_vehicle(self, vehicle_id, lane_id, init_state, controller_id):
        """
        車両を生成するメソッド
        Args:
        - vehicle_id: 車両ID
        - lane_id: 車両がスポーンする車線のID
        - init_state: 初期状態 [s, d, yaw, velocity, steering_angle]
        - controller_id: コントローラーID
        """
        controller_factory = ControllerFactory(self.config)
        controller = controller_factory.create_controller(controller_id)

        vehicle_config = self.config.vehicle

        if self.config.vehicle.model == "kinematic_bicycle":
            return KinematicBicycleModel(vehicle_id, lane_id, init_state, controller, vehicle_config)
        else:
            raise ValueError(f"Unknown vehicle model: {self.config.vehicle.model}")