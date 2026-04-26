from .base_vehicle import BaseVehicle
import numpy as np

class KinematicBicycleModel(BaseVehicle):
    """
    Kinematic Bicycle Modelを用いた車両クラス
    重心中心で滑り角を考慮する
        状態: [s, d, yaw, velocity, steering_angle]
        制御入力: [acceleration, steering_rate]
        運動方程式:
        ds/dt = velocity * cos(yaw) - slip_angle * sin(yaw)
        dd/dt = velocity * sin(yaw) + slip_angle * cos(yaw)
        dyaw/dt = velocity / length * tan(steering_angle)
        dvelocity/dt = acceleration
        dsteering_angle/dt = steering_rate
    """

    def get_dynamics(self, state, control_input):
        s, d, yaw, velocity, steering_angle = state
        acceleration, steering_rate = control_input

        slip_angle = np.arctan((self.length / 2) * np.tan(steering_angle) / self.length)

        ds = velocity * np.cos(yaw) - slip_angle * np.sin(yaw)
        dd = velocity * np.sin(yaw) + slip_angle * np.cos(yaw)
        dyaw = velocity / self.length * np.tan(steering_angle)
        dvelocity = acceleration
        dsteering_angle = steering_rate

        return np.array([ds, dd, dyaw, dvelocity, dsteering_angle], dtype=np.float64)