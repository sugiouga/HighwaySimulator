from .base_vehicle import BaseVehicle
import numpy as np

class KinematicBicycleModel(BaseVehicle):
    """
    Kinematic Bicycle Modelを用いた車両クラス
    重心中心で滑り角を考慮する
        状態: [x, y, yaw, velocity, steering_angle]
        制御入力: [acceleration, steering_rate]
        運動方程式:
        slip_angle = arctan((length/2) * tan(steering_angle) / length)
        dx/dt = velocity * cos(yaw + slip_angle)
        dy/dt = velocity * sin(yaw + slip_angle)
        dyaw/dt = velocity / length * tan(steering_angle)
        dvelocity/dt = acceleration
        dsteering_angle/dt = steering_rate
    """

    def get_dynamics(self, state, action):
        x, y, yaw, velocity, steering_angle = np.nan_to_num(
            np.asarray(state, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        acceleration, steering_rate = np.nan_to_num(
            np.asarray(action, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )

        velocity = float(np.clip(velocity, self.min_velocity, self.max_velocity))
        max_steer_rad = np.deg2rad(self.max_steering_angle)
        steering_angle = float(np.clip(steering_angle, -max_steer_rad, max_steer_rad))
        steering_rate = float(np.clip(steering_rate, -self.max_steering_rate, self.max_steering_rate))
        acceleration = float(np.clip(acceleration, self.min_acceleration, self.max_acceleration))

        # 滑り角を計算
        tan_steer = np.tan(steering_angle)
        slip_angle = np.arctan(0.5 * tan_steer)

        dx = velocity * np.cos(yaw + slip_angle)
        dy = velocity * np.sin(yaw + slip_angle)
        dyaw = velocity / max(self.length, 1e-6) * tan_steer
        dvelocity = acceleration
        dsteering_angle = steering_rate

        return np.nan_to_num(np.array([dx, dy, dyaw, dvelocity, dsteering_angle], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)