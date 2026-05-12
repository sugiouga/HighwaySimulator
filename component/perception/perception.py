from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from component.vehicle.base_vehicle import BaseVehicle
from manager.road_network import RoadNetwork

@dataclass
class SurroundingVehicles:
    """自車の周囲の車両を保持するコンテナクラス"""
    left_front: Optional[object] = None
    left_rear: Optional[object] = None
    right_front: Optional[object] = None
    right_rear: Optional[object] = None
    front: Optional[object] = None
    rear: Optional[object] = None

class Perception:
    def __init__(self, sensor_range: List[float]):
        self.sensor_range = sensor_range

    def observe(self, ego, all_vehicles, road_network) -> SurroundingVehicles:
        """
        全車両リストから、自車の長方形範囲内かつ関係ある車線の車両を抽出
        """

        # RoadNetworkから隣接車線IDを取得
        target_lanes = [
            ego.lane_id,
            road_network.get_left_lane(ego.lane_id),
            road_network.get_right_lane(ego.lane_id)
        ]

        surrounding_vehicles = SurroundingVehicles()
        for vehicle in all_vehicles:
            if vehicle.vehicle_id == ego.vehicle_id:
                continue  # 自車はスキップ

            if vehicle.lane_id not in target_lanes:
                continue  # 関係ない車線の車両はスキップ

            relative_position = self.calculate_relative_position(ego, vehicle)

            if self.is_within_sensor_range(relative_position):
                self.classify_surrounding_vehicles(ego, road_network, surrounding_vehicles, vehicle, relative_position)
        # return as a dict for compatibility with policies
        return {
            'left_front': surrounding_vehicles.left_front,
            'left_rear': surrounding_vehicles.left_rear,
            'right_front': surrounding_vehicles.right_front,
            'right_rear': surrounding_vehicles.right_rear,
            'front': surrounding_vehicles.front,
            'rear': surrounding_vehicles.rear,
        }

    def calculate_relative_position(self, ego: 'BaseVehicle', vehicle: 'BaseVehicle') -> List[float]:
        """自車と他車の相対位置を計算するメソッド"""
        rel_x = float(np.nan_to_num(vehicle.x - ego.x, nan=0.0, posinf=0.0, neginf=0.0))
        rel_y = float(np.nan_to_num(vehicle.y - ego.y, nan=0.0, posinf=0.0, neginf=0.0))
        return [rel_x, rel_y]

    def calculate_relative_velocity(self, ego: 'BaseVehicle', vehicle: 'BaseVehicle') -> List[float]:
        """自車と他車の相対速度を計算するメソッド"""
        rel_vx = float(np.nan_to_num(vehicle.vx - ego.vx, nan=0.0, posinf=0.0, neginf=0.0))
        rel_vy = float(np.nan_to_num(vehicle.vy - ego.vy, nan=0.0, posinf=0.0, neginf=0.0))
        return [rel_vx, rel_vy]

    def is_within_sensor_range(self, relative_position: List[float]) -> bool:
        """相対位置がセンサーの検出範囲内にあるかを判定するメソッド"""
        rel_x, rel_y = relative_position
        rel_x = float(np.nan_to_num(rel_x, nan=0.0, posinf=0.0, neginf=0.0))
        rel_y = float(np.nan_to_num(rel_y, nan=0.0, posinf=0.0, neginf=0.0))
        distance = float(np.hypot(rel_x, rel_y))
        sensor_limit = max(float(self.sensor_range[0]), float(self.sensor_range[1]))
        return distance <= sensor_limit

    def classify_surrounding_vehicles(self, ego, road_network, surrounding_vehicles, vehicle, relative_position):
        rel_x, rel_y = relative_position
        # 相対位置をyaw角に基づいて変換し、前後左右を判定
        cos_yaw = np.cos(ego.yaw)
        sin_yaw = np.sin(ego.yaw)
        rel_front = rel_x * cos_yaw + rel_y * sin_yaw  # 車両前方向への投影
        if vehicle.lane_id == ego.lane_id:
            if rel_front > 0:
                surrounding_vehicles.front = vehicle
            else:
                surrounding_vehicles.rear = vehicle
        elif vehicle.lane_id == road_network.get_left_lane(ego.lane_id):
            if rel_front > 0:
                surrounding_vehicles.left_front = vehicle
            else:
                surrounding_vehicles.left_rear = vehicle
        elif vehicle.lane_id == road_network.get_right_lane(ego.lane_id):
            if rel_front > 0:
                surrounding_vehicles.right_front = vehicle
            else:
                surrounding_vehicles.right_rear = vehicle