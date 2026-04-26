from dataclasses import dataclass
from typing import List, Optional
from component.vehicle.base_vehicle import BaseVehicle
from component.road_network.road_network import RoadNetwork

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
    def __init__(self, sensor_range: List[float, float]):
        self.sensor_range = sensor_range

    def observe(self, ego: 'BaseVehicle', road_network: 'RoadNetwork', all_vehicles: List['BaseVehicle']) -> SurroundingVehicles:
        """自車の周囲の車両を観測するメソッド"""
        surrounding_vehicles = SurroundingVehicles()

        for vehicle in all_vehicles:
            if vehicle is ego:
                continue

            relative_position = self.calculate_relative_position(ego, vehicle)

            if self.is_within_sensor_range(relative_position):
                self.add_surrounding_vehicles(surrounding_vehicles, relative_position, vehicle)

        return surrounding_vehicles

    def calculate_relative_position(self, ego: 'BaseVehicle', vehicle: 'BaseVehicle') -> List[float, float]:
        """自車と他車の相対位置を計算するメソッド"""
        rel_s = vehicle.s - ego.s
        rel_d = vehicle.d - ego.d
        return [rel_s, rel_d]

    def is_within_sensor_range(self, relative_position: List[float, float]) -> bool:
        """相対位置がセンサーの検出範囲内にあるかを判定するメソッド"""
        rel_s, rel_d = relative_position
        return (abs(rel_s) <= self.sensor_range[0]) and (abs(rel_d) <= self.sensor_range[1])

    def add_surrounding_vehicles(self, surrounding_vehicles: 'SurroundingVehicles', relative_position: List[float, float], vehicle: 'BaseVehicle'):
        """相対位置に基づいて周囲の車両を適切な属性に追加するメソッド"""
        rel_s, rel_d = relative_position

        if rel_s > 0:  # 前方
            if abs(rel_d) < 1.0:  # 正面
                surrounding_vehicles.front = vehicle
            elif rel_d >= 1.0:  # 右前
                surrounding_vehicles.right_front = vehicle
            else:  # 左前
                surrounding_vehicles.left_front = vehicle
        else:  # 後方
            if abs(rel_d) < 1.0:  # 正面
                surrounding_vehicles.rear = vehicle
            elif rel_d >= 1.0:  # 右後
                surrounding_vehicles.right_rear = vehicle
            else:  # 左後
                surrounding_vehicles.left_rear = vehicle