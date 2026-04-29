from dataclasses import dataclass
from typing import List, Optional
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
    def __init__(self, sensor_range: List[float, float]):
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
            if vehicle.id == ego.id:
                continue  # 自車はスキップ

            if vehicle.lane_id not in target_lanes:
                continue  # 関係ない車線の車両はスキップ

            relative_position = self.calculate_relative_position(ego, vehicle)

            if self.is_within_sensor_range(relative_position):
                self.classify_surrounding_vehicles(ego, road_network, surrounding_vehicles, vehicle, relative_position)
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

    def classify_surrounding_vehicles(self, ego, road_network, surrounding_vehicles, vehicle, relative_position):
        rel_s, rel_d = relative_position
        if vehicle.lane_id == ego.lane_id:
            if rel_s > 0:
                surrounding_vehicles.front = vehicle
            else:
                surrounding_vehicles.rear = vehicle
        elif vehicle.lane_id == road_network.get_left_lane(ego.lane_id):
            if rel_s > 0:
                surrounding_vehicles.left_front = vehicle
            else:
                surrounding_vehicles.left_rear = vehicle
        elif vehicle.lane_id == road_network.get_right_lane(ego.lane_id):
            if rel_s > 0:
                surrounding_vehicles.right_front = vehicle
            else:
                surrounding_vehicles.right_rear = vehicle