import numpy as np
from factory.vehicle_factory import VehicleFactory
from utils.safety_checker import SafetyChecker

class VehicleManager:
    def __init__(self, config):
        self.config = config
        self.factory = VehicleFactory(config.vehicle, config.policies)

    def initialize(self, traffic_manager):
        """シミュレーション開始時に初期車両をスポーンするメソッド"""
        for init_spawn in self.config.road_network.init_spawns:
            lane = traffic_manager.road_network.get_lane(init_spawn.lane_id)
            s = np.random.uniform(*init_spawn.s_range)
            d = init_spawn.d_start
            velocity = np.random.uniform(*init_spawn.velocity_range)
            policy_type = np.random.choice(
                list(init_spawn.policies_distribution.keys()),
                p=list(init_spawn.policies_distribution.values())
            )
            vehicle = self.factory.create_vehicle(vehicle_id=f"{len(traffic_manager.vehicles) + 1}_{policy_type}", lane_id=init_spawn.lane_id, init_state=[s, d, 0.0, velocity, 0.0], policy_id=policy_type)
            traffic_manager.add_vehicle(vehicle)

    def update(self, traffic_manager):
        """車両のスポーンと削除を管理するメソッド"""
        dt = traffic_manager.dt
        # スポーンポイントから車両をスポーンする
        self._handle_spawning(traffic_manager, dt)
        # 車線外にいる車両を削除する
        self._remove_out_of_bounds_vehicles(traffic_manager)
        # 車両の衝突を処理する
        self._handle_collisions(traffic_manager)
        # lane_idの更新
        self._update_lane_id(traffic_manager)

    def _handle_spawning(self, traffic_manager, dt):
        """スポーンポイントから車両をスポーンするメソッド"""
        for spawn_point in self.config.road_network.spawn_points:
            if np.random.rand() < spawn_point.arrive_rate * dt:
                lane = traffic_manager.road_network.get_lane(spawn_point.lane_id)
                s = np.random.uniform(spawn_point.s_start, spawn_point.s_start + 10.0) # スポーン位置の範囲を広げる
                d = spawn_point.d_start
                velocity = np.random.uniform(*spawn_point.velocity_range)
                policy_type = np.random.choice(
                    list(spawn_point.policies_distribution.keys()),
                    p=list(spawn_point.policies_distribution.values())
                )
                vehicle = self.factory.create_vehicle(vehicle_id=f"{len(traffic_manager.vehicles) + 1}_{policy_type}", lane_id=spawn_point.lane_id, init_state=[s, d, 0.0, velocity, 0.0], policy_id=policy_type)
                traffic_manager.add_vehicle(vehicle)

    def _remove_out_of_bounds_vehicles(self, traffic_manager):
        """車線外にいる車両を削除するメソッド"""
        vehicles_to_remove = SafetyChecker.check_out_of_bounds(traffic_manager)
        for vehicle in vehicles_to_remove:
            traffic_manager.vehicles.remove(vehicle)

    def _handle_collisions(self, traffic_manager):
        """車両の衝突を処理するメソッド"""
        vehicles_to_remove = SafetyChecker.check_collision(traffic_manager)
        for vehicle in vehicles_to_remove:
            traffic_manager.vehicles.remove(vehicle)

    def _update_lane_id(self, traffic_manager):
        for vehicle in traffic_manager.vehicles:
            lane = traffic_manager.road_network.get_lane(vehicle.lane_id)
            if lane:
                vehicle.update_lane_id(lane)