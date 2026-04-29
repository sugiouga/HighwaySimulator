import numpy as np
from factory.vehicle_factory import VehicleFactory

class VehicleManager:
    def __init__(self, config):
        self.config = config
        self.factory = VehicleFactory(config.vehicle, config.controllers)

    def initialize(self, traffic_manager):
        """シミュレーション開始時に初期車両をスポーンするメソッド"""
        for init_spawn in self.config.road_network.init_spawns:
            lane = traffic_manager.road_network.get_lane(init_spawn.lane_id)
            s = np.random.uniform(*init_spawn.s_range)
            d = init_spawn.d_start
            velocity = np.random.uniform(*init_spawn.velocity_range)
            controller_type = np.random.choice(
                list(init_spawn.controllers_distribution.keys()),
                p=list(init_spawn.controllers_distribution.values())
            )
            vehicle = self.factory.create_vehicle(vehicle_id=f"{len(traffic_manager.vehicles) + 1}_{controller_type}", lane_id=init_spawn.lane_id, init_state=[s, d, 0.0, velocity, 0.0], controller_id=controller_type)
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
        for vehicle in traffic_manager.vehicles:
            lane = traffic_manager.road_network.get_lane(vehicle.lane_id)
            if lane:
                vehicle.update_lane_id(lane)

    def _handle_spawning(self, traffic_manager, dt):
        """スポーンポイントから車両をスポーンするメソッド"""
        for spawn_point in self.config.road_network.spawn_points:
            if np.random.rand() < spawn_point.arrive_rate * dt:
                lane = traffic_manager.road_network.get_lane(spawn_point.lane_id)
                s = np.random.uniform(spawn_point.s_start, spawn_point.s_start + 10.0) # スポーン位置の範囲を広げる
                d = spawn_point.d_start
                velocity = np.random.uniform(*spawn_point.velocity_range)
                controller_type = np.random.choice(
                    list(spawn_point.controllers_distribution.keys()),
                    p=list(spawn_point.controllers_distribution.values())
                )
                vehicle = self.factory.create_vehicle(vehicle_id=f"{len(traffic_manager.vehicles) + 1}_{controller_type}", lane_id=spawn_point.lane_id, init_state=[s, d, 0.0, velocity, 0.0], controller_id=controller_type)
                traffic_manager.add_vehicle(vehicle)

    def _remove_out_of_bounds_vehicles(self, traffic_manager):
        """車線外にいる車両を削除するメソッド"""
        vehicles_to_remove = []
        for vehicle in traffic_manager.vehicles:
            # 車両の4隅の座標を計算する
            corners = vehicle.get_corners()
            # 4隅のいずれかが道路ネットワークの範囲外にある場合は削除対象とする
            if any(not traffic_manager.road_network.is_within_bounds(corner[0], corner [1]) for corner in corners):
                vehicles_to_remove.append(vehicle)
        for vehicle in vehicles_to_remove:
            traffic_manager.vehicles.remove(vehicle)

    def _handle_collisions(self, traffic_manager):
        """車両の衝突を処理するメソッド"""
        # 簡単な衝突処理: 車両同士が重なっている場合は両方とも削除する
        vehicles_to_remove = set()
        for i in range(len(traffic_manager.vehicles)):
            for j in range(i + 1, len(traffic_manager.vehicles)):
                v1 = traffic_manager.vehicles[i]
                v2 = traffic_manager.vehicles[j]
                if self._check_collision(v1, v2):
                    vehicles_to_remove.add(v1)
                    vehicles_to_remove.add(v2)
        for vehicle in vehicles_to_remove:
            traffic_manager.vehicles.remove(vehicle)