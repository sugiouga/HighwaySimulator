import numpy as np
from factory.vehicle_factory import VehicleFactory
from utils.safety_checker import SafetyChecker

class VehicleManager:
    def __init__(self, config):
        self.config = config
        self.factory = VehicleFactory(config)

    def initialize(self, traffic_manager):
        """シミュレーション開始時に初期車両をスポーンするメソッド"""
        for init_spawn in self.config.road_network.init_spawns:
            # support both dict and dataclass/object access
            lane_id = init_spawn.get('lane_id') if isinstance(init_spawn, dict) else getattr(init_spawn, 'lane_id')
            s_range = init_spawn.get('s_range') if isinstance(init_spawn, dict) else getattr(init_spawn, 's_range')
            d_start = init_spawn.get('d_start') if isinstance(init_spawn, dict) else getattr(init_spawn, 'd_start')
            velocity_range = init_spawn.get('velocity_range') if isinstance(init_spawn, dict) else getattr(init_spawn, 'velocity_range')
            policies_distribution = init_spawn.get('policies_distribution') if isinstance(init_spawn, dict) else getattr(init_spawn, 'policies_distribution')

            lane = traffic_manager.road_network.get_lane(lane_id)
            s = np.random.uniform(*s_range)
            d = d_start
            velocity = np.random.uniform(*velocity_range)
            policy_type = np.random.choice(list(policies_distribution.keys()), p=list(policies_distribution.values()))
            # s, d座標からx, y座標に変換
            x, y = lane.get_cartesian(s, d)
            vehicle = self.factory.create_vehicle(vehicle_id=f"{len(traffic_manager.vehicles) + 1}_{policy_type}", lane_id=lane_id, init_state=[x, y, 0.0, velocity, 0.0], policy_id=policy_type)
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
            # support dict or object
            arrive_rate = spawn_point.get('arrive_rate') if isinstance(spawn_point, dict) else getattr(spawn_point, 'arrive_rate')
            if np.random.rand() < arrive_rate * dt:
                lane_id = spawn_point.get('lane_id') if isinstance(spawn_point, dict) else getattr(spawn_point, 'lane_id')
                s_start = spawn_point.get('s_start') if isinstance(spawn_point, dict) else getattr(spawn_point, 's_start')
                d_start = spawn_point.get('d_start') if isinstance(spawn_point, dict) else getattr(spawn_point, 'd_start')
                velocity_range = spawn_point.get('velocity_range') if isinstance(spawn_point, dict) else getattr(spawn_point, 'velocity_range')
                policies_distribution = spawn_point.get('policies_distribution') if isinstance(spawn_point, dict) else getattr(spawn_point, 'policies_distribution')

                lane = traffic_manager.road_network.get_lane(lane_id)
                s = np.random.uniform(s_start, s_start + 10.0) # スポーン位置の範囲を広げる
                d = d_start
                velocity = np.random.uniform(*velocity_range)
                policy_type = np.random.choice(list(policies_distribution.keys()), p=list(policies_distribution.values()))
                # s, d座標からx, y座標に変換
                x, y = lane.get_cartesian(s, d)
                vehicle = self.factory.create_vehicle(vehicle_id=f"{len(traffic_manager.vehicles) + 1}_{policy_type}", lane_id=lane_id, init_state=[x, y, 0.0, velocity, 0.0], policy_id=policy_type)
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