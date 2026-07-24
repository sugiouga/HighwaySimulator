import numpy as np
from factory.vehicle_factory import VehicleFactory
from utils.safety_checker import SafetyChecker
from component.policy.ghost_policy import GhostPolicy

class VehicleManager:
    def __init__(self, config):
        self.config = config
        self.factory = VehicleFactory(config)

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

    def spawn_init_vehicles(self, traffic_manager):
        """初期車両をスポーンするメソッド"""
        for vehicle_config in self.config.road_network.init_vehicles:
            lane_id = vehicle_config.get('lane_id') if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'lane_id')
            s = vehicle_config.get('s') if isinstance(vehicle_config, dict) else getattr(vehicle_config, 's')
            d = vehicle_config.get('d') if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'd')
            velocity = vehicle_config.get('velocity') if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'velocity')
            policy_id = vehicle_config.get('policy_id') if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'policy_id')
            is_ego = vehicle_config.get('is_ego', False) if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'is_ego', False)

            lane = traffic_manager.road_network.get_lane(lane_id)
            x, y = lane.get_cartesian(s, d)
            vehicle = self.factory.create_vehicle(vehicle_id=vehicle_config.get('id', f"init_{len(traffic_manager.vehicles) + 1}"), lane_id=lane_id, init_state=[x, y, 0.0, velocity, 0.0], policy_id=policy_id, is_ego=is_ego)
            traffic_manager.add_vehicle(vehicle)

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
                min_spawn_gap = spawn_point.get('min_spawn_gap') if isinstance(spawn_point, dict) else getattr(spawn_point, 'min_spawn_gap', None)
                if min_spawn_gap is None:
                    min_spawn_gap = 2.0 * self.config.vehicle.length

                lane = traffic_manager.road_network.get_lane(lane_id)
                s = np.random.uniform(s_start, s_start + 10.0) # スポーン位置の範囲を広げる
                d = d_start

                # 前方に他車がいる場合はスポーンを見送る
                if self._is_vehicle_ahead(traffic_manager, lane, lane_id, s, min_spawn_gap):
                    continue

                velocity = np.random.uniform(*velocity_range)
                policy_type = np.random.choice(list(policies_distribution.keys()), p=list(policies_distribution.values()))
                # s, d座標からx, y座標に変換
                x, y = lane.get_cartesian(s, d)
                vehicle = self.factory.create_vehicle(vehicle_id=f"{len(traffic_manager.vehicles) + 1}_{policy_type}", lane_id=lane_id, init_state=[x, y, 0.0, velocity, 0.0], policy_id=policy_type, is_ego=False)
                traffic_manager.add_vehicle(vehicle)

    def _is_vehicle_ahead(self, traffic_manager, lane, lane_id, s, min_gap):
        """指定した車線・s位置から前方min_gap[m]以内に他車が存在するかを判定するメソッド"""
        for vehicle in traffic_manager.vehicles:
            if vehicle.lane_id != lane_id:
                continue
            vehicle_s, _ = lane.get_frenet(vehicle.x, vehicle.y)
            if 0 <= vehicle_s - s < min_gap:
                return True
        return False

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
        """Update each vehicle's lane_id.

        Priority:
        1) Ghost車両は固定（更新しない）
        2) 車線変更中でターゲット車線がある場合は、その車線IDを優先
        3) それ以外は現在位置から判定

        This checks all lanes in the road network and assigns the first lane
        for which `lane.is_within_bounds(vehicle.x, vehicle.y)` is True.
        If no lane matches, the vehicle's lane_id is left unchanged.
        """
        for vehicle in traffic_manager.vehicles:
            # Ghost車両はlane_idを固定する
            if isinstance(getattr(vehicle, "policy", None), GhostPolicy):
                continue

            assigned = False
            for lane in traffic_manager.road_network.lanes.values():
                try:
                    if lane.is_within_bounds(vehicle.x, vehicle.y):
                        vehicle.update_lane_id(lane)
                        assigned = True
                        break
                except Exception:
                    # keep existing lane_id on any unexpected error
                    continue
            if not assigned:
                # fallback: keep current lane_id (no change)
                pass