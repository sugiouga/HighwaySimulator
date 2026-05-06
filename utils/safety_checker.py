import numpy as np

class SafetyChecker:
    @staticmethod
    def check_collision(traffic_manager):
        """車両の衝突をチェックするメソッド"""
        # 1. 円近似による高速フィルタリング
        vehicles_to_remove = set()
        for i in range(len(traffic_manager.vehicles)):
            for j in range(i + 1, len(traffic_manager.vehicles)):
                v1 = traffic_manager.vehicles[i]
                v2 = traffic_manager.vehicles[j]
                dx = float(np.nan_to_num(v1.x - v2.x, nan=0.0, posinf=0.0, neginf=0.0))
                dy = float(np.nan_to_num(v1.y - v2.y, nan=0.0, posinf=0.0, neginf=0.0))
                distance = float(np.hypot(dx, dy))
                if distance < (v1.length + v1.width + v2.length + v2.width) / 4:
                    # 円近似で衝突の可能性がある場合は、精密判定を行う
                    if SafetyChecker._check_precise_collision(v1, v2):
                        vehicles_to_remove.add(v1)
                        vehicles_to_remove.add(v2)
        # return vehicles to remove as list
        return list(vehicles_to_remove)
    def _check_precise_collision(vehicle1, vehicle2):
        """2台の車両が衝突しているかどうかを精密に判定するメソッド"""
        # OBB（Oriented Bounding Box）を使用して、分離軸定理 (Separating Axis Theorem, SAT) に基づいて衝突判定を行う
        # 2つの長方形の各辺に垂直な軸を計算し、その軸に沿って投影したときに重なりがあるかどうかを判定する
        axes = SafetyChecker._get_axes(vehicle1) + SafetyChecker._get_axes(vehicle2)
        for axis in axes:
            projection1 = SafetyChecker._project_vehicle(vehicle1, axis)
            projection2 = SafetyChecker._project_vehicle(vehicle2, axis)
            if not SafetyChecker._overlap(projection1, projection2):
                return False  # 分離軸が見つかった場合は衝突していない
        return True  # 全ての軸で重なりがある場合は衝突している

    def _get_axes(vehicle):
        """車両のOBBの辺に垂直な軸を計算するメソッド"""
        # 車両の向きに基づいて、2つの軸を計算する
        cos_yaw = np.cos(vehicle.yaw)
        sin_yaw = np.sin(vehicle.yaw)
        axis1 = np.array([cos_yaw, sin_yaw])  # 車両の前方向
        axis2 = np.array([-sin_yaw, cos_yaw])  # 車両の横方向
        return [axis1, axis2]

    def _project_vehicle(vehicle, axis):
        """車両を指定した軸に投影するメソッド"""
        corners = vehicle.get_corners()
        projections = [np.dot(corner, axis) for corner in corners]
        projections = [float(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)) for p in projections]
        return [min(projections), max(projections)]

    def _overlap(projection1, projection2):
        """2つの投影が重なっているかどうかを判定するメソッド"""
        return projection1[0] < projection2[1] and projection1[1] > projection2[0]

    @staticmethod
    def check_out_of_bounds(traffic_manager):
        """車線外にいる車両をチェックするメソッド"""
        vehicles_to_remove = []
        for vehicle in traffic_manager.vehicles:
            # 車両の4隅の座標を計算する
            corners = vehicle.get_corners()
            # 4隅のいずれかが道路ネットワークの範囲外にある場合は削除対象とする
            if any(not traffic_manager.road_network.is_within_bounds(corner[0], corner[1]) for corner in corners):
                vehicles_to_remove.append(vehicle)
        return vehicles_to_remove