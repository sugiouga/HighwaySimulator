import pandas as pd
import numpy as np
from .base_observer import BaseObserver

class TTCObserver(BaseObserver):
    """
    Time To Collision (TTC)を評価する観測者クラス
    各車両の周囲の車両とのTTCを計算し、リスク評価を行う。
    """

    def observe(self, vehicles, road_network, current_time):
        for vehicle in vehicles:
            for other_vehicle in vehicles:
                if vehicle.vehicle_id == other_vehicle.vehicle_id:
                    continue  # 自分自身はスキップ

                # 車両間の相対位置と速度を計算
                relative_position = np.array([other_vehicle.x - vehicle.x, other_vehicle.y - vehicle.y])
                relative_velocity = np.array([other_vehicle.velocity * np.cos(other_vehicle.yaw) - vehicle.velocity * np.cos(vehicle.yaw),
                                              other_vehicle.velocity * np.sin(other_vehicle.yaw) - vehicle.velocity * np.sin(vehicle.yaw)])

                # TTCを計算
                ttc = self.calculate_ttc(relative_position, relative_velocity, vehicle.yaw)

                log_entry = {
                    'time': current_time,
                    'vehicle_id': vehicle.vehicle_id,
                    'other_vehicle_id': other_vehicle.vehicle_id,
                    'ttc': ttc
                }
                self.logs.append(log_entry)

    def calculate_ttc(self, relative_position, relative_velocity, vehicle_yaw):
        """TTCを計算するメソッド"""
        # 相対位置を車両の前方向に投影
        cos_yaw = np.cos(vehicle_yaw)
        sin_yaw = np.sin(vehicle_yaw)
        relative_distance_front = relative_position[0] * cos_yaw + relative_position[1] * sin_yaw
        
        # 相対速度を車両の前方向に投影
        relative_velocity_front = relative_velocity[0] * cos_yaw + relative_velocity[1] * sin_yaw
        
        if relative_velocity_front >= 0:  # 前方車両が同速以上の場合、TTCは無限大とする
            return float('inf')
        else:
            if relative_distance_front > 0:
                return relative_distance_front / (-relative_velocity_front)
            else:
                return 0  # すでに衝突している

    def save_logs(self, file_path):
        df = pd.DataFrame(self.logs)
        df.to_csv(file_path, index=False)