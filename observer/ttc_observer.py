import pandas as pd
import numpy as np
from .base_observer import BaseObserver

class TTCObserver(BaseObserver):
    """
    Time To Collision (TTC)を評価する観測者クラス
    各車両の周囲の車両とのTTCを計算し、リスク評価を行う。
    """

    def observe(self, vehicles, current_time):
        for vehicle in vehicles:
            for other_vehicle in vehicles:
                if vehicle.id == other_vehicle.id:
                    continue  # 自分自身はスキップ

                # 車両間の相対位置と速度を計算
                relative_position = np.array([other_vehicle.s - vehicle.s, other_vehicle.d - vehicle.d])
                relative_velocity = np.array([other_vehicle.velocity - vehicle.velocity])

                # TTCを計算
                ttc = self.calculate_ttc(relative_position, relative_velocity)

                log_entry = {
                    'time': current_time,
                    'vehicle_id': vehicle.id,
                    'other_vehicle_id': other_vehicle.id,
                    'ttc': ttc
                }
                self.logs.append(log_entry)

    def calculate_ttc(self, relative_position, relative_velocity):
        """TTCを計算するメソッド"""
        if relative_velocity[0] <= 0: # 前方車両が同速または遅い場合、TTCは無限大とする
            return float('inf')
        else:
            return relative_position[0] / relative_velocity[0]

    def save_logs(self, file_path):
        df = pd.DataFrame(self.logs)
        df.to_csv(file_path, index=False)