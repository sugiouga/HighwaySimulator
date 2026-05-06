import pandas as pd
from .base_observer import BaseObserver

class FuelConsumptionObserver(BaseObserver):
    """
    燃料消費量を評価する観測者クラス
    各車両の燃料消費量を計算し、リスク評価を行う。
    """

    def observe(self, vehicles, current_time):
        for vehicle in vehicles:
            fuel_consumption = self.calculate_fuel_consumption(vehicle)
            log_entry = {
                'time': current_time,
                'vehicle_id': vehicle.id,
                'fuel_consumption': fuel_consumption
            }
            self.logs.append(log_entry)

    def calculate_fuel_consumption(self, vehicle):
        """燃料消費量を計算するメソッド"""
        # ARRBモデルを使用して燃料消費量を計算する
        delta = self.config.arrb_model.delta
        gamma1 = self.config.arrb_model.gamma1
        gamma2 = self.config.arrb_model.gamma2
        d_1 = self.config.arrb_model.d_1
        d_2 = self.config.arrb_model.d_2
        d_3 = self.config.arrb_model.d_3

        mass = vehicle.mass
        speed = vehicle.velocity
        acceleration = vehicle.acceleration

        P_T = max(0, d_1 * speed + d_2 * speed**2 + d_3 * speed**3 + 0.001 * mass * acceleration * speed)  # トラクションパワー
        fuel_consumption = delta + gamma1 * P_T + gamma2 * 0.001 * mass * max(0, acceleration)**2 * speed  # 燃料消費量
        return fuel_consumption

    def save_logs(self, file_path):
        df = pd.DataFrame(self.logs)
        df.to_csv(file_path, index=False)