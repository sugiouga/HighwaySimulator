import pandas as pd
from .base_observer import BaseObserver

class JerkObserver(BaseObserver):
    """
    各車両の加速度の変化率（ジャーク）を計算し、リスク評価を行う。
    """

    def observe(self, vehicles, road_network, current_time):
        for vehicle in vehicles:
            jerk = self.calculate_jerk(vehicle)
            log_entry = {
                'time': current_time,
                'vehicle_id': vehicle.vehicle_id,
                'jerk': jerk
            }
            self.logs.append(log_entry)

    def calculate_jerk(self, vehicle):
        """ジャークを計算するメソッド"""
        # ジャークは加速度の変化率で定義されるため、前回の加速度と現在の加速度を使用して計算する
        if hasattr(vehicle, 'previous_acceleration'):
            previous_acceleration = vehicle.previous_acceleration
        else:
            previous_acceleration = 0.0  # 初回は加速度を0とする

        # obtain current acceleration from vehicle attribute or fallback to current_action[0]
        if getattr(vehicle, 'acceleration', None) is not None:
            current_acceleration = float(vehicle.acceleration)
        elif hasattr(vehicle, 'current_action') and vehicle.current_action is not None and len(getattr(vehicle, 'current_action', [])) > 0:
            current_acceleration = float(vehicle.current_action[0])
        else:
            current_acceleration = 0.0

        # time step is stored under config.simulation.time_step
        dt = float(getattr(getattr(self.config, 'simulation', None), 'time_step', 0.1))
        jerk = (current_acceleration - previous_acceleration) / max(dt, 1e-6)

        # 現在の加速度を保存しておく
        vehicle.previous_acceleration = current_acceleration

        return jerk

    def save_logs(self, file_path):
        df = pd.DataFrame(self.logs)
        df.to_csv(file_path, index=False)

    def get_current_jerk(self, vehicle_id):
        """車両の現在のジャークを取得するメソッド"""
        for log in reversed(self.logs):
            if log['vehicle_id'] == vehicle_id:
                return log['jerk']
        return None  # ジャークが見つからない場合はNoneを返す

    def get_previous_jerk(self, vehicle_id):
        """車両の前回のジャークを取得するメソッド"""
        found_current = False
        for log in reversed(self.logs):
            if log['vehicle_id'] == vehicle_id:
                if found_current:
                    return log['jerk']  # 前回のジャークを返す
                else:
                    found_current = True  # 現在のジャークを見つけたことを記録
        return None  # 前回のジャークが見つからない場合はNoneを返す