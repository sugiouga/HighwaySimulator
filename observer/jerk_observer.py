from base_observer import BaseObserver

class JerkObserver(BaseObserver):
    """
    各車両の加速度の変化率（ジャーク）を計算し、リスク評価を行う。
    """

    def observe(self, vehicles, current_time):
        for vehicle in vehicles:
            jerk = self.calculate_jerk(vehicle)
            log_entry = {
                'time': current_time,
                'vehicle_id': vehicle.id,
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

        current_acceleration = vehicle.acceleration
        jerk = (current_acceleration - previous_acceleration) / self.config.time_step

        # 現在の加速度を保存しておく
        vehicle.previous_acceleration = current_acceleration

        return jerk