import pandas as pd
from .base_observer import BaseObserver

class MetricsObserver(BaseObserver):

    def observe(self, vehicles, road_network, current_time):
        for vehicle in vehicles:
            log_entry = {
                'time': current_time,
                'vehicle_id': vehicle.vehicle_id,
                'lane_id': vehicle.lane_id,
                'x': vehicle.x,
                'y': vehicle.y,
                'yaw': vehicle.yaw,
                'velocity': vehicle.velocity,
                'steering_angle': vehicle.steering_angle,
                'acceleration': vehicle.acceleration,
                'steering_rate': vehicle.steering_rate
            }
            self.logs.append(log_entry)

    def save_logs(self, file_path):
        df = pd.DataFrame(self.logs)
        df.to_csv(file_path, index=False)

    def get_current_metrics(self, vehicle_id):
        """指定した車両の現在のメトリクスを取得するメソッド"""
        for log in reversed(self.logs):
            if log['vehicle_id'] == vehicle_id:
                return log
        return None  # メトリクスが見つからない場合はNoneを返す

    def get_previous_metrics(self, vehicle_id):
        """指定した車両の前回のメトリクスを取得するメソッド"""
        found_current = False
        for log in reversed(self.logs):
            if log['vehicle_id'] == vehicle_id:
                if found_current:
                    return log  # 前回のメトリクスを返す
                else:
                    found_current = True  # 現在のメトリクスを見つけたことを記録
        return None  # 前回のメトリクスが見つからない場合はNoneを返す
