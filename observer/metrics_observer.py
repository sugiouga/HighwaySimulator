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