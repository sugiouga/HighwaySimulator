from .base_observer import BaseObserver
from utils.safety_checker import SafetyChecker
import numpy as np

class TerminationObserver(BaseObserver):
    """シミュレーションの終了条件を評価するオブザーバー"""

    def observe(self, vehicles, road_network, current_time):
        """終了条件を評価し、必要に応じてシミュレーションを終了させる"""
        self.info = {}
        self.safety_checker = SafetyChecker()

        ego_vehicle = next((v for v in vehicles if v.is_ego), None)
        if ego_vehicle is not None:
            corners = ego_vehicle.get_corners()
            # 衝突判定
            for v in vehicles:
                if v is ego_vehicle:
                    continue
                try:
                    dx = float(np.nan_to_num(ego_vehicle.x - v.x, nan=0.0, posinf=0.0, neginf=0.0))
                    dy = float(np.nan_to_num(ego_vehicle.y - v.y, nan=0.0, posinf=0.0, neginf=0.0))
                    distance = float(np.hypot(dx, dy))
                    if distance < (ego_vehicle.length + ego_vehicle.width + v.length + v.width) / 4:
                        if SafetyChecker.check_precise_collision(ego_vehicle, v):
                            self.info = {"termination_reason": "collision"}
                            return
                except Exception:
                    # fallback: ignore errors in collision check
                    continue

            # 車線逸脱判定
            try:
                for corner in corners:
                    if not road_network.is_within_bounds(corner[0], corner[1]):
                        self.info = {"termination_reason": "lane_deviation"}
                        return
            except Exception:
                pass

            # goal check
            if ego_vehicle.state[0] > self.config.simulation.goal_x:
                self.info = {"termination_reason": "goal_reached"}


    def get_info(self):
        """終了条件の情報を返すメソッド"""
        return self.info