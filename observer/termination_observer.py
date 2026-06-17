from .base_observer import BaseObserver
from utils.safety_checker import SafetyChecker

class TerminationObserver(BaseObserver):
    """シミュレーションの終了条件を評価するオブザーバー"""

    def observe(self, vehicles, road_network, current_time):
        """終了条件を評価し、必要に応じてシミュレーションを終了させる"""
        self.info = {}
        self.safety_checker = SafetyChecker()

        ego_vehicle = next((v for v in vehicles if v.is_ego), None)
        if ego_vehicle is not None:
            # collision: check precise collision between ego and any other vehicle
            for v in vehicles:
                if v is ego_vehicle:
                    continue
                try:
                    if SafetyChecker._check_precise_collision(ego_vehicle, v):
                        self.info = {"termination_reason": "collision"}
                        return
                except Exception:
                    # fallback: ignore errors in collision check
                    continue

            # lane deviation: ego not within its assigned lane bounds
            try:
                lane = road_network.get_lane(ego_vehicle.lane_id)
                if lane is None or not lane.is_within_bounds(ego_vehicle.x, ego_vehicle.y):
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