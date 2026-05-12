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
            if self.safety_checker.check_collision(ego_vehicle, vehicles):
                self.info = {"termination_reason": "collision"}
            elif self.safety_checker.check_lane_deviation(ego_vehicle, road_network):
                self.info = {"termination_reason": "lane_deviation"}
            elif ego_vehicle.state[0] > self.config.simulation.goal_x:  # ゴール条件（例: x座標がgoal_xを超える）
                self.info = {"termination_reason": "goal_reached"}


    def get_info(self):
        """終了条件の情報を返すメソッド"""
        return self.info