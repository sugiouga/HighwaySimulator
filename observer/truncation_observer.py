from .base_observer import BaseObserver

class TruncationObserver(BaseObserver):
    """シミュレーションの終了条件を評価し、必要に応じてシミュレーションを終了させるオブザーバー"""

    def observe(self, vehicles, road_network, current_time):
        """終了条件を評価し、必要に応じてシミュレーションを終了させる"""
        self.info = {}
        if current_time >= self.config.simulation.total_time:
            self.info = {"termination_reason": "timeout"}

    def get_info(self):
        """終了条件の情報を返すメソッド"""
        return self.info