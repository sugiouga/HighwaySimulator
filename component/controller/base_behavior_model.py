from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseBehaviorModel(ABC):
    """
    意思決定モデルの基底クラス
    各車両の行動を決定するための抽象クラスで、具体的な行動モデルはこのクラスを継承して実装される。
    """

    @abstractmethod
    def compute_control(self, state: list, environment_info: Dict[str, Any]) -> List[float]:
        """
        周辺状況から車両の制御入力を計算する抽象メソッド
        Args:
        - state: 車両の状態 [s, d, yaw, velocity, steering_angle]
        - environment_info: 周囲の環境情報を表す辞書（例: 他の車両の状態、道路情報など）
        Returns:
        - control_input: 車両の制御入力を表すリスト（例: [acceleration, steering_rate]）
        """
        pass