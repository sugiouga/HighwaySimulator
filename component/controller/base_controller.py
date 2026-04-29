from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseController(ABC):
    """
    車両の制御器の基底クラス
    車両の制御器は、周囲の環境情報をもとに車両の制御入力を計算する役割を担う。
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