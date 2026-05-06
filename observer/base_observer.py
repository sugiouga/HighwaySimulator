from abc import ABC, abstractmethod

@abstractmethod
class BaseObserver(ABC):
    """
    観測者の基底クラス
    観測者は、車両の周囲の環境情報をもとに、車両の安全性やリスクを評価する役割を担う。
    """

    def __init__(self, config):
        """
        観測者の初期化を行う抽象メソッド
        Args:
        - config: 観測者の設定情報を含むオブジェクト
        """
        self.config = config
        self.logs = []

    @abstractmethod
    def observe(self, vehicles, current_time):
        """観測者が車両の状態を観測し、評価を行う抽象メソッド"""
        pass
