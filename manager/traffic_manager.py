from component.perception.perception import Perception
from manager.vehicle_manager import VehicleManager
from manager.road_network import RoadNetwork

class TrafficManager:
    def __init__(self, road_network, config, dt=0.1):
        self.road_network = road_network
        self.config = config
        self.dt = dt
        self.current_time = 0.0
        self.vehicles = []
        self.observers = []
        self.perceptions = {}

        self.road_network.reset()
        self.vehicle_manager = VehicleManager(self.config)

        for vehicle in self.vehicles:
            self._register_perception(vehicle)

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)
        self._register_perception(vehicle)

    def _register_perception(self, vehicle):
        if vehicle.vehicle_id not in self.perceptions:
            self.perceptions[vehicle.vehicle_id] = Perception(sensor_range=vehicle.policy.sensor_range)

    def add_observer(self, observer):
        self.observers.append(observer)

    def step(self):
        """シミュレーションの1ステップを進めるメソッド"""

        # 車両のスポーンと削除を管理する
        self.vehicle_manager.update(self)

        for vehicle in self.vehicles:
            self._register_perception(vehicle)

        # 1. 認識層
        # 各車両の周囲の車両を観測する
        observations = {}
        for vehicle in self.vehicles:
            observations[vehicle.vehicle_id] = self.perceptions[vehicle.vehicle_id].observe(vehicle, self.vehicles, self.road_network)

        # 2. 計画層
        # 各車両の制御入力を計算する
        for vehicle in self.vehicles:
            vehicle.plan(observations[vehicle.vehicle_id])

        # 3. 制御層
        # 各車両の状態を更新する
        for vehicle in self.vehicles:
            vehicle.update_state(self.dt)

        # 4. 後処理
        self.current_time += self.dt
        self._notify_observers()

    def _notify_observers(self):
        """観察者にシミュレーションの状態を通知するメソッド"""
        for observer in self.observers:
            observer.observe(self.vehicles, self.road_network, self.current_time)