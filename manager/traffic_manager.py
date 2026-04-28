from component.perception.perception import Perception

class TrafficManager:
    def __init__(self, road_network, dt=0.1):
        self.road_network = road_network
        self.dt = dt
        self.current_time = 0.0
        self.vehicles = []
        self.observers = []

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def add_observer(self, observer):
        self.observers.append(observer)

    def step(self):
        """シミュレーションの1ステップを進めるメソッド"""

        # 1. 認識層
        # 各車両の周囲の車両を観測する
        observations = {}
        for vehicle in self.vehicles:
            perception = Perception(sensor_range=vehicle.controller.sensor_range)
            observations[vehicle.id] = perception.observe(vehicle, self.road_network, self.vehicles)

        # 2. 計画層
        # 各車両の制御入力を計算する
        for vehicle in self.vehicles:
            vehicle.plan(observations[vehicle.id])

        # 3. 制御層
        # 各車両の状態を更新する
        for vehicle in self.vehicles:
            vehicle.update_state(self.dt)

        # 4. 後処理
        self.current_time += self.dt
        self._notify_observers()
        self._remove_out_of_bounds_vehicles()

    def _notify_observers(self):
        """観察者にシミュレーションの状態を通知するメソッド"""
        for observer in self.observers:
            observer.update(self.current_time, self.vehicles)

    def _remove_out_of_bounds_vehicles(self):
        """道路外に出た車両をシミュレーションから削除するメソッド"""
        self.vehicles = [v for v in self.vehicles if all(self.road_network.is_within_bounds(corner) for corner in v.get_corners())]