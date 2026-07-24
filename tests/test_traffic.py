"""
config.yamlに基づき、全シミュレーション時間を実行して交通流を確認するテストケース
Visualizerを用いてシミュレーションを描画することで、交通流の様子を目視で確認できるようにする
"""

import unittest
import sys
from pathlib import Path

# ワークスペースのルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import MasterConfig
from manager.traffic_manager import TrafficManager
from manager.road_network import RoadNetwork
from observer.visualizer import Visualizer


class TestTrafficFlow(unittest.TestCase):
    """交通流のテストケース"""

    @classmethod
    def setUpClass(cls):
        """テストクラスのセットアップ"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        cls.config = MasterConfig.from_yaml(str(config_path))

    def setUp(self):
        """各テストメソッドの実行前にセットアップ"""
        self.road_network = RoadNetwork(self.config)
        self.traffic_manager = TrafficManager(
            self.road_network,
            self.config,
            dt=self.config.simulation.time_step
        )

    def test_traffic_manager_initialization(self):
        """TrafficManagerが正しく初期化されるかを確認するテスト"""
        self.assertIsNotNone(self.traffic_manager)
        self.assertEqual(self.traffic_manager.current_time, 0.0)
        self.assertIsNotNone(self.traffic_manager.vehicles)
        self.assertIsNotNone(self.traffic_manager.road_network)

    def test_road_network_initialization(self):
        """RoadNetworkが正しく初期化されるかを確認するテスト"""
        self.assertIsNotNone(self.road_network)
        self.assertGreater(len(self.road_network.lanes), 0)
        for lane_id, lane in self.road_network.lanes.items():
            self.assertIsNotNone(lane)

    def test_simulation_runs_full_time(self):
        """シミュレーションが全時間実行できるかを確認するテスト"""
        total_time = self.config.simulation.total_time
        expected_steps = int(total_time / self.config.simulation.time_step)

        # シミュレーション実行
        step_count = 0
        while self.traffic_manager.current_time < total_time:
            self.traffic_manager.step()
            step_count += 1

        # 時間が正しく進んでいることを確認
        self.assertAlmostEqual(
            self.traffic_manager.current_time,
            total_time,
            places=1
        )
        self.assertEqual(step_count, expected_steps)

    def test_vehicles_spawn_and_exist(self):
        """車両がスポーンして存在することを確認するテスト"""
        # 初期化時点で車両が存在するはず
        if len(self.traffic_manager.vehicles) > 0:
            self.assertGreater(len(self.traffic_manager.vehicles), 0)
            for vehicle in self.traffic_manager.vehicles:
                self.assertIsNotNone(vehicle.vehicle_id)
                self.assertIsNotNone(vehicle.policy)

    def test_simulation_with_visualizer(self):
        """Visualizerを使用してシミュレーションを実行するテスト"""
        if self.config.visualization.enable:
            visualizer = Visualizer(self.config)
            self.traffic_manager.add_observer(visualizer)

            # 全シミュレーション時間を実行して可視化がエラーなく実行されるか確認
            total_time = self.config.simulation.total_time
            while self.traffic_manager.current_time < total_time:
                self.traffic_manager.step()

            # シミュレーションが完了したことを確認
            self.assertAlmostEqual(
                self.traffic_manager.current_time,
                total_time,
                places=1
            )

    def test_observers_are_notified(self):
        """オブザーバーが通知を受けることを確認するテスト"""
        class DummyObserver:
            def __init__(self):
                self.call_count = 0
                self.last_time = None

            def observe(self, vehicles, road_network, current_time):
                self.call_count += 1
                self.last_time = current_time

        dummy_observer = DummyObserver()
        self.traffic_manager.add_observer(dummy_observer)

        # いくつかのステップを実行
        steps = 5
        for _ in range(steps):
            self.traffic_manager.step()

        # オブザーバーが呼び出されたことを確認
        self.assertEqual(dummy_observer.call_count, steps)
        self.assertIsNotNone(dummy_observer.last_time)
        self.assertGreater(dummy_observer.last_time, 0.0)

    def test_vehicle_positions_change(self):
        """車両の位置が時間とともに変化することを確認するテスト"""
        if len(self.traffic_manager.vehicles) > 0:
            vehicle = self.traffic_manager.vehicles[0]
            initial_x = vehicle.x
            initial_y = vehicle.y

            # いくつかのステップを実行
            steps_to_run = 10
            for _ in range(steps_to_run):
                self.traffic_manager.step()

            # 少なくとも一つの車両の位置が変わっていることを確認
            vehicles_changed = any(
                v.x != initial_x or v.y != initial_y
                for v in self.traffic_manager.vehicles
            )
            # 車両が存在し、シミュレーションが進んでいることを確認
            self.assertGreater(len(self.traffic_manager.vehicles), 0)

    def test_time_step_consistency(self):
        """時間ステップが一貫していることを確認するテスト"""
        dt = self.config.simulation.time_step
        previous_time = self.traffic_manager.current_time

        for _ in range(5):
            self.traffic_manager.step()
            current_time = self.traffic_manager.current_time
            time_diff = current_time - previous_time
            self.assertAlmostEqual(time_diff, dt, places=5)
            previous_time = current_time


if __name__ == "__main__":
    unittest.main()
