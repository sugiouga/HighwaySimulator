#!/usr/bin/env python3
"""デバッグ用スクリプト：MOBIL車線変更の検査"""
import sys
sys.path.insert(0, '/'.join(__file__.split('/')[:-2]))

from utils.config_loader import MasterConfig
from manager.road_network import RoadNetwork
from manager.traffic_manager import TrafficManager

config = MasterConfig.from_yaml("config.yaml")

# ロードネットワークを設定
road_network = RoadNetwork(config)

# トラフィックマネージャーを作成
traffic_manager = TrafficManager(road_network, config, dt=0.1)

# シミュレーションを進める
print("=== Lane Configuration ===")
for lane_id, lane in road_network.lanes.items():
    print(f"Lane: {lane_id}")
    print(f"  Left lane: {lane.left_lane_id}")
    print(f"  Right lane: {lane.right_lane_id}")
    print()

print("=== Initial Vehicles ===")
for v in traffic_manager.vehicles:
    print(f"Vehicle {v.vehicle_id}: lane={v.lane_id}, x={v.x:.2f}, y={v.y:.2f}")

# 数ステップ進める
print("\n=== Stepping simulation ===")
for step in range(5):
    print(f"\nStep {step}:")
    traffic_manager.step()
    
    for v in traffic_manager.vehicles:
        if hasattr(v, 'policy') and hasattr(v.policy, '_active_lane_change'):
            print(f"  Vehicle {v.vehicle_id}: "
                  f"lane={v.lane_id}, "
                  f"active_lc={v.policy._active_lane_change}, "
                  f"x={v.x:.2f}, y={v.y:.2f}")
        else:
            print(f"  Vehicle {v.vehicle_id}: lane={v.lane_id}, x={v.x:.2f}, y={v.y:.2f}")

print("\n=== Checking perception output ===")
if traffic_manager.vehicles:
    v = traffic_manager.vehicles[0]
    obs = traffic_manager.perceptions[v.vehicle_id].observe(v, traffic_manager.vehicles, road_network)
    print(f"Vehicle {v.vehicle_id} observation:")
    print(f"  left_lane_id: {obs.get('left_lane_id')}")
    print(f"  right_lane_id: {obs.get('right_lane_id')}")
    print(f"  left_lane: {obs.get('left_lane')}")
    print(f"  right_lane: {obs.get('right_lane')}")
