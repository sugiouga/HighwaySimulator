#!/usr/bin/env python3
"""多項式軌跡ベースの車線変更制御を検証"""
import sys
sys.path.insert(0, '/'.join(__file__.split('/')[:-2]))

from utils.config_loader import MasterConfig
from manager.road_network import RoadNetwork
from manager.traffic_manager import TrafficManager

config = MasterConfig.from_yaml("config.yaml")
road_network = RoadNetwork(config)
traffic_manager = TrafficManager(road_network, config, dt=0.1)

print("=== Polynomial Trajectory-Based Lane Change Control ===\n")

# シミュレーションを進める
total_steps = 100
print(f"Running simulation for {total_steps * 0.1:.1f}s...\n")

lane_change_events = []

for step in range(total_steps):
    traffic_manager.step()
    
    # MOBIL車両の状態をトラッキング
    for v in traffic_manager.vehicles:
        if hasattr(v, 'policy') and hasattr(v.policy, '_active_lane_change'):
            if v.policy._active_lane_change:
                event = {
                    'step': step,
                    'time': step * 0.1,
                    'vehicle_id': v.vehicle_id,
                    'x': v.x,
                    'y': v.y,
                    'lane_id': v.lane_id,
                    'heading': v.yaw,
                    'velocity': v.velocity,
                    'steering_angle': v.steering_angle,
                    'lane_change_steps': v.policy._lane_change_steps,
                    'target_y': v.policy._lane_change_target_y,
                }
                lane_change_events.append(event)

if lane_change_events:
    print("=== Lane Change Events Detected ===\n")
    
    # イベントをグループ化（連続したステップをまとめる）
    grouped_events = []
    current_group = [lane_change_events[0]]
    
    for event in lane_change_events[1:]:
        if event['vehicle_id'] == current_group[0]['vehicle_id'] and event['step'] == current_group[-1]['step'] + 1:
            current_group.append(event)
        else:
            grouped_events.append(current_group)
            current_group = [event]
    grouped_events.append(current_group)
    
    for group_idx, group in enumerate(grouped_events):
        print(f"Lane Change #{group_idx + 1}:")
        start_event = group[0]
        end_event = group[-1]
        
        print(f"  Vehicle: {start_event['vehicle_id']}")
        print(f"  Start: t={start_event['time']:.2f}s, position=({start_event['x']:.2f}, {start_event['y']:.2f}), lane={start_event['lane_id']}")
        print(f"  End:   t={end_event['time']:.2f}s, position=({end_event['x']:.2f}, {end_event['y']:.2f}), lane={end_event['lane_id']}")
        print(f"  Duration: {end_event['time'] - start_event['time']:.2f}s ({len(group)} steps)")
        print(f"  Lateral displacement: {end_event['y'] - start_event['y']:.4f}m")
        print(f"  Target Y: {start_event['target_y']:.4f}m")
        print()
else:
    print("No lane change events detected.")
    print("\nVehicles at end of simulation:")
    for v in traffic_manager.vehicles[:5]:
        print(f"  {v.vehicle_id}: lane={v.lane_id}, y={v.y:.2f}m, policy={type(v.policy).__name__}")

print("\n=== Control Parameters ===")
for policy_name, policy in config.policies.items():
    if policy.type == "MOBIL":
        print(f"MOBIL Policy Configuration:")
        print(f"  lane_change_kp: {policy.parameters.lane_change_kp}")
        print(f"  lane_change_kd: {policy.parameters.lane_change_kd}")
        print(f"  lane_change_min_front_gap: {policy.parameters.lane_change_min_front_gap}m")
        print(f"  lane_change_min_rear_gap: {policy.parameters.lane_change_min_rear_gap}m")
