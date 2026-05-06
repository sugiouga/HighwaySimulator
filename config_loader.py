import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union

@dataclass(frozen=True)
class SimulationConfig:
    time_step: float
    total_time: float

@dataclass(frozen=True)
class VisualizationConfig:
    enable: bool
    screen_width: int
    screen_height: int
    screen_color: str
    caption: str
    ppm: float
    origin_x: float

@dataclass(frozen=True)
class RoadNetworkConfig:
    lane_width: float
    lanes: List[Dict[str, Any]]
    init_spawns: List[Dict[str, Any]]
    spawn_points: List[Dict[str, Any]]
@dataclass(frozen=True)
class ARRBModelParams:
    delta: float
    gamma1: float
    gamma2: float
    d_1: float
    d_2: float
    d_3: float
@dataclass(frozen=True)
class VehicleConfig:
    mass: float
    length: float
    width: float
    min_velocity: float
    max_velocity: float
    min_acceleration: float
    max_acceleration: float
    max_steering_angle: float
    max_steering_rate: float
    model: str = "kinematic_bicycle"

@dataclass(frozen=True)
class IDMParams:
    desired_velocity: float
    desired_time_headway: float
    min_spacing: float
    comfortable_deceleration: float

@dataclass(frozen=True)
class MPCParams:
    # MPCのパラメータを定義するクラス
    pass
@dataclass(frozen=True)
class RLParams:
    # 強化学習のパラメータを定義するクラス
    pass

@dataclass(frozen=True)
class RLMPCParams:
    # RLMPCのパラメータを定義するクラス
    pass

@dataclass(frozen=True)
class PolicyConfig:
    id: str
    type: str
    parameters: Union[IDMParams, MPCParams, RLParams, RLMPCParams]
    sensor_range: List[float]
    color: str

@dataclass
class MasterConfig:
    simulation: SimulationConfig
    visualization: VisualizationConfig
    road_network: RoadNetworkConfig
    vehicle: VehicleConfig
    policies: Dict[str, PolicyConfig] = field(default_factory=dict)
    arrb_model: ARRBModelParams = field(default=None)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'MasterConfig':
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # ポリシーのパラメータをPolicyConfigに変換
        policy_configs = {}
        policies_data = config_dict['policies']
        if isinstance(policies_data, list):
            policy_items = ((policy_info['policy_id'], policy_info) for policy_info in policies_data)
        else:
            policy_items = policies_data.items()

        for policy_name, policy_info in policy_items:
            if policy_info['type'] == 'IDM':
                parameters = IDMParams(**policy_info['parameters'])
            elif policy_info['type'] == 'MPC':
                parameters = MPCParams(**policy_info['parameters'])
            elif policy_info['type'] == 'RL':
                parameters = RLParams(**policy_info['parameters'])
            elif policy_info['type'] == 'RLMPC':
                parameters = RLMPCParams(**policy_info['parameters'])
            else:
                raise ValueError(f"Unknown policy type: {policy_info['type']}")

            policy_configs[policy_name] = PolicyConfig(
                id=policy_info.get('policy_id', policy_name),
                type=policy_info['type'],
                parameters=parameters,
                sensor_range=policy_info['sensor_range'] if isinstance(policy_info['sensor_range'], list) else [
                    policy_info['sensor_range']['front_distance'],
                    policy_info['sensor_range']['side_distance']
                ],
                color=policy_info.get('color', 'blue')
            )

        config_dict['policies'] = policy_configs

        return cls(
            simulation=SimulationConfig(**config_dict['simulation']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            road_network=RoadNetworkConfig(**config_dict['road_network']),
            vehicle=VehicleConfig(**config_dict['vehicle']),
            policies=config_dict['policies'],
            arrb_model=ARRBModelParams(**config_dict['arrb_model'])
        )
