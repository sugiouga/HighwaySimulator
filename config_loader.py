import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union

@dataclass(frozen=True)
class SimulationConfig:
    time_step: float
    total_time: float

@dataclass(frozen=True)
class RoadNetworkConfig:
    lane_width: float
    lanes: List[Dict[str, Any]]
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
    fuel_consumption_model: ARRBModelParams

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
class ControllerConfig:
    type: str
    parameters: Union[IDMParams, MPCParams, RLParams, RLMPCParams]
    sensor_range: List[float, float]
    spawn_probability: float

@dataclass
class MasterConfig:
    simulation: SimulationConfig
    road_network: RoadNetworkConfig
    vehicle: Dict[str, VehicleConfig]
    controllers: Dict[str, ControllerConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'MasterConfig':
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # ARRBモデルのパラメータをVehicleConfigに組み込む
        arrb_params = ARRBModelParams(**config_dict['arrb_model'])
        vehicle_configs = {}
        for vehicle_type, vehicle_info in config_dict['vehicle'].items():
            vehicle_configs[vehicle_type] = VehicleConfig(
                mass=vehicle_info['mass'],
                length=vehicle_info['length'],
                width=vehicle_info['width'],
                min_velocity=vehicle_info['min_velocity'],
                max_velocity=vehicle_info['max_velocity'],
                min_acceleration=vehicle_info['min_acceleration'],
                max_acceleration=vehicle_info['max_acceleration'],
                max_steering_angle=vehicle_info['max_steering_angle'],
                max_steering_rate=vehicle_info['max_steering_rate'],
                fuel_consumption_model=arrb_params
            )
        config_dict['vehicle'] = vehicle_configs

        # コントローラーのパラメータをControllerConfigに変換
        controller_configs = {}
        for controller_name, controller_info in config_dict['controllers'].items():
            if controller_info['type'] == 'IDM':
                parameters = IDMParams(**controller_info['parameters'])
            elif controller_info['type'] == 'MPC':
                parameters = MPCParams(**controller_info['parameters'])
            elif controller_info['type'] == 'RL':
                parameters = RLParams(**controller_info['parameters'])
            elif controller_info['type'] == 'RLMPC':
                parameters = RLMPCParams(**controller_info['parameters'])
            else:
                raise ValueError(f"Unknown controller type: {controller_info['type']}")

            controller_configs[controller_name] = ControllerConfig(
                type=controller_info['type'],
                parameters=parameters,
                sensor_range=controller_info['sensor_range'],
                spawn_probability=controller_info.get('spawn_probability', 0)  # デフォルトは0
            )

        config_dict['controllers'] = controller_configs

        return cls(
            simulation=SimulationConfig(**config_dict['simulation']),
            road_network=RoadNetworkConfig(**config_dict['road_network']),
            vehicle=config_dict['vehicle'],
            controllers=config_dict['controllers']
        )
