import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any

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

@dataclass(frozen=True)
class IDMConfig:
    desired_velocity: float
    desired_time_headway: float
    min_spacing: float
    comfortable_deceleration: float
@dataclass
class MasterConfig:
    vehicle: VehicleConfig
    idm: IDMConfig

    @classmethod
    def from_yaml(cls, file_path: str) -> 'MasterConfig':
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(vehicle=VehicleConfig(**config_dict['vehicle']),
                   idm=IDMConfig(**config_dict['idm'])
                   )