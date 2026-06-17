import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

@dataclass(frozen=True)
class SimulationConfig:
    time_step: float
    warmup_time: float
    total_time: float
    goal_x: float

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
    init_vehicles: List[Dict[str, Any]]
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
class MOBILParams:
    desired_velocity: float
    desired_time_headway: float
    min_spacing: float
    comfortable_deceleration: float
    politeness_factor: float
    acceleration_threshold: float
    # optional lane-change gap overrides
    lane_change_min_front_gap: Optional[float] = None
    lane_change_min_rear_gap: Optional[float] = None
    lane_change_cooldown_steps: Optional[int] = None
    lane_change_steering_smoothing: Optional[float] = None
    lane_change_duration: Optional[float] = None
    lane_change_pure_pursuit_gain: Optional[float] = None
    lane_change_minimum_lf: Optional[float] = None
    # PID制御ゲイン
    pid_kp: Optional[float] = None
    pid_ki: Optional[float] = None
    pid_kd: Optional[float] = None
    pid_integral_limit: Optional[float] = None

@dataclass(frozen=True)
class MPCParams:
    # MPCのパラメータを定義するクラス
    mpc_horizon: int
    mpc_time_step: float

@dataclass(frozen=True)
class RLParams:
    # 強化学習のパラメータを定義するクラス
    pass

@dataclass(frozen=True)
class RLMPCParams:
    # RLMPCのパラメータを定義するクラス
    mpc_horizon: int
    mpc_time_step: float

@dataclass(frozen=True)
class GhostParams:
    # ゴースト車両のパラメータを定義するクラス
    pass

@dataclass(frozen=True)
class PolicyConfig:
    id: str
    type: str
    parameters: Union[IDMParams, MOBILParams, MPCParams, RLParams, RLMPCParams, GhostParams]
    sensor_range: List[float]
    color: str

@dataclass(frozen=True)
class RewardParams:
    enabled: bool
    weight: float
    std: float
    target: float

@dataclass(frozen=True)
class RewardConfig:
    success_reward: float
    collision_penalty: float
    lane_deviation_penalty: float
    timeout_penalty: float
    y_position_reward: RewardParams
    target_velocity_reward: RewardParams
    following_vehicle_deceleration_penalty: RewardParams
    jerk_penalty: RewardParams

@dataclass(frozen=True)
class SACConfig:
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: Union[str, float] = "auto"
    target_update_interval: int = 1
    device: str = "auto"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MasterConfig:
    simulation: SimulationConfig
    visualization: VisualizationConfig
    road_network: RoadNetworkConfig
    vehicle: VehicleConfig
    policies: Dict[str, PolicyConfig] = field(default_factory=dict)
    arrb_model: ARRBModelParams = field(default=None)
    reward: RewardConfig = field(default=None)
    sac: SACConfig = field(default_factory=SACConfig)

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
            elif policy_info['type'] == 'MOBIL':
                parameters = MOBILParams(**policy_info['parameters'])
            elif policy_info['type'] == 'MPC':
                parameters = MPCParams(**policy_info['parameters'])
            elif policy_info['type'] == 'RL':
                parameters = RLParams(**policy_info['parameters'])
            elif policy_info['type'] == 'RLMPC':
                parameters = RLMPCParams(**policy_info['parameters'])
            elif policy_info['type'] == 'Ghost':
                parameters = GhostParams(**policy_info['parameters'])
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

        reward_config = RewardConfig(
            success_reward=config_dict['reward']['success_reward'],
            collision_penalty=config_dict['reward']['collision_penalty'],
            lane_deviation_penalty=config_dict['reward']['lane_deviation_penalty'],
            timeout_penalty=config_dict['reward']['timeout_penalty'],
            y_position_reward=RewardParams(**config_dict['reward']['y_position_reward']),
            target_velocity_reward=RewardParams(**config_dict['reward']['target_velocity_reward']),
            following_vehicle_deceleration_penalty=RewardParams(**config_dict['reward']['following_vehicle_deceleration_penalty']),
            jerk_penalty=RewardParams(**config_dict['reward']['jerk_penalty'])
        )
        config_dict['reward'] = reward_config

        sac_config = SACConfig(**config_dict.get('sac', {}))
        config_dict['sac'] = sac_config

        return cls(
            simulation=SimulationConfig(**config_dict['simulation']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            road_network=RoadNetworkConfig(**config_dict['road_network']),
            vehicle=VehicleConfig(**config_dict['vehicle']),
            policies=config_dict['policies'],
            arrb_model=ARRBModelParams(**config_dict['arrb_model']),
            reward=config_dict['reward'],
            sac=config_dict['sac']
        )
