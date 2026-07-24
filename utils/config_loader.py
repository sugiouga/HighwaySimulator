import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Tuple

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
    frame_rate: int  # フレームレートを追加

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
    acceleration_penalty: RewardParams
    jerk_penalty: RewardParams
    steering_angle_penalty: RewardParams
    steering_rate_penalty: RewardParams

@dataclass(frozen=True)
class SACConfig:
    policy: str = "MlpPolicy"
    total_timesteps: int = 200000
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
    callback: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RLMPCWeights:
    w_pos: float = 1.0
    w_yaw: float = 0.0
    w_vel: float = 1.0
    w_accel: float = 0.5
    w_steer: float = 0.5
    w_steer_rate: float = 0.5

@dataclass(frozen=True)
class RLMPCWeightLimits:
    # DRLが学習するMPCコスト重みの範囲 [min, max]
    w_pos: Tuple[float, float] = (0.1, 50.0)
    w_yaw: Tuple[float, float] = (0.0, 10.0)
    w_vel: Tuple[float, float] = (0.1, 50.0)
    w_accel: Tuple[float, float] = (0.01, 10.0)
    w_steer: Tuple[float, float] = (0.01, 10.0)
    w_steer_rate: Tuple[float, float] = (0.01, 10.0)

@dataclass(frozen=True)
class RLMPCCBFEllipseConfig:
    use_dynamic_length: bool = False
    fixed_length_margin: float = 10.0
    width_margin: float = 1.0

@dataclass(frozen=True)
class RLMPCCBFConfig:
    # 制御バリア関数（CBF）による安全制約の設定
    enabled: bool = True
    max_obstacles: int = 4  # CBFで衝突回避対象とする周辺車両の最大数
    nearby_vehicle_range: float = 20.0  # CBF対象を探索する半径[m]
    gamma: float = 0.7  # 車間楕円CBFの減衰係数 (0 < gamma <= 1)
    ellipse: RLMPCCBFEllipseConfig = field(default_factory=RLMPCCBFEllipseConfig)
    lane_margin: float = 0.0  # 車線離脱防止CBFのマージン[m]
    lane_gamma: float = 0.7  # 車線離脱防止CBFの減衰係数 (0 < gamma <= 1)
    lane_slack_penalty: float = 1000.0  # 車線離脱防止CBFのスラック変数ペナルティ重み
    # 車線離脱防止CBFの許容範囲は、合流車線・本線を合わせた範囲からシグモイド関数で
    # 本線のみの範囲へ滑らかに遷移させる（合流完了後は本線への収束を強制する）
    lane_sigmoid_steepness: float = 0.3  # シグモイド関数の傾き（大きいほど急峻）
    lane_sigmoid_transition_x: Optional[float] = None  # 遷移中心のx座標[m]。未指定時は合流車線終端を自動使用

@dataclass(frozen=True)
class RLMPCConfig:
    # DRLが意思決定（目標横位置・目標速度・計画時間・MPC重み）を行い、
    # 5次多項式で軌道を生成、MPCで追従制御するパイプライン用の設定
    mpc_horizon: int = 10
    mpc_time_step: float = 0.1
    planning_time_min: float = 1.0
    planning_time_max: float = 4.0
    weights: RLMPCWeights = field(default_factory=RLMPCWeights)
    weight_limits: RLMPCWeightLimits = field(default_factory=RLMPCWeightLimits)
    cbf: RLMPCCBFConfig = field(default_factory=RLMPCCBFConfig)

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
    rlmpc: RLMPCConfig = field(default_factory=RLMPCConfig)

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
            acceleration_penalty=RewardParams(**config_dict['reward']['acceleration_penalty']),
            jerk_penalty=RewardParams(**config_dict['reward']['jerk_penalty']),
            steering_angle_penalty=RewardParams(**config_dict['reward']['steering_angle_penalty']),
            steering_rate_penalty=RewardParams(**config_dict['reward']['steering_rate_penalty'])
        )
        config_dict['reward'] = reward_config

        sac_config = SACConfig(**config_dict.get('sac', {}))
        config_dict['sac'] = sac_config

        rlmpc_dict = dict(config_dict.get('rlmpc', {}))
        rlmpc_weights = RLMPCWeights(**rlmpc_dict.pop('weights', {}))

        weight_limits_dict = rlmpc_dict.pop('weight_limits', {})
        rlmpc_weight_limits = RLMPCWeightLimits(**{k: tuple(v) for k, v in weight_limits_dict.items()})

        cbf_dict = dict(rlmpc_dict.pop('cbf', {}))
        ellipse_dict = cbf_dict.pop('ellipse', {})
        rlmpc_cbf = RLMPCCBFConfig(ellipse=RLMPCCBFEllipseConfig(**ellipse_dict), **cbf_dict)

        rlmpc_config = RLMPCConfig(
            weights=rlmpc_weights,
            weight_limits=rlmpc_weight_limits,
            cbf=rlmpc_cbf,
            **rlmpc_dict,
        )
        config_dict['rlmpc'] = rlmpc_config

        return cls(
            simulation=SimulationConfig(**config_dict['simulation']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            road_network=RoadNetworkConfig(**config_dict['road_network']),
            vehicle=VehicleConfig(**config_dict['vehicle']),
            policies=config_dict['policies'],
            arrb_model=ARRBModelParams(**config_dict['arrb_model']),
            reward=config_dict['reward'],
            sac=config_dict['sac'],
            rlmpc=config_dict['rlmpc']
        )
