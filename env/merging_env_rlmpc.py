import numpy as np
from gymnasium import spaces

from env.merging_env import MergingEnv
from manager.road_network import RoadNetwork
from component.planner.quintic_polynomial import QuinticPolynomial
from component.controller.mpc_controller import MPCController

_WEIGHT_NAMES = ('w_pos', 'w_yaw', 'w_vel', 'w_accel', 'w_steer', 'w_steer_rate')


class MergingEnvRLMPC(MergingEnv):
    """DRLが意思決定、5次多項式で経路計画、MPCで追従制御を行う階層型のGym環境

    行動空間: [target_y, target_v, planning_time, w_pos, w_yaw, w_vel, w_accel, w_steer, w_steer_rate]
        - target_y: 目標横位置[m]。本線車線の中心／合流車線の中心の2択。
          SACは連続値のBox行動空間しか扱えないため、連続値のまま学習させ、
          step()内で「本線中心」「合流車線中心」のどちらか近い方へスナップする。
        - target_v: 目標速度[m/s]
        - planning_time: 目標状態に到達するまでの計画時間[s]
        - w_pos, w_yaw, w_vel, w_accel, w_steer, w_steer_rate: MPCのコスト重み。
          DRLが毎ステップ出力し、config.rlmpc.weight_limitsの範囲でMPCに渡す。

    DRLエージェントはこの9次元の「意思決定」のみを出力する。
    step() の内部で毎ステップ、現在状態から目標状態への5次多項式軌道を生成し、
    MPCの時間刻みでサンプリングした参照軌道をMPCで追従して
    実際の制御入力 [acceleration, steering_rate] を求め、基底クラスのstep()に渡す。

    MPCはconfig.rlmpc.cbfが有効な場合、以下の安全制約を課す。
        - 周辺車両（自車から近い順にmax_obstacles台）を楕円形の安全領域で守る衝突回避CBF
        - 走行可能範囲を「合流車線+本線」から「本線のみ」へシグモイド関数で滑らかに
          遷移させる車線離脱防止CBF（合流完了後は本線への収束を強制する）

    MPC（CasADi/ipopt）を毎シミュレーションステップ解くため、
    [acceleration, steering_rate] を直接学習する MergingEnv より計算コストが大きい。
    """

    def __init__(self, config):
        super().__init__(config)

        rlmpc_config = config.rlmpc
        self._planning_time_min = float(rlmpc_config.planning_time_min)
        self._planning_time_max = float(rlmpc_config.planning_time_max)
        self._mpc_dt = float(rlmpc_config.mpc_time_step)
        self._mpc_horizon = int(rlmpc_config.mpc_horizon)

        # 目標横位置は「本線車線の中心」「合流車線の中心」の2択に固定する
        main_center_y, merge_center_y, merge_lane_end_x = self._compute_lane_geometry(config)
        self._lane_center_candidates = (main_center_y, merge_center_y)
        y_low, y_high = sorted(self._lane_center_candidates)

        weight_limits = rlmpc_config.weight_limits
        weight_low = [getattr(weight_limits, name)[0] for name in _WEIGHT_NAMES]
        weight_high = [getattr(weight_limits, name)[1] for name in _WEIGHT_NAMES]

        self.action_space = spaces.Box(
            low=np.array([y_low, self._v_min, self._planning_time_min, *weight_low], dtype=np.float32),
            high=np.array([y_high, self._v_max, self._planning_time_max, *weight_high], dtype=np.float32),
            dtype=np.float32,
        )

        model_params = {
            'wheel_base': self.config.vehicle.length,
            'min_speed': self.config.vehicle.min_velocity,
            'max_speed': self.config.vehicle.max_velocity,
            'max_acceleration': self.config.vehicle.max_acceleration,
            'min_acceleration': self.config.vehicle.min_acceleration,
            'max_steer_angle': np.deg2rad(self.config.vehicle.max_steering_angle),
            # steering_rateはKinematicBicycleModel/MergingEnvの行動空間と同じく
            # 単位変換せずそのままの値を上下限として扱う
            'max_steer_rate': self.config.vehicle.max_steering_rate,
            'vehicle_length': self.config.vehicle.length,
            'vehicle_width': self.config.vehicle.width,
        }
        weights = {name: getattr(rlmpc_config.weights, name) for name in _WEIGHT_NAMES}

        cbf_config = rlmpc_config.cbf
        self._cbf_enabled = bool(cbf_config.enabled)
        self._cbf_nearby_range = float(cbf_config.nearby_vehicle_range)
        self._cbf_max_obstacles = int(cbf_config.max_obstacles) if self._cbf_enabled else 0

        # 車線離脱防止CBFの走行可能範囲:
        #   上限は常に本線車線の中心（y=main_center_y）
        #   下限は合流車線の中心（early、合流完了前）から本線車線の中心
        #     （late、合流完了後）へシグモイド関数で滑らかに遷移する
        #   （合流完了後は上限・下限とも本線中心に収束し、本線中心への追従を強制する）
        lane_early_y_min = min(main_center_y, merge_center_y)
        lane_early_y_max = max(main_center_y, merge_center_y)
        lane_late_y_min = main_center_y
        lane_late_y_max = main_center_y
        lane_merge_end_x = (
            float(cbf_config.lane_sigmoid_transition_x)
            if cbf_config.lane_sigmoid_transition_x is not None
            else float(merge_lane_end_x)
        )

        cbf_params = None
        if self._cbf_enabled:
            cbf_params = {
                'enabled': True,
                'max_obstacles': self._cbf_max_obstacles,
                'gamma': cbf_config.gamma,
                'ellipse': {
                    'use_dynamic_length': cbf_config.ellipse.use_dynamic_length,
                    'fixed_length_margin': cbf_config.ellipse.fixed_length_margin,
                    'width_margin': cbf_config.ellipse.width_margin,
                },
                'lane_early_y_min': lane_early_y_min,
                'lane_early_y_max': lane_early_y_max,
                'lane_late_y_min': lane_late_y_min,
                'lane_late_y_max': lane_late_y_max,
                'lane_merge_end_x': lane_merge_end_x,
                'lane_sigmoid_steepness': cbf_config.lane_sigmoid_steepness,
                'lane_margin': cbf_config.lane_margin,
                'lane_gamma': cbf_config.lane_gamma,
                'lane_slack_penalty': cbf_config.lane_slack_penalty,
            }

        self.mpc = MPCController(
            horizon=self._mpc_horizon,
            dt=self._mpc_dt,
            model_params=model_params,
            weights=weights,
            cbf_params=cbf_params,
        )

    @staticmethod
    def _compute_lane_geometry(config):
        """本線・合流車線の中心y座標と、合流車線の終端x座標を返す

        戻り値: (main_center_y, merge_center_y, merge_lane_end_x)

        ego車両の初期車線（road_network.init_vehiclesのis_ego車両）を合流車線とみなし、
        その隣接車線（adj_left/adj_rightで接続された車線）を本線とみなす。
        merge_lane_end_xは、車線離脱防止CBFのシグモイド遷移の中心として使う
        （lane_sigmoid_transition_xが未指定の場合のデフォルト値）。
        """
        road_network = RoadNetwork(config)

        ego_lane_id = None
        for vehicle_config in config.road_network.init_vehicles:
            is_ego = vehicle_config.get('is_ego', False) if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'is_ego', False)
            if is_ego:
                ego_lane_id = vehicle_config.get('lane_id') if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'lane_id')
                break
        if ego_lane_id is None:
            raise ValueError("road_network.init_vehiclesにis_ego=Trueの車両が見つかりません")

        merge_lane = road_network.get_lane(ego_lane_id)
        if merge_lane is None:
            raise ValueError(f"lane_id={ego_lane_id}の車線が見つかりません")

        main_lane = road_network.get_left_lane(ego_lane_id) or road_network.get_right_lane(ego_lane_id)
        if main_lane is None:
            raise ValueError(f"lane_id={ego_lane_id}に隣接する本線車線が見つかりません")

        main_center_y = float(np.mean(main_lane.waypoints[:, 1]))
        merge_center_y = float(np.mean(merge_lane.waypoints[:, 1]))
        merge_lane_end_x = float(np.max(merge_lane.waypoints[:, 0]))
        return main_center_y, merge_center_y, merge_lane_end_x

    def _collect_nearby_obstacles(self):
        """CBFの衝突回避対象とする周辺車両（自車から近い順にmax_obstacles台）を集める"""
        if self._cbf_max_obstacles <= 0:
            return []

        ego = self.ego_vehicle
        candidates = []
        for vehicle in self.traffic_manager.vehicles:
            if vehicle.vehicle_id == ego.vehicle_id:
                continue
            distance = float(np.hypot(vehicle.x - ego.x, vehicle.y - ego.y))
            if distance <= self._cbf_nearby_range:
                candidates.append((distance, vehicle))
        candidates.sort(key=lambda item: item[0])

        obstacles = []
        for _, vehicle in candidates[: self._cbf_max_obstacles]:
            obstacles.append({
                'x': float(vehicle.x),
                'y': float(vehicle.y),
                'vx': float(vehicle.velocity * np.cos(vehicle.yaw)),
                'vy': float(vehicle.velocity * np.sin(vehicle.yaw)),
                'yaw': float(vehicle.yaw),
            })
        return obstacles

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.mpc.reset()
        # Visualizerに対し、MPC/CBFが実際に適用されるRLMPCパイプラインであることを伝える
        # （直接[accel, steering_rate]を学習するMergingEnvではCBFは実際には適用されないため）
        # visualization.enable=falseの場合はself.visualizerが定義されないためgetattrで参照する
        visualizer = getattr(self, 'visualizer', None)
        if visualizer is not None:
            visualizer.rlmpc_active = True
        return observation, info

    def step(self, action):
        values = [float(v) for v in np.asarray(action, dtype=np.float64).reshape(-1)]
        raw_target_y, target_v, planning_time = values[0:3]
        weights = dict(zip(_WEIGHT_NAMES, values[3:9]))

        # 連続値で受け取った目標横位置を、本線／合流車線の中心のうち近い方へスナップする
        target_y = min(self._lane_center_candidates, key=lambda center: abs(center - raw_target_y))
        planning_time = float(np.clip(planning_time, self._planning_time_min, self._planning_time_max))

        ego_state = self.ego_vehicle.state  # [x, y, yaw, v, steering_angle]
        start = (float(ego_state[0]), float(ego_state[1]), float(ego_state[2]))
        end = (0.0, target_y, 0.0)  # x, yawは未使用（QuinticPolynomial側で終端xを再計算する）

        calc_point = QuinticPolynomial.generate_trajectory(
            start=start,
            end=end,
            T=planning_time,
            v0=float(ego_state[3]),
            vT=target_v,
            a0=0.0,
        )
        ref_trajectory = [
            calc_point(min(i * self._mpc_dt, planning_time))
            for i in range(self._mpc_horizon + 1)
        ]

        obstacles = self._collect_nearby_obstacles()
        result = self.mpc.solve(x0=list(ego_state), ref_trajectory=ref_trajectory, weights=weights, obstacles=obstacles)
        low_level_action = self.mpc.get_first_action(result) if result is not None else [0.0, 0.0]

        # Visualizerに5次多項式の目標軌道・MPCの予測軌道・DRLの目標位置/速度を渡す（可視化用）
        visualizer = getattr(self, 'visualizer', None)
        if visualizer is not None:
            # ref_trajectoryはMPCホライズン長（mpc_horizon*mpc_dt）で打ち切られているため、
            # 可視化用にはplanning_time全体をカバーする軌道を別途サンプリングし、
            # 目標位置まで描画されるようにする
            num_vis_points = max(self._mpc_horizon + 1, 30)
            full_trajectory = [
                calc_point(t) for t in np.linspace(0.0, planning_time, num_vis_points)
            ]
            visualizer.quintic_reference_trajectory = [(point[0], point[1]) for point in full_trajectory]
            visualizer.mpc_prediction_trajectory = (
                list(zip(result['x'], result['y'])) if result is not None else []
            )
            # calc_point(planning_time)で計画時間ちょうどの目標状態を厳密に評価する
            # （ref_trajectoryはMPCホライズン長で打ち切られており、
            # planning_timeがホライズンより長い場合は目標点まで到達していないため）
            target_x, target_y_exact, _target_yaw, target_v_exact = calc_point(planning_time)
            visualizer.drl_target_point = (target_x, target_y_exact)
            visualizer.drl_target_velocity = target_v_exact

        return super().step(low_level_action)
