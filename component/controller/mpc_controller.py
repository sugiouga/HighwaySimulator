from typing import Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

_WEIGHT_NAMES = ('w_pos', 'w_yaw', 'w_vel', 'w_accel', 'w_steer', 'w_steer_rate')


class MPCController:
    """5次多項式で生成した参照軌道を追従するモデル予測制御（MPC）コントローラ

    component.vehicle.kinematic_bicycle_model.KinematicBicycleModel と
    同一の運動モデル（重心基準・滑り角考慮のkinematic bicycle model）を内部で使用する。
        状態: [x, y, yaw, velocity, steering_angle]
        制御入力: [acceleration, steering_rate]

    コスト重み（w_pos等）は最適化のパラメータとして持ち、solve()の度に
    値を差し替えられる（DRLが重みを学習・出力するケースを想定）。

    cbf_params が有効な場合、以下2種類の制御バリア関数（CBF）による
    安全制約を追加する。
        - 周辺車両を楕円形の安全領域で守る衝突回避CBF（ハード制約）
        - 走行可能な車線範囲からの逸脱を防ぐ車線離脱防止CBF（スラック付きソフト制約）

    最適化問題はコンストラクタで一度だけ構築し、毎ステップ solve() で
    初期状態・参照軌道・重み・障害物情報をパラメータとして差し替えて再計算する
    （レシーディングホライズン方式）。
    """

    def __init__(
        self,
        horizon: int,
        dt: float,
        model_params: Dict,
        weights: Optional[Dict] = None,
        cbf_params: Optional[Dict] = None,
    ):
        self.horizon = horizon
        self.dt = dt
        self.last_solution: Optional[Dict[str, np.ndarray]] = None

        self.L = float(model_params['wheel_base'])
        self.lr = float(model_params.get('lr', self.L * 0.5))
        self.min_speed = float(model_params['min_speed'])
        self.max_speed = float(model_params['max_speed'])
        self.max_accel = float(model_params['max_acceleration'])
        self.min_accel = float(model_params['min_acceleration'])
        self.max_steer = float(model_params['max_steer_angle'])  # [rad]
        self.max_steer_rate = float(model_params['max_steer_rate'])
        self.vehicle_length = float(model_params.get('vehicle_length', self.L))
        self.vehicle_width = float(model_params.get('vehicle_width', 1.8))
        self.max_decel = abs(float(model_params.get('min_acceleration', -3.0)))

        self.default_weights = {
            'w_pos': 1.0, 'w_yaw': 0.0, 'w_vel': 1.0,
            'w_accel': 0.5, 'w_steer': 0.5, 'w_steer_rate': 0.5,
        }
        if weights:
            self.default_weights.update(weights)

        cbf_params = cbf_params or {}
        self.cbf_enabled = bool(cbf_params.get('enabled', False))
        self.max_obstacles = int(cbf_params.get('max_obstacles', 0)) if self.cbf_enabled else 0
        self.cbf_gamma = float(cbf_params.get('gamma', 0.7))
        ellipse_params = cbf_params.get('ellipse', {})
        self.use_dynamic_length = bool(ellipse_params.get('use_dynamic_length', False))
        self.fixed_length_margin = float(ellipse_params.get('fixed_length_margin', 10.0))
        self.ellipse_width_margin = float(ellipse_params.get('width_margin', 1.0))

        _lane_keys = ('lane_early_y_min', 'lane_early_y_max', 'lane_late_y_min', 'lane_late_y_max', 'lane_merge_end_x')
        self.lane_bounds_enabled = self.cbf_enabled and all(key in cbf_params for key in _lane_keys)
        if self.lane_bounds_enabled:
            # 走行可能範囲は、合流車線+本線を合わせた範囲（lane_early_*）から
            # 本線のみの範囲（lane_late_*）へ、x座標に関するシグモイド関数で滑らかに遷移する
            self.lane_early_y_min = float(cbf_params['lane_early_y_min'])
            self.lane_early_y_max = float(cbf_params['lane_early_y_max'])
            self.lane_late_y_min = float(cbf_params['lane_late_y_min'])
            self.lane_late_y_max = float(cbf_params['lane_late_y_max'])
            self.lane_merge_end_x = float(cbf_params['lane_merge_end_x'])
            self.lane_sigmoid_steepness = float(cbf_params.get('lane_sigmoid_steepness', 0.3))
            self.lane_margin = float(cbf_params.get('lane_margin', 0.0))
            self.lane_gamma = float(cbf_params.get('lane_gamma', 0.7))
            self.lane_slack_penalty = float(cbf_params.get('lane_slack_penalty', 1000.0))

        self._build_problem()

    def _build_problem(self):
        horizon = self.horizon
        opti = ca.Opti()

        x = opti.variable(horizon + 1)
        y = opti.variable(horizon + 1)
        yaw = opti.variable(horizon + 1)
        v = opti.variable(horizon + 1)
        accel = opti.variable(horizon)
        steer = opti.variable(horizon)
        steer_rate = opti.variable(horizon)

        # 初期状態・参照軌道・コスト重みはパラメータとして持ち、solve()の度に値だけ差し替える
        x0_param = opti.parameter(4)      # x, y, yaw, v
        steer0_param = opti.parameter(1)  # 現在の操舵角
        ref_param = opti.parameter(4, horizon + 1)  # 各列が (ref_x, ref_y, ref_yaw, ref_v)
        weight_params = {name: opti.parameter(1) for name in _WEIGHT_NAMES}

        opti.subject_to(x[0] == x0_param[0])
        opti.subject_to(y[0] == x0_param[1])
        opti.subject_to(yaw[0] == x0_param[2])
        opti.subject_to(v[0] == x0_param[3])

        slack = opti.variable(horizon + 1) if self.lane_bounds_enabled else None

        prev_steer = steer0_param
        for t in range(horizon):
            # 滑り角: KinematicBicycleModelと同じ arctan((lr/L) * tan(steer))
            slip_angle = ca.atan2(self.lr * ca.tan(steer[t]), self.L)

            opti.subject_to(x[t + 1] == x[t] + v[t] * ca.cos(yaw[t] + slip_angle) * self.dt)
            opti.subject_to(y[t + 1] == y[t] + v[t] * ca.sin(yaw[t] + slip_angle) * self.dt)
            opti.subject_to(yaw[t + 1] == yaw[t] + (v[t] / self.L) * ca.tan(steer[t]) * self.dt)
            opti.subject_to(v[t + 1] == v[t] + accel[t] * self.dt)

            opti.subject_to(opti.bounded(self.min_accel, accel[t], self.max_accel))
            opti.subject_to(opti.bounded(-self.max_steer, steer[t], self.max_steer))
            opti.subject_to(opti.bounded(self.min_speed, v[t + 1], self.max_speed))

            # 操舵角速度は前ステップの操舵角（t=0では現在の実車の操舵角）からの差分
            opti.subject_to(steer_rate[t] == (steer[t] - prev_steer) / self.dt)
            opti.subject_to(opti.bounded(-self.max_steer_rate, steer_rate[t], self.max_steer_rate))
            prev_steer = steer[t]

        # ---- 車線離脱防止CBF（シグモイド関数で走行可能範囲を表現、スラック付きソフト制約） ----
        if self.lane_bounds_enabled:
            opti.subject_to(slack >= 0)

            def sigmoid_bounds(x_t):
                # x座標に関するシグモイド関数で、合流車線+本線の範囲(early)から
                # 本線のみの範囲(late)へ滑らかに遷移するy_min, y_maxを返す
                s = 1.0 / (1.0 + ca.exp(-self.lane_sigmoid_steepness * (x_t - self.lane_merge_end_x)))
                y_min_t = self.lane_early_y_min + (self.lane_late_y_min - self.lane_early_y_min) * s
                y_max_t = self.lane_early_y_max + (self.lane_late_y_max - self.lane_early_y_max) * s
                return y_min_t - self.lane_margin, y_max_t + self.lane_margin

            h_lower = []
            h_upper = []
            for t in range(horizon + 1):
                y_min_t, y_max_t = sigmoid_bounds(x[t])
                h_lower.append(y[t] - y_min_t)
                h_upper.append(y_max_t - y[t])

            opti.subject_to(h_lower[0] + slack[0] >= 0)
            opti.subject_to(h_upper[0] + slack[0] >= 0)
            for t in range(horizon):
                # 離散CBF制約: h_{t+1} >= gamma * h_t （スラックにより逸脱を許容しつつ強くペナルティ）
                opti.subject_to(h_lower[t + 1] - self.lane_gamma * h_lower[t] + slack[t + 1] >= 0)
                opti.subject_to(h_upper[t + 1] - self.lane_gamma * h_upper[t] + slack[t + 1] >= 0)

        # ---- 周辺車両との楕円CBF（衝突回避、ハード制約） ----
        # 自車・障害物それぞれを向きθ_iを持つ楕円とみなし、
        #   h(x) = D(x) - (R1(x) + R2(x))
        # で衝突バリア関数を定義する。
        #   D: 自車・障害物の中心間距離
        #   R_i: 楕円iの中心から見て、相手方向 d = (D中心を結ぶ単位ベクトル) への実効半径
        #        A_i = R(θ_i) diag(1/a_i^2, 1/b_i^2) R(θ_i)^T として R_i = 1 / sqrt(d^T A_i d)
        # h > 0: 2つの楕円が重ならない（安全）, h <= 0: 重なる可能性（危険）
        obs_params = None
        if self.max_obstacles > 0:
            obs_x0 = opti.parameter(self.max_obstacles)
            obs_y0 = opti.parameter(self.max_obstacles)
            obs_vx = opti.parameter(self.max_obstacles)
            obs_vy = opti.parameter(self.max_obstacles)
            obs_yaw = opti.parameter(self.max_obstacles)
            obs_params = (obs_x0, obs_y0, obs_vx, obs_vy, obs_yaw)

            b = self.ellipse_width_margin + self.vehicle_width / 2.0

            def effective_radius(a, b, theta, d_x, d_y):
                cos_t = ca.cos(theta)
                sin_t = ca.sin(theta)
                a_xx = (cos_t ** 2) / (a ** 2) + (sin_t ** 2) / (b ** 2)
                a_yy = (sin_t ** 2) / (a ** 2) + (cos_t ** 2) / (b ** 2)
                a_xy = (1.0 / (a ** 2) - 1.0 / (b ** 2)) * sin_t * cos_t
                quad_form = a_xx * d_x ** 2 + 2.0 * a_xy * d_x * d_y + a_yy * d_y ** 2
                return 1.0 / ca.sqrt(quad_form + 1e-9)

            for i in range(self.max_obstacles):
                h_prev = None
                for t in range(horizon + 1):
                    obs_x_t = obs_x0[i] + obs_vx[i] * t * self.dt
                    obs_y_t = obs_y0[i] + obs_vy[i] * t * self.dt

                    dx = obs_x_t - x[t]
                    dy = obs_y_t - y[t]
                    D = ca.sqrt(dx ** 2 + dy ** 2 + 1e-6)
                    d_x = dx / D
                    d_y = dy / D

                    if self.use_dynamic_length:
                        stopping_distance = (v[t] ** 2) / (2.0 * self.max_decel)
                        a = stopping_distance + self.vehicle_length / 2.0
                    else:
                        a = self.fixed_length_margin + self.vehicle_length / 2.0

                    r1 = effective_radius(a, b, yaw[t], d_x, d_y)      # 自車楕円の実効半径
                    r2 = effective_radius(a, b, obs_yaw[i], d_x, d_y)  # 障害物楕円の実効半径
                    h_t = D - (r1 + r2)

                    if t == 0:
                        opti.subject_to(h_t >= 0)
                    else:
                        # 離散CBF制約: h_{t+1} >= gamma * h_t
                        opti.subject_to(h_t - self.cbf_gamma * h_prev >= 0)
                    h_prev = h_t

        # ---- コスト関数 ----
        cost = 0
        for t in range(horizon):
            cost += weight_params['w_pos'] * ((x[t] - ref_param[0, t]) ** 2 + (y[t] - ref_param[1, t]) ** 2)
            cost += weight_params['w_yaw'] * (yaw[t] - ref_param[2, t]) ** 2
            cost += weight_params['w_vel'] * (v[t] - ref_param[3, t]) ** 2
            cost += weight_params['w_accel'] * accel[t] ** 2
            cost += weight_params['w_steer'] * steer[t] ** 2
            cost += weight_params['w_steer_rate'] * steer_rate[t] ** 2
        if self.lane_bounds_enabled:
            cost += self.lane_slack_penalty * ca.sumsqr(slack)
        opti.minimize(cost)

        opti.solver('ipopt', {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})

        self._opti = opti
        self._vars = {
            'x': x, 'y': y, 'yaw': yaw, 'v': v,
            'accel': accel, 'steer': steer, 'steer_rate': steer_rate,
        }
        if self.lane_bounds_enabled:
            self._vars['slack'] = slack
        self._x0_param = x0_param
        self._steer0_param = steer0_param
        self._ref_param = ref_param
        self._weight_params = weight_params
        self._obs_params = obs_params

    def solve(
        self,
        x0: List[float],
        ref_trajectory: List[Tuple[float, float, float, float]],
        weights: Optional[Dict[str, float]] = None,
        obstacles: Optional[List[Dict[str, float]]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Args:
            x0: 現在状態 [x, y, yaw, velocity, steering_angle]
            ref_trajectory: 追従したい参照軌道 [(x, y, yaw, v), ...]
            weights: MPCコスト重み。省略時はコンストラクタに渡したデフォルト値を使用
            obstacles: 衝突回避対象の周辺車両 [{'x','y','vx','vy','yaw'}, ...]
                （max_obstacles件を超える分は無視。CBFが無効、または未指定の場合は考慮しない）

        Returns:
            'x','y','yaw','v','accel','steer','steer_rate' を含む解の辞書。
            ソルバーが失敗し、かつ直前の解も存在しない場合は None。
        """
        horizon = self.horizon
        ref_array = np.zeros((4, horizon + 1))
        for t in range(horizon + 1):
            point = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            ref_array[:, t] = point[:4]

        self._opti.set_value(self._x0_param, x0[:4])
        self._opti.set_value(self._steer0_param, float(x0[4]) if len(x0) > 4 else 0.0)
        self._opti.set_value(self._ref_param, ref_array)

        merged_weights = dict(self.default_weights)
        if weights:
            merged_weights.update(weights)
        for name, param in self._weight_params.items():
            self._opti.set_value(param, float(merged_weights[name]))

        if self._obs_params is not None:
            obs_x0, obs_y0, obs_vx, obs_vy, obs_yaw = self._obs_params
            # 未使用スロットはego位置から遠く離れたダミー障害物として扱い、CBF制約を常に非活性にする
            far_x0 = np.full(self.max_obstacles, float(x0[0]) - 1.0e4)
            far_y0 = np.full(self.max_obstacles, float(x0[1]))
            far_vx = np.zeros(self.max_obstacles)
            far_vy = np.zeros(self.max_obstacles)
            far_yaw = np.zeros(self.max_obstacles)
            for i, obs in enumerate((obstacles or [])[: self.max_obstacles]):
                far_x0[i] = float(obs['x'])
                far_y0[i] = float(obs['y'])
                far_vx[i] = float(obs.get('vx', 0.0))
                far_vy[i] = float(obs.get('vy', 0.0))
                far_yaw[i] = float(obs.get('yaw', 0.0))
            self._opti.set_value(obs_x0, far_x0)
            self._opti.set_value(obs_y0, far_y0)
            self._opti.set_value(obs_vx, far_vx)
            self._opti.set_value(obs_vy, far_vy)
            self._opti.set_value(obs_yaw, far_yaw)

        self._set_warm_start(x0, ref_array)

        try:
            sol = self._opti.solve()
            result = {name: np.atleast_1d(sol.value(var)) for name, var in self._vars.items()}
            self.last_solution = result
            return result
        except RuntimeError:
            return self.last_solution

    def _set_warm_start(self, x0: List[float], ref_array: np.ndarray) -> None:
        horizon = self.horizon

        def shift(values, length, fallback):
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size >= length:
                return np.concatenate([arr[1:length], arr[length - 1:length]])
            return fallback

        if self.last_solution is not None:
            x_guess = shift(self.last_solution['x'], horizon + 1, ref_array[0].copy())
            y_guess = shift(self.last_solution['y'], horizon + 1, ref_array[1].copy())
            yaw_guess = shift(self.last_solution['yaw'], horizon + 1, ref_array[2].copy())
            v_guess = shift(self.last_solution['v'], horizon + 1, ref_array[3].copy())
            accel_guess = shift(self.last_solution['accel'], horizon, np.zeros(horizon))
            steer_guess = shift(self.last_solution['steer'], horizon, np.zeros(horizon))
            steer_rate_guess = shift(self.last_solution['steer_rate'], horizon, np.zeros(horizon))
            slack_guess = (
                shift(self.last_solution['slack'], horizon + 1, np.zeros(horizon + 1))
                if self.lane_bounds_enabled and 'slack' in self.last_solution
                else np.zeros(horizon + 1)
            )
        else:
            x_guess, y_guess = ref_array[0].copy(), ref_array[1].copy()
            yaw_guess, v_guess = ref_array[2].copy(), ref_array[3].copy()
            accel_guess = np.zeros(horizon)
            steer_guess = np.zeros(horizon)
            steer_rate_guess = np.zeros(horizon)
            slack_guess = np.zeros(horizon + 1)

        x_guess[0], y_guess[0], yaw_guess[0], v_guess[0] = x0[0], x0[1], x0[2], x0[3]

        self._opti.set_initial(self._vars['x'], x_guess)
        self._opti.set_initial(self._vars['y'], y_guess)
        self._opti.set_initial(self._vars['yaw'], yaw_guess)
        self._opti.set_initial(self._vars['v'], v_guess)
        self._opti.set_initial(self._vars['accel'], accel_guess)
        self._opti.set_initial(self._vars['steer'], steer_guess)
        self._opti.set_initial(self._vars['steer_rate'], steer_rate_guess)
        if self.lane_bounds_enabled:
            self._opti.set_initial(self._vars['slack'], slack_guess)

    @staticmethod
    def get_first_action(result: Dict[str, np.ndarray]) -> List[float]:
        """解の最初のステップの制御入力 [acceleration, steering_rate] を取り出す"""
        return [float(result['accel'][0]), float(result['steer_rate'][0])]

    def reset(self) -> None:
        """エピソード切り替え時などにウォームスタート履歴をクリアする"""
        self.last_solution = None
