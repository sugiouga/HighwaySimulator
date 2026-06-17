from .base_policy import BasePolicy
from .idm_policy import IDMPolicy
from typing import List, Optional, Tuple
import numpy as np


class MOBILPolicy(BasePolicy):
    """MOBIL（Minimizing Overall Braking Induced by Lane changes）ポリシーの実装"""

    def __init__(self, vehicle_config, mobil_config):
        self.vehicle_config = vehicle_config
        self.mobil_config = mobil_config

        self.desired_velocity = mobil_config.desired_velocity
        self.desired_time_headway = mobil_config.desired_time_headway
        self.min_spacing = mobil_config.min_spacing
        self.comfortable_deceleration = mobil_config.comfortable_deceleration
        self.politeness_factor = mobil_config.politeness_factor
        self.acceleration_threshold = getattr(mobil_config, "acceleration_threshold", 0.0)

        # lane-change gap config: allow overriding required front/rear gaps
        self.lane_change_min_front_gap = getattr(mobil_config, "lane_change_min_front_gap", None)
        self.lane_change_min_rear_gap = getattr(mobil_config, "lane_change_min_rear_gap", None)
        _cooldown_raw = getattr(mobil_config, "lane_change_cooldown_steps", None)
        self.lane_change_cooldown_steps = int(_cooldown_raw) if (_cooldown_raw is not None) else 0

        # Pure Pursuit制御ゲイン
        _pc_gain = getattr(mobil_config, "lane_change_pure_pursuit_gain", None)
        self.lane_change_pure_pursuit_gain = float(_pc_gain) if (_pc_gain is not None) else 1.0
        _min_lf = getattr(mobil_config, "lane_change_minimum_lf", None)
        self.lane_change_minimum_lf = float(_min_lf) if (_min_lf is not None) else 5.0  # 最小ルックアヘッド距離[m]

        self.max_acceleration = vehicle_config.max_acceleration
        self.max_deceleration = abs(getattr(vehicle_config, "min_acceleration", -3.0))
        self.max_steering_rate = getattr(vehicle_config, "max_steering_rate", 0.0)

        self.idm_policy = IDMPolicy(vehicle_config, mobil_config)
        # lane change persistent state
        self._active_lane_change = False
        self._active_direction: Optional[str] = None
        self._lane_change_target_lane = None
        self._lane_change_start_x: Optional[float] = None
        self._lane_change_start_y: Optional[float] = None
        self._lane_change_start_yaw: Optional[float] = None
        self._lane_change_poly_coeffs: Optional[Tuple[float, float, float, float, float, float]] = None
        self._lane_change_steps = 0
        self._lane_change_timeout = 50  # step limit to avoid stuck state
        self._lane_change_cooldown_remaining = 0
        # PID制御用の状態変数
        self._lane_change_error_integral = 0.0
        self._lane_change_prev_error: Optional[float] = None
        self._lane_change_filtered_error_derivative = 0.0
        self._lane_change_prev_steering_rate: Optional[float] = None
        # Trajectory tracking parameters
        _look = getattr(mobil_config, "lane_change_lookahead_distance", None)
        self._lane_change_lookahead_distance = float(_look) if (_look is not None) else 20.0
        _trans = getattr(mobil_config, "lane_change_transition_length", None)
        self._lane_change_transition_length = float(_trans) if (_trans is not None) else 50.0
        _smooth = getattr(mobil_config, "lane_change_steering_smoothing", None)
        self._lane_change_steering_smoothing = float(_smooth) if (_smooth is not None) else 0.6
        
        # PID制御パラメータ
        _kp = getattr(mobil_config, "pid_kp", None)
        self.pid_kp = float(_kp) if (_kp is not None) else 0.5  # 比例ゲイン
        _ki = getattr(mobil_config, "pid_ki", None)
        self.pid_ki = float(_ki) if (_ki is not None) else 0.01  # 積分ゲイン
        _kd = getattr(mobil_config, "pid_kd", None)
        self.pid_kd = float(_kd) if (_kd is not None) else 0.1   # 微分ゲイン
        _pil = getattr(mobil_config, "pid_integral_limit", None)
        self.pid_integral_limit = float(_pil) if (_pil is not None) else 10.0  # 積分値の上限
        _pdt = getattr(mobil_config, "pid_dt", None)
        self.pid_dt = float(_pdt) if (_pdt is not None) else 0.1  # 制御周期

    def action(self, state: list, environment_info: dict) -> list:
        """MOBILに基づいて制御入力を計算するメソッド。"""
        if not self._active_lane_change and self._lane_change_cooldown_remaining > 0:
            self._lane_change_cooldown_remaining -= 1
            return self._lane_keep_action(state, environment_info)

        # If a lane change is already active, continue it until completion
        if self._active_lane_change:
            cont_action = self._continue_lane_change(state, environment_info)
            if cont_action is not None:
                return cont_action
            # if continuation returns None, lane-change finished; fall through to normal decision

        # otherwise evaluate incentives
        left_score, left_action = self._lane_change_action(state, environment_info, direction="left")
        right_score, right_action = self._lane_change_action(state, environment_info, direction="right")

        candidates: List[Tuple[float, List[float]]] = []
        if left_action is not None:
            candidates.append((left_score, left_action, "left"))
        if right_action is not None:
            candidates.append((right_score, right_action, "right"))

        if candidates:
            best_score, best_action, best_dir = max(candidates, key=lambda item: item[0])
            if best_score > 0.0:
                # start persistent lane change
                self._start_lane_change(best_dir, best_action, state, environment_info)
                return best_action

        return self._lane_keep_action(state, environment_info)

    def _lane_keep_action(self, state: list, environment_info: dict) -> List[float]:
        """非合流時は自車線中心を追従しつつ、縦方向はIDMで制御する。"""
        ego_vehicle = self._state_as_vehicle_proxy(state)
        front = environment_info.get("front")
        acceleration = float(np.clip(self._idm_acceleration(ego_vehicle, front), self.vehicle_config.min_acceleration, self.vehicle_config.max_acceleration))
        ego_lane = environment_info.get("ego_lane")
        steering_rate = self._pure_pursuit_steering_to_lane_center(state, ego_lane)
        return [acceleration, steering_rate]

    def _lane_change_action(self, state: list, environment_info: dict, direction: str) -> Tuple[float, Optional[List[float]]]:
        # Check if target lane exists by verifying the lane_id is not None
        target_lane_id = environment_info.get(f"{direction}_lane_id")
        if target_lane_id is None:
            return float("-inf"), None

        ego_vehicle = self._state_as_vehicle_proxy(state)
        current_front = environment_info.get("front")
        current_rear = environment_info.get("rear")
        target_front = environment_info.get(f"{direction}_front")
        target_rear = environment_info.get(f"{direction}_rear")

        ego_before = self._idm_acceleration(ego_vehicle, current_front)
        ego_after = self._idm_acceleration(ego_vehicle, target_front)

        # impact on target-lane rear vehicle (we're joining, so it gets us ahead)
        if target_rear is not None:
            target_rear_before = self._idm_acceleration(target_rear, target_front)
            target_rear_after = self._idm_acceleration(target_rear, ego_vehicle)
        else:
            target_rear_before = 0.0
            target_rear_after = 0.0

        # impact on current-lane rear vehicle (we're leaving, so it has the same front vehicle)
        # but this is neutral unless there's a gap opening; typically no change in front
        if current_rear is not None:
            current_rear_before = self._idm_acceleration(current_rear, current_front)
            current_rear_after = self._idm_acceleration(current_rear, current_front)  # front unchanged
        else:
            current_rear_before = 0.0
            current_rear_after = 0.0

        # incentive includes both neighbors: target-lane rear and current-lane rear
        incentive = ego_after - ego_before + self.politeness_factor * ((target_rear_after - target_rear_before) + (current_rear_after - current_rear_before))
        incentive -= float(self.acceleration_threshold)

        if not self._is_lane_change_safe(target_rear, target_front, ego_vehicle):
            return incentive, None

        target_lane = self._get_target_lane(environment_info, direction)
        steering_rate = self._pure_pursuit_steering_to_quintic(state, target_lane)

        acceleration = float(np.clip(ego_after, self.vehicle_config.min_acceleration, self.vehicle_config.max_acceleration))
        return incentive, [acceleration, steering_rate]

    def _start_lane_change(self, direction: str, action: List[float], state: list, environment_info: dict):
        """Mark lane change as active and store target information."""
        self._active_lane_change = True
        self._active_direction = direction
        self._lane_change_target_lane = self._get_target_lane(environment_info, direction)
        self._lane_change_start_x = self._get_state_value(state, 0)
        self._lane_change_start_y = self._get_state_value(state, 1)
        self._lane_change_start_yaw = self._get_state_value(state, 2)
        target_y = self._lane_center_y(self._lane_change_target_lane)
        self._lane_change_poly_coeffs = self._build_quintic_coefficients(self._lane_change_start_y, target_y)
        self._lane_change_steps = 0
        # PID制御用の状態変数初期化
        self._lane_change_error_integral = 0.0
        self._lane_change_prev_error = None
        self._lane_change_filtered_error_derivative = 0.0
        self._lane_change_prev_steering_rate = None

    def _continue_lane_change(self, state: list, environment_info: dict) -> Optional[List[float]]:
        """Continue issuing lane-change control using Pure Pursuit toward the quintic trajectory."""
        self._lane_change_steps += 1
        if self._lane_change_steps > self._lane_change_timeout:
            self._finish_lane_change(cooldown=True)
            return None

        # compute steering to follow the target lane centerline
        if self._lane_change_target_lane is None or self._lane_change_poly_coeffs is None or self._lane_change_start_x is None:
            self._active_lane_change = False
            return None

        # current vehicle state
        ego_vehicle = self._state_as_vehicle_proxy(state)
        current_x = self._get_state_value(state, 0)
        current_y = self._get_state_value(state, 1)

        # Pure Pursuit steering to the quintic polynomial trajectory
        steering_rate = self._pure_pursuit_steering_to_quintic(state, self._lane_change_target_lane)

        # maintain acceleration according to IDM for comfort
        front = environment_info.get("front")
        acceleration = float(np.clip(self._idm_acceleration(ego_vehicle, front), self.vehicle_config.min_acceleration, self.vehicle_config.max_acceleration))

        # stop condition: near the end of the polynomial trajectory and close to target lane center
        tau = self._get_lane_change_tau(current_x)
        _, target_y, _, _ = self._evaluate_quintic_profile(tau)
        lateral_error = float(np.nan_to_num(target_y - current_y, nan=0.0, posinf=0.0, neginf=0.0))
        target_lane_completed = tau >= 0.999 and abs(lateral_error) <= max(self._lane_change_target_lane.width * 0.25, 0.5)
        if target_lane_completed:
            self._finish_lane_change(cooldown=True)
            return None

        return [acceleration, steering_rate]

    def _finish_lane_change(self, cooldown: bool = False):
        self._active_lane_change = False
        self._active_direction = None
        self._lane_change_target_lane = None
        self._lane_change_start_x = None
        self._lane_change_start_y = None
        self._lane_change_start_yaw = None
        self._lane_change_poly_coeffs = None
        self._lane_change_steps = 0
        if cooldown:
            self._lane_change_cooldown_remaining = max(int(self.lane_change_cooldown_steps), 0)

    def _pure_pursuit_steering_to_quintic(self, state: list, target_lane) -> float:
        """PID制御で5次多項式軌跡を追従。"""
        if target_lane is None or self._lane_change_poly_coeffs is None or self._lane_change_start_x is None:
            return 0.0

        current_x = self._get_state_value(state, 0)
        current_y = self._get_state_value(state, 1)
        current_yaw = self._get_state_value(state, 2)
        current_velocity = self._get_state_value(state, 3)
        current_steering = self._get_state_value(state, 4)

        # 現在のtauを計算
        tau = self._get_lane_change_tau(current_x)
        _, target_y_current, _, _ = self._evaluate_quintic_profile(tau)

        # 横方向誤差を計算
        lateral_error = float(np.nan_to_num(target_y_current - current_y, nan=0.0, posinf=0.0, neginf=0.0))

        # PID制御を計算
        steering_rate = self._calculate_pid_steering(lateral_error, current_steering)
        
        return float(np.clip(steering_rate, -self.max_steering_rate, self.max_steering_rate))

    def _calculate_pid_steering(self, lateral_error: float, current_steering: float) -> float:
        """PID制御ロジック: 横方向誤差からステアリングレートを計算"""
        # 比例項 (P)
        p_term = self.pid_kp * lateral_error
        
        # 積分項 (I)
        self._lane_change_error_integral += lateral_error * self.pid_dt
        self._lane_change_error_integral = float(np.clip(
            self._lane_change_error_integral, 
            -self.pid_integral_limit, 
            self.pid_integral_limit
        ))
        i_term = self.pid_ki * self._lane_change_error_integral
        
        # 微分項 (D)
        if self._lane_change_prev_error is not None:
            error_derivative = (lateral_error - self._lane_change_prev_error) / max(self.pid_dt, 1e-6)
            # Low-pass filter on derivative to reduce noise
            self._lane_change_filtered_error_derivative = (
                0.7 * self._lane_change_filtered_error_derivative + 
                0.3 * error_derivative
            )
        else:
            self._lane_change_filtered_error_derivative = 0.0
        
        d_term = self.pid_kd * self._lane_change_filtered_error_derivative
        self._lane_change_prev_error = lateral_error
        
        # PID出力（ステアリング角の目標値）
        steering_command_deg = p_term + i_term + d_term
        
        # ステアリングレートを計算
        dt = max(self.pid_dt, 1e-3)
        steering_rate = (steering_command_deg - current_steering) / dt
        
        return steering_rate

    def _pure_pursuit_steering_to_lane_center(self, state: list, lane) -> float:
        """非合流時の車線中心追従（PID制御）。"""
        if lane is None:
            return 0.0

        current_y = self._get_state_value(state, 1)
        current_steering = self._get_state_value(state, 4)
        target_y = self._lane_center_y(lane)
        
        # 横方向誤差を計算
        lateral_error = float(np.nan_to_num(target_y - current_y, nan=0.0, posinf=0.0, neginf=0.0))

        # PID制御を計算
        steering_rate = self._calculate_pid_steering(lateral_error, current_steering)
        
        return float(np.clip(steering_rate, -self.max_steering_rate, self.max_steering_rate))

    def _build_quintic_coefficients(self, y0: float, y1: float) -> Tuple[float, float, float, float, float, float]:
        dy = float(y1 - y0)
        return (
            float(y0),
            0.0,
            0.0,
            10.0 * dy,
            -15.0 * dy,
            6.0 * dy,
        )

    def _evaluate_quintic_profile(self, tau: float) -> Tuple[float, float, float, float]:
        if self._lane_change_start_x is None or self._lane_change_poly_coeffs is None:
            return 0.0, 0.0, 0.0, 0.0

        tau = float(np.clip(tau, 0.0, 1.0))
        a0, a1, a2, a3, a4, a5 = self._lane_change_poly_coeffs
        y = a0 + a1 * tau + a2 * tau ** 2 + a3 * tau ** 3 + a4 * tau ** 4 + a5 * tau ** 5
        x = float(self._lane_change_start_x + self._lane_change_transition_length * tau)
        dy_dtau = a1 + 2.0 * a2 * tau + 3.0 * a3 * tau ** 2 + 4.0 * a4 * tau ** 3 + 5.0 * a5 * tau ** 4
        d2y_dtau2 = 2.0 * a2 + 6.0 * a3 * tau + 12.0 * a4 * tau ** 2 + 20.0 * a5 * tau ** 3
        dx_dtau = max(self._lane_change_transition_length, 1e-3)
        dy_dx = dy_dtau / dx_dtau
        d2y_dx2 = d2y_dtau2 / (dx_dtau ** 2)
        curvature = d2y_dx2 / max((1.0 + dy_dx ** 2) ** 1.5, 1e-6)
        return x, y, dy_dx, curvature

    def _get_lane_change_tau(self, current_x: float) -> float:
        """合流進度を0-1で返す（移動距離ベース）"""
        if self._lane_change_start_x is None:
            return 0.0
        transition_length = max(self._lane_change_transition_length, 1e-3)
        traveled_x = current_x - self._lane_change_start_x
        return float(np.clip(traveled_x / transition_length, 0.0, 1.0))

    def _is_lane_change_safe(self, target_rear, target_front, ego_vehicle) -> bool:
        """Check safety of lane change by ensuring sufficient gaps ahead and behind

        Conditions:
        - if there's no rear vehicle in target lane, ok for rear-side check
        - if there's no front vehicle in target lane, ok for front-side check
        - require gap ahead >= min_spacing + ego_vel * desired_time_headway
        - require gap behind >= min_spacing + rear_vel * desired_time_headway
        - also ensure the rear vehicle would not need to brake harder than comfortable_deceleration
        """
        # check rear vehicle safety (braking demand)
        if target_rear is not None:
            rear_after = self._idm_acceleration(target_rear, ego_vehicle)
            if rear_after < -float(self.comfortable_deceleration):
                return False

        # check distance gaps
        ego_x = float(np.nan_to_num(getattr(ego_vehicle, "x", 0.0), nan=0.0))
        ego_y = float(np.nan_to_num(getattr(ego_vehicle, "y", 0.0), nan=0.0))
        ego_yaw = float(np.nan_to_num(getattr(ego_vehicle, "yaw", 0.0), nan=0.0))
        ego_v = float(np.nan_to_num(getattr(ego_vehicle, "velocity", 0.0), nan=0.0))

        # required gaps
        # allow config override: explicit numeric gaps take precedence
        if self.lane_change_min_front_gap is not None:
            required_ahead = float(self.lane_change_min_front_gap)
        else:
            required_ahead = max(float(self.min_spacing), ego_v * float(self.desired_time_headway))
        if target_front is not None:
            front_x = float(np.nan_to_num(getattr(target_front, "x", ego_x), nan=ego_x))
            front_y = float(np.nan_to_num(getattr(target_front, "y", ego_y), nan=ego_y))
            # projection of vector (front - ego) onto ego heading
            rel_xf = front_x - ego_x
            rel_yf = front_y - ego_y
            gap_front = rel_xf * np.cos(ego_yaw) + rel_yf * np.sin(ego_yaw)
            if gap_front < required_ahead:
                return False

        if target_rear is not None:
            rear_x = float(np.nan_to_num(getattr(target_rear, "x", ego_x), nan=ego_x))
            rear_y = float(np.nan_to_num(getattr(target_rear, "y", ego_y), nan=ego_y))
            rear_v = float(np.nan_to_num(getattr(target_rear, "velocity", 0.0), nan=0.0))
            # projection of vector (ego - rear) onto ego heading
            rel_xr = ego_x - rear_x
            rel_yr = ego_y - rear_y
            gap_rear = rel_xr * np.cos(ego_yaw) + rel_yr * np.sin(ego_yaw)
            if self.lane_change_min_rear_gap is not None:
                required_rear = float(self.lane_change_min_rear_gap)
            else:
                required_rear = max(float(self.min_spacing), rear_v * float(self.desired_time_headway))
            if gap_rear < required_rear:
                return False

        return True

    def _get_target_lane(self, environment_info: dict, direction: str):
        lane = environment_info.get(f"{direction}_lane")
        return lane

    def _lane_center_y(self, lane) -> float:
        if lane is None:
            return 0.0
        if hasattr(lane, "waypoints"):
            return float(np.mean(np.asarray(lane.waypoints, dtype=np.float64)[:, 1]))
        return 0.0

    def _state_as_vehicle_proxy(self, state: list):
        class VehicleProxy:
            def __init__(self, x, y, yaw, velocity):
                self.x = x
                self.y = y
                self.yaw = yaw
                self.velocity = velocity

        return VehicleProxy(
            self._get_state_value(state, 0),
            self._get_state_value(state, 1),
            self._get_state_value(state, 2),
            self._get_state_value(state, 3),
        )

    def _idm_acceleration(self, ego_vehicle, lead_vehicle) -> float:
        velocity = float(np.nan_to_num(getattr(ego_vehicle, "velocity", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        desired_velocity = max(float(self.desired_velocity), 1e-3)

        if lead_vehicle is None:
            acceleration = self.max_acceleration * (1.0 - (velocity / desired_velocity) ** 4)
            return float(np.clip(np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0), self.vehicle_config.min_acceleration, self.vehicle_config.max_acceleration))

        lead_velocity = float(np.nan_to_num(getattr(lead_vehicle, "velocity", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        ego_x = float(np.nan_to_num(getattr(ego_vehicle, "x", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        ego_y = float(np.nan_to_num(getattr(ego_vehicle, "y", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        lead_x = float(np.nan_to_num(getattr(lead_vehicle, "x", ego_x), nan=ego_x, posinf=ego_x, neginf=ego_x))
        lead_y = float(np.nan_to_num(getattr(lead_vehicle, "y", ego_y), nan=ego_y, posinf=ego_y, neginf=ego_y))
        ego_yaw = float(np.nan_to_num(getattr(ego_vehicle, "yaw", 0.0), nan=0.0, posinf=0.0, neginf=0.0))

        rel_x = lead_x - ego_x
        rel_y = lead_y - ego_y
        longitudinal_gap = rel_x * np.cos(ego_yaw) + rel_y * np.sin(ego_yaw)
        ego_length = float(getattr(ego_vehicle, "length", self.vehicle_config.length))
        lead_length = float(getattr(lead_vehicle, "length", ego_length))
        gap = max(longitudinal_gap - 0.5 * (ego_length + lead_length), 0.1)
        delta_v = velocity - lead_velocity

        max_acceleration = max(float(self.max_acceleration), 1e-6)
        comfortable_deceleration = max(float(self.comfortable_deceleration), 1e-6)
        denom = 2.0 * np.sqrt(max_acceleration * comfortable_deceleration)
        s_alpha = self.min_spacing + max(0.0, velocity * self.desired_time_headway + (velocity * delta_v) / max(denom, 1e-6))
        v_ratio = np.clip(velocity / desired_velocity, -20.0, 20.0)
        interaction = np.clip((s_alpha / gap) ** 2, 0.0, 1e6)
        acceleration = max_acceleration * (1.0 - v_ratio ** 4 - interaction)

        return float(np.clip(np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0), self.vehicle_config.min_acceleration, self.vehicle_config.max_acceleration))

    @staticmethod
    def _get_state_value(state: list, index: int) -> float:
        return float(np.nan_to_num(state[index], nan=0.0, posinf=0.0, neginf=0.0))


MobilPolicy = MOBILPolicy