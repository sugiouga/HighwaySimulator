import pygame
import numpy as np
from observer.base_observer import BaseObserver

class Visualizer(BaseObserver):
    def __init__(self, config):
        super().__init__(config)
        pygame.init()
        self.visualization_config = config.visualization
        self.screen_width = self.visualization_config.screen_width
        self.screen_height = self.visualization_config.screen_height
        self.screen_color = self.visualization_config.screen_color
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.frame_rate = self.visualization_config.frame_rate

        # 描画パラメータの設定
        pygame.display.set_caption(self.visualization_config.caption)
        self.ppm = self.visualization_config.ppm  # ピクセルあたりのメートル数
        self.origin_x = self.visualization_config.origin_x  # 描画開始位置(ピクセル)
        self.font = pygame.font.SysFont("Consolas", 14)
        self.clock = pygame.time.Clock()
        self.road_network_initialized = False

        # RLMPCパイプライン（MergingEnvRLMPC）使用時のみMergingEnvRLMPC側からTrueに設定される。
        # 直接[accel, steering_rate]を学習する経路（MergingEnv）ではMPC/CBFが実際には
        # 適用されていないため、CBFの可視化はrlmpc_active=Trueのときのみ行う
        self.rlmpc_active = False

        # 車線離脱防止CBFの上下限カーブ（ピクセル座標）。初回update()時に計算してキャッシュする
        self.lane_cbf_initialized = False
        self.lane_cbf_lower_points = None
        self.lane_cbf_upper_points = None

        # 自動運転車（ego）のこれまでの走行軌跡（ピクセル座標）
        self.ego_trajectory_points = []

        # RLMPCパイプライン使用時、MergingEnvRLMPC.step()から毎ステップ設定される
        # 5次多項式の目標軌道 [(x, y), ...] とMPCの予測軌道 [(x, y), ...]（ワールド座標）
        self.quintic_reference_trajectory = []
        self.mpc_prediction_trajectory = []

        # DRLが意思決定した目標位置(x, y)と目標速度[m/s]（ワールド座標）
        self.drl_target_point = None
        self.drl_target_velocity = None

    def observe(self, vehicles, road_network, current_time):
        self.update(vehicles, road_network, current_time)

    def update(self, vehicles, road_network, current_time):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # 道路ネットワークのピクセル座標を初回時に計算・保存
        if not self.road_network_initialized:
            self._compute_road_pixel_waypoints(road_network)
            self.road_network_initialized = True

        # 車線離脱防止CBFの上下限カーブを初回時に計算・保存
        if not self.lane_cbf_initialized:
            self._compute_lane_cbf_curve(road_network)
            self.lane_cbf_initialized = True

        # 画面を背景色でクリア
        self.screen.fill(self.screen_color)

        # 道路を描画
        for lane in road_network.lanes.values():
            self._draw_road(lane)

        rlmpc_config = getattr(self.config, 'rlmpc', None)
        # RLMPCパイプライン使用時（rlmpc_active=True）かつCBFが有効なときのみ描画する
        cbf_enabled = self.rlmpc_active and rlmpc_config is not None and rlmpc_config.cbf.enabled
        ego_vehicle = next((vehicle for vehicle in vehicles if getattr(vehicle, 'is_ego', False)), None)

        # 車線離脱防止CBFの上下限を描画。CBFが実際に適用されるego車両が
        # 存在するときのみ描画する
        if cbf_enabled:
            self._draw_lane_cbf()

        # 衝突回避CBFの対象車両を特定（egoから近い順にmax_obstacles台）
        cbf_targets = self._compute_cbf_target_vehicles(vehicles)

        # egoの走行軌跡を記録・描画
        if ego_vehicle is not None:
            self._record_ego_trajectory(ego_vehicle)
        self._draw_ego_trajectory()

        # RLMPCパイプライン使用時、5次多項式の目標軌道とMPCの予測軌道、
        # DRLが意思決定した目標位置・目標速度を描画
        if self.rlmpc_active:
            self._draw_quintic_trajectory()
            self._draw_mpc_prediction()
            self._draw_drl_target()

        # 車両を描画
        for vehicle in vehicles:
            if cbf_enabled and vehicle is ego_vehicle:
                # 自車楕円（h=D-(R1+R2)のR1側）。CBFが実際に適用される
                # 周辺車両が存在するときのみ描画する
                self._draw_ellipse_cbf(vehicle, color=(220, 20, 60))
            elif vehicle in cbf_targets:
                # 障害物楕円（h=D-(R1+R2)のR2側）
                self._draw_ellipse_cbf(vehicle, color=(30, 144, 255))
            self._draw_vehicle(vehicle)

        # 時刻を表示
        time_text = self.font.render(f"Time: {current_time:.1f} s", True, (0, 0, 0))
        self.screen.blit(time_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.frame_rate)  # フレームレートを30FPSに設定

    def _compute_road_pixel_waypoints(self, road_network):
        """道路のピクセル座標を計算して保存するメソッド
        Args
        - road_network: 道路ネットワークオブジェクト
        """
        for lane in road_network.lanes.values():
            # 左側の境界線をピクセル座標に変換
            for waypoint in lane.left_waypoints:
                x, y = waypoint
                pixel_x = self.origin_x + x * self.ppm
                pixel_y = self.screen_height // 2 - y * self.ppm
                lane.left_pixel_waypoints.append((pixel_x, pixel_y))

            # 右側の境界線をピクセル座標に変換
            for waypoint in lane.right_waypoints:
                x, y = waypoint
                pixel_x = self.origin_x + x * self.ppm
                pixel_y = self.screen_height // 2 - y * self.ppm
                lane.right_pixel_waypoints.append((pixel_x, pixel_y))

            # 中心線をピクセル座標に変換
            for waypoint in lane.waypoints:
                x, y = waypoint
                pixel_x = self.origin_x + x * self.ppm
                pixel_y = self.screen_height // 2 - y * self.ppm
                lane.center_pixel_waypoints.append((pixel_x, pixel_y))

    def _draw_road(self, lane):
        """道路を描画するメソッド
        Args
        - lane: 描画する車線オブジェクト
        """
        # 保存されたピクセル座標を使用して描画
        left_pixel_waypoints = lane.left_pixel_waypoints
        right_pixel_waypoints = lane.right_pixel_waypoints
        center_pixel_waypoints = lane.center_pixel_waypoints

        # 車線を塗りつぶされた多角形として描画
        if len(left_pixel_waypoints) > 1 and len(right_pixel_waypoints) > 1:
            lane_polygon = left_pixel_waypoints + right_pixel_waypoints[::-1]
            pygame.draw.polygon(self.screen, (200, 200, 200), lane_polygon)

        # 左側の境界線を描画（黄色）
        if len(left_pixel_waypoints) > 1:
            pygame.draw.lines(self.screen, (255, 255, 0), False, left_pixel_waypoints, 2)

        # 右側の境界線を描画（黄色）
        if len(right_pixel_waypoints) > 1:
            pygame.draw.lines(self.screen, (255, 255, 0), False, right_pixel_waypoints, 2)

        # 車線の中心線を描画（白色）
        if len(center_pixel_waypoints) > 1:
            pygame.draw.lines(self.screen, (255, 255, 255), False, center_pixel_waypoints, 1)

    def _find_ego_lane_id(self):
        """config.road_network.init_vehiclesからego車両の初期車線IDを取得する"""
        for vehicle_config in self.config.road_network.init_vehicles:
            is_ego = vehicle_config.get('is_ego', False) if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'is_ego', False)
            if is_ego:
                return vehicle_config.get('lane_id') if isinstance(vehicle_config, dict) else getattr(vehicle_config, 'lane_id')
        return None

    def _compute_lane_cbf_curve(self, road_network):
        """車線離脱防止CBF（シグモイド関数）の上下限カーブをピクセル座標として計算・保存する
        Args
        - road_network: 道路ネットワークオブジェクト
        """
        rlmpc_config = getattr(self.config, 'rlmpc', None)
        if rlmpc_config is None or not rlmpc_config.cbf.enabled:
            return

        ego_lane_id = self._find_ego_lane_id()
        if ego_lane_id is None:
            return

        merge_lane = road_network.get_lane(ego_lane_id)
        main_lane = road_network.get_left_lane(ego_lane_id) or road_network.get_right_lane(ego_lane_id)
        if merge_lane is None or main_lane is None:
            return

        cbf = rlmpc_config.cbf
        main_center_y = float(np.mean(main_lane.waypoints[:, 1]))
        merge_center_y = float(np.mean(merge_lane.waypoints[:, 1]))
        # 上限は常に本線車線の中心。下限は合流車線の中心（early）から
        # 本線車線の中心（late）へシグモイド関数で遷移する（MPCController側と合わせる）
        early_y_min, early_y_max = min(main_center_y, merge_center_y), max(main_center_y, merge_center_y)
        late_y_min, late_y_max = main_center_y, main_center_y

        merge_end_x = cbf.lane_sigmoid_transition_x
        if merge_end_x is None:
            merge_end_x = float(np.max(merge_lane.waypoints[:, 0]))

        # 本線は複数の車線セグメント（例: main_1_1/main_1_2/main_1_3）に分割されている
        # 場合があるため、本線車線と同じ中心y座標を持つ全セグメントを本線とみなし、
        # その終端まで車線制約カーブを描画する
        main_segments = [
            lane for lane in road_network.lanes.values()
            if abs(float(np.mean(lane.waypoints[:, 1])) - main_center_y) < 1e-6
        ]
        main_x_min = min(float(np.min(lane.waypoints[:, 0])) for lane in main_segments)
        main_x_max = max(float(np.max(lane.waypoints[:, 0])) for lane in main_segments)

        x_start = float(min(main_x_min, np.min(merge_lane.waypoints[:, 0])))
        x_end = float(max(main_x_max, np.max(merge_lane.waypoints[:, 0])))
        x_values = np.linspace(x_start, x_end, 200)

        sigmoid = 1.0 / (1.0 + np.exp(-cbf.lane_sigmoid_steepness * (x_values - merge_end_x)))
        y_min = early_y_min + (late_y_min - early_y_min) * sigmoid - cbf.lane_margin
        y_max = early_y_max + (late_y_max - early_y_max) * sigmoid + cbf.lane_margin

        self.lane_cbf_lower_points = [
            (self.origin_x + x * self.ppm, self.screen_height // 2 - y * self.ppm)
            for x, y in zip(x_values, y_min)
        ]
        self.lane_cbf_upper_points = [
            (self.origin_x + x * self.ppm, self.screen_height // 2 - y * self.ppm)
            for x, y in zip(x_values, y_max)
        ]

    def _draw_lane_cbf(self):
        """車線離脱防止CBFの上下限を描画するメソッド"""
        if self.lane_cbf_lower_points:
            pygame.draw.lines(self.screen, (30, 144, 255), False, self.lane_cbf_lower_points, 2)
        if self.lane_cbf_upper_points:
            pygame.draw.lines(self.screen, (220, 20, 60), False, self.lane_cbf_upper_points, 2)

    def _record_ego_trajectory(self, ego_vehicle):
        """egoのこれまでの走行軌跡をピクセル座標として記録するメソッド
        Args
        - ego_vehicle: 自動運転車（ego）の車両オブジェクト
        """
        pixel_x = self.origin_x + ego_vehicle.x * self.ppm
        pixel_y = self.screen_height // 2 - ego_vehicle.y * self.ppm
        self.ego_trajectory_points.append((pixel_x, pixel_y))

    def _draw_ego_trajectory(self):
        """egoのこれまでの走行軌跡を赤線で描画するメソッド"""
        if len(self.ego_trajectory_points) > 1:
            pygame.draw.lines(self.screen, (200, 0, 0), False, self.ego_trajectory_points, 2)

    def _world_points_to_pixels(self, points):
        """(x, y)のワールド座標リストをピクセル座標リストに変換する"""
        return [
            (self.origin_x + x * self.ppm, self.screen_height // 2 - y * self.ppm)
            for x, y in points
        ]

    def _draw_quintic_trajectory(self):
        """5次多項式で生成した目標軌道を緑線で描画するメソッド"""
        if len(self.quintic_reference_trajectory) < 2:
            return
        points = self._world_points_to_pixels(self.quintic_reference_trajectory)
        pygame.draw.lines(self.screen, (0, 180, 0), False, points, 2)
        pygame.draw.circle(self.screen, (0, 180, 0), points[0], 4)
        pygame.draw.circle(self.screen, (0, 180, 0), points[-1], 4)

    def _draw_mpc_prediction(self):
        """MPCの予測軌道を橙線で描画するメソッド"""
        if len(self.mpc_prediction_trajectory) < 2:
            return
        points = self._world_points_to_pixels(self.mpc_prediction_trajectory)
        pygame.draw.lines(self.screen, (255, 140, 0), False, points, 2)
        pygame.draw.circle(self.screen, (255, 140, 0), points[-1], 4)

    def _draw_drl_target(self):
        """DRLが意思決定した目標位置を緑三角マーカーで、目標速度を吹き出しで描画するメソッド"""
        if self.drl_target_point is None:
            return

        pixel_x, pixel_y = self._world_points_to_pixels([self.drl_target_point])[0]
        size = 6
        triangle = [
            (pixel_x, pixel_y - size),
            (pixel_x - size, pixel_y + size),
            (pixel_x + size, pixel_y + size),
        ]
        pygame.draw.polygon(self.screen, (0, 180, 0), triangle)
        pygame.draw.polygon(self.screen, (0, 100, 0), triangle, 1)

        if self.drl_target_velocity is not None:
            self._draw_speed_bubble(pixel_x, pixel_y - size - 4, self.drl_target_velocity)

    def _compute_cbf_target_vehicles(self, vehicles):
        """衝突回避CBFの対象車両（egoから近い順にmax_obstacles台）の集合を返す
        Args
        - vehicles: 車両オブジェクトのリスト
        """
        rlmpc_config = getattr(self.config, 'rlmpc', None)
        if not self.rlmpc_active or rlmpc_config is None or not rlmpc_config.cbf.enabled:
            return set()

        ego_vehicle = next((vehicle for vehicle in vehicles if getattr(vehicle, 'is_ego', False)), None)
        if ego_vehicle is None:
            return set()

        cbf = rlmpc_config.cbf
        candidates = []
        for vehicle in vehicles:
            if vehicle is ego_vehicle:
                continue
            distance = float(np.hypot(vehicle.x - ego_vehicle.x, vehicle.y - ego_vehicle.y))
            if distance <= cbf.nearby_vehicle_range:
                candidates.append((distance, vehicle))
        candidates.sort(key=lambda item: item[0])

        return {vehicle for _, vehicle in candidates[: cbf.max_obstacles]}

    def _draw_ellipse_cbf(self, vehicle, color=(220, 20, 60)):
        """車両を守る楕円形の衝突回避CBF安全領域を描画するメソッド
        Args
        - vehicle: 描画する車両オブジェクト
        - color: 楕円の描画色
        """
        ellipse_cfg = self.config.rlmpc.cbf.ellipse
        if ellipse_cfg.use_dynamic_length:
            max_decel = abs(self.config.vehicle.min_acceleration)
            stopping_distance = (vehicle.velocity ** 2) / (2.0 * max_decel) if max_decel > 0 else 0.0
            a = stopping_distance + self.config.vehicle.length / 2.0
        else:
            a = ellipse_cfg.fixed_length_margin + self.config.vehicle.length / 2.0
        b = ellipse_cfg.width_margin + self.config.vehicle.width / 2.0

        # 楕円を多角形で近似し、車両のヨー角に合わせて回転
        num_points = 24
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        local_points = np.stack([a * np.cos(angles), b * np.sin(angles)], axis=1)
        cos_yaw, sin_yaw = np.cos(vehicle.yaw), np.sin(vehicle.yaw)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        world_points = local_points @ rotation.T + np.array([vehicle.x, vehicle.y])

        pixel_points = [
            (self.origin_x + x * self.ppm, self.screen_height // 2 - y * self.ppm)
            for x, y in world_points
        ]
        self._draw_dashed_polygon(pixel_points, color, width=2)

    def _draw_dashed_line(self, start, end, color, width=2, dash_length=6, gap_length=4):
        """始点から終点まで点線を描画するメソッド"""
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-6:
            return
        direction = segment / length
        step = dash_length + gap_length
        distance = 0.0
        while distance < length:
            seg_start = start + direction * distance
            seg_end = start + direction * min(distance + dash_length, length)
            pygame.draw.line(self.screen, color, seg_start, seg_end, width)
            distance += step

    def _draw_dashed_polygon(self, points, color, width=2, dash_length=6, gap_length=4):
        """多角形の輪郭を点線で描画するメソッド"""
        num_points = len(points)
        for i in range(num_points):
            self._draw_dashed_line(points[i], points[(i + 1) % num_points], color, width, dash_length, gap_length)

    def _draw_vehicle(self, vehicle):
        """車両を描画するメソッド
        Args
        - vehicle: 描画する車両オブジェクト
        """
        # 車両のコーナー座標を取得
        corners = vehicle.get_corners()

        # ピクセル座標に変換
        pixel_corners = []
        for corner in corners:
            x, y = corner
            pixel_x = self.origin_x + x * self.ppm
            pixel_y = self.screen_height // 2 - y * self.ppm
            pixel_corners.append((pixel_x, pixel_y))

        # 車両を多角形として描画
        pygame.draw.polygon(self.screen, vehicle.color, pixel_corners)

        # 車両の輪郭を黒色で描画
        pygame.draw.polygon(self.screen, (0, 0, 0), pixel_corners, 2)

        # 車両の頭上に速度を表示
        self._draw_vehicle_speed(vehicle)

    def _draw_vehicle_speed(self, vehicle):
        """車両の頭上に速度を白い吹き出しで表示するメソッド
        Args
        - vehicle: 描画する車両オブジェクト
        """
        center_pixel_x = self.origin_x + vehicle.x * self.ppm
        center_pixel_y = self.screen_height // 2 - vehicle.y * self.ppm
        top_pixel_y = center_pixel_y - (vehicle.width / 2) * self.ppm - 4
        self._draw_speed_bubble(center_pixel_x, top_pixel_y, vehicle.velocity)

    def _draw_speed_bubble(self, anchor_pixel_x, anchor_pixel_y, speed):
        """指定したピクセル座標の直上に速度を白い吹き出しで表示するメソッド
        Args
        - anchor_pixel_x, anchor_pixel_y: 吹き出し下端の基準ピクセル座標
        - speed: 表示する速度[m/s]
        """
        speed_text = self.font.render(f"{speed:.1f} m/s", True, (0, 0, 0))
        text_rect = speed_text.get_rect(midbottom=(anchor_pixel_x, anchor_pixel_y))

        # 吹き出し（白背景）を描画
        bubble_rect = text_rect.inflate(12, 6)
        bubble_surface = pygame.Surface(bubble_rect.size, pygame.SRCALPHA)
        bubble_local_rect = bubble_surface.get_rect()
        pygame.draw.rect(bubble_surface, (255, 255, 255, 230), bubble_local_rect, border_radius=6)
        pygame.draw.rect(bubble_surface, (0, 0, 0, 180), bubble_local_rect, width=1, border_radius=6)
        self.screen.blit(bubble_surface, bubble_rect)

        # 吹き出しの上に速度テキストを描画
        self.screen.blit(speed_text, text_rect)