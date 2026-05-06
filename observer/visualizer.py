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

        # 描画パラメータの設定
        pygame.display.set_caption(self.visualization_config.caption)
        self.ppm = self.visualization_config.ppm  # ピクセルあたりのメートル数
        self.origin_x = self.visualization_config.origin_x  # 描画開始位置(ピクセル)
        self.font = pygame.font.SysFont("Consolas", 14)
        self.clock = pygame.time.Clock()
        self.road_network_initialized = False

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

        # 画面を背景色でクリア
        self.screen.fill(self.screen_color)

        # 道路を描画
        for lane in road_network.lanes.values():
            self._draw_road(lane)

        # 車両を描画
        for vehicle in vehicles:
            self._draw_vehicle(vehicle)

        # 時刻を表示
        time_text = self.font.render(f"Time: {current_time:.1f} s", True, (0, 0, 0))
        self.screen.blit(time_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)  # フレームレートを30FPSに設定

    def _compute_road_pixel_waypoints(self, road_network):
        """道路のピクセル座標を計算して保存するメソッド
        Args
        - road_network: 道路ネットワークオブジェクト
        """
        for lane in road_network.lanes.values():
            # 左側の境界線をピクセル座標に変換
            lane.left_pixel_waypoints = []
            for waypoint in lane.left_waypoints:
                x, y = waypoint
                pixel_x = self.origin_x + x * self.ppm
                pixel_y = self.screen_height // 2 - y * self.ppm
                lane.left_pixel_waypoints.append((pixel_x, pixel_y))

            # 右側の境界線をピクセル座標に変換
            lane.right_pixel_waypoints = []
            for waypoint in lane.right_waypoints:
                x, y = waypoint
                pixel_x = self.origin_x + x * self.ppm
                pixel_y = self.screen_height // 2 - y * self.ppm
                lane.right_pixel_waypoints.append((pixel_x, pixel_y))

            # 中心線をピクセル座標に変換
            lane.center_pixel_waypoints = []
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
