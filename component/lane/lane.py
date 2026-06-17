import numpy as np
from scipy.interpolate import CubicSpline

class Lane:
    def __init__(self, lane_id, waypoints, width):
        self.lane_id = lane_id
        self.width = width
        # 入力ウェイポイントを密にして、スプラインの精度を確保する
        self.waypoints = self._densify_waypoints(np.array(waypoints))

        # 累積距離sの計算
        diffs = np.diff(self.waypoints, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        self.s_coords = np.concatenate(([0], np.cumsum(distances)))

        # CubicSplineを使用してsからx, yへの変換関数を作成
        self.x_spline = CubicSpline(self.s_coords, self.waypoints[:, 0])
        self.y_spline = CubicSpline(self.s_coords, self.waypoints[:, 1])

        # 隣接車線のID
        self.left_lane_id = None
        self.right_lane_id = None

        # 左右の境界線ウェイポイントを計算
        self._compute_boundary_waypoints()

    def _compute_boundary_waypoints(self):
        """左右の境界線ウェイポイントを計算するメソッド"""
        self.left_waypoints = []
        self.right_waypoints = []
        self.left_pixel_waypoints = []
        self.right_pixel_waypoints = []
        self.center_pixel_waypoints = []

        for s in self.s_coords:
            # 左側の境界線（d = width/2）
            x_left, y_left = self.get_cartesian(s, self.width / 2)
            self.left_waypoints.append([x_left, y_left])

            # 右側の境界線（d = -width/2）
            x_right, y_right = self.get_cartesian(s, -self.width / 2)
            self.right_waypoints.append([x_right, y_right])

        self.left_waypoints = np.array(self.left_waypoints)
        self.right_waypoints = np.array(self.right_waypoints)

    def _densify_waypoints(self, waypoints, max_spacing=0.1):
        """入力ウェイポイントを線形補間して、隣接点間の距離がmax_spacing以下になるように増やす。"""
        if len(waypoints) < 2:
            return waypoints

        new_points = [waypoints[0].tolist()]
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            seg = p1 - p0
            seg_len = np.linalg.norm(seg)
            if seg_len == 0:
                continue
            # 分割数（各区間の長さが max_spacing 以下になるように）
            parts = int(np.ceil(seg_len / max_spacing))
            for k in range(1, parts + 1):
                t = k / parts
                pt = p0 + t * seg
                new_points.append(pt.tolist())

        return np.array(new_points)

    def get_cartesian(self, s, d):
        """s, d座標をx, y座標に変換する関数
        Args
        - s: 累積距離
        - d: 車線中心からの横方向距離
        """
        x_center = self.x_spline(s)
        y_center = self.y_spline(s)

        # 車線の接線ベクトルを計算
        dx_ds = self.x_spline.derivative()(s)
        dy_ds = self.y_spline.derivative()(s)
        tangent = np.array([dx_ds, dy_ds])
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent /= tangent_norm
        else:
            tangent = np.array([0, 0])
        # 車線の法線ベクトルを計算
        normal = np.array([-tangent[1], tangent[0]])
        # x, y座標を計算
        x = x_center + d * normal[0]
        y = y_center + d * normal[1]
        return x, y

    def get_frenet(self, x, y):
        """x, y座標をs, d座標に変換する関数
        Args
        - x: x座標
        - y: y座標
        """
        # 車線の接線ベクトルを計算
        dx_ds = self.x_spline.derivative()(self.s_coords)
        dy_ds = self.y_spline.derivative()(self.s_coords)
        tangents = np.vstack((dx_ds, dy_ds)).T
        tangent_norms = np.linalg.norm(tangents, axis=1)
        tangents[tangent_norms > 0] /= tangent_norms[tangent_norms > 0][:, np.newaxis]

        # 車線の法線ベクトルを計算
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]

        # x, y座標から最も近い車線上の点を見つける
        diffs = self.waypoints - np.array([x, y])
        distances = np.linalg.norm(diffs, axis=1)
        closest_index = np.argmin(distances)

        # s座標は最も近い点のs座標
        s = self.s_coords[closest_index]

        # d座標は、最も近い点からの距離を法線方向に投影した値
        if tangent_norms[closest_index] > 0:
            normal_vector = normals[closest_index]
            d = np.dot(diffs[closest_index], normal_vector)
        else:
            d = 0.0

        return s, d

    def is_within_bounds(self, x, y):
        """座標が車線の範囲内にあるかをチェックするメソッド"""
        s, d = self.get_frenet(x, y)
        return 0 <= s <= 0.99 * self.s_coords[-1] and -self.width / 2 <= d <= self.width / 2