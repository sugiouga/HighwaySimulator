import numpy as np
from scipy.interpolate import CubicSpline

class Lane:
    def __init__(self, lane_id, waypoints, width):
        self.lane_id = lane_id
        self.width = width
        self.waypoints = np.array(waypoints)

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

        # 車線の法線ベクトルを計算
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]

        # 各waypointに対してs, dを計算
        s_values = self.s_coords
        d_values = np.zeros_like(s_values)

        for i in range(len(s_values)):
            waypoint = self.waypoints[i]
            normal = normals[i]
            d_values[i] = np.dot(np.array([x, y]) - waypoint, normal)

        # 最も近いwaypointを見つける
        closest_index = np.argmin(np.linalg.norm(self.waypoints - np.array([x, y]), axis=1))
        s_closest = s_values[closest_index]
        d_closest = d_values[closest_index]

        return s_closest, d_closest