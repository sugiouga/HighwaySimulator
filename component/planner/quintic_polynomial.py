import math
from typing import Callable, Tuple

import numpy as np


class QuinticPolynomial:
    """始点・終点の位置と速度・加速度条件から目標軌道を生成するクラス

    x(t) は4次多項式（終端加速度0を仮定）、y(x) は5次多項式で近似する。
    """

    @staticmethod
    def generate_trajectory(
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        T: float,
        v0: float = 0.0,
        vT: float = 0.0,
        a0: float = 0.0,
    ) -> Callable[[float], Tuple[float, float, float, float]]:
        """
        Args:
            start: 始点の (x, y, yaw)
            end: 終点の (x, y, yaw)。x, yaw は未使用で yaw は常に0を仮定する
            T: 始点から終点に到達するまでの計画時間[s]
            v0: 始点の速度[m/s]
            vT: 終点の目標速度[m/s]
            a0: 始点の加速度[m/s^2]

        Returns:
            calc_point(t) -> (x, y, yaw, v) を返すコールバック（0 <= t <= T を想定）
        """
        x0, y0, yaw0 = start
        _, yT, _ = end
        vx0 = v0 * math.cos(yaw0)

        # ---------- x(t) の4次多項式係数を算出（終端加速度0を仮定） ----------
        a0_coef = float(x0)
        a1_coef = float(vx0)
        a2_coef = float(a0 / 2.0)

        M = np.array([[3 * T**2, 4 * T**3],
                      [6 * T, 12 * T**2]])
        b = np.array([vT - (a1_coef + 2.0 * a2_coef * T),
                      -2.0 * a2_coef])
        a3_coef, a4_coef = (float(v) for v in np.linalg.solve(M, b))

        XT = (a0_coef + a1_coef * T + a2_coef * T**2 +
              a3_coef * T**3 + a4_coef * T**4)

        # ---------- y(x) の5次多項式係数を算出 ----------
        slope0 = math.tan(yaw0)
        slopeT = 0.0
        k0 = 0.0
        kT = 0.0

        X0 = x0
        A = np.zeros((6, 6))
        Bv = np.zeros(6)
        A[0, :] = [1, X0, X0**2, X0**3, X0**4, X0**5]
        Bv[0] = y0
        A[1, :] = [0, 1, 2 * X0, 3 * X0**2, 4 * X0**3, 5 * X0**4]
        Bv[1] = slope0
        A[2, :] = [0, 0, 2, 6 * X0, 12 * X0**2, 20 * X0**3]
        Bv[2] = k0
        A[3, :] = [1, XT, XT**2, XT**3, XT**4, XT**5]
        Bv[3] = yT
        A[4, :] = [0, 1, 2 * XT, 3 * XT**2, 4 * XT**3, 5 * XT**4]
        Bv[4] = slopeT
        A[5, :] = [0, 0, 2, 6 * XT, 12 * XT**2, 20 * XT**3]
        Bv[5] = kT
        b0, b1, b2, b3, b4, b5 = (float(v) for v in np.linalg.solve(A, Bv))

        def calc_point(t: float) -> Tuple[float, float, float, float]:
            x = (a0_coef + a1_coef * t + a2_coef * t**2 +
                 a3_coef * t**3 + a4_coef * t**4)
            dx = (a1_coef + 2 * a2_coef * t +
                  3 * a3_coef * t**2 + 4 * a4_coef * t**3)

            y = b0 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4 + b5 * x**5
            dy_dx = b1 + 2 * b2 * x + 3 * b3 * x**2 + 4 * b4 * x**3 + 5 * b5 * x**4
            dy = dy_dx * dx  # dy/dt = dy/dx * dx/dt

            yaw = math.atan2(dy, dx)
            v = math.hypot(dx, dy)
            return x, y, yaw, v

        return calc_point
