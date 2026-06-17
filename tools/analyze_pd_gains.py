#!/usr/bin/env python3
"""PD制御ゲイン調整の効果を検証"""
import numpy as np

def estimate_lateral_acceleration(kp, kd, max_steering_rate=10.0, vehicle_length=4.5, velocity=10.0):
    """
    PD制御のゲインから最大横加速度を推定
    
    仮定:
    - ステアリング角の最大値: max_steering_rate * dt
    - 横加速度: v^2 * sin(δ) / L ≈ v^2 * δ / L (小角近似)
    """
    
    # 典型的な横誤差
    max_lateral_error = 1.75  # 車線幅の半分
    
    # PD制御による最大ステアリングレート
    # steering_rate = kp * error - kd * steering_angle
    # 最悪の場合: steering_rate = kp * max_error
    max_steering_rate_from_pd = kp * max_lateral_error
    
    # 制約を考慮
    actual_steering_rate = min(max_steering_rate_from_pd, max_steering_rate)
    
    # 時間ステップで積分してステアリング角を推定
    dt = 0.1
    max_steering_angle = actual_steering_rate * dt  # [deg/step]
    max_steering_angle_rad = np.radians(max_steering_angle)
    
    # 小角近似での横加速度
    lateral_accel = velocity**2 * max_steering_angle_rad / vehicle_length
    
    return {
        'kp': kp,
        'kd': kd,
        'max_lateral_error': max_lateral_error,
        'max_steering_rate_from_pd': max_steering_rate_from_pd,
        'actual_steering_rate': actual_steering_rate,
        'max_steering_angle_deg': max_steering_angle,
        'lateral_accel_ms2': lateral_accel,
        'lateral_accel_g': lateral_accel / 9.81
    }

if __name__ == "__main__":
    print("=== PD制御ゲイン調整の効果分析 ===\n")
    
    test_cases = [
        ("OLD (KP=0.5, KD=0.25)", 0.5, 0.25),
        ("NEW (KP=0.1, KD=0.05)", 0.1, 0.05),
        ("CONSERVATIVE (KP=0.05, KD=0.02)", 0.05, 0.02),
        ("AGGRESSIVE (KP=0.2, KD=0.1)", 0.2, 0.1),
    ]
    
    print(f"{'Case':<30} {'KP':>6} {'KD':>6} {'Max δ_rate':>10} {'Max δ':>8} {'a_lateral':>10} {'a_lateral':>10}")
    print(f"{'':30} {'':>6} {'':>6} {'[deg/s]':>10} {'[deg]':>8} {'[m/s²]':>10} {'[G]':>10}")
    print("-" * 100)
    
    for name, kp, kd in test_cases:
        result = estimate_lateral_acceleration(kp, kd)
        print(f"{name:<30} {result['kp']:>6.3f} {result['kd']:>6.3f} "
              f"{result['actual_steering_rate']:>10.2f} "
              f"{result['max_steering_angle_deg']:>8.4f} "
              f"{result['lateral_accel_ms2']:>10.4f} "
              f"{result['lateral_accel_g']:>10.4f}")
    
    print("\n=== 推奨値 ===")
    print("- 快適な横加速度: 2-4 m/s² (0.2-0.4 G)")
    print("- 安全上許容できる横加速度: 5-8 m/s² (0.5-0.8 G)")
    print("- 乗用車の最大横加速度: 約 10 m/s² (1.0 G)")
    print("\n★ NEW設定 (KP=0.1, KD=0.05) が推奨されます\n")
