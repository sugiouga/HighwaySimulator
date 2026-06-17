#!/usr/bin/env python3
"""5次多項式の軌跡を分析"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_quintic(y0, y1, duration_steps=30):
    """5次多項式の特性を分析
    
    Parameters:
    - y0: 初期y位置
    - y1: 最終y位置
    - duration_steps: 目標達成までのステップ数
    """
    
    dy = y1 - y0
    
    # 5次多項式の係数
    a0 = y0
    a1 = 0.0
    a2 = 0.0
    a3 = 10.0 * dy
    a4 = -15.0 * dy
    a5 = 6.0 * dy
    
    print(f"=== Quintic Polynomial Analysis ===")
    print(f"Initial y: {y0}, Final y: {y1}, Δy: {dy}")
    print(f"Duration steps: {duration_steps}")
    print(f"Coefficients: a0={a0}, a1={a1}, a2={a2}, a3={a3:.4f}, a4={a4:.4f}, a5={a5:.4f}\n")
    
    # τ(0-1)での評価
    taus = np.linspace(0, 1, 101)
    
    # y値の計算
    y_vals = a0 + a1*taus + a2*taus**2 + a3*taus**3 + a4*taus**4 + a5*taus**5
    
    # 1階導関数: y' = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
    y_prime = a1 + 2*a2*taus + 3*a3*taus**2 + 4*a4*taus**3 + 5*a5*taus**4
    
    # 2階導関数: y'' = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
    y_double_prime = 2*a2 + 6*a3*taus + 12*a4*taus**2 + 20*a5*taus**3
    
    # 3階導関数（ジャーク）: y''' = 6*a3 + 24*a4*t + 60*a5*t^2
    y_triple_prime = 6*a3 + 24*a4*taus + 60*a5*taus**2
    
    # 分析
    print("Key Values:")
    print(f"  y(0) = {y_vals[0]:.4f}, y(1) = {y_vals[-1]:.4f}")
    print(f"  y'(0) = {y_prime[0]:.4f}, y'(1) = {y_prime[-1]:.4f}")
    print(f"  y''(0) = {y_double_prime[0]:.4f}, y''(1) = {y_double_prime[-1]:.4f}")
    print()
    
    # 極値を探す
    print("Critical Points (y' = 0):")
    sign_changes = np.where(np.diff(np.sign(y_prime)))[0]
    for idx in sign_changes:
        print(f"  τ ≈ {taus[idx]:.3f}, y ≈ {y_vals[idx]:.4f}")
    if len(sign_changes) == 0:
        print("  None (monotonic)")
    print()
    
    # 変曲点を探す
    print("Inflection Points (y'' = 0):")
    sign_changes_2 = np.where(np.diff(np.sign(y_double_prime)))[0]
    for idx in sign_changes_2:
        print(f"  τ ≈ {taus[idx]:.3f}, y' ≈ {y_prime[idx]:.4f}")
    print()
    
    # 曲率を計算
    curvature = y_double_prime / (1 + y_prime**2)**1.5
    max_curvature = np.max(np.abs(curvature))
    print(f"Maximum Curvature: {max_curvature:.6f}")
    print()
    
    # ジャークを計算
    jerk = y_triple_prime
    max_jerk = np.max(np.abs(jerk))
    print(f"Maximum Jerk (3rd derivative): {max_jerk:.6f}")
    print()
    
    # 横加速度を計算（速度一定と仮定）
    v = 10.0  # m/s
    lateral_accel = v**2 * curvature
    max_lateral_accel = np.max(np.abs(lateral_accel))
    print(f"Maximum Lateral Acceleration (v={v}m/s): {max_lateral_accel:.4f} m/s^2")
    print()
    
    # プロット
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # y軌跡
    steps = np.arange(0, duration_steps+1)
    tau_steps = steps / duration_steps
    y_steps = a0 + a1*tau_steps + a2*tau_steps**2 + a3*tau_steps**3 + a4*tau_steps**4 + a5*tau_steps**5
    
    axes[0].plot(taus, y_vals, 'b-', label='y(τ)', linewidth=2)
    axes[0].scatter(tau_steps, y_steps, color='red', s=20, label='Discrete steps', zorder=5)
    axes[0].set_ylabel('y Position [m]')
    axes[0].set_title('Quintic Polynomial Lane Change Trajectory')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axhline(y=y0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=y1, color='gray', linestyle='--', alpha=0.5)
    
    # 1階・2階導関数
    axes[1].plot(taus, y_prime, 'g-', label="y'(τ)", linewidth=2)
    axes[1].plot(taus, y_double_prime, 'r-', label="y''(τ)", linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].set_ylabel('Derivative Value')
    axes[1].set_title('First and Second Derivatives')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 曲率とジャーク
    axes[2].plot(taus, curvature, 'purple', label='Curvature', linewidth=2)
    axes[2].plot(taus, jerk/np.max(np.abs(jerk)), 'orange', label='Normalized Jerk', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2].set_xlabel('Normalized Time τ')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Curvature and Jerk')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('quintic_analysis.png', dpi=100)
    print("Plot saved to: quintic_analysis.png\n")
    
    # 単調性の確認
    print("Monotonicity Check:")
    if np.all(np.diff(y_vals) >= 0):
        print("  ✓ y(τ) is strictly monotonically increasing (no oscillation)")
    elif np.all(np.diff(y_vals) <= 0):
        print("  ✓ y(τ) is strictly monotonically decreasing (no oscillation)")
    else:
        print("  ✗ y(τ) has turning points (possible oscillation)")
    print()

# テストケース
if __name__ == "__main__":
    print("\n### Case 1: Lane change from 0m to 3.5m (typical merge) ###\n")
    analyze_quintic(0.0, 3.5, duration_steps=30)
    
    print("\n" + "="*60)
    print("\n### Case 2: Lane change from -3.5m to 0m (merge back) ###\n")
    analyze_quintic(-3.5, 0.0, duration_steps=30)
    
    print("\n" + "="*60)
    print("\n### Case 3: Small lane change from 0m to 1.5m ###\n")
    analyze_quintic(0.0, 1.5, duration_steps=30)
