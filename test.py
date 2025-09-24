import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# 只有在实际运行时才需要 SciPy 和 Matplotlib
try:
    from scipy.integrate import solve_ivp
    has_scipy = True
except Exception:
    has_scipy = False

try:
    import matplotlib.pyplot as plt
    has_plot = True
except Exception:
    has_plot = False

@dataclass
class Parameters:
    r: float  # 生长速率
    K: float  # 突变点容量
    c: float  # 抑制项系数

@dataclass
class Result:
    t: np.ndarray
    x: np.ndarray
    dxdt: np.ndarray
    state: str
    reason: str

def dxdt(t: float, x: float, p: Parameters) -> float:
    """
    动力系统微分方程：
    dx/dt = r x (1 - x / K) - c x^2 / (1 + x^2)
    """
    r, K, c = p.r, p.K, p.c
    return r * x * (1.0 - x / K) - (c * x**2) / (1.0 + x**2)

def simulate(p: Parameters, x0: float, t_span: Tuple[float, float], method: str = "RK45",
             dt: float | None = None) -> Result:
    """
    使用 solve_ivp 进行仿真，若 dt 非 None，则生成等间距的采样点并在这些点处返回解。
    """
    if not has_scipy:
        raise RuntimeError("需要 scipy 库中的 solve_ivp 来执行积分，请安装 SciPy。")

    def f(t, x):
        return dxdt(t, x[0], p)

    t0, tf = t_span

    if dt is not None:
        # 以等间距采样点进行求解
        t_eval = np.arange(t0, tf + dt, dt)
    else:
        t_eval = None  # 让 solve_ivp 自动决定

    sol = solve_ivp(fun=f, t_span=(t0, tf), y0=[x0], method=method, t_eval=t_eval, vectorized=False)

    # 计算 dx/dt 的近似值
    x_vals = sol.y[0]
    t_vals = sol.t
    dx_vals = np.array([dxdt(t, x, p) for t, x in zip(t_vals, x_vals)])

    # 简单状态判定
    state, reason = classify_state(x_vals, dx_vals, t_vals, p)

    return Result(t=t_vals, x=x_vals, dxdt=dx_vals, state=state, reason=reason)

def classify_state(x_vals: np.ndarray, dx_vals: np.ndarray, t_vals: np.ndarray, p: Parameters) -> Tuple[str, str]:
    """
    简单的状态分类示例：
    - 若 x 趋于稳定且 dx/dt 近似 0，则 Normal
    - 若 x 超出某个正负阈值（如 [-epsilon, K+epsilon] 之外），警告/错误
    - 若存在发散趋势（dx/dt 不再趋近 0，且 x 趋向无界），标为 Warning/Error
    你可以根据实际需求扩展更精细的判定。
    """
    # 基本阈值设置
    eps = 1e-6
    x_min, x_max = np.min(x_vals), np.max(x_vals)

    # 简单判断：若解在区间内并趋于定值且 dxdt ~ 0
    if np.all(np.abs(dx_vals) < 1e-3):
        # 检查是否接近稳态解
        if abs(x_vals[-1] - x_vals[0]) < 1e-3:
            return "Normal", "系统已收敛到稳态"
        else:
            return "Normal", "系统接近稳态"

    # 观察是否越界
    # 这里假设若 x 超过 K 的两倍或负数，视为异常
    if x_max > 2.0 * p.K or x_min < -0.1 * p.K:
        return "Warning", "解出现极端值，可能需要进一步分析"

    # 简单默认
    return "Normal", "运行中"

def main():
    # 示例参数与初始条件
    p = Parameters(r=1.2, K=50.0, c=0.8)
    x0 = 2.0
    t_span = (0.0, 50.0)

    # 选择是否使用 dt 着色的采样点
    dt = 0.05  # 也可以设为 None 使用自适应
    result = simulate(p, x0, t_span, method="RK45", dt=dt)

    print(f"状态: {result.state}, 原因: {result.reason}")
    print(f"最终 x = {result.x[-1]:.4f}, dx/dt = {result.dxdt[-1]:.4f}")

    # 简单绘图
    if has_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(result.t, result.x, label="x(t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("动力系统迭代解 x(t)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(result.t, result.dxdt, label="dx/dt")
        plt.xlabel("t")
        plt.ylabel("dx/dt")
        plt.title("导数 dx/dt(t)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Matplotlib 未安装，无法绘制图形。")

if __name__ == "__main__":
    main()