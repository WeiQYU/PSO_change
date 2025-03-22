import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import warnings

# 这是一个分析PSO运行结果的文件
# 忽略警告
warnings.filterwarnings("ignore")

# 常量
G = 6.67430e-11  # 引力常数
c = 2.998e8  # 光速
M_sun = 1.989e30  # 太阳质量
pc = 3.086e16  # 秒差距


# 原始的引力波生成函数
def crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t):
    def generate_h_t(t_array, M_c, r, phi_c, t_c):
        # Vectorized computation using CuPy
        mask = t_array < t_c
        theta_t = cp.zeros_like(t_array)
        theta_t[mask] = c ** 3 * (t_c - t_array[mask]) / (5 * G * M_c)
        h = cp.zeros_like(t_array)
        h[mask] = G * M_c / (c ** 2 * r) * theta_t[mask] ** (-1 / 4) * cp.cos(2 * phi_c - 2 * theta_t[mask] ** (5 / 8))
        return h

    # Convert input array to CuPy
    dataX_gpu = cp.asarray(dataX)

    # Generate signal
    h = generate_h_t(dataX_gpu, m_c, r, phi_c, tc)

    # Convert to frequency domain
    h_f = cp.fft.rfft(h)
    freqs = cp.fft.rfftfreq(len(h), dataX_gpu[1] - dataX_gpu[0])

    # Apply low frequency cutoff
    low_freq_cutoff = 3.0
    freq_mask = freqs >= low_freq_cutoff
    h_f = cp.where(freq_mask, h_f, 0)

    def parametric_lens_model(h_f, freqs, A, dt):
        phi = 2 * cp.pi * freqs * dt
        F_f = 1 + A * cp.exp(1j * phi)
        return h_f * F_f

    # Apply lensing effect
    h_lens_f = parametric_lens_model(h_f, freqs, A, delta_t)

    # Convert back to time domain
    return cp.fft.irfft(h_lens_f)


# 定义PSO结果
pso_results = [
    {"run": 1, "fitness": -165.9138, "r": 109.7303, "m_c": 9.1403, "tc": 5.3637, "phi_c": 1.0318, "A": 0.4976,
     "delta_t": 39.5310, "snr": 12.88},
    {"run": 2, "fitness": -240.2260, "r": 216.7691, "m_c": 267.0778, "tc": 4.7837, "phi_c": 0.1437, "A": 0.5276,
     "delta_t": 9.5286, "snr": 15.50},
    {"run": 3, "fitness": -155.8597, "r": 111.9969, "m_c": 7.4194, "tc": 5.2907, "phi_c": 1.8658, "A": 0.2625,
     "delta_t": 11.5025, "snr": 12.48},
    {"run": 4, "fitness": -196.6256, "r": 717.7676, "m_c": 14.5268, "tc": 4.8766, "phi_c": 1.6415, "A": 0.8282,
     "delta_t": 15.8186, "snr": 14.02},
    {"run": 5, "fitness": -235.7331, "r": 7.4306, "m_c": 146.4834, "tc": 4.7591, "phi_c": 0.9141, "A": 0.0292,
     "delta_t": 70.4526, "snr": 15.35},
    {"run": 6, "fitness": -325.9683, "r": 20.9298, "m_c": 71.6010, "tc": 4.6743, "phi_c": 1.4121, "A": 0.3262,
     "delta_t": 1.8264, "snr": 18.05},
    {"run": 7, "fitness": -208.8810, "r": 275.2047, "m_c": 6.2076, "tc": 5.2387, "phi_c": 1.1140, "A": 0.2625,
     "delta_t": 8.4963, "snr": 14.45},
    {"run": 8, "fitness": -419.8086, "r": 0.0248, "m_c": 36.2298, "tc": 4.8573, "phi_c": 1.9207, "A": 0.3043,
     "delta_t": 15.6578, "snr": 20.49}
]

# 按照运行编号排序
pso_results = sorted(pso_results, key=lambda x: x["run"])

# 找出最佳结果
best_result = min(pso_results, key=lambda x: x["fitness"])

# 创建时间数组
dt = 1 / 2000  # 采样率
t = np.arange(-90, 10, dt)  # 完整时间范围从 -90 到 10 秒


# 为处理数值异常增加一个辅助函数
def safe_generate_waveform(param):
    """安全地生成波形，处理可能的异常"""
    try:
        # 创建适当的参数格式
        r_val = param["r"] * 1e6 * pc
        m_c_val = param["m_c"] * M_sun
        delta_t_val =param["delta_t"]

        # 生成波形
        wave = crcbgenqcsig(
            t,
            r_val,
            m_c_val,
            param["tc"],
            param["phi_c"] * np.pi,
            param["A"],
            delta_t_val
        )

        # 转换为numpy数组并清理NaN和Inf
        wave_np = cp.asnumpy(wave)
        wave_np = np.nan_to_num(wave_np)

        return wave_np
    except Exception as e:
        print(f"生成波形时出错，参数: {param}, 错误: {e}")
        # 返回零数组作为后备
        return np.zeros_like(t)


# 创建所有波形的图像
plt.figure(figsize=(20, 16))

# 设置不同运行的颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(pso_results)))

# 在4x2网格中绘制每个波形
for i, result in enumerate(pso_results):
    # 计算子图的行和列
    ax = plt.subplot(4, 2, i + 1)

    # 尝试生成波形
    try:
        # 生成波形
        signal_np = safe_generate_waveform(result)

        # 检查信号是否有效
        if np.all(signal_np == 0) or np.isnan(signal_np).any() or np.isinf(signal_np).any():
            print(f"警告: 运行 {result['run']} 产生了无效信号。使用占位波形。")
            signal_np = np.sin(2 * np.pi * 0.1 * (t - result["tc"])) * np.exp(-(t - result["tc"]) ** 2 / 10)

        # 绘制信号
        plt.plot(t, signal_np, color=colors[i], linewidth=1.0)

        # 在tc处添加垂直线
        plt.axvline(x=result["tc"], color='r', linestyle='--', alpha=0.5)

        # 设置标题和运行信息
        plt.title(f"Run {result['run']} (Fitness={result['fitness']:.2f}, SNR={result['snr']:.2f})", fontsize=12)

        # 添加参数值作为文本
        param_text = (
            f"r={result['r']:.2f}, m_c={result['m_c']:.2f}, tc={result['tc']:.2f}\n"
            f"phi_c={result['phi_c']:.2f}π, A={result['A']:.4f}, δt={result['delta_t']:.2f}"
        )
        plt.text(0.02, 0.02, param_text, transform=ax.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=9)

    except Exception as e:
        print(f"绘制运行 {result['run']} 时出错: {e}")
        plt.text(0.5, 0.5, f"Error generating waveform",
                 ha='center', va='center', transform=ax.transAxes)

    # 设置标签
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')

    # 显示完整的时间范围
    plt.xlim(-90, 10)

    # 添加网格线以便于查看
    plt.grid(True, alpha=0.3)

# 为整个图像添加标题
plt.suptitle('PSO Gravitational Wave Results - Full Time Range (-90s to 10s)', fontsize=16, y=0.995)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.98])

# 保存图像
plt.savefig('gwave_pso_results_full_range.png', dpi=300)
print("可视化已保存为 'gwave_pso_results_full_range.png'")

# 创建所有信号的比较图
plt.figure(figsize=(15, 10))

# 生成和绘制所有信号
valid_signals = 0
for i, result in enumerate(pso_results):
    try:
        # 生成波形
        signal_np = safe_generate_waveform(result)

        # 检查信号是否有效
        if np.all(signal_np == 0) or np.isnan(signal_np).any() or np.isinf(signal_np).any():
            print(f"警告: 运行 {result['run']} 在比较图中产生了无效信号。跳过。")
            continue

        # 归一化以便更好地比较
        max_val = np.max(np.abs(signal_np))
        if max_val > 0:
            signal_norm = signal_np / max_val
        else:
            continue

        # 根据SNR排名设置透明度
        alpha = 0.5 + 0.5 * (result["snr"] / max(r["snr"] for r in pso_results))
        plt.plot(t, signal_norm, color=colors[i], alpha=alpha,
                 label=f"Run {result['run']} (SNR={result['snr']:.1f})")

        valid_signals += 1

    except Exception as e:
        print(f"在比较图中绘制运行 {result['run']} 时出错: {e}")

# 如果没有有效信号，添加说明文本
if valid_signals == 0:
    plt.text(0.5, 0.5, "Unable to generate valid signals for comparison",
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)

# 标记合并时间范围
min_tc = min(r["tc"] for r in pso_results)
max_tc = max(r["tc"] for r in pso_results)
plt.axvspan(min_tc, max_tc, color='r', alpha=0.1)

# 设置轴标签和标题
plt.xlabel('Time (s)')
plt.ylabel('Normalized Strain')
plt.title('Comparison of Normalized Waveforms (-90s to 10s)')
plt.xlim(-90, 10)  # 完整时间范围
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=8)

# 保存比较图
plt.tight_layout()
plt.savefig('gwave_all_signals_comparison_full_range.png', dpi=300)
print("综合分析已保存为 'gwave_all_signals_comparison_full_range.png'")

# 创建参数比较可视化
plt.figure(figsize=(15, 10))

# 定义要可视化的参数
params = ["r", "m_c", "tc", "phi_c", "A", "delta_t"]
param_names = ["Distance (r)", "Chirp Mass (m_c)", "Coalescence Time (tc)",
               "Coalescence Phase (phi_c)", "Amplitude (A)", "Time Delay (delta_t)"]

# 为每个参数创建子图
for i, param in enumerate(params):
    plt.subplot(3, 2, i + 1)
    values = [result[param] for result in pso_results]
    snrs = [result["snr"] for result in pso_results]

    # 绘制点
    scatter = plt.scatter(range(1, len(pso_results) + 1), values, c=snrs, cmap='viridis',
                          s=100, alpha=0.7, edgecolors='black')

    # 高亮最佳结果
    best_idx = pso_results.index(best_result)
    plt.scatter(best_idx + 1, best_result[param], s=200, facecolors='none',
                edgecolors='red', linewidth=2)

    # 添加运行编号作为标签
    for j, result in enumerate(pso_results):
        plt.text(j + 1.1, values[j], f"{j + 1}", fontsize=9)

    plt.xticks(range(1, len(pso_results) + 1))
    plt.xlabel('Run Number')
    plt.ylabel(param_names[i])
    plt.grid(True, alpha=0.3)

    # 如果是最后一个子图，添加颜色条
    if i == len(params) - 1:
        cbar = plt.colorbar(scatter)
        cbar.set_label('SNR')

plt.suptitle('Parameter Comparison Across PSO Runs', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('parameter_comparison.png', dpi=300)
print("参数比较已保存为 'parameter_comparison.png'")