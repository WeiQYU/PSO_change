```python
def estimate_distance_from_amplitude(template_params, observed_signal, dataX, sampFreq, psdHigh):
    """
    基于振幅匹配估计距离参数

    原理：引力波振幅与距离成反比 h ∝ 1/r
    因此 h_observed / h_template = r_template / r_observed
    所以 r_observed = r_template * h_template / h_observed

    Parameters:
    -----------
    template_params : dict
        模板参数 (除距离外的其他参数)
    observed_signal : array
        观测信号
    dataX : array
        时间序列
    sampFreq : float
        采样频率
    psdHigh : array
        功率谱密度

    Returns:
    --------
    estimated_distance_log10 : float
        估计的距离参数 (log10 scale)
    confidence : float
        估计的置信度
    """

    print("开始基于振幅的距离估计...")

    # 提取参数
    m_c = template_params['m_c']
    tc = template_params['tc']
    phi_c = template_params['phi_c']
    A = template_params['A']
    delta_t = template_params['delta_t']
    use_lensing = template_params.get('use_lensing', A >= 0.01)

    # 设置参考距离 (log10 scale) - 选择一个合理的中间值
    reference_distance_log10 = 3.0  # 10^3 = 1000 Mpc

    print(f"使用参考距离: {10 ** reference_distance_log10:.1f} Mpc")

    # 生成参考距离的模板信号
    template_signal = crcbgenqcsig(
        dataX, reference_distance_log10, m_c, tc, phi_c, A, delta_t,
        use_lensing=use_lensing
    )

    # 归一化模板信号
    template_normalized, template_norm_factor = normsig4psd_pycbc(
        template_signal, sampFreq, psdHigh, 1.0
    )

    if template_norm_factor == 0 or np.all(template_normalized == 0):
        print("警告: 模板信号生成失败")
        return reference_distance_log10, 0.0

    # 计算观测信号与模板的最优振幅比
    optimal_amplitude = innerprodpsd(observed_signal, template_normalized, sampFreq, psdHigh)

    if abs(optimal_amplitude) < 1e-15:
        print("警告: 振幅匹配失败")
        return reference_distance_log10, 0.0

    # 根据振幅比估计距离
    # h_observed / h_template = optimal_amplitude
    # r_observed = r_template / optimal_amplitude (因为 h ∝ 1/r)

    amplitude_ratio = abs(optimal_amplitude)
    estimated_distance_mpc = (10 ** reference_distance_log10) / amplitude_ratio
    estimated_distance_log10 = np.log10(estimated_distance_mpc)

    # 计算置信度 - 基于匹配质量
    scaled_template = optimal_amplitude * template_normalized
    match_quality = pycbc_calculate_match(scaled_template, observed_signal, sampFreq, psdHigh)
    confidence = max(0.0, min(1.0, match_quality))

    print(f"振幅比: {amplitude_ratio:.6f}")
    print(f"估计距离: {estimated_distance_mpc:.1f} Mpc (log10: {estimated_distance_log10:.4f})")
    print(f"匹配质量: {match_quality:.4f}")
    print(f"置信度: {confidence:.4f}")

    # 合理性检查
    if estimated_distance_mpc < 10 or estimated_distance_mpc > 50000:
        print(f"警告: 估计距离 {estimated_distance_mpc:.1f} Mpc 超出合理范围")
        # 如果估计结果不合理，返回一个保守的估计
        estimated_distance_log10 = 3.5  # 3162 Mpc
        confidence = 0.1

    return estimated_distance_log10, confidence


def amplitude_based_distance_refinement(pso_params, dataX, observed_signal, sampFreq, psdHigh):
    """
    基于振幅的距离参数精炼

    Parameters:
    -----------
    pso_params : dict
        PSO优化得到的参数
    dataX : array
        时间序列
    observed_signal : array
        观测信号
    sampFreq : float
        采样频率
    psdHigh : array
        功率谱密度

    Returns:
    --------
    refined_params : dict
        精炼后的参数
    refinement_info : dict
        精炼过程信息
    """

    print("=" * 50)
    print("开始基于振幅的距离参数精炼")
    print("=" * 50)

    # 原始参数
    original_r = pso_params['r']
    original_distance_mpc = 10 ** original_r

    print(f"原始距离估计: {original_distance_mpc:.1f} Mpc (log10: {original_r:.4f})")

    # 准备模板参数 (除距离外)
    template_params = {
        'm_c': pso_params['m_c'],
        'tc': pso_params['tc'],
        'phi_c': pso_params['phi_c'],
        'A': pso_params['A'],
        'delta_t': pso_params['delta_t'],
        'use_lensing': pso_params['A'] >= 0.01
    }

    try:
        # 基于振幅估计距离
        estimated_r, confidence = estimate_distance_from_amplitude(
            template_params, observed_signal, dataX, sampFreq, psdHigh
        )

        estimated_distance_mpc = 10 ** estimated_r

        # 验证精炼结果
        refined_signal = crcbgenqcsig(
            dataX, estimated_r, template_params['m_c'], template_params['tc'],
            template_params['phi_c'], template_params['A'], template_params['delta_t'],
            use_lensing=template_params['use_lensing']
        )

        # 归一化并匹配振幅
        refined_signal_norm, norm_factor = normsig4psd_pycbc(refined_signal, sampFreq, psdHigh, 1.0)
        if norm_factor > 0:
            optimal_amp = innerprodpsd(observed_signal, refined_signal_norm, sampFreq, psdHigh)
            refined_signal_final = optimal_amp * refined_signal_norm

            # 计算最终匹配度
            final_match = pycbc_calculate_match(refined_signal_final, observed_signal, sampFreq, psdHigh)

            # 计算原始匹配度进行比较
            original_signal = crcbgenqcsig(
                dataX, original_r, template_params['m_c'], template_params['tc'],
                template_params['phi_c'], template_params['A'], template_params['delta_t'],
                use_lensing=template_params['use_lensing']
            )
            original_signal_norm, orig_norm = normsig4psd_pycbc(original_signal, sampFreq, psdHigh, 1.0)
            if orig_norm > 0:
                orig_amp = innerprodpsd(observed_signal, original_signal_norm, sampFreq, psdHigh)
                original_signal_final = orig_amp * original_signal_norm
                original_match = pycbc_calculate_match(original_signal_final, observed_signal, sampFreq, psdHigh)
            else:
                original_match = 0.0

            print(f"距离精炼结果:")
            print(f"  原始距离: {original_distance_mpc:.1f} Mpc, 匹配度: {original_match:.4f}")
            print(f"  精炼距离: {estimated_distance_mpc:.1f} Mpc, 匹配度: {final_match:.4f}")
            print(f"  改进程度: {final_match - original_match:.4f}")

            # 判断是否接受精炼结果
            improvement_threshold = 0.01  # 至少改进1%
            if final_match > original_match + improvement_threshold and confidence > 0.5:
                status = 'success'
                use_refined = True
                print("✓ 距离精炼成功，采用新的距离估计")
            else:
                status = 'no_improvement'
                use_refined = False
                print("△ 距离精炼未带来显著改进，保持原始估计")
        else:
            status = 'normalization_failed'
            use_refined = False
            final_match = 0.0
            print("✗ 精炼信号归一化失败")

    except Exception as e:
        print(f"✗ 距离精炼过程出错: {str(e)}")
        status = 'error'
        use_refined = False
        estimated_r = original_r
        estimated_distance_mpc = original_distance_mpc
        confidence = 0.0
        final_match = 0.0

    # 准备返回结果
    if use_refined:
        refined_params = pso_params.copy()
        refined_params['r'] = estimated_r
        final_distance = estimated_distance_mpc
        final_r = estimated_r
    else:
        refined_params = pso_params.copy()
        final_distance = original_distance_mpc
        final_r = original_r

    refinement_info = {
        'status': status,
        'original_distance_mpc': original_distance_mpc,
        'original_distance_log10': original_r,
        'estimated_distance_mpc': estimated_distance_mpc,
        'estimated_distance_log10': estimated_r,
        'final_distance_mpc': final_distance,
        'final_distance_log10': final_r,
        'confidence': confidence,
        'improvement_used': use_refined,
        'final_match': final_match if 'final_match' in locals() else 0.0
    }

    print(f"最终使用距离: {final_distance:.1f} Mpc")
    print("=" * 50)

    return refined_params, refinement_info
```