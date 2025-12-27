import numpy as np
from sklearn.preprocessing import StandardScaler

def spectral_residual_transform(series):
    """
    实现论文IV.A节的谱残差预处理
    输入: 时间序列 (n_samples, n_features)
    输出: 显著性图 (n_samples, n_features)
    """
    def _sr_1d(x):
        """处理一维时间序列"""
        # 检查是否有NaN或Inf
        if np.any(~np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        n = len(x)
        
        # 步骤1: 傅里叶变换，获取振幅谱
        fft = np.fft.fft(x)
        amplitude = np.abs(fft)
        
        # 防止log的数值不稳定，添加更大的epsilon
        log_amplitude = np.log(amplitude + 1e-5)
        
        # 步骤2: 计算平均谱（使用卷积平滑）
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        avg_spectrum = np.convolve(log_amplitude, kernel, mode='same')
        
        # 步骤3: 计算谱残差
        spectral_residual = log_amplitude - avg_spectrum
        
        # 步骤4: 逆傅里叶变换，获取显著性图
        phase = np.angle(fft)
        combined = np.exp(spectral_residual + 1j * phase)
        saliency_map = np.abs(np.fft.ifft(combined))
        
        # 重要：对显著性图进行归一化，防止数值溢出
        # 使用L2范数归一化
        max_val = np.max(np.abs(saliency_map))
        if max_val > 1e-8:
            saliency_map = saliency_map / (max_val + 1e-8)
        else:
            saliency_map = saliency_map * 0  # 如果全是0则保持0
        
        return saliency_map
    
    # 对每个特征分别应用SR
    n_samples, n_features = series.shape
    saliency_maps = np.zeros_like(series)
    
    for i in range(n_features):
        saliency_maps[:, i] = _sr_1d(series[:, i])
    
    return saliency_maps

def sliding_window_sequence(data, window_size, step_size=1):
    """
    实现论文公式(14)-(15)的滑动窗口切分
    输入: 数据 (n_samples, n_features)
    输出: 窗口序列 (n_windows, window_size, n_features)
    """
    n_samples = len(data)
    windows = []
    
    for i in range(0, n_samples - window_size + 1, step_size):
        window = data[i:i + window_size]
        windows.append(window)
    
    return np.array(windows)

def preprocess_data_for_evoaae(data, window_size=100, step_size=10, apply_spectral_residual=True):
    """
    完整的EvoAAE预处理流程（论文IV.A节）
    1. 数据验证和清理
    2. 谱残差转换
    3. 归一化
    4. 滑动窗口切分
    """
    # 数据验证和清理
    print("数据验证...")
    data = np.asarray(data, dtype=np.float32)
    
    # 替换NaN和Inf值
    if np.any(~np.isfinite(data)):
        print(f"  发现 {np.sum(~np.isfinite(data))} 个非有限值，进行清理...")
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 显示数据统计信息
    print(f"  数据范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
    print(f"  数据均值: {np.mean(data):.6f}, 标准差: {np.std(data):.6f}")
    
    if apply_spectral_residual:
        print("应用谱残差预处理...")
        processed_data = spectral_residual_transform(data)
        print(f"  谱残差后范围: [{np.min(processed_data):.6f}, {np.max(processed_data):.6f}]")
    else:
        processed_data = data
    
    # 归一化
    print("归一化数据...")
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(processed_data)
    print(f"  归一化后范围: [{np.min(normalized_data):.6f}, {np.max(normalized_data):.6f}]")
    print(f"  归一化后均值: {np.mean(normalized_data):.6f}, 标准差: {np.std(normalized_data):.6f}")
    
    # 滑动窗口切分
    print(f"应用滑动窗口切分 (窗口大小={window_size}, 步长={step_size})...")
    windows = sliding_window_sequence(normalized_data, window_size, step_size)
    
    print(f"原始数据形状: {data.shape}")
    print(f"处理后窗口形状: {windows.shape}")
    print(f"窗口数据范围: [{np.min(windows):.6f}, {np.max(windows):.6f}]")
    
    return windows, scaler