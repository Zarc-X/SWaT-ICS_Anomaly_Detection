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
        n = len(x)
        
        # 步骤1: 傅里叶变换，获取振幅谱
        fft = np.fft.fft(x)
        amplitude = np.abs(fft)
        log_amplitude = np.log(amplitude + 1e-8)
        
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
    1. 谱残差转换
    2. 归一化
    3. 滑动窗口切分
    """
    if apply_spectral_residual:
        print("应用谱残差预处理...")
        processed_data = spectral_residual_transform(data)
    else:
        processed_data = data
    
    # 归一化
    print("归一化数据...")
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(processed_data)
    
    # 滑动窗口切分
    print(f"应用滑动窗口切分 (窗口大小={window_size}, 步长={step_size})...")
    windows = sliding_window_sequence(normalized_data, window_size, step_size)
    
    print(f"原始数据形状: {data.shape}")
    print(f"处理后窗口形状: {windows.shape}")
    
    return windows, scaler