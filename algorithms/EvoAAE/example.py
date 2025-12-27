import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体支持
from matplotlib_config import setup_chinese_font
setup_chinese_font()

from evoaae_model import EvoAAE

def generate_synthetic_data(n_samples=10000, n_features=10, anomaly_ratio=0.05):
    """生成合成数据用于测试"""
    np.random.seed(42)
    
    # 生成正常数据（多元高斯分布）
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_samples
    )
    
    # 添加一些周期性模式
    t = np.linspace(0, 20 * np.pi, n_samples)
    for i in range(n_features):
        normal_data[:, i] += 0.5 * np.sin(t + i * np.pi / n_features)
    
    # 添加一些异常（占总数据的5%）
    n_anomalies = int(anomaly_ratio * n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # 随机选择一个特征并添加大偏移
        feature_idx = np.random.randint(0, n_features)
        normal_data[idx, feature_idx] += np.random.uniform(5, 10)
    
    # 创建标签（0=正常，1=异常）
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1
    
    return normal_data, labels

def main():
    """主函数：演示如何使用EvoAAE"""
    print("EvoAAE - 进化对抗自编码器示例")
    
    # 1. 生成示例数据
    print("生成合成数据...")
    data, labels = generate_synthetic_data(n_samples=5000, n_features=8, anomaly_ratio=0.05)
    
    print(f"数据形状: {data.shape}")
    print(f"异常比例: {np.sum(labels) / len(labels) * 100:.1f}%")
    
    # 2. 划分训练集和测试集（训练只用正常数据）
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    # 训练集：正常数据
    X_train = data[normal_indices[:3000]]
    
    # 测试集：正常+异常数据
    X_test = np.vstack([
        data[normal_indices[3000:3500]],  # 正常数据
        data[anomaly_indices[:150]]       # 异常数据
    ])
    y_test = np.hstack([
        np.zeros(500),  # 正常标签
        np.ones(150)    # 异常标签
    ])
    
    print(f"\n训练集: {len(X_train)} 个正常样本")
    print(f"测试集: {len(X_test)} 个样本 ({np.sum(y_test)} 个异常)")
    
    # 3. 配置和训练EvoAAE
    config = {
        'preprocessing': {
            'window_size': 50,  # 较小的窗口用于演示
            'step_size': 10,
            'apply_spectral_residual': True
        },
        'model': {
            'latent_dim': 8,
            'kl_beta': 0.5,
            'adv_weight_latent': 1.0,
            'adv_weight_data': 1.0
        }
    }
    
    # 创建模型
    evoaae = EvoAAE(config)
    
    # 训练模型（简化参数以加快速度）
    print("\n开始训练模型...")
    history = evoaae.fit(
        X_train,
        epochs=30,
        pso_pop_size=10,
        pso_max_iter=10
    )
    
    # 4. 评估模型
    print("\n评估模型性能...")
    metrics = evoaae.evaluate(X_test, y_test)
    
    # 5. 可视化结果
    results = evoaae.detect_anomalies(X_test)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_title('训练损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 重构误差分布
    axes[0, 1].hist(results['reconstruction_errors'], bins=50, alpha=0.7, color='blue')
    axes[0, 1].axvline(results['threshold'], color='red', linestyle='--', 
                      label=f'阈值: {results["threshold"]:.4f}')
    axes[0, 1].set_title('重构误差分布')
    axes[0, 1].set_xlabel('重构误差')
    axes[0, 1].set_ylabel('频率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 重构误差时间序列
    axes[1, 0].plot(results['reconstruction_errors'], alpha=0.7, label='重构误差')
    axes[1, 0].axhline(results['threshold'], color='red', linestyle='--', 
                      label=f'阈值: {results["threshold"]:.4f}')
    
    # 标记异常点
    anomaly_indices_plot = np.where(results['anomalies'])[0]
    axes[1, 0].scatter(anomaly_indices_plot, 
                      results['reconstruction_errors'][anomaly_indices_plot],
                      color='red', s=30, label='检测到的异常', zorder=5)
    
    axes[1, 0].set_title('重构误差时间序列')
    axes[1, 0].set_xlabel('样本索引')
    axes[1, 0].set_ylabel('重构误差')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 真实标签 vs 预测标签
    axes[1, 1].plot(y_test, 'g-', alpha=0.7, label='真实标签')
    axes[1, 1].plot(results['anomalies'].astype(float), 'b--', alpha=0.7, label='预测标签')
    axes[1, 1].set_title('真实标签 vs 预测标签')
    axes[1, 1].set_xlabel('样本索引')
    axes[1, 1].set_ylabel('标签 (0=正常, 1=异常)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('evoaae_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 6. 打印摘要
    print("\n" + "=" * 60)
    print("EvoAAE实现完成！")
    print("=" * 60)
    print(f"设备: {evoaae.device}")
    print(f"检测到的异常: {results['anomaly_count']} ({results['anomaly_percentage']:.1f}%)")
    
    if 'precision' in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return evoaae, history, metrics

if __name__ == "__main__":
    model, history, metrics = main()