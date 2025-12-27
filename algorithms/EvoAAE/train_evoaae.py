import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import pickle
from datetime import datetime
# import torch

# # 强制使用CPU而不是GPU（避免CUDA问题）
# torch.cuda.is_available = lambda: False
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 设置matplotlib中文字体支持
from .matplotlib_config import setup_chinese_font
setup_chinese_font()

from .evoaae_model import EvoAAE

def load_swat_data(normal_path, attack_path):
    """加载SWaT数据集"""
    print("加载数据...")
    
    # 加载Excel文件
    normal_data = pd.read_excel(normal_path)
    attack_data = pd.read_excel(attack_path)

    # 强制转换为数值类型，无法转换的值设为NaN后再填充，避免FFT遇到非数值
    normal_data = normal_data.apply(pd.to_numeric, errors='coerce')
    attack_data = attack_data.apply(pd.to_numeric, errors='coerce')

    # 用前向/后向填充处理缺失，再用0兜底
    normal_data = normal_data.ffill().bfill().fillna(0.0)
    attack_data = attack_data.ffill().bfill().fillna(0.0)
    
    print(f"正常数据形状: {normal_data.shape}")
    print(f"攻击数据形状: {attack_data.shape}")
    
    # 转换为numpy数组
    X_train = normal_data.values.astype(np.float32)
    X_test = attack_data.values.astype(np.float32)
    
    # 为测试集创建标签（假设attack数据包含异常）
    # 注意：如果你的数据集有标签列，请相应修改
    y_test = np.ones(len(X_test))  # 假设attack数据全是异常
    
    return X_train, X_test, y_test, list(normal_data.columns)


def save_intermediate_results(output_dir, X_train, X_test, y_test, 
                             history, metrics, results, config, feature_names,
                             model=None, y_test_original=None):
    """
    保存所有中间结果和模型信息
    
    Args:
        output_dir: 输出目录
        X_train: 训练数据
        X_test: 测试数据
        y_test: 测试标签（窗口化后）
        history: 训练历史
        metrics: 评估指标
        results: 检测结果
        config: 模型配置
        feature_names: 特征名称列表
        model: 训练好的模型对象（可选）
        y_test_original: 原始的测试标签（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("保存中间结果和模型信息...")
    print("=" * 80)
    
    # 1. 保存预处理后的数据
    data_dir = os.path.join(output_dir, "preprocessed_data")
    os.makedirs(data_dir, exist_ok=True)
    
    np.savez(os.path.join(data_dir, "train_data.npz"), 
             X_train=X_train, feature_names=np.array(feature_names))
    print(f" 训练数据已保存到: {os.path.join(data_dir, 'train_data.npz')}")
    
    np.savez(os.path.join(data_dir, "test_data.npz"), 
             X_test=X_test, y_test=y_test, feature_names=np.array(feature_names))
    print(f" 测试数据已保存到: {os.path.join(data_dir, 'test_data.npz')}")
    
    # 如果有原始标签，也保存它
    if y_test_original is not None:
        np.savez(os.path.join(data_dir, "test_labels_original.npz"), 
                 y_test_original=y_test_original)
        print(f" 原始测试标签已保存到: {os.path.join(data_dir, 'test_labels_original.npz')}")
    
    # 2. 保存训练历史
    history_dir = os.path.join(output_dir, "training_history")
    os.makedirs(history_dir, exist_ok=True)
    
    # 将history字典中的numpy数组转换为列表以便JSON序列化
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            history_serializable[key] = value.tolist()
        else:
            history_serializable[key] = value
    
    with open(os.path.join(history_dir, "history.json"), 'w', encoding='utf-8') as f:
        json.dump(history_serializable, f, indent=2)
    print(f" 训练历史已保存到: {os.path.join(history_dir, 'history.json')}")
    
    # 保存loss和val_loss为CSV便于查看
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['loss']) + 1),
        'loss': history['loss']
    })
    if 'val_loss' in history:
        history_df['val_loss'] = history['val_loss']
    
    history_df.to_csv(os.path.join(history_dir, "history.csv"), index=False)
    print(f" 训练历史CSV已保存到: {os.path.join(history_dir, 'history.csv')}")
    
    # 3. 保存检测结果
    results_dir = os.path.join(output_dir, "detection_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存重构误差和异常检测结果
    np.savez(os.path.join(results_dir, "anomaly_detection.npz"),
             reconstruction_errors=results['reconstruction_errors'],
             anomalies=results['anomalies'],
             threshold=np.array([results['threshold']]))
    print(f" 异常检测结果已保存到: {os.path.join(results_dir, 'anomaly_detection.npz')}")
    
    # 4. 保存评估指标
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, (np.ndarray, np.generic)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value
    
    with open(os.path.join(results_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    print(f" 评估指标已保存到: {os.path.join(results_dir, 'metrics.json')}")
    
    # 5. 保存配置信息
    config_dir = os.path.join(output_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    with open(os.path.join(config_dir, "model_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f" 模型配置已保存到: {os.path.join(config_dir, 'model_config.json')}")
    
    # 6. 保存数据集信息
    dataset_info = {
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'feature_count': X_train.shape[1],
        'feature_names': feature_names,
        'train_samples_class_0': int(np.sum(y_test == 0)),
        'train_samples_class_1': int(np.sum(y_test == 1)),
    }
    
    with open(os.path.join(config_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f" 数据集信息已保存到: {os.path.join(config_dir, 'dataset_info.json')}")
    
    # 7. 保存模型对象（使用pickle）
    if model is not None:
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            with open(os.path.join(model_dir, "evoaae_model.pkl"), 'wb') as f:
                pickle.dump(model, f)
            print(f" 模型对象已保存到: {os.path.join(model_dir, 'evoaae_model.pkl')}")
        except Exception as e:
            print(f"⚠ 警告: 模型对象保存失败 - {str(e)}")
    
    # 8. 保存执行摘要
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_epochs': len(history['loss']),
        'final_loss': float(history['loss'][-1]),
        'anomaly_detection_threshold': float(results['threshold']),
        'anomalies_detected': int(results['anomaly_count']),
        'anomaly_percentage': float(results['anomaly_percentage']),
        'metrics': metrics_json
    }
    
    with open(os.path.join(output_dir, "execution_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f" 执行摘要已保存到: {os.path.join(output_dir, 'execution_summary.json')}")
    
    print("\n所有中间结果已保存完成！")
    print(f"保存位置: {output_dir}")



def main():
    """主函数：在SWaT数据集上训练和测试EvoAAE"""
    print("=" * 80)
    print("EvoAAE - 在SWaT数据集上进行异常检测")
    print("=" * 80)
    
    # 数据路径（相对于项目根目录）
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    normal_path = os.path.join(base_dir, "processed_data", "SWaT_Dataset_Normal_v0_interval5_win20_step10_mean_max.xlsx")
    attack_path = os.path.join(base_dir, "processed_data", "SWaT_Dataset_Attack_v0_interval5_win20_step10_mean_max.xlsx")
    
    # 1. 加载数据
    X_train, X_test, y_test_original, feature_names = load_swat_data(normal_path, attack_path)
    
    print(f"\n特征数量: {len(feature_names)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 2. 配置EvoAAE模型
    config = {
        'preprocessing': {
            'window_size': 100,  # 可根据需要调整
            'step_size': 20,     # 可根据需要调整
            'apply_spectral_residual': True
        },
        'model': {
            'latent_dim': 16,    # 潜在空间维度
            'kl_beta': 0.5,      # KL散度权重
            'adv_weight_latent': 1.0,
            'adv_weight_data': 1.0
        }
    }
    
    # 3. 创建并训练模型
    print("\n" + "=" * 80)
    print("初始化EvoAAE模型...")
    print("=" * 80)
    evoaae = EvoAAE(config)
    
    print("\n开始训练模型...")
    print("这可能需要一些时间，请耐心等待...")
    
    # 训练模型（可根据需要调整参数）
    history = evoaae.fit(
        X_train,
        epochs=50,           # 训练轮数
        pso_pop_size=15,     # PSO种群大小
        pso_max_iter=20      # PSO最大迭代次数
    )
    
    # 4. 在测试集上评估
    print("\n" + "=" * 80)
    print("在测试集上评估模型...")
    print("=" * 80)
    
    # 检测异常
    results = evoaae.detect_anomalies(X_test)
    
    # 调整y_test长度以匹配检测结果
    # 因为滑动窗口会改变序列长度
    # 保留与检测结果相同长度的标签
    window_size = config['preprocessing']['window_size']
    step_size = config['preprocessing']['step_size']
    n_windows = len(results['reconstruction_errors'])
    
    # 为每个窗口分配标签（使用窗口的最后一个样本的标签，或窗口内任何异常都标记为异常）
    y_test = np.zeros(n_windows, dtype=y_test_original.dtype)
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, len(y_test_original))
        # 如果窗口内有异常样本，则标记该窗口为异常
        if np.any(y_test_original[start_idx:end_idx] > 0):
            y_test[i] = 1
        else:
            y_test[i] = 0
    
    print(f"原始测试集大小: {X_test.shape[0]}")
    print(f"窗口化后测试集大小: {len(y_test)}")
    print(f"检测结果长度: {len(results['reconstruction_errors'])}")
    
    # 评估指标（如果有真实标签）
    metrics = evoaae.evaluate(X_test, y_test)
    
    # 5. 可视化结果
    print("\n生成可视化结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 训练损失曲线
    axes[0, 0].plot(history['loss'], label='训练损失', linewidth=2)
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0, 0].set_title('训练损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 重构误差分布
    axes[0, 1].hist(results['reconstruction_errors'], bins=100, alpha=0.7, 
                    color='skyblue', edgecolor='black')
    axes[0, 1].axvline(results['threshold'], color='red', linestyle='--', 
                      linewidth=2, label=f'阈值: {results["threshold"]:.4f}')
    axes[0, 1].set_title('重构误差分布', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('重构误差', fontsize=12)
    axes[0, 1].set_ylabel('频率', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 重构误差时间序列（只显示前5000个样本以提高可读性）
    max_samples = min(5000, len(results['reconstruction_errors']))
    axes[1, 0].plot(results['reconstruction_errors'][:max_samples], 
                   alpha=0.7, linewidth=1, label='重构误差')
    axes[1, 0].axhline(results['threshold'], color='red', linestyle='--', 
                      linewidth=2, label=f'阈值: {results["threshold"]:.4f}')
    
    # 标记异常点
    anomaly_indices_plot = np.where(results['anomalies'][:max_samples])[0]
    if len(anomaly_indices_plot) > 0:
        axes[1, 0].scatter(anomaly_indices_plot, 
                          results['reconstruction_errors'][:max_samples][anomaly_indices_plot],
                          color='red', s=20, alpha=0.6, label='检测到的异常', zorder=5)
    
    axes[1, 0].set_title(f'重构误差时间序列（前{max_samples}个样本）', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('样本索引', fontsize=12)
    axes[1, 0].set_ylabel('重构误差', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 检测结果统计
    axes[1, 1].axis('off')
    
    # 显示评估指标
    metrics_text = f"""
    ===== Model Performance =====
    
    Device: {evoaae.device}
    
    Dataset:
    • Training set size: {X_train.shape[0]} samples
    • Test set size: {X_test.shape[0]} samples
    • Number of features: {X_train.shape[1]}
    
    Detection Results:
    • Anomalies detected: {results['anomaly_count']} 
    • Anomaly percentage: {results['anomaly_percentage']:.2f}%
    • Detection threshold: {results['threshold']:.6f}
    
    Performance Metrics:
    """
    
    if 'precision' in metrics:
        metrics_text += f"• Precision: {metrics['precision']:.4f}\n"
        metrics_text += f"• Recall: {metrics['recall']:.4f}\n"
        metrics_text += f"• F1-Score: {metrics['f1_score']:.4f}\n"
    
    if 'auc' in metrics:
        metrics_text += f"• AUC: {metrics['auc']:.4f}\n"
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = os.path.join(base_dir, "results", "EvoAAE")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, 'evoaae_swat_results.png'), 
                dpi=150, bbox_inches='tight')
    print(f"结果图已保存到: {os.path.join(output_dir, 'evoaae_swat_results.png')}")
    
    plt.show()
    
    # 保存结果
    output_dir = os.path.join(base_dir, "results", "EvoAAE")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有中间结果
    save_intermediate_results(
        output_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_test_original=y_test_original,
        history=history,
        metrics=metrics,
        results=results,
        config=config,
        feature_names=feature_names,
        model=evoaae
    )
    
    # 6. 保存检测结果
    # 处理长度不匹配的问题：滑动窗口可能导致结果长度与原始测试集长度不同
    result_len = len(results['reconstruction_errors'])
    y_test_matched = y_test[:result_len] if len(y_test) > result_len else np.pad(
        y_test, (0, result_len - len(y_test)), mode='constant', constant_values=0
    )
    
    print(f"\n调试信息:")
    print(f"  重构误差长度: {len(results['reconstruction_errors'])}")
    print(f"  异常标签长度: {len(results['anomalies'])}")
    print(f"  原始y_test长度: {len(y_test)}")
    print(f"  匹配后y_test长度: {len(y_test_matched)}")
    
    results_df = pd.DataFrame({
        'reconstruction_error': results['reconstruction_errors'],
        'is_anomaly': results['anomalies'].astype(int),
        'true_label': y_test_matched
    })
    
    results_csv_path = os.path.join(output_dir, 'evoaae_detection_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f" 检测结果已保存到: {results_csv_path}")
    
    # 7. 打印最终摘要
    print("\n" + "=" * 80)
    print("训练和测试完成！")
    print("=" * 80)
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"检测到的异常: {results['anomaly_count']} ({results['anomaly_percentage']:.2f}%)")
    
    if 'precision' in metrics:
        print(f"\nPrecision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC: {metrics['auc']:.4f}")
    
    print("\n所有结果已保存到 results/EvoAAE 目录")
    
    return evoaae, history, metrics, results

if __name__ == "__main__":
    model, history, metrics, results = main()
