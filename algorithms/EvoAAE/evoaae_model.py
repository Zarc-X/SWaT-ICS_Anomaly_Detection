import torch
import numpy as np
from .preprocessing import preprocess_data_for_evoaae, sliding_window_sequence
from .models import AdversarialAutoencoderWithDualDiscriminator
from .pso_optimizer import BinaryPSO

class EvoAAE:
    """
    论文中的完整EvoAAE系统
    结合了谱残差预处理、卷积对抗自编码器和二进制PSO优化
    """
    def __init__(self, config=None, device=None):
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 默认配置
        self.default_config = {
            'preprocessing': {
                'window_size': 100,
                'step_size': 10,
                'apply_spectral_residual': True
            },
            'pso_options': {
                'batch_size': [1024, 2048, 4096, 6144],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
                'optimizer': ['adam', 'adamax', 'rmsprop', 'adadelta'],
                'n_layers': [3, 4, 5, 6],
                'n_kernels': [2, 4, 8, 16, 32, 64, 128, 256],
                'kernel_size': [1, 2, 3, 4],
                'normalization': ['batchnorm', 'none'],
                'activation': ['relu', 'sigmoid', 'tanh', 'none']
            },
            'model': {
                'latent_dim': 8,
                'kl_beta': 1.0,
                'adv_weight_latent': 1.0,
                'adv_weight_data': 1.0,
                'conv_channels': [32, 64],
                'kernel_sizes': [3, 3]
            }
        }
        
        # 合并配置
        if config:
            self._merge_configs(self.default_config, config)
        
        self.config = self.default_config
        
        # 模型组件
        self.preprocessor = None
        self.model = None
        self.best_params = None
        self.training_history = None
        
    def _merge_configs(self, base, update):
        """递归合并配置字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def fit(self, X, epochs=50, pso_pop_size=20, pso_max_iter=30):
        """
        训练EvoAAE模型（论文完整流程）
        1. 预处理数据
        2. PSO优化超参数
        3. 训练最终模型
        """
        print("=" * 60)
        print("开始训练EvoAAE模型")
        print("=" * 60)
        
        # 步骤1: 数据预处理
        print("\n1. 数据预处理...")
        windows, self.preprocessor = preprocess_data_for_evoaae(
            X,
            window_size=self.config['preprocessing']['window_size'],
            step_size=self.config['preprocessing']['step_size'],
            apply_spectral_residual=self.config['preprocessing']['apply_spectral_residual']
        )
        
        # 数据集划分
        n_windows = len(windows)
        train_size = int(0.7 * n_windows)
        val_size = int(0.15 * n_windows)
        
        indices = np.random.permutation(n_windows)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        X_train = windows[train_idx]
        X_val = windows[val_idx]
        X_test = windows[test_idx]
        
        print(f"训练集: {len(X_train)} 个窗口")
        print(f"验证集: {len(X_val)} 个窗口")
        print(f"测试集: {len(X_test)} 个窗口")
        
        # 步骤2: PSO优化（论文核心贡献）
        print("\n2. PSO超参数优化...")
        
        # 适应度函数
        def fitness_function(params):
            # 从PSO参数构建模型配置
            model_config = {
                'input_dim': X_train.shape[2],
                'seq_len': X_train.shape[1],
                'latent_dim': self.config['model']['latent_dim'],
                'conv_channels': [params['n_kernels']] * params['n_layers'],
                'kernel_sizes': [params['kernel_size']] * params['n_layers'],
                'kl_beta': self.config['model']['kl_beta'],
                'adv_weight_latent': self.config['model']['adv_weight_latent'],
                'adv_weight_data': self.config['model']['adv_weight_data']
            }
            
            # 创建模型
            model = AdversarialAutoencoderWithDualDiscriminator(model_config, self.device).to_device()
            
            # 设置优化器
            lr = params['learning_rate']
            model.compile_optimizers(
                enc_dec_lr=lr,
                latent_disc_lr=lr,
                data_disc_lr=lr
            )
            
            # 快速训练几轮评估适应度
            batch_size = min(params['batch_size'], len(X_train) - (len(X_train) % min(params['batch_size'], len(X_train))))
            batch_size = max(batch_size, 2)  # 确保至少2个样本
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            
            # 训练少量epochs
            history = model.fit(
                X_train_tensor,
                epochs=5,
                batch_size=batch_size,
                validation_data=X_val_tensor,
                verbose=0
            )
            
            # 适应度 = 验证集重构误差 + 判别器损失（论文公式）
            fitness = np.mean(history['val_recon_loss'][-2:]) + \
                     np.mean(history['val_ld_loss'][-2:]) + \
                     np.mean(history['val_xd_loss'][-2:])
            
            return fitness
        
        # 执行PSO优化
        pso = BinaryPSO(
            param_options=self.config['pso_options'],
            pop_size=pso_pop_size,
            max_iter=pso_max_iter
        )
        
        self.best_params, best_fitness = pso.optimize(fitness_function)
        
        # 步骤3: 使用最佳参数训练最终模型
        print("\n3. 使用最佳参数训练最终模型...")
        
        # 构建最终模型配置
        final_config = {
            'input_dim': X_train.shape[2],
            'seq_len': X_train.shape[1],
            'latent_dim': self.config['model']['latent_dim'],
            'conv_channels': [self.best_params['n_kernels']] * self.best_params['n_layers'],
            'kernel_sizes': [self.best_params['kernel_size']] * self.best_params['n_layers'],
            'kl_beta': self.config['model']['kl_beta'],
            'adv_weight_latent': self.config['model']['adv_weight_latent'],
            'adv_weight_data': self.config['model']['adv_weight_data']
        }
        
        # 创建最终模型
        self.model = AdversarialAutoencoderWithDualDiscriminator(final_config, self.device).to_device()
        
        # 设置优化器
        lr = self.best_params['learning_rate']
        self.model.compile_optimizers(
            enc_dec_lr=lr,
            latent_disc_lr=lr,
            data_disc_lr=lr
        )
        
        # 完整训练
        batch_size = min(self.best_params['batch_size'], len(X_train))
        batch_size = batch_size - (len(X_train) % batch_size)  # 确保能被整除
        batch_size = max(batch_size, 2)  # 确保至少2个样本
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        print(f"最终模型配置:")
        print(f"  - 批量大小: {batch_size}")
        print(f"  - 学习率: {lr}")
        print(f"  - 优化器: {self.best_params['optimizer']}")
        print(f"  - 卷积层数: {self.best_params['n_layers']}")
        print(f"  - 每层卷积核数: {self.best_params['n_kernels']}")
        print(f"  - 卷积核大小: {self.best_params['kernel_size']}")
        print(f"  - 归一化: {self.best_params['normalization']}")
        print(f"  - 激活函数: {self.best_params['activation']}")
        
        self.training_history = self.model.fit(
            X_train_tensor,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=X_val_tensor,
            verbose=1
        )
        
        print("\n训练完成！")
        
        return self.training_history
    
    def detect_anomalies(self, X, threshold_percentile=95):
        """
        检测异常（论文异常检测方法）
        基于重构误差的阈值方法
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 预处理输入数据
        if self.config['preprocessing']['apply_spectral_residual']:
            # 使用训练时的预处理流程
            # 注意：实际应用中应保存训练时的scaler
            from sklearn.preprocessing import StandardScaler
            
            # 谱残差处理
            from .preprocessing import spectral_residual_transform
            X_sr = spectral_residual_transform(X)
            
            # 使用训练时的scaler
            if hasattr(self.preprocessor, 'transform'):
                X_scaled = self.preprocessor.transform(X_sr)
            else:
                X_scaled = X_sr
        else:
            # 简单标准化
            X_scaled = self.preprocessor.transform(X)
        
        # 滑动窗口切分
        windows = sliding_window_sequence(
            X_scaled,
            window_size=self.config['preprocessing']['window_size'],
            step_size=self.config['preprocessing']['step_size']
        )
        
        # 计算重构误差
        reconstruction_errors = self.model.compute_reconstruction_error(windows)
        
        # 计算异常阈值（基于训练数据分布）
        # 在实际应用中，这里应该使用训练数据的重构误差分布
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        # 标记异常
        anomalies = reconstruction_errors > threshold
        
        # 返回结果
        results = {
            'reconstruction_errors': reconstruction_errors,
            'threshold': threshold,
            'anomalies': anomalies,
            'anomaly_count': np.sum(anomalies),
            'anomaly_percentage': np.mean(anomalies) * 100
        }
        
        return results
    
    def evaluate(self, X, y_true=None):
        """
        评估模型性能
        如果提供真实标签，计算precision, recall, F1分数
        """
        # 检测异常
        results = self.detect_anomalies(X)
        
        metrics = {
            'threshold': results['threshold'],
            'anomaly_count': results['anomaly_count'],
            'anomaly_percentage': results['anomaly_percentage'],
            'mean_reconstruction_error': np.mean(results['reconstruction_errors'])
        }
        
        # 如果有真实标签，计算分类指标
        if y_true is not None:
            # 确保标签与检测结果形状一致
            if len(y_true) != len(results['reconstruction_errors']):
                print(f"警告: 标签长度({len(y_true)})与检测结果长度({len(results['reconstruction_errors'])})不匹配")
                return metrics
            
            # 计算分类指标
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # 二元分类：正常=0，异常=1
            y_pred = results['anomalies'].astype(int)
            
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
            
            print(f"评估结果:")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - F1-Score: {metrics['f1_score']:.4f}")
            print(f"  - 检测到的异常: {metrics['anomaly_count']}/{np.sum(y_true)}")
        
        return metrics