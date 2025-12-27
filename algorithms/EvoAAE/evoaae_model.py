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
    
    def fit(self, X_normal, X_anomaly=None, epochs=50, pso_pop_size=20, pso_max_iter=30):
        """
        训练EvoAAE模型（修改策略：只用正常数据训练）
        1. 预处理正常数据
        2. PSO优化超参数（在正常数据上）
        3. 训练最终模型
        4. 在验证集（正常数据）上计算阈值
        
        Args:
            X_normal: 正常数据（训练数据）
            X_anomaly: 异常数据（可选，用于后续测试，不用于训练）
            epochs: 训练轮数
            pso_pop_size: PSO种群大小
            pso_max_iter: PSO最大迭代次数
        """
        print("=" * 60)
        print("开始训练EvoAAE模型（仅用正常数据）")
        print("=" * 60)
        
        # 步骤1: 预处理正常数据
        print("\n1. 预处理正常数据...")
        windows_normal, self.preprocessor = preprocess_data_for_evoaae(
            X_normal,
            window_size=self.config['preprocessing']['window_size'],
            step_size=self.config['preprocessing']['step_size'],
            apply_spectral_residual=self.config['preprocessing']['apply_spectral_residual']
        )
        
        # 仅在正常数据上划分训练/验证集
        n_windows = len(windows_normal)
        train_size = int(0.8 * n_windows)  # 80%训练，20%验证
        val_size = n_windows - train_size
        
        indices = np.random.permutation(n_windows)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]
        
        X_train = windows_normal[train_idx]
        X_val = windows_normal[val_idx]
        
        print("训练集和验证集已准备")
        print(f"  训练集: {len(X_train)} 个窗口")
        print(f"  验证集: {len(X_val)} 个窗口")
        
        # 如果提供了异常数据，也预处理它
        if X_anomaly is not None:
            print("\n预处理异常数据...")
            windows_anomaly, _ = preprocess_data_for_evoaae(
                X_anomaly,
                window_size=self.config['preprocessing']['window_size'],
                step_size=self.config['preprocessing']['step_size'],
                apply_spectral_residual=self.config['preprocessing']['apply_spectral_residual']
            )
            print(f"异常数据已预处理: {len(windows_anomaly)} 个窗口")
            X_test = windows_anomaly
        else:
            print("\n未提供异常数据用于测试")
            X_test = None
        
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
        
        # 步骤4: 在验证集（正常数据）上计算异常检测阈值
        print("\n4. 计算异常检测阈值...")
        print("   使用验证集（正常数据）的重建误差来设定阈值...")
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        self.model.eval()
        val_recon_errors = self.model.compute_reconstruction_error(X_val_tensor)
        
        # 使用95%ile作为阈值（假设95%的正常数据重建误差都在这个值以下）
        self.threshold = np.percentile(val_recon_errors, 95)
        val_mean = np.mean(val_recon_errors)
        val_median = np.median(val_recon_errors)
        val_min = np.min(val_recon_errors)
        val_max = np.max(val_recon_errors)
        
        print(f"阈值已计算: {self.threshold:.6f}")
        print(f"   验证集重建误差统计:")
        print(f"     - 最小值: {val_min:.6f}")
        print(f"     - 最大值: {val_max:.6f}")
        print(f"     - 平均值: {val_mean:.6f}")
        print(f"     - 中位数: {val_median:.6f}")
        print(f"     - 95%ile: {self.threshold:.6f}")
        
        return self.training_history
    
    def detect_anomalies(self, X):
        """
        检测异常（使用在训练时计算的阈值）
        基于重构误差的阈值方法
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        if not hasattr(self, 'threshold') or self.threshold is None:
            raise ValueError("阈值未计算，请确保在fit方法中已设置阈值")
        
        # 预处理输入数据
        X_scaled = self.preprocessor.transform(X)
        
        # 滑动窗口切分
        windows = sliding_window_sequence(
            X_scaled,
            window_size=self.config['preprocessing']['window_size'],
            step_size=self.config['preprocessing']['step_size']
        )
        
        # 计算重构误差
        reconstruction_errors = self.model.compute_reconstruction_error(windows)
        
        # 安全检查：处理NaN和Inf值
        reconstruction_errors = np.nan_to_num(reconstruction_errors, nan=0.0, posinf=1e10, neginf=0.0)
        
        # 使用训练时计算的阈值标记异常
        anomalies = reconstruction_errors > self.threshold
        
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