# EvoAAE 使用指南（中文）

本目录实现论文 *Evolutionary_Adversarial_Autoencoder_for_Unsupervised_Anomaly_Detection_of_Industrial_Internet_of_Things* 的 EvoAAE 模型，并提供在 SWaT 数据集上的端到端流程：预处理 → PSO 超参搜索 → 模型训练 → 阈值计算 → 异常检测与评估。

## 1. 环境依赖

- Python ≥ 3.8
- 主要包：`torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`
- 可参考已导出的依赖快照 `evoaae.txt`：
  ```bash
  pip install -r algorithms/EvoAAE/evoaae.txt
  ```

## 2. 数据准备

- 预期放置在项目根目录的 `processed_data/` 下：
  - `SWaT_Dataset_Normal_v0_interval5_win20_step10_mean_max.xlsx`
  - `SWaT_Dataset_Attack_v0_interval5_win20_step10_mean_max.xlsx`
- 表头需包含 `Label` 列（"Normal"/"Attack"），其余为 51 维传感器/执行器特征。

## 3. 快速运行（SWaT 示例）

在项目根目录执行：
```bash
python -m algorithms.EvoAAE.train_evoaae
```
运行流程：
1) 加载 Normal/Attack 数据；
2) 仅用正常数据训练（异常数据仅用于测试/评估）；
3) PSO 搜索超参；
4) 以最佳超参全量训练；
5) 在正常验证集上估计重构误差阈值；
6) 对攻击集窗口进行检测并计算指标。

输出位置（默认 `results/EvoAAE/`）：
- `evoaae_detection_results.csv`：每个窗口的重构误差、检测标签、真值。
- `detection_results/anomaly_detection.npz`：重构误差、阈值、异常掩码。
- `detection_results/metrics.json`：Precision/Recall/F1 等。
- `training_history/`：`history.json`、`history.csv`。
- `config/`：`model_config.json`、`dataset_info.json`。
- `model/`：`evoaae_model.pkl`（pickle 版模型）。
- `evoaae_swat_results.png`：训练与检测可视化。

## 4. 主要可调参数

在 `train_evoaae.py` 的 `config` 中修改：
- 预处理：`window_size`、`step_size`、`apply_spectral_residual`（是否启用谱残差）。
- 模型：`latent_dim`、`kl_beta`、`adv_weight_latent`、`adv_weight_data`。
- PSO：`pso_pop_size`、`pso_max_iter`，以及 `evoaae_model.py` 中 `pso_options`（批量大小、学习率、卷积层数、卷积核数、核大小、归一化、激活等搜索空间）。
- 训练轮数：`epochs`。

## 5. 在自有数据上的使用

示例流程（仅正常数据训练）：
```python
from algorithms.EvoAAE.evoaae_model import EvoAAE
import numpy as np

# X_normal: 正常样本 (n_samples, n_features)
# X_anomaly: 可选的异常样本，用于测试

config = {
    'preprocessing': {
        'window_size': 100,
        'step_size': 20,
        'apply_spectral_residual': False,
    },
    'model': {
        'latent_dim': 16,
        'kl_beta': 0.5,
        'adv_weight_latent': 0.1,
        'adv_weight_data': 0.1,
    }
}

model = EvoAAE(config)
model.fit(X_normal, X_anomaly=None, epochs=50, pso_pop_size=15, pso_max_iter=20)

# 检测
results = model.detect_anomalies(X_normal)  # 或其他待测数据
print(results['threshold'], results['anomaly_count'])
```

如果只想做检测（模型已训练且持有 scaler、阈值）：
```python
# 已训练好的 model
results = model.detect_anomalies(X_new)
```

## 6. 注意事项

- 阈值默认取 max(90% 分位, 均值+2σ) 的重构误差，偏保守，高精度但可能牺牲召回，可按需调整。
- PSO 搜索得到的批量大小在最终训练会被整除修正，若要严格保持一致可手动设定 `batch_size`。
- 训练中若出现梯度范数警告，可适当降低学习率或收紧梯度裁剪。
- 开启谱残差（`apply_spectral_residual=True`）更突出突变信号；关闭则更稳健于噪声。
- 窗口 `step_size` 越小，检测粒度越细但计算量更大。

## 7. 相关文件

- 模型与训练核心：`evoaae_model.py`, `models.py`
- 预处理：`preprocessing.py`
- PSO 搜索：`pso_optimizer.py`
- SWaT 端到端脚本：`train_evoaae.py`
- 可视化字体配置：`matplotlib_config.py`

如需复现实验，直接运行第 3 节命令即可生成完整结果与指标。
