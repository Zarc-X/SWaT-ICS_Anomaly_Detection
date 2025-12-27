# SWaT Dataset Preprocessing

## 功能
将 SWaT 原始 Excel 数据进行：
- 采样（降采样）
- 连续/二值数据聚合
- 分帧窗口采样
- 输出处理后的 Excel 和 `.npz` 文件

## 输入
- 目录下的 SWaT Excel 文件，例如：
- data/SWaT_Dataset_Attack_v0.xlsx
- data/SWaT_Dataset_Normal_v0.xlsx

## 输出
- 文件会保存到指定目录（默认 `processed_data`）：
  - Excel 文件：`<原文件名>_interval<采样秒>_win<窗口长度>_step<步长>_<连续方法>_<二值方法>.xlsx`
  - npz 文件：`<原文件名>_interval<采样秒>_win<窗口长度>_step<步长>_<连续方法>_<二值方法>.npz`

`.npz` 文件包含：
- `windows` → shape `(num_windows, window_size, m_features)`  
- `labels` → shape `(num_windows,)`，对应窗口最后一个点的标签

## 命令行使用示例
```bash
python swat_preprocess.py \
    --input_dir "SWaT-ICS Anomaly Detection/data" \
    --output_dir "processed_data" \
    --interval 5 \
    --window 20 \
    --step 10 \
    --cont_method mean \
    --bin_method max
