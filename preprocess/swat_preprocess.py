import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
from sklearn.preprocessing import MinMaxScaler

# 忽略 openpyxl 的扩展警告
warnings.simplefilter("ignore", UserWarning)


def process_file(file_path, output_dir, sample_interval=1, window_size=10, step_size=5,
                 continuous_method="mean", binary_method="max"):
    """
    处理单个 SWaT 文件，自动找到 'Timestamp' 行并兼容空格/大小写
    """
    # 读取 Excel，不设置 header
    df = pd.read_excel(file_path, header=None)

    # 查找 'Timestamp' 行，去掉空格，忽略大小写
    col_idx_candidates = df[df.iloc[:, 0].astype(str).str.strip().str.lower() == 'timestamp'].index
    if len(col_idx_candidates) == 0:
        raise ValueError(f"No 'Timestamp' row found in {file_path}")
    col_idx = col_idx_candidates[0]

    # 设置列名并去掉说明行
    df.columns = df.iloc[col_idx].str.strip()
    df = df.iloc[col_idx + 1:].reset_index(drop=True)

    # 转换 Timestamp 列
    timestamp_col = 'Timestamp'
    if timestamp_col not in df.columns:
        raise ValueError(f"'Timestamp' column not found after processing {file_path}")
    label_col = df.columns[-1]

    # 明确 dayfirst=True 解析日期
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', dayfirst=True)
    if df[timestamp_col].isna().any():
        raise ValueError(f"Some timestamps could not be parsed in {file_path}")

    # 分离数据
    timestamps = df[timestamp_col]
    df_values = df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')  # 转为 float
    labels = df[label_col].values

    # 区分连续和二值列
    binary_cols = df_values.nunique() <= 2
    continuous_cols = ~binary_cols

    # 采样
    sample_idx = np.arange(0, len(df_values), sample_interval)
    sampled_timestamps = timestamps.iloc[sample_idx].reset_index(drop=True)
    sampled_labels = labels[sample_idx]
    sampled_values = pd.DataFrame(index=sampled_timestamps)

    # 使用列名列表避免布尔索引对齐问题
    cont_cols = df_values.columns[continuous_cols]
    bin_cols = df_values.columns[binary_cols]

    # 连续列处理
    if len(cont_cols) > 0:
        # Min-Max 归一化
        scaler = MinMaxScaler()
        cont_values = df_values[cont_cols].iloc[sample_idx].values
        cont_values_norm = scaler.fit_transform(cont_values)
        sampled_values[cont_cols] = cont_values_norm

    # 二值列处理
    if len(bin_cols) > 0:
        sampled_values[bin_cols] = df_values[bin_cols].iloc[sample_idx].values

    # 分窗采样
    windows = []
    window_labels = []
    for start in tqdm(range(0, len(sampled_values) - window_size + 1, step_size),
                      desc=f"Processing {os.path.basename(file_path)}"):
        end = start + window_size
        windows.append(sampled_values.iloc[start:end].values)
        window_labels.append(sampled_labels[end - 1])  # 最新点标签

    windows = np.array(windows)
    window_labels = np.array(window_labels)

    # 输出文件名
    base_name = Path(file_path).stem
    output_base = f"{base_name}_interval{sample_interval}_win{window_size}_step{step_size}_{continuous_method}_{binary_method}"

    # 保存 Excel
    excel_out = Path(output_dir) / f"{output_base}.xlsx"
    sampled_values["Label"] = sampled_labels
    sampled_values.to_excel(excel_out, index=False)

    # 保存 npz
    npz_out = Path(output_dir) / f"{output_base}.npz"
    np.savez_compressed(npz_out, windows=windows, labels=window_labels)

    print(f"Saved: {excel_out} and {npz_out}")


def main():
    parser = argparse.ArgumentParser(description="SWaT Dataset Preprocessing")
    parser.add_argument("--input_dir", type=str, required=True, help="Input folder with SWaT Excel files")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output folder")
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval (frames)")
    parser.add_argument("--window", type=int, default=10, help="Window size (frames)")
    parser.add_argument("--step", type=int, default=5, help="Step size for windows")
    parser.add_argument("--cont_method", type=str, default="mean", choices=["mean","median"], help="Continuous value aggregation method")
    parser.add_argument("--bin_method", type=str, default="max", choices=["max"], help="Binary value aggregation method")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    excel_files = list(Path(args.input_dir).glob("*.xlsx"))

    if len(excel_files) == 0:
        raise ValueError(f"No Excel files found in {args.input_dir}")

    # 并行处理
    with ProcessPoolExecutor() as executor:
        futures = []
        for file in excel_files:
            futures.append(executor.submit(process_file, file, args.output_dir,
                                           args.interval, args.window, args.step,
                                           args.cont_method, args.bin_method))
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
