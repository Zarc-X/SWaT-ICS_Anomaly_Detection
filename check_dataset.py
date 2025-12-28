import numpy as np
import pandas as pd

# 加载数据
file_path = "processed_data/SWaT_Dataset_Attack_v0_interval5_win20_step10_mean_max.npz"
data = np.load(file_path)

windows = data['windows']   # shape: (num_windows, window_size, num_features)
labels = data['labels']     # shape: (num_windows,)

# 设置 NumPy 打印选项，不省略任何元素
np.set_printoptions(threshold=np.inf, precision=3, suppress=True)

# 查看前 3 个窗口
for i in range(3):
    print(f"\n=== Window {i} - Label: {labels[i]} ===")
    
    # NumPy 原始矩阵
    print("NumPy array view:")
    print(windows[i])
    
    # 转成 pandas DataFrame 打印整齐
    df_window = pd.DataFrame(windows[i], columns=[f"F{j}" for j in range(windows.shape[2])])
    df_window['Label'] = labels[i]  # 可选：每行加标签
    print("\nPandas DataFrame view:")
    print(df_window)

    # 可选：输出每个窗口到 CSV
    # csv_path = f"window_{i}.csv"
    # df_window.to_csv(csv_path, index=False)
    # print(f"Saved CSV: {csv_path}")
