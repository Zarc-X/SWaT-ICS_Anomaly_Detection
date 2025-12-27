"""
matplotlib中文字体配置模块
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams

def setup_chinese_font():
    """
    设置matplotlib支持中文显示
    """
    # 尝试使用系统可用的中文字体
    try:
        # 优先尝试使用 SimHei（黑体）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['font.serif'] = ['SimHei', 'DejaVu Serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        try:
            # 如果SimHei不可用，尝试 Microsoft YaHei
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['font.serif'] = ['Microsoft YaHei', 'DejaVu Serif']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            try:
                # 如果都不可用，尝试 SimSun
                plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
                plt.rcParams['font.serif'] = ['SimSun', 'DejaVu Serif']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                print("警告: 未能设置中文字体，图表中文字可能无法正常显示")
                # 使用默认英文字体
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

    # 设置其他图表参数
    rcParams['figure.figsize'] = (12, 8)
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
