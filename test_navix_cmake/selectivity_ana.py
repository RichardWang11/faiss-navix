import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Matplotlib 中文显示设置 ---
# 确保您的系统安装了支持中文的字体，例如 'SimHei'
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"无法设置中文字体，绘图标签可能显示不正确。错误: {e}")

# --- 加载数据 ---
# 请将 'your_results.csv' 替换为您实际的 CSV 文件名
# 这里我们假设您的 CSV 文件名为 navix_comparison_results_20250811_033216.csv
try:
    # 尝试读取文件，如果文件不存在，则使用示例数据
    csv_file_path = '/home/wjl/faiss-navix/test_navix_cmake/navix_comparison_results_20250811_033216.csv'
    df = pd.read_csv(csv_file_path)
    print(f"成功从 '{csv_file_path}' 文件加载了 {len(df)} 条数据。")
except FileNotFoundError:
    print("警告: 未找到指定的 CSV 文件。将使用示例数据进行绘图。")
    # 如果找不到文件，可以使用一个模拟的 DataFrame 来演示
    data = {'Selectivity': np.random.choice([0.528, 0.299, 0.120, 0.038, 0.012], 10000, p=[0.5, 0.3, 0.15, 0.04, 0.01])}
    df = pd.DataFrame(data)


# --- 绘制 Selectivity 分布直方图 ---
plt.figure(figsize=(10, 6))

# 使用 plt.hist 来创建直方图
# bins 参数可以调整，以更好地匹配论文中的图表样式
plt.hist(df['Selectivity'], bins=50, edgecolor='black', alpha=0.7)

# --- 图表美化 ---
plt.title('EM 查询的选择性 (Selectivity) 分布', fontsize=16)
plt.xlabel('选择性 (Selectivity)', fontsize=12)
plt.ylabel('查询数量 (Number of queries)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 将 y 轴设置为对数刻度，以更好地观察低频数值
plt.yscale('log')

# 显示图表
plt.show()

# --- 打印唯一的 Selectivity 值及其频率 ---
print("\n唯一 Selectivity 值及其出现频率:")
print(df['Selectivity'].value_counts().sort_index())
