import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取数据
df = pd.read_csv('/home/wjl/faiss-navix/test_navix_cmake/navix_comparison_results_20250809_122645.csv')

# 查看数据基本信息
print("数据基本信息：")
print(df.head())
print("\n数据列名：")
print(df.columns.tolist())
print("\n数据统计：")
print(df.describe())

# 设置图表样式（移除中文字体设置）
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 创建图表
plt.figure(figsize=(12, 8))

# 按selectivity排序
df_sorted = df.sort_values('Selectivity')

# 分别绘制navix和hnsw的性能曲线
plt.plot(df_sorted['Selectivity'], 
         df_sorted['Navix (us)'], 
         marker='o', 
         linewidth=2, 
         markersize=6,
         label='NAVIX',
         alpha=0.8,
         color='blue')

plt.plot(df_sorted['Selectivity'], 
         df_sorted['HNSW+Filter (us)'], 
         marker='s', 
         linewidth=2, 
         markersize=6,
         label='HNSW+Filter',
         alpha=0.8,
         color='red')

# 设置图表属性（使用英文）
plt.xlabel('Selectivity', fontsize=14, fontweight='bold')
plt.ylabel('Query Time (microseconds)', fontsize=14, fontweight='bold')
plt.title('NAVIX vs HNSW+Filter Performance Comparison\nSelectivity vs Query Time', fontsize=16, fontweight='bold')

# 设置x轴范围为0到1
plt.xlim(0, 1)
plt.ylim(bottom=0)

# 添加网格和图例
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='best')

# 美化坐标轴
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 计算性能差异
avg_navix = df['Navix (us)'].mean()
avg_hnsw = df['HNSW+Filter (us)'].mean()

if avg_hnsw < avg_navix:
    improvement = ((avg_navix - avg_hnsw) / avg_navix) * 100
    plt.text(0.02, 0.98, f'HNSW+Filter avg. {improvement:.1f}% faster', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
             fontsize=11, verticalalignment='top')
else:
    improvement = ((avg_hnsw - avg_navix) / avg_hnsw) * 100
    plt.text(0.02, 0.98, f'NAVIX avg. {improvement:.1f}% faster', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             fontsize=11, verticalalignment='top')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建更详细的分析图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. 主要对比图（散点图）
ax1.scatter(df['Selectivity'], df['Navix (us)'], alpha=0.6, label='NAVIX', color='blue', s=30)
ax1.scatter(df['Selectivity'], df['HNSW+Filter (us)'], alpha=0.6, label='HNSW+Filter', color='red', s=30)
ax1.set_xlabel('Selectivity')
ax1.set_ylabel('Query Time (microseconds)')
ax1.set_title('Performance Scatter Plot: Selectivity vs Query Time')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 1)

# 2. 按selectivity分组的箱型图
# 创建selectivity分组
df['selectivity_group'] = pd.cut(df['Selectivity'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# 重构数据用于箱型图
plot_data = []
for _, row in df.iterrows():
    plot_data.append({'Selectivity Group': row['selectivity_group'], 'Method': 'NAVIX', 'Query Time': row['Navix (us)']})
    plot_data.append({'Selectivity Group': row['selectivity_group'], 'Method': 'HNSW+Filter', 'Query Time': row['HNSW+Filter (us)']})

plot_df = pd.DataFrame(plot_data)
sns.boxplot(data=plot_df, x='Selectivity Group', y='Query Time', hue='Method', ax=ax2)
ax2.set_title('Performance Distribution by Selectivity Range')
ax2.set_ylabel('Query Time (microseconds)')

# 3. 加速比图
speedup_corrected = df['HNSW+Filter (us)'] / df['Navix (us)']
ax3.scatter(df['Selectivity'], speedup_corrected, alpha=0.6, color='green', s=30)
ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (no difference)')
ax3.set_xlabel('Selectivity')
ax3.set_ylabel('Speed Ratio (HNSW time / NAVIX time)')
ax3.set_title('HNSW vs NAVIX Performance Ratio')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim(0, 1)

# 4. 性能统计柱状图
methods = ['NAVIX', 'HNSW+Filter']
avg_times = [df['Navix (us)'].mean(), df['HNSW+Filter (us)'].mean()]
std_times = [df['Navix (us)'].std(), df['HNSW+Filter (us)'].std()]

bars = ax4.bar(methods, avg_times, yerr=std_times, capsize=5, alpha=0.7, 
               color=['blue', 'red'], error_kw={'alpha': 0.8})
ax4.set_ylabel('Average Query Time (microseconds)')
ax4.set_title('Average Performance Comparison (with std dev)')
ax4.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值
for bar, avg_time in zip(bars, avg_times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + std_times[avg_times.index(avg_time)],
             f'{avg_time:.0f}μs', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 生成详细的性能分析报告
print("\n=== NAVIX vs HNSW+Filter Performance Analysis Report ===\n")

print("NAVIX Method Statistics:")
print(f"  Average query time: {df['Navix (us)'].mean():.2f} microseconds")
print(f"  Min query time: {df['Navix (us)'].min():.2f} microseconds")
print(f"  Max query time: {df['Navix (us)'].max():.2f} microseconds")
print(f"  Standard deviation: {df['Navix (us)'].std():.2f} microseconds")
print()

print("HNSW+Filter Method Statistics:")
print(f"  Average query time: {df['HNSW+Filter (us)'].mean():.2f} microseconds")
print(f"  Min query time: {df['HNSW+Filter (us)'].min():.2f} microseconds")
print(f"  Max query time: {df['HNSW+Filter (us)'].max():.2f} microseconds")
print(f"  Standard deviation: {df['HNSW+Filter (us)'].std():.2f} microseconds")
print()

print("Selectivity Statistics:")
print(f"  Min selectivity: {df['Selectivity'].min():.4f}")
print(f"  Max selectivity: {df['Selectivity'].max():.4f}")
print(f"  Average selectivity: {df['Selectivity'].mean():.4f}")
print()

# 性能比较
navix_avg = df['Navix (us)'].mean()
hnsw_avg = df['HNSW+Filter (us)'].mean()

if hnsw_avg < navix_avg:
    improvement = ((navix_avg - hnsw_avg) / navix_avg) * 100
    print(f"HNSW+Filter vs NAVIX average performance improvement: {improvement:.1f}%")
    print(f"HNSW+Filter is {navix_avg/hnsw_avg:.2f}x faster on average")
else:
    degradation = ((hnsw_avg - navix_avg) / hnsw_avg) * 100
    print(f"NAVIX vs HNSW+Filter average performance improvement: {degradation:.1f}%")
    print(f"NAVIX is {hnsw_avg/navix_avg:.2f}x faster on average")

print()

# 分析不同selectivity范围下的性能
print("=== Performance Analysis by Selectivity Range ===")
selectivity_ranges = [
    (0, 0.1, "Very Low Selectivity (0-0.1)"),
    (0.1, 0.3, "Low Selectivity (0.1-0.3)"),
    (0.3, 0.5, "Medium Selectivity (0.3-0.5)"),
    (0.5, 1.0, "High Selectivity (0.5-1.0)")
]

for low, high, label in selectivity_ranges:
    mask = (df['Selectivity'] >= low) & (df['Selectivity'] < high)
    subset = df[mask]
    
    if len(subset) > 0:
        navix_avg_range = subset['Navix (us)'].mean()
        hnsw_avg_range = subset['HNSW+Filter (us)'].mean()
        
        print(f"\n{label} (samples: {len(subset)}):")
        print(f"  NAVIX average time: {navix_avg_range:.2f} microseconds")
        print(f"  HNSW+Filter average time: {hnsw_avg_range:.2f} microseconds")
        
        if hnsw_avg_range < navix_avg_range:
            improvement = ((navix_avg_range - hnsw_avg_range) / navix_avg_range) * 100
            print(f"  HNSW+Filter is {improvement:.1f}% faster in this range")
        else:
            degradation = ((hnsw_avg_range - navix_avg_range) / hnsw_avg_range) * 100
            print(f"  NAVIX is {degradation:.1f}% faster in this range")

print("\n=== Analysis Complete ===")