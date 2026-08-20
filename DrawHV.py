# import matplotlib.subplots as plt
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 准备已知的统计量数据 (请替换为您真实的数值)
# ---------------------------------------------------------

# 格式要求：
# 'whislo': 最小值 (Lower whisker)
# 'q1'    : 25%分位数 (First quartile)
# 'med'   : 中位数 (Median)
# 'q3'    : 75%分位数 (Third quartile)
# 'whishi': 最大值 (Upper whisker)
# 'label' : X轴上的标签

# # HV 数据 (左上图)
# stats_hv = [
#     {'label': 'BLF-NSGA-II', 'whislo': 0.5629, 'q1': 0.6653, 'med': 0.72655, 'q3': 1.0165, 'whishi': 1.2298},
#     {'label': 'RL-NSGA-II-HD', 'whislo': 0.8376, 'q1': 0.9008, 'med': 0.94865, 'q3': 1.0184, 'whishi': 1.1744},
#     {'label': 'HD-NSGA-II', 'whislo': 0.8243, 'q1': 1.0163, 'med': 1.0771, 'q3': 1.1455, 'whishi': 1.1759}
# ]
#
# # IGD 数据 (右上图)
# stats_igd = [
#     {'label': 'BLF-NSGA-II', 'whislo': 0.2418, 'q1': 0.2695, 'med': 0.34535, 'q3': 0.4668, 'whishi': 0.5593},
#     {'label': 'RL-NSGA-II-HD', 'whislo': 0.1786, 'q1': 0.1872, 'med': 0.2039, 'q3': 0.221, 'whishi': 0.2419},
#     {'label': 'HD-NSGA-II', 'whislo': 0.0811, 'q1': 0.0948, 'med': 0.10895, 'q3': 0.1211, 'whishi': 0.1397}
# ]
#
# # Time 数据 (左下图)
# stats_time = [
#     {'label': 'BLF-NSGA-II', 'whislo': 556.75, 'q1': 656.18, 'med': 775.825, 'q3': 845.17, 'whishi': 928.07},
#     {'label': 'RL-NSGA-II-HD', 'whislo': 2185.82, 'q1': 2236.15, 'med': 2316.745, 'q3': 2403.44, 'whishi': 2758.41},
#     {'label': 'HD-NSGA-II', 'whislo': 1397.08, 'q1': 1438.05, 'med': 1516.45, 'q3': 1704.07, 'whishi': 1796.65}
# ]
#
# # Memory 数据 (右下图)
# stats_mem = [
#     {'label': 'BLF-NSGA-II', 'whislo': 3.12, 'q1': 3.13, 'med': 3.13, 'q3': 3.14, 'whishi': 3.16},
#     {'label': 'RL-NSGA-II-HD', 'whislo': 3.84, 'q1': 3.85, 'med': 3.87, 'q3': 3.89, 'whishi': 3.93},
#     {'label': 'HD-NSGA-II', 'whislo': 3.32, 'q1': 3.34, 'med': 3.35, 'q3': 3.35, 'whishi': 3.36}
# ]

# HV 数据 (左上图)
stats_hv = [
    {'label': 'HD-MOGA', 'whislo': 0.1301, 'q1': 0.2835, 'med': 0.58175, 'q3': 1.0341, 'whishi': 1.331},
    {'label': 'HD-SPEA2', 'whislo': 1.0843, 'q1': 1.1242, 'med': 1.14185, 'q3': 1.2008, 'whishi': 1.2404},
    {'label': 'HD-NSGA-III', 'whislo': 0.9361, 'q1': 1.0307, 'med': 1.082, 'q3': 1.1254, 'whishi': 1.1566},
    {'label': 'HD-MOEA/D', 'whislo': 0.8326, 'q1': 0.9116, 'med': 1.01625, 'q3': 1.1397, 'whishi': 1.2404},
    {'label': 'HD-IEBA', 'whislo': 0.1662, 'q1': 0.2442, 'med': 0.26545, 'q3': 0.3522, 'whishi': 0.4886},
    {'label': 'HD-NSDBO', 'whislo': 0.5967, 'q1': 0.9078, 'med': 1.0459, 'q3': 1.0789, 'whishi': 1.1612},
    {'label': 'HD-NSGA-II', 'whislo': 0.8243, 'q1': 1.0163, 'med': 1.0771, 'q3': 1.1455, 'whishi': 1.1759}
]

# IGD 数据 (右上图)
stats_igd = [
    {'label': 'HD-MOGA', 'whislo': 0.5474, 'q1': 0.6733, 'med': 1.02005, 'q3': 1.0509, 'whishi': 1.1341},
    {'label': 'HD-SPEA2', 'whislo': 0.1347, 'q1': 0.1534, 'med': 0.1668, 'q3': 0.1828, 'whishi': 0.2162},
    {'label': 'HD-NSGA-III', 'whislo': 0.1754, 'q1': 0.1817, 'med': 0.1873, 'q3': 0.1993, 'whishi': 0.2116},
    {'label': 'HD-MOEA/D', 'whislo': 0.2364, 'q1': 0.3018, 'med': 0.37655, 'q3': 0.4182, 'whishi': 0.8897},
    {'label': 'HD-IEBA', 'whislo': 0.5625, 'q1': 0.6932, 'med': 0.71105, 'q3': 0.8785, 'whishi': 0.9519},
    {'label': 'HD-NSDBO', 'whislo': 0.1686, 'q1': 0.2263, 'med': 0.27125, 'q3': 0.3052, 'whishi': 0.4614},
    {'label': 'HD-NSGA-II', 'whislo': 0.0811, 'q1': 0.0948, 'med': 0.10895, 'q3': 0.1211, 'whishi': 0.1397}
]

# Time 数据 (左下图)
stats_time = [
    {'label': 'HD-MOGA', 'whislo': 1524.76, 'q1': 1537.82, 'med': 1582.13, 'q3': 1605.35, 'whishi': 1630.47},
    {'label': 'HD-SPEA2', 'whislo': 1997.11, 'q1': 2001.52, 'med': 2013.615, 'q3': 2026.49, 'whishi': 2151.34},
    {'label': 'HD-NSGA-III', 'whislo': 1765.11, 'q1': 1784.41, 'med': 1792.435, 'q3': 1889.01, 'whishi': 1906.86},
    {'label': 'HD-MOEA/D', 'whislo': 808.67, 'q1': 877.25, 'med': 916.21, 'q3': 922.2, 'whishi': 943.19},
    {'label': 'HD-IEBA', 'whislo': 616.98, 'q1': 680.89, 'med': 755.965, 'q3': 776.07, 'whishi': 861.5},
    {'label': 'HD-NSDBO', 'whislo': 1452.89, 'q1': 1486.45, 'med': 1510.11, 'q3': 1538.05, 'whishi': 1577.58},
    {'label': 'HD-NSGA-II', 'whislo': 1397.08, 'q1': 1438.05, 'med': 1516.45, 'q3': 1704.07, 'whishi': 1796.65}
]

# Memory 数据 (右下图)
stats_mem = [
    {'label': 'HD-MOGA', 'whislo': 2.17, 'q1': 2.18, 'med': 2.18, 'q3': 2.18, 'whishi': 2.19},
    {'label': 'HD-SPEA2', 'whislo': 4.07, 'q1': 4.07, 'med': 4.075, 'q3': 4.08, 'whishi': 4.09},
    {'label': 'HD-NSGA-III', 'whislo': 3.66, 'q1': 3.67, 'med': 3.675, 'q3': 3.68, 'whishi': 3.88},
    {'label': 'HD-MOEA/D', 'whislo': 2.35, 'q1': 2.67, 'med': 3.595, 'q3': 3.83, 'whishi': 4.31},
    {'label': 'HD-IEBA', 'whislo': 1.43, 'q1': 1.45, 'med': 1.47, 'q3': 1.48, 'whishi': 1.54},
    {'label': 'HD-NSDBO', 'whislo': 2.65, 'q1': 2.71, 'med': 2.76, 'q3': 2.77, 'whishi': 2.77},
    {'label': 'HD-NSGA-II', 'whislo': 3.32, 'q1': 3.34, 'med': 3.35, 'q3': 3.35, 'whishi': 3.36}
]

# 强烈对比，经典多分类配色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#ffff00']

# ---------------------------------------------------------
# 2. 创建画布与子图 (1行2列)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 12))
ax1, ax2, ax3, ax4 = axes.flatten()
# ---------------------------------------------------------
# 3. 定义一个绘制标准化箱线图的函数 (使用 bxp 方法)
# ---------------------------------------------------------
def create_custom_boxplot_from_stats(ax, stats_data, ylabel, sub_title, y_limits):
    # 使用 ax.bxp 绘制基于预计算统计量的箱线图
    bplot = ax.bxp(
        stats_data,
        patch_artist=True,  # 允许填充颜色
        widths=0.6,  # 箱体宽度
        showfliers=False,  # 如果没有离群值，可以关掉
        # 设置箱线图各部分的样式
        boxprops=dict(color='black', linewidth=1),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1),
        # 图中的红线呈现点划线样式 (-.)
        medianprops=dict(color='red', linestyle='-.', linewidth=1.2)
    )

    # 遍历每个箱体并为其填充对应的颜色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # 坐标轴和文本设置
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(y_limits)  # 设置Y轴范围以匹配原图

    # 旋转X轴标签，防止重叠
    ax.tick_params(axis='x', rotation=35, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # 在子图正下方添加标题
    ax.text(0.5, -0.35, sub_title, transform=ax.transAxes,
            ha='center', va='top', fontsize=13)


# ---------------------------------------------------------
# 4. 分别绘制两张图
# ---------------------------------------------------------
# 左图：HV metrics
create_custom_boxplot_from_stats(ax1, stats_hv, 'HV', '(a) HV metrics ', y_limits=(0, 1.5))

# 右图：IGD metrics
create_custom_boxplot_from_stats(ax2, stats_igd, 'IGD', '(b) IGD metrics', y_limits=(0, 1.2))

# 左图：Time metrics
create_custom_boxplot_from_stats(ax3, stats_time, 'Time/s', '(c) Time metrics ', y_limits=(500, 2500))

# 右图：Mem metrics
create_custom_boxplot_from_stats(ax4, stats_mem, 'Memory/MB', '(d) Memory metrics', y_limits=(1, 5))

# ---------------------------------------------------------
# 5. 调整布局并添加总说明文字
# ---------------------------------------------------------
# 预留底部空间给旋转的X轴标签和子图标题
plt.subplots_adjust(bottom=0.3,hspace=0.45, wspace=0.3)

# 模拟原图底部带有黄色高亮背景的总标题
# fig.text(
#     0.5, 0.05,
#     'Fig. 10. Box plots of HVs and IGDs in 15 independent runs.',
#     ha='center', va='bottom', fontsize=12, weight='bold',
#     bbox=dict(facecolor='#fde466', edgecolor='none', pad=3.0)
# )

# 显示图表
plt.show()