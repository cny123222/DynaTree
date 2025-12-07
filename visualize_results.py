"""
KnormPress 性能对比可视化工具

这个脚本用于可视化不同 keep_ratio 对各项性能指标的影响。
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 实验数据（来自optimized_test.py的真实结果）
keep_ratios = [1.0, 0.9, 0.8, 0.7]
compression_pcts = [0, 10, 20, 30]  # 压缩百分比

# 平均性能数据
ttft_values = [0.0623, 0.0080, 0.0066, 0.0063]  # Time To First Token (秒)
tpot_values = [0.0125, 0.0129, 0.0130, 0.0131]  # Time Per Output Token (秒)
throughput_values = [84.16, 77.69, 76.79, 76.45]  # 吞吐量 (tokens/sec)
ppl_values = [75.03, 75.03, 75.03, 75.03]  # 困惑度

# 计算相对于baseline的改进百分比
ttft_improvement = [(ttft_values[0] - v) / ttft_values[0] * 100 for v in ttft_values]
throughput_change = [(v - throughput_values[0]) / throughput_values[0] * 100 for v in throughput_values]
ppl_change = [(v - ppl_values[0]) / ppl_values[0] * 100 for v in ppl_values]


def create_comprehensive_plots():
    """创建综合性能对比图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('KnormPress 性能对比分析 - 不同 keep_ratio 的影响', 
                 fontsize=16, fontweight='bold')
    
    # 图1: TTFT 对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(compression_pcts, ttft_values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('压缩率 (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('TTFT (秒)', fontsize=12, fontweight='bold')
    ax1.set_title('首Token生成时间 (TTFT)\n越低越好', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上显示数值
    for i, (bar, val) in enumerate(zip(bars1, ttft_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}s\n({keep_ratios[i]:.1f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 图2: TPOT 对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(compression_pcts, tpot_values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('压缩率 (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('TPOT (秒)', fontsize=12, fontweight='bold')
    ax2.set_title('每Token平均时间 (TPOT)\n越低越好', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars2, tpot_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}s\n({keep_ratios[i]:.1f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 图3: 吞吐量对比
    ax3 = axes[0, 2]
    bars3 = ax3.bar(compression_pcts, throughput_values, 
                    color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('压缩率 (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('吞吐量 (tokens/sec)', fontsize=12, fontweight='bold')
    ax3.set_title('生成吞吐量\n越高越好', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars3, throughput_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}\n({keep_ratios[i]:.1f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 图4: TTFT改进百分比
    ax4 = axes[1, 0]
    colors4 = ['green' if v > 0 else 'red' for v in ttft_improvement]
    bars4 = ax4.bar(compression_pcts, ttft_improvement, color=colors4, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('压缩率 (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('改进百分比 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('TTFT 改进率\n相对于 baseline (keep_ratio=1.0)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars4, ttft_improvement)):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va=va, fontsize=11, fontweight='bold')
    
    # 图5: 吞吐量变化百分比
    ax5 = axes[1, 1]
    colors5 = ['green' if v > 0 else 'red' for v in throughput_change]
    bars5 = ax5.bar(compression_pcts, throughput_change, color=colors5, alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('压缩率 (%)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('变化百分比 (%)', fontsize=12, fontweight='bold')
    ax5.set_title('吞吐量变化\n相对于 baseline', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars5, throughput_change)):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va=va, fontsize=11, fontweight='bold')
    
    # 图6: 困惑度对比
    ax6 = axes[1, 2]
    line = ax6.plot(compression_pcts, ppl_values, marker='o', linewidth=3, 
                    markersize=12, color='#e74c3c', markeredgecolor='black', 
                    markeredgewidth=2, label='PPL')
    ax6.fill_between(compression_pcts, [p*0.95 for p in ppl_values], 
                      [p*1.05 for p in ppl_values], alpha=0.2, color='red')
    ax6.set_xlabel('压缩率 (%)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('困惑度 (PPL)', fontsize=12, fontweight='bold')
    ax6.set_title('模型质量 (困惑度)\n越低越好 - 保持稳定是关键', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.legend(fontsize=11)
    
    for i, (x, y) in enumerate(zip(compression_pcts, ppl_values)):
        ax6.text(x, y + 2, f'{y:.2f}\n({keep_ratios[i]:.1f})', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_summary_comparison():
    """创建对比总结图"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(keep_ratios))
    width = 0.2
    
    # 归一化数据（相对于baseline）
    ttft_norm = [v / ttft_values[0] * 100 for v in ttft_values]
    tpot_norm = [v / tpot_values[0] * 100 for v in tpot_values]
    throughput_norm = [v / throughput_values[0] * 100 for v in throughput_values]
    ppl_norm = [v / ppl_values[0] * 100 for v in ppl_values]
    
    bars1 = ax.bar(x - 1.5*width, ttft_norm, width, label='TTFT', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, tpot_norm, width, label='TPOT', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, throughput_norm, width, label='吞吐量', 
                   color='#9b59b6', alpha=0.8, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, ppl_norm, width, label='PPL', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # 添加100%基线
    ax.axhline(y=100, color='black', linestyle='--', linewidth=2, alpha=0.5, 
               label='Baseline (100%)')
    
    ax.set_xlabel('Keep Ratio (压缩率)', fontsize=14, fontweight='bold')
    ax.set_ylabel('相对于 Baseline 的百分比 (%)', fontsize=14, fontweight='bold')
    ax.set_title('KnormPress 性能综合对比\n(所有指标相对于 keep_ratio=1.0 的百分比)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{kr:.1f}\n({int(cp)}%压缩)' 
                        for kr, cp in zip(keep_ratios, compression_pcts)])
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_tradeoff_plot():
    """创建性能权衡图（速度 vs 质量）"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # TTFT改进 vs PPL变化
    scatter = ax.scatter(ttft_improvement, ppl_change, s=500, 
                        c=compression_pcts, cmap='RdYlGn_r', 
                        alpha=0.7, edgecolors='black', linewidth=2)
    
    # 添加标签
    for i, (x, y) in enumerate(zip(ttft_improvement, ppl_change)):
        ax.annotate(f'keep_ratio={keep_ratios[i]:.1f}\n压缩{compression_pcts[i]}%\nTPOT={tpot_values[i]:.4f}s',
                   (x, y), fontsize=11, fontweight='bold',
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 理想区域（右下角）
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='PPL无变化')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='TTFT无改进')
    
    # 标注理想区域
    ax.text(50, -2, '理想区域\n高速度改进 + 低质量损失', 
           fontsize=14, fontweight='bold', color='green',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.5))
    
    ax.set_xlabel('TTFT 改进百分比 (%) - 越高越好', fontsize=14, fontweight='bold')
    ax.set_ylabel('PPL 变化百分比 (%) - 越接近0越好', fontsize=14, fontweight='bold')
    ax.set_title('性能权衡分析：速度改进 vs 模型质量\nKnormPress在保持质量的同时大幅提升速度', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('压缩率 (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_detailed_table():
    """创建详细数据表格"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 表格数据
    table_data = [
        ['Keep Ratio', '压缩率', 'TTFT (秒)', 'TTFT改进', 'TPOT (秒)', 
         '吞吐量 (tok/s)', '吞吐量变化', 'PPL', 'PPL变化'],
    ]
    
    for i in range(len(keep_ratios)):
        table_data.append([
            f'{keep_ratios[i]:.2f}',
            f'{compression_pcts[i]}%',
            f'{ttft_values[i]:.4f}',
            f'{ttft_improvement[i]:+.1f}%',
            f'{tpot_values[i]:.4f}',
            f'{throughput_values[i]:.2f}',
            f'{throughput_change[i]:+.1f}%',
            f'{ppl_values[i]:.2f}',
            f'{ppl_change[i]:+.1f}%',
        ])
    
    # 创建表格
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.1, 0.1, 0.12, 0.12, 0.12, 0.13, 0.13, 0.09, 0.11])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(9):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # 设置数据行颜色
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    for i in range(1, 5):
        for j in range(9):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i-1])
            cell.set_alpha(0.3)
            if j in [3, 6, 8]:  # 改进/变化列
                if '+' in table_data[i][j]:
                    cell.set_text_props(weight='bold', color='green')
                elif '-' in table_data[i][j]:
                    cell.set_text_props(weight='bold', color='red')
    
    plt.title('KnormPress 性能详细数据表', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("生成性能对比图表...")
    
    # 创建所有图表
    fig1 = create_comprehensive_plots()
    fig1.savefig('knormpress_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: knormpress_comprehensive.png (综合性能对比)")
    
    fig2 = create_summary_comparison()
    fig2.savefig('knormpress_summary.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: knormpress_summary.png (对比总结)")
    
    fig3 = create_tradeoff_plot()
    fig3.savefig('knormpress_tradeoff.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: knormpress_tradeoff.png (性能权衡分析)")
    
    fig4 = create_detailed_table()
    fig4.savefig('knormpress_table.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: knormpress_table.png (详细数据表)")
    
    print("\n所有图表已生成！")
    print("\n关键发现：")
    print(f"1. TTFT 改进: {ttft_improvement[1]:.1f}% (keep_ratio=0.9)")
    print(f"2. TTFT 改进: {ttft_improvement[2]:.1f}% (keep_ratio=0.8)")
    print(f"3. 吞吐量变化: {throughput_change[1]:.1f}% (keep_ratio=0.9)")
    print(f"4. PPL 变化: {ppl_change[1]:.1f}% (完全保持)")
    print(f"\n推荐配置: keep_ratio=0.9 (压缩10%)")
    print(f"  - TTFT降低 87.2%")
    print(f"  - 吞吐量仅下降 7.7%")
    print(f"  - PPL完全保持")
    
    # 显示图表（可选）
    # plt.show()

