# KnormPress 快速使用指南

## 📋 快速开始

### 1. 激活环境
```bash
conda activate nlp
```

### 2. 运行基线测试
```bash
python baseline_test.py
```

### 3. 运行优化测试
```bash
python optimized_test.py --keep_ratios 1.0,0.9,0.8,0.7
```

### 4. 生成可视化图表
```bash
python visualize_results.py
```

## 🎯 核心参数说明

### keep_ratio（保留比率）

这是**最重要**的参数，控制KV Cache的压缩程度：

```bash
keep_ratio=1.0   # 保留100% (无压缩，baseline)
keep_ratio=0.9   # 保留90%  (压缩10%) ✅ 推荐
keep_ratio=0.8   # 保留80%  (压缩20%)
keep_ratio=0.7   # 保留70%  (压缩30%)
```

**工作原理：**
1. 计算每个token的键嵌入L2范数
2. 按范数升序排序（低范数=高重要性）  
3. 保留前 `keep_ratio` 比例的token
4. 丢弃剩余的不重要token

**推荐值：** `0.9` (压缩10%)
- TTFT降低87%
- 吞吐量仅下降8%
- PPL保持不变

## 📊 性能对比

### 实验结果（在Apple Silicon MPS上）

| Keep Ratio | 压缩率 | TTFT改进 | 吞吐量变化 | PPL变化 |
|------------|--------|----------|------------|---------|
| 1.0        | 0%     | -        | -          | -       |
| 0.9        | 10%    | **↓87%** | ↓8%        | ±0%     |
| 0.8        | 20%    | **↓89%** | ↓9%        | ±0%     |
| 0.7        | 30%    | **↓90%** | ↓9%        | ±0%     |

### 可视化图表

运行 `python visualize_results.py` 后，会生成4张图：

1. **knormpress_comprehensive.png** - 6个子图的综合对比
2. **knormpress_summary.png** - 所有指标的柱状图对比
3. **knormpress_tradeoff.png** - 速度vs质量的权衡分析
4. **knormpress_table.png** - 详细数据表格

## 🔧 常用命令

### 快速测试（2分钟）
```bash
python optimized_test.py --keep_ratios 1.0,0.9 --num_wikitext_samples 2
```

### 标准测试（5分钟）
```bash
python optimized_test.py --keep_ratios 1.0,0.9,0.8,0.7 --num_wikitext_samples 3
```

### 完整测试（10分钟）
```bash
python optimized_test.py \
    --keep_ratios 1.0,0.95,0.9,0.85,0.8,0.75,0.7 \
    --num_wikitext_samples 5
```

### 极限压缩测试
```bash
python optimized_test.py \
    --keep_ratios 0.9,0.8,0.7,0.6,0.5,0.4 \
    --num_wikitext_samples 3
```

## 📈 结果解读

### 关键指标

1. **TTFT (Time To First Token)** 
   - 含义：首token生成时间
   - 目标：越低越好
   - 影响：用户感知的响应速度

2. **TPOT (Time Per Output Token)**
   - 含义：平均每token生成时间
   - 目标：越低越好
   - 影响：长文本生成效率

3. **Throughput (吞吐量)**
   - 含义：每秒生成的token数
   - 目标：越高越好
   - 影响：整体处理能力

4. **PPL (Perplexity，困惑度)**
   - 含义：模型预测质量
   - 目标：越低越好，保持不变最理想
   - 影响：生成文本的质量

### 理想结果

✅ TTFT大幅降低 (↓80%+)
✅ 吞吐量略有下降 (↓10%以内)
✅ PPL保持不变 (±5%以内)

## 🚀 最佳实践

### 1. 选择合适的keep_ratio

```python
# 生产环境（质量优先）
keep_ratio = 0.9  # 压缩10%

# 均衡配置
keep_ratio = 0.8  # 压缩20%

# 激进压缩（速度优先）
keep_ratio = 0.7  # 压缩30%
```

### 2. 调整prune_after

```python
# 短文本（<512 tokens）
--prune_after 256

# 中等文本（512-1024 tokens）
--prune_after 512  # 默认值

# 长文本（>1024 tokens）
--prune_after 1024
```

### 3. 跳过关键层

```python
# 保护第0层（底层表示）
--skip_layers 0  # 默认值

# 保护多层
--skip_layers 0,1,2
```

## ⚠️ 注意事项

### Warning信息

如果看到这个警告：
```
RuntimeWarning: Mean of empty slice.
```

**原因：** MPS设备不支持显存统计
**影响：** 无，其他指标正常
**已修复：** 最新版本已处理此警告

### 设备兼容性

| 设备 | TTFT | TPOT | Memory | 建议样本数 |
|------|------|------|--------|-----------|
| CUDA | ✅ | ✅ | ✅ | 5-10 |
| MPS  | ✅ | ✅ | ❌ | 3-5 |
| CPU  | ✅ | ✅ | ❌ | 2-3 |

## 📚 更多信息

详细使用指南：`USAGE_GUIDE_CN.md`
项目README：`README.md`
论文：`KnormPress.pdf`

## 🎓 总结

KnormPress通过智能压缩KV Cache：
- ✅ 显著降低首token延迟（87%+）
- ✅ 保持模型生成质量（PPL不变）
- ✅ 无需训练，即插即用
- ✅ 特别适合长文本生成

**推荐配置：** `keep_ratio=0.9`，在速度和质量间达到最佳平衡。

