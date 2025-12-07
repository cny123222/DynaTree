# 📚 文档索引

本项目包含完整的中文文档，帮助您理解和使用 KnormPress。

## 🎯 快速导航

### 新手入门
1. **[QUICK_START_CN.md](QUICK_START_CN.md)** - 快速使用指南
   - ⏱️ 5分钟快速上手
   - 📋 常用命令速查
   - 🎯 参数说明
   - 📊 结果解读

### 详细说明
2. **[EXPLANATION_CN.md](EXPLANATION_CN.md)** - 详细中文解释
   - 🔍 Benchmark逻辑详解
   - ⚠️ Warning信息分析
   - 🎛️ keep_ratio参数详解
   - 📊 横向对比与可视化
   - 💡 实际应用示例

### 完整指南
3. **[USAGE_GUIDE_CN.md](USAGE_GUIDE_CN.md)** - 完整使用指南
   - 📖 Benchmark逻辑完整流程
   - 🔧 所有参数详解
   - 🎨 性能优化建议
   - ❓ 常见问题解答
   - 📈 详细实验结果

### 项目文档
4. **[README.md](README.md)** - 项目主文档
   - 📝 项目概述
   - 🚀 环境配置
   - 📊 实验结果表格
   - 💻 核心代码说明
   - 📚 参考文献

## 📁 文件结构

```
CS2602-LLM-Inference-Acceleration/
│
├── 📄 核心代码
│   ├── baseline_test.py          # 基线性能测试
│   ├── optimized_test.py         # KnormPress优化测试
│   ├── kv_compress.py            # KV Cache压缩算法
│   └── custom_generate.py        # 自定义生成函数
│
├── 📊 可视化
│   ├── visualize_results.py      # 生成对比图表
│   ├── knormpress_comprehensive.png   # 综合对比图
│   ├── knormpress_summary.png         # 总结图
│   ├── knormpress_tradeoff.png        # 权衡分析图
│   └── knormpress_table.png           # 数据表格
│
├── 📚 中文文档
│   ├── QUICK_START_CN.md         # 快速开始 ⭐
│   ├── EXPLANATION_CN.md         # 详细解释 ⭐
│   ├── USAGE_GUIDE_CN.md         # 完整指南
│   ├── README.md                 # 项目主文档
│   └── INDEX_CN.md               # 本文件
│
└── 📖 参考资料
    ├── lab-instruction.md        # 作业要求
    ├── KnormPress.pdf            # 论文PDF
    └── l2compress/               # 参考实现
```

## 🎓 学习路径

### 路径A：快速使用（推荐新手）
```
1. QUICK_START_CN.md    (5分钟)
   ↓
2. 运行 baseline_test.py
   ↓
3. 运行 optimized_test.py
   ↓
4. 运行 visualize_results.py
   ↓
5. 查看生成的图表
```

### 路径B：深入理解
```
1. README.md           (了解项目)
   ↓
2. EXPLANATION_CN.md   (理解原理)
   ↓
3. USAGE_GUIDE_CN.md   (详细用法)
   ↓
4. 阅读核心代码
   ↓
5. 自己运行实验
```

### 路径C：论文写作
```
1. 运行完整实验
   ↓
2. 生成所有图表
   ↓
3. 阅读 EXPLANATION_CN.md
   ↓
4. 参考 README.md 的结果分析
   ↓
5. 撰写实验报告
```

## 🔍 按需查找

### 我想知道...

#### "如何快速运行测试？"
→ 查看 [QUICK_START_CN.md](QUICK_START_CN.md) 的"快速开始"部分

#### "benchmark是如何工作的？"
→ 查看 [EXPLANATION_CN.md](EXPLANATION_CN.md) 的"问题1"部分

#### "Warning是什么原因？"
→ 查看 [EXPLANATION_CN.md](EXPLANATION_CN.md) 的"问题2"部分

#### "keep_ratio参数具体做什么？"
→ 查看 [EXPLANATION_CN.md](EXPLANATION_CN.md) 的"问题4"部分

#### "如何生成可视化图表？"
→ 查看 [QUICK_START_CN.md](QUICK_START_CN.md) 的"生成可视化图表"部分

#### "不同压缩率的效果对比？"
→ 查看 [README.md](README.md) 的"性能对比表格"部分

#### "如何选择最佳参数？"
→ 查看 [USAGE_GUIDE_CN.md](USAGE_GUIDE_CN.md) 的"最佳实践"部分

#### "遇到错误怎么办？"
→ 查看 [USAGE_GUIDE_CN.md](USAGE_GUIDE_CN.md) 的"常见问题"部分

## 📊 图表说明

### 生成的可视化图表

运行 `python visualize_results.py` 后生成：

1. **knormpress_comprehensive.png**
   - 6个子图综合对比
   - 包含所有关键指标
   - 适合：详细分析

2. **knormpress_summary.png**
   - 归一化柱状图
   - 直观对比所有配置
   - 适合：快速了解

3. **knormpress_tradeoff.png**
   - 速度vs质量散点图
   - 展示性能权衡
   - 适合：论文图表

4. **knormpress_table.png**
   - 详细数据表格
   - 精确数值
   - 适合：报告附录

## 🎯 核心概念速查

### KnormPress是什么？
一种基于L2范数的KV Cache压缩算法，通过保留重要token来加速推理。

### keep_ratio是什么？
KV Cache保留比例，控制压缩程度。
- 1.0 = 无压缩
- 0.9 = 保留90%，压缩10% ✅推荐
- 0.8 = 保留80%，压缩20%

### 为什么TTFT会降低？
因为压缩减少了需要处理的KV Cache大小，降低了注意力计算复杂度。

### 为什么PPL不变？
因为保留的是低L2范数的重要token，这些token对应高注意力分数，是预测的关键。

### 推荐配置是什么？
```python
keep_ratio = 0.9
prune_after = 512
skip_layers = [0]
```

## 📞 获取帮助

### 按优先级：

1. **查看文档**
   - QUICK_START_CN.md (快速问题)
   - EXPLANATION_CN.md (原理问题)
   - USAGE_GUIDE_CN.md (使用问题)

2. **运行示例**
   ```bash
   python baseline_test.py
   python optimized_test.py --keep_ratios 1.0,0.9
   python visualize_results.py
   ```

3. **查看代码注释**
   - 所有核心代码都有详细中文注释

4. **参考论文**
   - KnormPress.pdf

## ✅ 检查清单

### 开始前确认：
- [ ] conda环境已激活 (`conda activate nlp`)
- [ ] 依赖已安装 (`torch`, `transformers`, `datasets`)
- [ ] 模型可下载 (需要网络连接)

### 运行实验：
- [ ] baseline_test.py 运行成功
- [ ] optimized_test.py 运行成功
- [ ] 至少测试3个keep_ratio值
- [ ] 生成了可视化图表

### 理解结果：
- [ ] 理解TTFT的含义和改进
- [ ] 理解keep_ratio如何工作
- [ ] 知道为什么PPL保持不变
- [ ] 能够解释不同配置的权衡

## 🎉 完成后

恭喜！您现在已经：
- ✅ 理解了KnormPress的工作原理
- ✅ 成功运行了基线和优化测试
- ✅ 生成了完整的性能对比图表
- ✅ 掌握了如何选择最佳参数

### 下一步：
1. 撰写实验报告
2. 将结果整理到README
3. 准备论文或演示文稿
4. （可选）尝试其他压缩算法

---

**提示：** 这个索引文档会持续更新。建议收藏以便快速查找！

