# 快速开始 - 论文实验

## 📋 目录
1. [环境配置](#环境配置)
2. [检查模型](#检查模型)
3. [运行实验](#运行实验)
4. [实验说明](#实验说明)
5. [故障排除](#故障排除)

---

## 🔧 环境配置

### 方法1: 自动配置（推荐）

```bash
# 进入项目目录
cd /root/LLM-Efficient-Reasoning

# 给脚本执行权限
chmod +x setup_environment.sh

# 运行配置脚本
bash setup_environment.sh
```

这个脚本会：
- ✅ 检测Python版本
- ✅ 自动安装PyTorch（根据CUDA版本）
- ✅ 安装所有依赖
- ✅ 验证安装
- ✅ 创建必要的目录

---

### 方法2: 手动配置

```bash
# 1. 创建conda环境（可选但推荐）
conda create -n nlp python=3.11 -y
conda activate nlp

# 2. 安装PyTorch（根据你的CUDA版本选择）
# CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

---

## 🔍 检查模型

在运行实验前，先检查模型是否已下载：

```bash
python check_models.py
```

**输出示例**：
```
============================================================
Checking: pythia-2.8b
============================================================
✓ Found at: /mnt/disk1/models/pythia-2.8b
  ✓ Valid model (has config.json)

============================================================
Checking: pythia-70m
============================================================
✓ Found at: /mnt/disk1/models/pythia-70m
  ✓ Valid model (has config.json)
```

### 如果模型不存在：

**选项1**: 从HuggingFace自动下载（首次运行时会自动下载）
- 需要网络连接
- 会下载到 `~/.cache/huggingface/`

**选项2**: 手动下载（如果有现成的模型）
```bash
# 下载到本地
python download_pythia_2.8b.py
python download_pythia_small.py
```

---

## 🚀 运行实验

### 一键运行所有实验

```bash
# 给脚本执行权限
chmod +x run_experiments.sh

# 运行实验脚本（交互式，可以选择运行哪些实验）
bash run_experiments.sh
```

脚本会提示你选择运行哪些实验：
```
Run Experiment 1? (y/n) y
Run Experiment 2? (y/n) y
Run Experiment 3? (y/n) y
Run Experiment 4? (y/n) n
```

---

### 手动运行单个实验

#### Experiment 1: 主要性能对比（必须）⭐⭐⭐
```bash
python spec_decode/benchmark_tree_vs_linear.py \
    --target-model EleutherAI/pythia-2.8b \
    --draft-model EleutherAI/pythia-70m \
    --max-new-tokens 100 \
    --num-samples 10 \
    --save
```
- **时间**: ~30分钟
- **输出**: `results/final_experiments/exp1_main_comparison.json`
- **用途**: 论文Table 2（主要结果表格）

#### Experiment 2: 参数扫描可视化（必须）⭐⭐⭐
```bash
# 使用已有的参数搜索数据生成图表
python papers/analyze_tree_search_results.py \
    results/tree_param_search_20251231_140952.json
```
- **时间**: ~5分钟
- **输出**: 终端输出分析结果
- **用途**: 论文Figure 2（参数分析）

#### Experiment 3: 消融实验（必须）⭐⭐⭐
```bash
python spec_decode/ablation_pruning.py \
    --target-model EleutherAI/pythia-2.8b \
    --draft-model EleutherAI/pythia-70m \
    --depth 3 --branch 3 \
    --max-new-tokens 100 \
    --output results/ablation_pruning.json
```
- **时间**: ~20分钟
- **输出**: `results/ablation_pruning.json`
- **用途**: 论文Table 3（消融研究）

#### Experiment 4: 长序列测试（可选）⭐
```bash
python spec_decode/benchmark_tree_vs_linear.py \
    --target-model EleutherAI/pythia-2.8b \
    --draft-model EleutherAI/pythia-70m \
    --max-new-tokens 100 200 500 \
    --num-samples 5 \
    --save
```
- **时间**: ~40分钟
- **输出**: 不同长度的性能数据
- **用途**: 论文Table 4或Discussion部分

---

## 📊 实验说明

### 实验优先级

| 实验 | 优先级 | 预计时间 | 论文用途 | 必要性 |
|------|--------|----------|----------|--------|
| Exp 1 | P0 | 30分钟 | Table 2 主要结果 | ✅ 必须 |
| Exp 2 | P0 | 5分钟 | Figure 2 参数分析 | ✅ 必须 |
| Exp 3 | P0 | 20分钟 | Table 3 消融实验 | ✅ 必须 |
| Exp 4 | P1 | 40分钟 | Table 4 或讨论 | ⭐ 重要 |

### 时间安排建议

**如果只有4小时**：
1. 运行 Exp 1 (30分钟)
2. 运行 Exp 3 (20分钟)
3. 分析 Exp 2 已有数据 (5分钟)
4. 开始写论文

**如果有6-8小时**：
1. 运行所有实验 (1.5小时)
2. 写论文 (4-6小时)
3. 润色和检查 (1小时)

---

## 🔧 故障排除

### 问题1: CUDA out of memory

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
```bash
# 1. 减少样本数
--num-samples 5  # 改为 3

# 2. 减少生成长度
--max-new-tokens 100  # 改为 50

# 3. 使用更小的模型（用于测试）
--target-model EleutherAI/pythia-1.4b
```

---

### 问题2: 模型下载太慢

**症状**: 下载HuggingFace模型很慢或失败

**解决**:
```bash
# 设置镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或者使用预下载的模型
# 把模型路径改为本地路径
--target-model /path/to/local/pythia-2.8b
```

---

### 问题3: ImportError

**症状**: `ImportError: No module named 'xxx'`

**解决**:
```bash
# 重新安装依赖
pip install -r requirements.txt

# 或者单独安装缺失的包
pip install transformers accelerate torch
```

---

### 问题4: 脚本权限错误

**症状**: `Permission denied`

**解决**:
```bash
# 给脚本执行权限
chmod +x setup_environment.sh
chmod +x run_experiments.sh
chmod +x check_models.py
```

---

## 📁 输出文件说明

运行实验后，结果会保存在：

```
results/final_experiments/YYYYMMDD_HHMMSS/
├── exp1_main_comparison.json      # 主要对比结果
├── exp2_param_sweep.pdf           # 参数扫描图表
├── exp3_ablation.json             # 消融实验结果
└── exp4_sequence_length.json      # 长序列测试结果
```

### 如何查看结果

```bash
# 查看JSON结果
cat results/final_experiments/*/exp1_main_comparison.json | python -m json.tool

# 或者用分析脚本
python papers/analyze_tree_search_results.py results/tree_param_search_*.json
```

---

## 🎯 快速检查清单

配置环境前：
- [ ] 确认Python 3.9+
- [ ] 确认有CUDA（如果用GPU）
- [ ] 确认有足够磁盘空间（模型~20GB）

运行实验前：
- [ ] 环境配置完成（`bash setup_environment.sh`）
- [ ] 模型检查通过（`python check_models.py`）
- [ ] 创建输出目录（脚本会自动创建）

运行实验后：
- [ ] 检查输出文件存在
- [ ] 检查JSON文件格式正确
- [ ] 记录实验配置和结果

---

## 📞 需要帮助？

如果遇到问题：

1. **检查日志**: 查看终端输出的错误信息
2. **检查GPU**: `nvidia-smi` 查看GPU状态
3. **检查空间**: `df -h` 查看磁盘空间
4. **重新安装**: 删除环境重新安装

```bash
# 完全重置环境
conda deactivate
conda env remove -n nlp
bash setup_environment.sh
```

---

**文档版本**: v1.0  
**最后更新**: 2026-01-02

祝实验顺利！🚀

