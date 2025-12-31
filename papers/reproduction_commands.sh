#!/bin/bash
# ============================================================================
# Speculative Decoding Paper - Reproduction Script
# ============================================================================
# 
# 使用方法:
#   chmod +x reproduction_commands.sh
#   ./reproduction_commands.sh [experiment_name]
#
# 可选实验:
#   all        - 运行所有实验 (默认)
#   main       - 运行主实验 (全面 benchmark)
#   longseq    - 运行长序列测试
#   help       - 显示帮助
#
# ============================================================================

set -e

# 配置路径
PROJECT_ROOT="/mnt/disk1/ljm/LLM-Efficient-Reasoning"
TARGET_MODEL="/mnt/disk1/models/pythia-2.8b"
DRAFT_MODEL="/mnt/disk1/models/pythia-70m"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# 检查环境
check_environment() {
    echo_info "检查环境..."
    [ -d "$PROJECT_ROOT" ] || { echo "项目目录不存在"; exit 1; }
    [ -d "$TARGET_MODEL" ] || { echo "Target 模型不存在"; exit 1; }
    [ -d "$DRAFT_MODEL" ] || { echo "Draft 模型不存在"; exit 1; }
    echo_info "环境检查通过!"
}

# 主实验: 全面 benchmark
run_main_experiment() {
    echo_info "=========================================="
    echo_info "运行主实验: 全面性能对比"
    echo_info "=========================================="
    
    cd "$PROJECT_ROOT"
    
    python spec_decode/benchmark_comprehensive.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --max-new-tokens 500 1000 2000 \
        --max-cache-lens 256 512 1024 \
        --k-value 5 \
        --num-samples 3 \
        --output-json benchmark_comprehensive_results.json \
        --output-plot papers/figures/paper_fig7_comprehensive.png
    
    echo_info "主实验完成!"
}

# 长序列测试
run_longseq_experiment() {
    echo_info "=========================================="
    echo_info "运行长序列测试"
    echo_info "=========================================="
    
    cd "$PROJECT_ROOT"
    
    python spec_decode/benchmark_long_sequence.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --max-new-tokens 500 1000 2000 \
        --max-cache-lens 256 512 1024 \
        --k-value 5 \
        --output-json benchmark_long_seq_results.json \
        --output-plot papers/figures/paper_fig6_long_seq.png
    
    echo_info "长序列测试完成!"
}

# 运行所有实验
run_all() {
    check_environment
    run_main_experiment
    run_longseq_experiment
    
    echo_info "=========================================="
    echo_info "所有实验完成!"
    echo_info "=========================================="
    echo_info "结果文件:"
    echo_info "  - benchmark_comprehensive_results.json"
    echo_info "  - benchmark_long_seq_results.json"
    echo_info "图表文件:"
    echo_info "  - papers/figures/paper_fig6_long_seq.png"
    echo_info "  - papers/figures/paper_fig7_comprehensive.png"
}

# 显示帮助
show_help() {
    echo "用法: $0 [experiment_name]"
    echo ""
    echo "可选实验:"
    echo "  all      - 运行所有实验 (默认)"
    echo "  main     - 运行主实验 (全面 benchmark)"
    echo "  longseq  - 运行长序列测试"
    echo "  help     - 显示帮助"
}

# 主入口
main() {
    local experiment="${1:-all}"
    
    case "$experiment" in
        all)     run_all ;;
        main)    check_environment; run_main_experiment ;;
        longseq) check_environment; run_longseq_experiment ;;
        help|--help|-h) show_help ;;
        *)       echo "未知实验: $experiment"; show_help; exit 1 ;;
    esac
}

main "$@"
