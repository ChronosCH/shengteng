#!/bin/bash
# ======================================================================
# ASL手语识别项目训练脚本（无 conda run 版本，默认 DEBUG 输出）
# - 直接调用指定 Python 解释器
# - 实时无缓冲输出（PYTHONUNBUFFERED / python -u）
# - 默认 DEBUG 模式并开启 --verbose
# ======================================================================

set -euo pipefail

# --- 颜色 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- 默认配置 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Python 解释器优先级（可用 PYTHON_BIN 覆盖）
PYTHON_BIN_DEFAULT="/root/miniconda3/envs/mind/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "$PYTHON_BIN_DEFAULT" ]]; then
    PYTHON_BIN="$PYTHON_BIN_DEFAULT"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="python"
  fi
fi

# 训练参数
MODEL_VARIANT="medium"
EXPERIMENT_CONFIG="full_training"
RESUME_TRAINING=""
GPU_DEVICE="0"
BATCH_SIZE=""
LEARNING_RATE=""
EPOCHS=""
WORKERS=""
LOG_LEVEL="DEBUG"   # 默认 DEBUG
SAVE_CHECKPOINT_EVERY="5"
VALIDATE_EVERY="1"

# --- 输出工具 ---
print_header(){ echo -e "${CYAN}=========================================${NC}\n${CYAN}$1${NC}\n${CYAN}=========================================${NC}"; }
print_info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
print_warning(){ echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error(){ echo -e "${RED}[ERROR]${NC} $1"; }
print_success(){ echo -e "${GREEN}[SUCCESS]${NC} $1"; }

show_help(){
  cat <<EOF
ASL手语识别训练脚本（无 conda run）

用法: $0 [选项]

基本选项:
  -h, --help              显示帮助
  -v, --variant VARIANT   模型变体 [small|medium|large] (默认: medium)
  -e, --experiment EXP    实验配置 [quick_test|full_training|large_model|ultra_fast] (默认: full_training)
  -r, --resume PATH       从检查点恢复训练
  --python PATH           指定 Python 解释器（默认自动探测）

训练参数:
  --gpu DEVICE            GPU 设备ID (默认: 0，-1 表示 CPU)
  --batch-size SIZE       批次大小
  --learning-rate LR      学习率
  --epochs N              训练轮数
  --workers N             数据加载进程数
  --log-level LEVEL       日志级别 [DEBUG|INFO|WARNING|ERROR] (默认: DEBUG)

实用工具:
  --setup                 仅设置环境
  --analyze               运行数据分析
  --list-checkpoints      列出可用检查点
EOF
}

check_environment(){
  print_info "检查运行环境..."
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    print_error "未找到 Python: $PYTHON_BIN"
    exit 1
  fi
  if [[ ! -f "$SCRIPT_DIR/main.py" ]]; then
    print_error "未找到 main.py"
    exit 1
  fi
  print_success "环境检查通过 (Python: $PYTHON_BIN $($PYTHON_BIN -V 2>&1))"
}

setup_cuda_env(){
  if [[ "$GPU_DEVICE" != "-1" ]]; then
    print_info "配置GPU环境 (设备: $GPU_DEVICE)"
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
    if command -v nvidia-smi >/dev/null 2>&1; then
      print_info "GPU信息:"
      nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | head -n 1
    else
      print_warning "未检测到 nvidia-smi"
    fi
  else
    print_info "使用CPU模式"
    export CUDA_VISIBLE_DEVICES=""
  fi
}

setup_logging(){
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/training_${MODEL_VARIANT}_${EXPERIMENT_CONFIG}_${TIMESTAMP}.log"
  export ASL_LOG_FILE="$LOG_FILE"
  export PYTHONUNBUFFERED=1
  print_info "日志将保存到: $LOG_FILE"
}

build_train_command(){
  local cmd="$PYTHON_BIN -u -X faulthandler $SCRIPT_DIR/main.py train"
  cmd+=" --variant $MODEL_VARIANT"
  cmd+=" --experiment $EXPERIMENT_CONFIG"
  if [[ "$GPU_DEVICE" = "-1" ]]; then
    cmd+=" --cpu"
  else
    cmd+=" --device-id $GPU_DEVICE"
  fi
  [[ -n "$RESUME_TRAINING" ]] && cmd+=" --resume $RESUME_TRAINING"
  [[ -n "$BATCH_SIZE" ]] && cmd+=" --batch-size $BATCH_SIZE"
  [[ -n "$LEARNING_RATE" ]] && cmd+=" --learning-rate $LEARNING_RATE"
  [[ -n "$EPOCHS" ]] && cmd+=" --epochs $EPOCHS"
  [[ -n "$WORKERS" ]] && cmd+=" --workers $WORKERS"
  [[ -n "$VALIDATE_EVERY" ]] && cmd+=" --eval-interval $VALIDATE_EVERY"
  # DEBUG 模式自动开启详细日志
  if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
    cmd+=" --verbose"
  fi
  echo "$cmd"
}

show_config(){
  print_header "训练配置"
  echo -e "${PURPLE}环境配置:${NC}"
  echo "  Python: $PYTHON_BIN ($($PYTHON_BIN -V 2>&1))"
  echo "  项目目录: $SCRIPT_DIR"
  echo "  GPU设备: $GPU_DEVICE"
  echo ""
  echo -e "${PURPLE}训练参数:${NC}"
  echo "  模型变体: $MODEL_VARIANT"
  echo "  实验配置: $EXPERIMENT_CONFIG"
  echo "  恢复训练: ${RESUME_TRAINING:-'否'}"
  echo "  日志级别: $LOG_LEVEL (verbose=on)"
  echo ""
  echo -e "${PURPLE}超参数 (如设置):${NC}"
  [[ -n "$BATCH_SIZE" ]] && echo "  批次大小: $BATCH_SIZE"
  [[ -n "$LEARNING_RATE" ]] && echo "  学习率: $LEARNING_RATE"
  [[ -n "$EPOCHS" ]] && echo "  训练轮数: $EPOCHS"
  [[ -n "$WORKERS" ]] && echo "  工作进程: $WORKERS"
  echo ""
}

list_checkpoints(){
  local d="$SCRIPT_DIR/checkpoints"
  print_header "可用检查点"
  if [[ ! -d "$d" ]]; then
    print_warning "检查点目录不存在: $d"; return
  fi
  local arr=($(find "$d" -name "*.ckpt" -type f 2>/dev/null | sort -t_ -k2 -n))
  if [[ ${#arr[@]} -eq 0 ]]; then
    print_info "暂无可用检查点"; return
  fi
  echo "找到 ${#arr[@]} 个检查点:"; for f in "${arr[@]}"; do
    echo "  $(basename "$f")"
  done
}

run_analysis(){
  print_header "运行数据分析"
  check_environment
  "$PYTHON_BIN" -u "$SCRIPT_DIR/main.py" analyze
  print_success "数据分析完成!"
}

parse_arguments(){
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) show_help; exit 0;;
      -v|--variant) MODEL_VARIANT="$2"; shift 2;;
      -e|--experiment) EXPERIMENT_CONFIG="$2"; shift 2;;
      -r|--resume) RESUME_TRAINING="$2"; shift 2;;
      --python) PYTHON_BIN="$2"; shift 2;;
      --gpu) GPU_DEVICE="$2"; shift 2;;
      --batch-size) BATCH_SIZE="$2"; shift 2;;
      --learning-rate) LEARNING_RATE="$2"; shift 2;;
      --epochs) EPOCHS="$2"; shift 2;;
      --workers) WORKERS="$2"; shift 2;;
      --log-level) LOG_LEVEL="$2"; shift 2;;
      --setup) "$PYTHON_BIN" -u "$SCRIPT_DIR/main.py" setup; exit 0;;
      --analyze) run_analysis; exit 0;;
      --list-checkpoints) list_checkpoints; exit 0;;
      --quick) EXPERIMENT_CONFIG="quick_test"; shift;;
      --full) EXPERIMENT_CONFIG="full_training"; shift;;
      --large) EXPERIMENT_CONFIG="large_model"; MODEL_VARIANT="large"; shift;;
      *) print_error "未知参数: $1"; echo "使用 --help 查看帮助信息"; exit 1;;
    esac
  done
}

validate_parameters(){
  [[ "$MODEL_VARIANT" =~ ^(small|medium|large)$ ]] || { print_error "无效模型变体: $MODEL_VARIANT"; exit 1; }
  [[ "$EXPERIMENT_CONFIG" =~ ^(quick_test|full_training|large_model|ultra_fast)$ ]] || { print_error "无效实验配置: $EXPERIMENT_CONFIG"; exit 1; }
  if [[ -n "$RESUME_TRAINING" && ! -f "$RESUME_TRAINING" ]]; then
    print_error "检查点文件不存在: $RESUME_TRAINING"; exit 1
  fi
  [[ "$GPU_DEVICE" =~ ^-?[0-9]+$ ]] || { print_error "无效GPU设备ID: $GPU_DEVICE"; exit 1; }
}

run_training(){
  print_header "开始ASL手语识别训练"
  check_environment
  show_config
  setup_cuda_env
  setup_logging
  local train_cmd
  train_cmd=$(build_train_command)
  echo -e "${YELLOW}即将开始训练，按Enter继续，Ctrl+C取消...${NC}"; read -r
  print_info "执行命令: $train_cmd"
  print_info "训练开始时间: $(date)"
  local start_time=$(date +%s)
  # 同时输出到终端与日志
  if eval "$train_cmd" 2>&1 | tee "$ASL_LOG_FILE"; then
    local end_time=$(date +%s)
    local d=$((end_time-start_time))
    print_success "训练完成! 用时 $((d/3600))时$(((d%3600)/60))分$((d%60))秒"
    print_info "日志文件: $ASL_LOG_FILE"
  else
    print_error "训练失败！详见日志: $ASL_LOG_FILE"; exit 1
  fi
}

main(){
  parse_arguments "$@"
  validate_parameters
  run_training
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
