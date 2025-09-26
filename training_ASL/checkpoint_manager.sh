#!/bin/bash
# ==============================================================================
# ASL项目检查点管理脚本
# 用于查找、管理和恢复训练检查点
# ==============================================================================

set -euo pipefail

# --- 颜色输出配置 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- 默认配置参数 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
ENV_NAME="asl_env"

# --- 辅助函数 ---
print_header() {
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}=========================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# --- 帮助函数 ---
show_help() {
    cat << EOF
ASL项目检查点管理脚本

用法: $0 [命令] [选项]

命令:
  list, ls               列出所有检查点
  info CHECKPOINT        显示检查点详细信息
  latest                 显示最新的检查点
  best                   显示最佳检查点（基于验证准确率）
  clean [--keep=N]       清理旧检查点，保留最新的N个
  resume CHECKPOINT      从指定检查点恢复训练
  backup [TARGET_DIR]    备份所有检查点到指定目录
  restore SOURCE_DIR     从备份目录恢复检查点

选项:
  -h, --help             显示此帮助信息
  --checkpoint-dir DIR   指定检查点目录 (默认: ./checkpoints)
  --env ENV_NAME         指定conda环境名称 (默认: asl_env)
  --format FORMAT        输出格式 [table|json|simple] (默认: table)
  --sort FIELD           排序字段 [time|epoch|size|name] (默认: time)
  --reverse              反向排序

清理选项:
  --keep N               保留最新的N个检查点 (默认: 5)
  --dry-run              仅显示将要删除的文件，不实际删除

示例:
  $0 list                          # 列出所有检查点
  $0 latest                        # 显示最新检查点
  $0 info asl_model_epoch_10.ckpt  # 查看检查点详情
  $0 resume latest                 # 从最新检查点恢复训练
  $0 clean --keep=3                # 只保留最新的3个检查点
  $0 backup /backup/checkpoints    # 备份检查点到指定目录
  $0 --format=json list            # 以JSON格式列出检查点

EOF
}

# --- 检查点发现和分析函数 ---
find_checkpoints() {
    local sort_field="${1:-time}"
    local reverse="${2:-false}"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        print_warning "检查点目录不存在: $CHECKPOINT_DIR"
        return 1
    fi
    
    local checkpoints=()
    while IFS= read -r -d '' file; do
        checkpoints+=("$file")
    done < <(find "$CHECKPOINT_DIR" -name "*.ckpt" -type f -print0 2>/dev/null)
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        print_info "未找到检查点文件"
        return 1
    fi
    
    # 排序
    case "$sort_field" in
        "time")
            if [ "$reverse" = "true" ]; then
                printf '%s\n' "${checkpoints[@]}" | sort -t/ -k2
            else
                printf '%s\n' "${checkpoints[@]}" | sort -t/ -k2 -r
            fi
            ;;
        "epoch")
            if [ "$reverse" = "true" ]; then
                printf '%s\n' "${checkpoints[@]}" | sort -t_ -k3 -n
            else
                printf '%s\n' "${checkpoints[@]}" | sort -t_ -k3 -nr
            fi
            ;;
        "size")
            if [ "$reverse" = "true" ]; then
                printf '%s\n' "${checkpoints[@]}" | xargs ls -lS | awk '{print $9}' | grep "\.ckpt$"
            else
                printf '%s\n' "${checkpoints[@]}" | xargs ls -lrS | awk '{print $9}' | grep "\.ckpt$"
            fi
            ;;
        "name")
            if [ "$reverse" = "true" ]; then
                printf '%s\n' "${checkpoints[@]}" | sort
            else
                printf '%s\n' "${checkpoints[@]}" | sort -r
            fi
            ;;
        *)
            printf '%s\n' "${checkpoints[@]}"
            ;;
    esac
}

# --- 获取检查点信息 ---
get_checkpoint_info() {
    local checkpoint_path="$1"
    local format="${2:-table}"
    
    if [ ! -f "$checkpoint_path" ]; then
        print_error "检查点文件不存在: $checkpoint_path"
        return 1
    fi
    
    local filename=$(basename "$checkpoint_path")
    local size=$(du -h "$checkpoint_path" 2>/dev/null | cut -f1 || echo "未知")
    local date=$(stat -c %y "$checkpoint_path" 2>/dev/null | cut -d' ' -f1 || echo "未知")
    local time=$(stat -c %y "$checkpoint_path" 2>/dev/null | cut -d' ' -f2 | cut -d. -f1 || echo "未知")
    
    # 从文件名提取epoch信息
    local epoch="未知"
    if [[ "$filename" =~ epoch_([0-9]+) ]]; then
        epoch="${BASH_REMATCH[1]}"
    fi
    
    # 查找对应的训练状态文件
    local state_file="${checkpoint_path%.*}_state.json"
    local accuracy="未知"
    local loss="未知"
    
    if [ -f "$state_file" ]; then
        if command -v python >/dev/null 2>&1; then
            accuracy=$(python -c "
import json
try:
    with open('$state_file', 'r') as f:
        data = json.load(f)
    print(data.get('best_accuracy', '未知'))
except:
    print('未知')
" 2>/dev/null || echo "未知")
        fi
    fi
    
    case "$format" in
        "json")
            cat << EOF
{
  "filename": "$filename",
  "path": "$checkpoint_path",
  "epoch": "$epoch",
  "size": "$size",
  "date": "$date",
  "time": "$time",
  "accuracy": "$accuracy",
  "loss": "$loss"
}
EOF
            ;;
        "simple")
            echo "$filename (Epoch: $epoch, Size: $size, Date: $date $time)"
            ;;
        "table"|*)
            printf "%-30s | %-8s | %-8s | %-12s | %-10s | %-8s\n" \
                "$filename" "$epoch" "$size" "$date $time" "$accuracy" "$loss"
            ;;
    esac
}

# --- 列出检查点 ---
list_checkpoints() {
    local format="${1:-table}"
    local sort_field="${2:-time}"
    local reverse="${3:-false}"
    
    print_header "检查点列表"
    
    local checkpoints
    if ! checkpoints=$(find_checkpoints "$sort_field" "$reverse"); then
        return 1
    fi
    
    if [ "$format" = "table" ]; then
        echo -e "${PURPLE}检查点目录: $CHECKPOINT_DIR${NC}"
        echo ""
        printf "%-30s | %-8s | %-8s | %-12s | %-10s | %-8s\n" \
            "文件名" "Epoch" "大小" "创建时间" "准确率" "损失"
        printf "%.30s-+-%.8s-+-%.8s-+-%.12s-+-%.10s-+-%.8s\n" \
            "------------------------------" "--------" "--------" "------------" "----------" "--------"
    elif [ "$format" = "json" ]; then
        echo "["
    fi
    
    local count=0
    while IFS= read -r checkpoint; do
        if [ "$format" = "json" ] && [ $count -gt 0 ]; then
            echo ","
        fi
        get_checkpoint_info "$checkpoint" "$format"
        ((count++))
    done <<< "$checkpoints"
    
    if [ "$format" = "json" ]; then
        echo "]"
    elif [ "$format" = "table" ]; then
        echo ""
        echo -e "${GREEN}总计: $count 个检查点${NC}"
    fi
}

# --- 获取最新检查点 ---
get_latest_checkpoint() {
    local checkpoints
    if ! checkpoints=$(find_checkpoints "time" "false"); then
        return 1
    fi
    
    local latest=$(echo "$checkpoints" | head -n1)
    echo "$latest"
}

# --- 获取最佳检查点 ---
get_best_checkpoint() {
    print_info "搜索最佳检查点（基于验证准确率）..."
    
    local best_checkpoint=""
    local best_accuracy=0
    
    local checkpoints
    if ! checkpoints=$(find_checkpoints "time" "false"); then
        return 1
    fi
    
    while IFS= read -r checkpoint; do
        local state_file="${checkpoint%.*}_state.json"
        if [ -f "$state_file" ] && command -v python >/dev/null 2>&1; then
            local accuracy=$(python -c "
import json
try:
    with open('$state_file', 'r') as f:
        data = json.load(f)
    acc = float(data.get('best_accuracy', 0))
    print(acc)
except:
    print(0)
" 2>/dev/null || echo "0")
            
            if (( $(echo "$accuracy > $best_accuracy" | bc -l 2>/dev/null || echo "0") )); then
                best_accuracy=$accuracy
                best_checkpoint=$checkpoint
            fi
        fi
    done <<< "$checkpoints"
    
    if [ -n "$best_checkpoint" ]; then
        echo "$best_checkpoint"
    else
        print_warning "无法找到最佳检查点，返回最新检查点"
        get_latest_checkpoint
    fi
}

# --- 清理检查点 ---
clean_checkpoints() {
    local keep="${1:-5}"
    local dry_run="${2:-false}"
    
    print_header "清理检查点"
    
    local checkpoints
    if ! checkpoints=$(find_checkpoints "time" "false"); then
        return 1
    fi
    
    local checkpoint_array=()
    while IFS= read -r checkpoint; do
        checkpoint_array+=("$checkpoint")
    done <<< "$checkpoints"
    
    local total=${#checkpoint_array[@]}
    
    if [ $total -le $keep ]; then
        print_info "当前检查点数量 ($total) 不超过保留数量 ($keep)，无需清理"
        return 0
    fi
    
    local to_delete=$((total - keep))
    print_info "将删除 $to_delete 个旧检查点，保留最新的 $keep 个"
    
    echo ""
    echo "将要删除的文件:"
    for i in $(seq $keep $((total - 1))); do
        local checkpoint="${checkpoint_array[$i]}"
        local filename=$(basename "$checkpoint")
        echo "  - $filename"
    done
    
    if [ "$dry_run" = "true" ]; then
        print_warning "这是预演模式，没有实际删除任何文件"
        return 0
    fi
    
    echo ""
    echo -e "${YELLOW}确认删除这些文件吗？ [y/N]${NC}"
    read -r confirmation
    
    if [[ "$confirmation" =~ ^[Yy]$ ]]; then
        local deleted=0
        for i in $(seq $keep $((total - 1))); do
            local checkpoint="${checkpoint_array[$i]}"
            if rm "$checkpoint" 2>/dev/null; then
                print_success "已删除: $(basename "$checkpoint")"
                ((deleted++))
                
                # 同时删除对应的状态文件
                local state_file="${checkpoint%.*}_state.json"
                if [ -f "$state_file" ]; then
                    rm "$state_file" 2>/dev/null
                fi
            else
                print_error "删除失败: $(basename "$checkpoint")"
            fi
        done
        
        print_success "成功删除 $deleted 个检查点文件"
    else
        print_info "操作已取消"
    fi
}

# --- 从检查点恢复训练 ---
resume_training() {
    local checkpoint_ref="$1"
    local checkpoint_path=""
    
    # 处理特殊引用
    case "$checkpoint_ref" in
        "latest")
            checkpoint_path=$(get_latest_checkpoint)
            ;;
        "best")
            checkpoint_path=$(get_best_checkpoint)
            ;;
        *)
            # 检查是否是完整路径
            if [ -f "$checkpoint_ref" ]; then
                checkpoint_path="$checkpoint_ref"
            # 检查是否在检查点目录中
            elif [ -f "$CHECKPOINT_DIR/$checkpoint_ref" ]; then
                checkpoint_path="$CHECKPOINT_DIR/$checkpoint_ref"
            else
                print_error "找不到检查点: $checkpoint_ref"
                return 1
            fi
            ;;
    esac
    
    if [ -z "$checkpoint_path" ] || [ ! -f "$checkpoint_path" ]; then
        print_error "无效的检查点路径: $checkpoint_path"
        return 1
    fi
    
    print_header "从检查点恢复训练"
    print_info "检查点文件: $(basename "$checkpoint_path")"
    
    # 显示检查点信息
    get_checkpoint_info "$checkpoint_path" "table"
    
    echo ""
    echo -e "${YELLOW}确认从此检查点恢复训练吗？ [Y/n]${NC}"
    read -r confirmation
    
    if [[ "$confirmation" =~ ^[Nn]$ ]]; then
        print_info "操作已取消"
        return 0
    fi
    
    # 调用训练脚本
    local train_script="$SCRIPT_DIR/train.sh"
    if [ -f "$train_script" ]; then
        print_info "启动训练脚本..."
        "$train_script" --resume "$checkpoint_path"
    else
        print_info "训练脚本不存在，使用Python直接调用..."
        conda run -n "$ENV_NAME" python main.py train --resume "$checkpoint_path"
    fi
}

# --- 备份检查点 ---
backup_checkpoints() {
    local target_dir="$1"
    
    if [ -z "$target_dir" ]; then
        target_dir="./checkpoint_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    
    print_header "备份检查点"
    print_info "源目录: $CHECKPOINT_DIR"
    print_info "目标目录: $target_dir"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        print_error "检查点目录不存在: $CHECKPOINT_DIR"
        return 1
    fi
    
    mkdir -p "$target_dir"
    
    local files_copied=0
    while IFS= read -r -d '' file; do
        cp "$file" "$target_dir/"
        ((files_copied++))
    done < <(find "$CHECKPOINT_DIR" -type f \( -name "*.ckpt" -o -name "*.json" \) -print0)
    
    print_success "成功备份 $files_copied 个文件到: $target_dir"
}

# --- 主函数 ---
main() {
    local command=""
    local format="table"
    local sort_field="time"
    local reverse="false"
    local keep_count="5"
    local dry_run="false"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --checkpoint-dir)
                CHECKPOINT_DIR="$2"
                shift 2
                ;;
            --env)
                ENV_NAME="$2"
                shift 2
                ;;
            --format)
                format="$2"
                shift 2
                ;;
            --sort)
                sort_field="$2"
                shift 2
                ;;
            --reverse)
                reverse="true"
                shift
                ;;
            --keep)
                keep_count="$2"
                shift 2
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            list|ls)
                command="list"
                shift
                ;;
            info)
                command="info"
                checkpoint_file="$2"
                shift 2
                ;;
            latest)
                command="latest"
                shift
                ;;
            best)
                command="best"
                shift
                ;;
            clean)
                command="clean"
                shift
                ;;
            resume)
                command="resume"
                checkpoint_ref="$2"
                shift 2
                ;;
            backup)
                command="backup"
                backup_target="${2:-}"
                shift
                [ -n "${2:-}" ] && shift
                ;;
            restore)
                command="restore"
                restore_source="$2"
                shift 2
                ;;
            *)
                if [ -z "$command" ]; then
                    print_error "未知命令: $1"
                    echo "使用 --help 查看帮助信息"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 执行命令
    case "$command" in
        "list"|"")
            list_checkpoints "$format" "$sort_field" "$reverse"
            ;;
        "info")
            if [ -z "${checkpoint_file:-}" ]; then
                print_error "请指定检查点文件名"
                exit 1
            fi
            print_header "检查点信息"
            get_checkpoint_info "$CHECKPOINT_DIR/$checkpoint_file" "$format"
            ;;
        "latest")
            print_header "最新检查点"
            local latest=$(get_latest_checkpoint)
            if [ -n "$latest" ]; then
                get_checkpoint_info "$latest" "$format"
            fi
            ;;
        "best")
            print_header "最佳检查点"
            local best=$(get_best_checkpoint)
            if [ -n "$best" ]; then
                get_checkpoint_info "$best" "$format"
            fi
            ;;
        "clean")
            clean_checkpoints "$keep_count" "$dry_run"
            ;;
        "resume")
            if [ -z "${checkpoint_ref:-}" ]; then
                print_error "请指定检查点文件或引用(latest/best)"
                exit 1
            fi
            resume_training "$checkpoint_ref"
            ;;
        "backup")
            backup_checkpoints "${backup_target:-}"
            ;;
        "restore")
            if [ -z "${restore_source:-}" ]; then
                print_error "请指定恢复源目录"
                exit 1
            fi
            print_info "恢复功能待实现..."
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
