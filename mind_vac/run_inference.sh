#!/bin/bash

# 手语识别 + LLM 推理脚本
# 使用方法: ./run_inference.sh [视频路径] [可选: --use-llm]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
VIDEO_PATH="${1:-test/1}"
CHECKPOINT="slr_mindspore.ckpt"
DICT_PATH="gloss_dict.npy"
DEVICE="CPU"
OUTPUT_DIR="./output_dir"
USE_LLM="${2}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}手语识别推理脚本${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "vac_cslr" ]]; then
    echo -e "${YELLOW}警告: 当前未激活 vac_cslr 环境${NC}"
    echo -e "${YELLOW}正在尝试激活...${NC}"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate vac_cslr
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}错误: 无法激活 vac_cslr 环境${NC}"
        echo -e "${YELLOW}请手动运行: conda activate vac_cslr${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Conda环境已激活: $CONDA_DEFAULT_ENV${NC}"

# 检查视频路径
if [[ ! -d "$VIDEO_PATH" ]]; then
    echo -e "${RED}错误: 视频路径不存在: $VIDEO_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 视频路径: $VIDEO_PATH${NC}"

# 检查模型文件
if [[ ! -f "$CHECKPOINT" ]]; then
    echo -e "${RED}错误: 模型文件不存在: $CHECKPOINT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 模型文件: $CHECKPOINT${NC}"

# 检查词典文件
if [[ ! -f "$DICT_PATH" ]]; then
    echo -e "${RED}错误: 词典文件不存在: $DICT_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 词典文件: $DICT_PATH${NC}"

# 构建命令
CMD="python inference.py \
    --video-path $VIDEO_PATH \
    --checkpoint $CHECKPOINT \
    --dict-path $DICT_PATH \
    --device $DEVICE \
    --output $OUTPUT_DIR"

# 检查是否使用LLM
if [[ "$USE_LLM" == "--use-llm" ]]; then
    # 检查API密钥
    if [[ -z "$DASHSCOPE_API_KEY" ]]; then
        echo -e "${YELLOW}警告: 未设置DASHSCOPE_API_KEY环境变量${NC}"
        echo -e "${YELLOW}LLM功能可能无法使用${NC}"
        echo ""
        read -p "是否继续? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ API密钥已配置${NC}"
    fi
    CMD="$CMD --use-llm"
    echo -e "${GREEN}✓ LLM功能: 已启用${NC}"
else
    echo -e "${YELLOW}○ LLM功能: 未启用 (添加 --use-llm 参数启用)${NC}"
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}开始推理...${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# 运行推理
eval $CMD

# 检查结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}推理完成!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "${GREEN}结果已保存到: $OUTPUT_DIR${NC}"
    echo ""
    
    # 显示结果文件
    if [[ -f "$OUTPUT_DIR/inference_result.txt" ]]; then
        echo -e "${GREEN}--- 推理结果预览 ---${NC}"
        cat "$OUTPUT_DIR/inference_result.txt"
        echo ""
    fi
else
    echo ""
    echo -e "${RED}推理失败${NC}"
    exit 1
fi
