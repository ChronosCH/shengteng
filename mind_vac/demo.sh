#!/bin/bash

# 演示脚本 - 展示完整的使用流程
# 使用方法: ./demo.sh

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo -e "${CYAN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   手语识别 + LLM句子生成 - 完整演示                         ║
║   Sign Language Recognition + LLM Sentence Generation        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BLUE}本演示将展示:${NC}"
echo "  1. 环境检查"
echo "  2. 基本推理(仅识别)"
echo "  3. LLM推理(生成完整句子)"
echo "  4. 结果对比"
echo ""

read -p "按Enter开始演示..." dummy

# ============ 环境检查 ============
echo ""
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}步骤 1/4: 环境检查${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 检查conda环境
echo -n "检查conda环境... "
if [[ "$CONDA_DEFAULT_ENV" == "vac_cslr" ]]; then
    echo -e "${GREEN}✓ vac_cslr${NC}"
else
    echo -e "${YELLOW}⚠ 当前环境: $CONDA_DEFAULT_ENV${NC}"
    echo -e "${YELLOW}建议切换到: vac_cslr${NC}"
fi

# 检查Python
echo -n "检查Python... "
if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    echo -e "${GREEN}✓ $python_version${NC}"
else
    echo -e "${RED}✗ 未找到${NC}"
    exit 1
fi

# 检查必要文件
echo -n "检查模型文件... "
if [[ -f "slr_mindspore.ckpt" ]]; then
    echo -e "${GREEN}✓ slr_mindspore.ckpt${NC}"
else
    echo -e "${RED}✗ 未找到${NC}"
    exit 1
fi

echo -n "检查词典文件... "
if [[ -f "gloss_dict.npy" ]]; then
    echo -e "${GREEN}✓ gloss_dict.npy${NC}"
else
    echo -e "${RED}✗ 未找到${NC}"
    exit 1
fi

echo -n "检查测试视频... "
if [[ -d "test/1" ]]; then
    frame_count=$(ls test/1/*.png 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ test/1 (${frame_count}帧)${NC}"
else
    echo -e "${RED}✗ 未找到${NC}"
    exit 1
fi

# 检查API密钥
echo -n "检查API密钥... "
if [[ -n "$DASHSCOPE_API_KEY" ]]; then
    echo -e "${GREEN}✓ 已配置${NC}"
    LLM_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ 未配置${NC}"
    echo -e "  ${YELLOW}LLM功能将不可用,仅演示基本识别${NC}"
    LLM_AVAILABLE=false
fi

echo ""
read -p "按Enter继续..." dummy

# ============ 基本推理 ============
echo ""
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}步骤 2/4: 基本推理(仅手语识别)${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${CYAN}运行命令:${NC}"
echo "python inference.py \\"
echo "    --video-path test/1 \\"
echo "    --checkpoint slr_mindspore.ckpt \\"
echo "    --dict-path gloss_dict.npy \\"
echo "    --device CPU \\"
echo "    --output ./output_dir"
echo ""

read -p "按Enter运行..." dummy
echo ""

python inference.py \
    --video-path test/1 \
    --checkpoint slr_mindspore.ckpt \
    --dict-path gloss_dict.npy \
    --device CPU \
    --output ./output_dir

echo ""
echo -e "${GREEN}基本推理完成!${NC}"
echo ""
echo -e "${CYAN}识别结果:${NC}"
if [[ -f "./output_dir/inference_result.txt" ]]; then
    cat ./output_dir/inference_result.txt | head -3
else
    echo -e "${RED}结果文件未生成${NC}"
fi

echo ""
read -p "按Enter继续..." dummy

# ============ LLM推理 ============
if [[ "$LLM_AVAILABLE" == true ]]; then
    echo ""
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}步骤 3/4: LLM推理(生成完整句子)${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${CYAN}运行命令:${NC}"
    echo "python inference.py \\"
    echo "    --video-path test/1 \\"
    echo "    --checkpoint slr_mindspore.ckpt \\"
    echo "    --dict-path gloss_dict.npy \\"
    echo "    --device CPU \\"
    echo "    --output ./output_dir \\"
    echo "    --use-llm"
    echo ""
    
    read -p "按Enter运行..." dummy
    echo ""
    
    python inference.py \
        --video-path test/1 \
        --checkpoint slr_mindspore.ckpt \
        --dict-path gloss_dict.npy \
        --device CPU \
        --output ./output_dir \
        --use-llm
    
    echo ""
    echo -e "${GREEN}LLM推理完成!${NC}"
else
    echo ""
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}步骤 3/4: LLM推理(跳过 - API未配置)${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}要启用LLM功能,请配置API密钥:${NC}"
    echo -e "  1. 运行: ${CYAN}./setup_api.sh${NC}"
    echo -e "  2. 或设置: ${CYAN}export DASHSCOPE_API_KEY='your_key'${NC}"
fi

echo ""
read -p "按Enter继续..." dummy

# ============ 结果展示 ============
echo ""
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}步骤 4/4: 结果展示${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ -f "./output_dir/inference_result.txt" ]]; then
    echo -e "${CYAN}完整结果 (inference_result.txt):${NC}"
    echo -e "${BLUE}──────────────────────────────────────────────────${NC}"
    cat ./output_dir/inference_result.txt
    echo -e "${BLUE}──────────────────────────────────────────────────${NC}"
else
    echo -e "${RED}未找到结果文件${NC}"
fi

echo ""

if [[ -f "./output_dir/inference_result.json" ]]; then
    echo -e "${CYAN}JSON结果:${NC}"
    echo -e "${BLUE}──────────────────────────────────────────────────${NC}"
    cat ./output_dir/inference_result.json
    echo -e "${BLUE}──────────────────────────────────────────────────${NC}"
fi

# ============ 总结 ============
echo ""
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}演示总结${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}✓ 演示完成!${NC}"
echo ""
echo -e "${CYAN}主要功能:${NC}"
echo "  ✓ 手语视频识别"
if [[ "$LLM_AVAILABLE" == true ]]; then
    echo "  ✓ LLM完整句子生成"
    echo "  ✓ 中英文对译"
else
    echo "  ○ LLM完整句子生成 (未配置)"
fi
echo ""

echo -e "${CYAN}输出文件:${NC}"
echo "  📄 ./output_dir/inference_result.txt"
echo "  📄 ./output_dir/inference_result.json"
echo ""

echo -e "${CYAN}下一步建议:${NC}"
echo "  1. 查看完整文档: ${BLUE}cat README.md${NC}"
echo "  2. 快速入门指南: ${BLUE}cat QUICKSTART.md${NC}"
echo "  3. 查看使用示例: ${BLUE}python example_usage.py${NC}"
if [[ "$LLM_AVAILABLE" == false ]]; then
    echo "  4. 配置API密钥: ${BLUE}./setup_api.sh${NC}"
fi
echo "  5. 测试其他视频: ${BLUE}./run_inference.sh test/2 --use-llm${NC}"
echo ""

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}感谢使用! 🎉${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
