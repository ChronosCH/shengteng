#!/bin/bash

# 通义千问API密钥配置助手
# 使用方法: ./setup_api.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   通义千问API密钥配置助手                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# 检查是否已有API密钥
if [[ -n "$DASHSCOPE_API_KEY" ]]; then
    echo -e "${GREEN}✓ 检测到现有API密钥${NC}"
    echo -e "当前密钥: ${YELLOW}${DASHSCOPE_API_KEY:0:8}...${NC}"
    echo ""
    read -p "是否要更新API密钥? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}保持现有配置${NC}"
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}请选择配置方式:${NC}"
echo "1) 临时配置 (仅当前会话有效)"
echo "2) 永久配置 (写入 ~/.bashrc)"
echo "3) 创建 .env 文件"
echo ""
read -p "请选择 (1-3): " -n 1 -r choice
echo ""
echo ""

# 获取API密钥
echo -e "${YELLOW}请输入你的通义千问API密钥:${NC}"
echo -e "${YELLOW}(可从 https://dashscope.console.aliyun.com/apiKey 获取)${NC}"
echo ""
read -p "API密钥: " api_key

if [[ -z "$api_key" ]]; then
    echo -e "${RED}错误: API密钥不能为空${NC}"
    exit 1
fi

# 简单验证API密钥格式
if [[ ${#api_key} -lt 20 ]]; then
    echo -e "${YELLOW}警告: API密钥长度似乎不正确${NC}"
    read -p "是否继续? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

case $choice in
    1)
        # 临时配置
        export DASHSCOPE_API_KEY="$api_key"
        echo -e "${GREEN}✓ API密钥已临时设置${NC}"
        echo -e "${YELLOW}注意: 此配置仅在当前终端会话有效${NC}"
        echo ""
        echo -e "当前会话可以直接使用:"
        echo -e "${BLUE}python inference.py --video-path test/1 --checkpoint slr_mindspore.ckpt --dict-path gloss_dict.npy --use-llm${NC}"
        ;;
    
    2)
        # 永久配置
        bashrc="$HOME/.bashrc"
        
        # 检查是否已存在配置
        if grep -q "DASHSCOPE_API_KEY" "$bashrc"; then
            # 更新现有配置
            sed -i "s|export DASHSCOPE_API_KEY=.*|export DASHSCOPE_API_KEY=\"$api_key\"|" "$bashrc"
            echo -e "${GREEN}✓ 已更新 ~/.bashrc 中的API密钥${NC}"
        else
            # 添加新配置
            echo "" >> "$bashrc"
            echo "# 通义千问API密钥" >> "$bashrc"
            echo "export DASHSCOPE_API_KEY=\"$api_key\"" >> "$bashrc"
            echo -e "${GREEN}✓ 已将API密钥添加到 ~/.bashrc${NC}"
        fi
        
        # 立即生效
        export DASHSCOPE_API_KEY="$api_key"
        
        echo -e "${YELLOW}注意: 新终端会话将自动加载此配置${NC}"
        echo -e "${YELLOW}当前会话已立即生效${NC}"
        echo ""
        echo -e "立即生效命令 (或重启终端):"
        echo -e "${BLUE}source ~/.bashrc${NC}"
        ;;
    
    3)
        # 创建.env文件
        env_file=".env"
        
        if [[ -f "$env_file" ]]; then
            echo -e "${YELLOW}警告: .env 文件已存在${NC}"
            read -p "是否覆盖? (y/n) " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        cat > "$env_file" << EOF
# 通义千问API配置
DASHSCOPE_API_KEY=$api_key

# 可选: 指定使用的模型 (默认: qwen-plus)
# 可选值: qwen-turbo, qwen-plus, qwen-max
QWEN_MODEL=qwen-plus
EOF
        
        echo -e "${GREEN}✓ 已创建 .env 文件${NC}"
        echo ""
        echo -e "${YELLOW}注意: 需要在代码中加载 .env 文件${NC}"
        echo -e "或手动导出环境变量:"
        echo -e "${BLUE}export \$(cat .env | xargs)${NC}"
        
        # 同时临时导出
        export DASHSCOPE_API_KEY="$api_key"
        echo ""
        echo -e "${GREEN}✓ 当前会话已临时生效${NC}"
        ;;
    
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo -e "${GREEN}配置完成!${NC}"
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo ""

# 测试API连接
echo -e "${YELLOW}是否测试API连接? (y/n)${NC}"
read -p "> " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}正在测试API连接...${NC}"
    echo ""
    
    if command -v python &> /dev/null; then
        # 尝试运行测试
        python -c "
import os
import requests

api_key = os.environ.get('DASHSCOPE_API_KEY')
if not api_key:
    print('❌ 环境变量未设置')
    exit(1)

url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}
payload = {
    'model': 'qwen-turbo',
    'input': {
        'messages': [{'role': 'user', 'content': 'Hi'}]
    },
    'parameters': {
        'result_format': 'message'
    }
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    if response.status_code == 200:
        print('✓ API连接成功!')
        print('  你的API密钥有效,可以正常使用')
    else:
        print(f'❌ API调用失败: {response.status_code}')
        print(f'  响应: {response.text[:200]}')
except Exception as e:
    print(f'❌ 连接失败: {str(e)}')
" 2>/dev/null
        
        if [[ $? -eq 0 ]]; then
            echo ""
            echo -e "${GREEN}API测试成功!${NC}"
        else
            echo ""
            echo -e "${YELLOW}API测试失败,请检查:${NC}"
            echo "1. API密钥是否正确"
            echo "2. 网络连接是否正常"
            echo "3. 阿里云账户是否有余额"
        fi
    else
        echo -e "${YELLOW}未找到Python,跳过API测试${NC}"
    fi
fi

echo ""
echo -e "${GREEN}现在可以使用以下命令进行推理:${NC}"
echo ""
echo -e "${BLUE}# 使用辅助脚本${NC}"
echo -e "${BLUE}./run_inference.sh test/1 --use-llm${NC}"
echo ""
echo -e "${BLUE}# 或直接使用Python${NC}"
echo -e "${BLUE}python inference.py --video-path test/1 --checkpoint slr_mindspore.ckpt --dict-path gloss_dict.npy --use-llm${NC}"
echo ""
