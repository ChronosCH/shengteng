#!/bin/bash

# 手语识别系统快速启动脚本
# 这个脚本会帮助你快速开始项目开发

set -e

echo "🚀 手语识别系统开发环境设置"
echo "================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Python版本
echo -e "${BLUE}检查Python环境...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
if [[ $python_version < "3.11" ]]; then
    echo -e "${RED}错误: 需要Python 3.11或更高版本，当前版本: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python版本: $python_version${NC}"

# 创建虚拟环境
echo -e "${BLUE}创建Python虚拟环境...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✓ 虚拟环境已创建${NC}"
else
    echo -e "${YELLOW}虚拟环境已存在${NC}"
fi

# 激活虚拟环境
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# 安装基础依赖
echo -e "${BLUE}安装基础依赖包...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 检查MindSpore安装
echo -e "${BLUE}检查MindSpore安装...${NC}"
if python -c "import mindspore" 2>/dev/null; then
    mindspore_version=$(python -c "import mindspore; print(mindspore.__version__)")
    echo -e "${GREEN}✓ MindSpore版本: $mindspore_version${NC}"
else
    echo -e "${YELLOW}⚠ MindSpore未安装，尝试安装CPU版本...${NC}"
    pip install mindspore
fi

# 创建必要目录
echo -e "${BLUE}创建项目目录结构...${NC}"
mkdir -p data/{raw,processed,annotations}
mkdir -p models/{checkpoints,exports}
mkdir -p logs/{training,inference}
mkdir -p temp
mkdir -p training/{configs,scripts}
echo -e "${GREEN}✓ 目录结构已创建${NC}"

# 检查系统健康状态
echo -e "${BLUE}检查系统健康状态...${NC}"
python health_check.py

# 提供后续步骤指导
echo ""
echo -e "${GREEN}🎉 开发环境设置完成！${NC}"
echo ""
echo -e "${BLUE}下一步建议：${NC}"
echo ""
echo -e "${YELLOW}1. 数据准备${NC}"
echo "   - 下载CSL-Daily数据集"
echo "   - 运行数据预处理：python training/data_preprocessing.py"
echo ""
echo -e "${YELLOW}2. 开始训练${NC}"
echo "   - 配置训练参数：编辑 training/configs/cslr_config.json"
echo "   - 开始训练：python training/train_cslr.py"
echo ""
echo -e "${YELLOW}3. 华为昇腾优化${NC}"
echo "   - 安装昇腾驱动和MindSpore-Ascend版本"
echo "   - 运行昇腾优化训练：python training/train_cslr_ascend.py"
echo ""
echo -e "${YELLOW}4. 系统启动${NC}"
echo "   - 部署模型：python training/deploy_models.py"
echo "   - 启动系统：./deploy.sh"
echo ""
echo -e "${BLUE}📚 详细文档：${NC}"
echo "   - 开发指南：docs/development-guide.md"
echo "   - 用户手册：docs/user-guide.md"
echo "   - 部署文档：docs/deployment.md"
echo ""
echo -e "${GREEN}祝你开发顺利！ 🚀${NC}"
