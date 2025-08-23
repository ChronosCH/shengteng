#!/bin/bash
# 增强版CE-CSL手语识别系统快速启动脚本

echo "🚀 启动增强版CE-CSL手语识别系统..."

# 检查conda环境
if ! conda env list | grep -q "shengteng"; then
    echo "❌ 错误: 找不到shengteng conda环境"
    echo "请先创建conda环境: conda create -n shengteng python=3.11"
    exit 1
fi

# 激活环境
echo "📦 激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate shengteng

# 检查依赖
echo "🔍 检查依赖包..."
python -c "import fastapi, uvicorn, numpy, pydantic" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 安装依赖包..."
    pip install fastapi uvicorn numpy pydantic python-multipart
fi

# 创建必要目录
mkdir -p temp/video_uploads
mkdir -p logs

echo "🌐 启动后端服务..."
# 启动后端服务
python simple_enhanced_server.py &
SERVER_PID=$!

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 3

# 检查服务状态
if curl -s http://localhost:8001/api/health > /dev/null; then
    echo "✅ 后端服务启动成功！"
    echo ""
    echo "🎯 测试地址:"
    echo "   - API服务: http://localhost:8001"
    echo "   - 健康检查: http://localhost:8001/api/health"
    echo "   - 测试页面: file://$(pwd)/frontend/enhanced-cecsl-test.html"
    echo ""
    echo "📝 使用说明:"
    echo "   1. 在浏览器中打开测试页面"
    echo "   2. 检查服务状态是否正常"
    echo "   3. 上传手语视频文件进行识别"
    echo "   4. 查看识别结果和统计信息"
    echo ""
    echo "🛑 停止服务: Ctrl+C 或 kill $SERVER_PID"
    
    # 保持脚本运行
    wait $SERVER_PID
else
    echo "❌ 后端服务启动失败！"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
