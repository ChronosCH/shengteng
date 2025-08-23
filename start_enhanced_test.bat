@echo off
echo ====================================
echo 启动增强版CE-CSL手语识别Web应用测试
echo ====================================

:: 激活conda环境
echo 正在激活shengteng环境...
call conda activate shengteng
if %errorlevel% neq 0 (
    echo 错误: 无法激活shengteng环境
    echo 请确保已创建shengteng conda环境
    pause
    exit /b 1
)

:: 检查Python环境
echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未正确安装或配置
    pause
    exit /b 1
)

:: 检查MindSpore
echo 检查MindSpore...
python -c "import mindspore; print('MindSpore版本:', mindspore.__version__)"
if %errorlevel% neq 0 (
    echo 警告: MindSpore未安装，将使用模拟模式
)

:: 检查模型文件
echo 检查模型文件...
if not exist "training\output\enhanced_cecsl_final_model.ckpt" (
    echo 错误: 模型文件不存在: training\output\enhanced_cecsl_final_model.ckpt
    echo 请先运行训练脚本生成模型
    pause
    exit /b 1
)

if not exist "training\output\enhanced_vocab.json" (
    echo 错误: 词汇表文件不存在: training\output\enhanced_vocab.json
    echo 请先运行训练脚本生成词汇表
    pause
    exit /b 1
)

echo 模型文件检查通过 ✓

:: 启动后端服务
echo 启动后端服务...
echo 按 Ctrl+C 停止服务
echo ====================================

start "SignAvatar Backend" python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

:: 等待服务启动
echo 等待后端服务启动...
timeout /t 5 /nobreak > nul

:: 运行测试
echo 运行集成测试...
python test_enhanced_integration.py

pause
