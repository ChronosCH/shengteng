@echo off
echo ========================================
echo 启动增强版CE-CSL手语识别Web应用
echo ========================================

:: 激活conda环境
echo 正在激活shengteng环境...
call conda activate shengteng
if %errorlevel% neq 0 (
    echo 错误: 无法激活shengteng环境
    echo 请先运行: conda create -n shengteng python=3.8
    echo 然后安装依赖: pip install -r requirements.txt
    pause
    exit /b 1
)

:: 检查Python环境
echo 检查Python环境...
python --version

:: 检查关键依赖
echo 检查关键依赖...
python -c "import fastapi, uvicorn; print('FastAPI/Uvicorn: OK')" 2>nul || (
    echo 正在安装FastAPI和Uvicorn...
    pip install fastapi uvicorn
)

python -c "import mindspore; print('MindSpore版本:', mindspore.__version__)" 2>nul || (
    echo 警告: MindSpore未安装，将使用模拟模式
    echo 如需使用真实模型，请安装MindSpore
)

:: 检查模型文件
echo 检查模型文件...
if not exist "training\output\enhanced_cecsl_final_model.ckpt" (
    echo 警告: 模型文件不存在: training\output\enhanced_cecsl_final_model.ckpt
    echo 将使用模拟模式运行
) else (
    echo 模型文件检查通过 ✓
)

if not exist "training\output\enhanced_vocab.json" (
    echo 警告: 词汇表文件不存在: training\output\enhanced_vocab.json
    echo 将使用默认词汇表
) else (
    echo 词汇表文件检查通过 ✓
)

:: 创建必要的目录
if not exist "temp" mkdir temp
if not exist "temp\sign_results" mkdir temp\sign_results
if not exist "logs" mkdir logs

echo ========================================
echo 启动服务...
echo ========================================

:: 启动后端服务
echo 启动后端服务 (端口 8000)...
start "增强版CE-CSL后端服务" cmd /k "conda activate shengteng && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

:: 等待后端服务启动
echo 等待后端服务启动...
timeout /t 3 /nobreak > nul

:: 检查前端目录
if exist "frontend" (
    cd frontend
    
    :: 检查是否有Node.js
    node --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo 检测到Node.js，启动前端开发服务器...
        
        :: 检查package.json
        if exist "package.json" (
            :: 安装依赖（如果需要）
            if not exist "node_modules" (
                echo 安装前端依赖...
                npm install
            )
            
            :: 启动前端服务
            echo 启动前端服务 (端口 5173)...
            start "前端开发服务器" cmd /k "npm run dev"
            
            :: 等待前端服务启动
            timeout /t 3 /nobreak > nul
            
            echo ========================================
            echo 服务启动完成！
            echo ========================================
            echo 前端地址: http://localhost:5173
            echo 后端API: http://localhost:8000
            echo 增强版CE-CSL测试页面: http://localhost:5173/enhanced-cecsl-test.html
            echo API文档: http://localhost:8000/api/docs
            echo ========================================
            
            :: 自动打开浏览器
            timeout /t 2 /nobreak > nul
            start http://localhost:5173/enhanced-cecsl-test.html
        ) else (
            echo 前端package.json不存在，只启动后端服务
        )
    ) else (
        echo Node.js未安装，只启动后端服务
        echo 请访问后端API文档: http://localhost:8000/api/docs
    )
    
    cd ..
) else (
    echo 前端目录不存在，只启动后端服务
)

:: 打开测试页面
echo 打开增强版CE-CSL测试页面...
start http://localhost:8000/docs

echo ========================================
echo 运行状态检查...
echo ========================================

:: 等待服务完全启动
timeout /t 5 /nobreak > nul

:: 运行测试脚本
echo 运行集成测试...
python test_enhanced_integration.py

echo ========================================
echo 启动完成！
echo ========================================
echo 后端服务: http://localhost:8000
echo API文档: http://localhost:8000/api/docs
echo 测试页面: file://%cd%\frontend\enhanced-cecsl-test.html
echo ========================================
echo 按任意键退出...
pause > nul
