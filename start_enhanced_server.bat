@echo off
title 增强版CE-CSL手语识别系统
color 0A

echo ========================================
echo    增强版CE-CSL手语识别系统启动器
echo ========================================
echo.

echo [1/6] 检查conda环境...
conda env list | findstr "shengteng" >nul
if errorlevel 1 (
    echo ❌ 错误: 找不到shengteng conda环境
    echo 请先创建conda环境: conda create -n shengteng python=3.11
    pause
    exit /b 1
)

echo [2/6] 激活conda环境...
call conda activate shengteng

echo [3/6] 检查Python依赖包...
python -c "import fastapi, uvicorn, numpy, pydantic" 2>nul
if errorlevel 1 (
    echo 📥 安装Python依赖包...
    pip install fastapi uvicorn numpy pydantic python-multipart
)

echo [4/6] 检查Node.js依赖包...
cd frontend
if not exist "node_modules" (
    echo 📥 安装Node.js依赖包...
    npm install
)

echo [5/6] 创建必要目录...
cd ..
if not exist "backend\uploads" mkdir "backend\uploads"
if not exist "logs" mkdir "logs"

rem 在启动服务前，尝试释放被占用的 8001 端口以及已运行的后端进程
echo 🔎 检查并释放端口 8001...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do (
    echo 终止占用 8001 的进程 PID=%%a
    taskkill /F /PID %%a >nul 2>&1
)

rem 终止已有的 main_simple.py 后端进程（如有）
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| findstr "main_simple.py"') do (
    echo 终止已有后端进程 %%i
    taskkill /f /pid %%i 2>nul
)

echo 等待端口释放...
timeout /t 2 /nobreak >nul

echo [6/6] 启动服务...
echo.

echo 🌐 启动后端服务 (端口: 8001)...
cd backend
start "后端服务" cmd /c "python -m uvicorn main_simple:app --host 0.0.0.0 --port 8001 --log-level info"

echo ⏳ 等待后端服务启动...
timeout /t 8 /nobreak >nul

echo 🌐 启动前端服务 (端口: 5173)...
cd ../frontend
start "前端服务" cmd /c "set VITE_WS_URL=ws://localhost:8001/ws/sign-recognition && npm run dev"

echo.
echo ✅ 服务启动完成！
echo.
echo 🎯 访问地址:
echo    - 前端界面: http://localhost:5173
echo    - 后端API: http://localhost:8001
echo    - 健康检查: http://localhost:8001/api/health
echo.
echo 📝 使用说明:
echo    1. 等待前端页面完全加载
echo    2. 进入识别页面
echo    3. 使用增强版视频识别功能
echo    4. 上传手语视频文件进行识别
echo.
echo 🛑 停止服务: 关闭对应的命令行窗口
echo.

echo 🎉 系统已启动，请在浏览器中访问: http://localhost:5173
pause
