@echo off
title 重启增强版CE-CSL后端服务
color 0C

echo ========================================
echo    重启增强版CE-CSL后端服务
echo ========================================
echo.

echo [1/4] 查找并终止现有后端进程...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| findstr "main_simple.py"') do (
    echo 终止进程 %%i
    taskkill /f /pid %%i 2>nul
)

echo [2/4] 等待进程完全终止...
timeout /t 2 /nobreak >nul

echo [3/4] 切换到后端目录...
cd /d "d:\shengteng\backend"

echo [4/4] 启动新的后端服务...
echo.
echo 🌐 启动后端服务 (端口: 8001)...
echo    - HTTP API: http://localhost:8001
echo    - WebSocket: ws://localhost:8001/ws/sign-recognition
echo    - 测试端点: ws://localhost:8001/ws/test
echo.

call conda activate shengteng
python main_simple.py

pause
