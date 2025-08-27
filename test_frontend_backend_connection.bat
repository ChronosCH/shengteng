@echo off
echo ========================================
echo 测试前后端连接
echo ========================================
echo.

echo 1. 检查后端服务状态...
netstat -an | findstr :8000
if %errorlevel%==0 (
    echo ✅ 后端服务正在运行 (端口8000)
) else (
    echo ❌ 后端服务未运行，请先启动后端服务
    echo 运行命令: python backend/main.py
    pause
    exit /b 1
)

echo.
echo 2. 测试连接...
python test_connection.py

echo.
echo 3. 如果连接正常，可以启动前端服务:
echo    cd frontend
echo    npm run dev
echo.
pause
