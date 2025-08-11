@echo off
echo 启动SignAvatar Web系统...
echo.
echo 正在启动前端服务器...
cd /d "%~dp0frontend"
start "SignAvatar Frontend" npm run dev
echo.
echo 前端服务器启动中，请等待...
timeout /t 5 /nobreak >nul
echo.
echo 系统启动完成！
echo.
echo 可以访问以下页面：
echo 主页: http://localhost:3000
echo 基础Avatar: http://localhost:3000/avatar
echo 专业Avatar: http://localhost:3000/avatar-pro
echo 高质量Avatar: http://localhost:3000/avatar-hq
@echo off
echo ===================================
echo      启动Avatar优化演示系统
echo ===================================
echo.
echo 正在启动开发服务器...
cd frontend
npm run dev
echo.
echo 演示页面:
echo - 基础Avatar: http://localhost:3000/avatar
echo - 真人级Avatar: http://localhost:3000/avatar-advanced  
echo - Avatar对比: http://localhost:3000/avatar-compare
echo.
pause
echo.
echo 按任意键退出...
pause >nul
