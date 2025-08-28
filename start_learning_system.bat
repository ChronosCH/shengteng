@echo off
chcp 65001 >nul
echo.
echo ==========================================
echo    🎓 手语学习训练系统启动器
echo ==========================================
echo.
echo 🌟 专业的手语学习平台
echo 📚 功能特色:
echo   • 系统化学习路径
echo   • 互动式手语练习  
echo   • 实时进度跟踪
echo   • 成就系统激励
echo   • 个性化推荐
echo.

:MENU
echo 请选择启动方式:
echo.
echo [1] 🚀 完整系统启动 (推荐)
echo     启动后端+前端+自动打开浏览器
echo.
echo [2] 🔧 仅启动后端服务
echo     仅启动API服务，可通过Postman等工具测试
echo.
echo [3] 🎨 仅启动前端服务  
echo     需要后端服务已运行
echo.
echo [4] 📚 查看API文档
echo     在浏览器中查看完整API文档
echo.
echo [5] 🔍 检查系统状态
echo     检查依赖和服务状态
echo.
echo [0] ❌ 退出
echo.

set /p choice="请输入选择 (0-5): "

if "%choice%"=="1" goto FULL_START
if "%choice%"=="2" goto BACKEND_ONLY
if "%choice%"=="3" goto FRONTEND_ONLY
if "%choice%"=="4" goto SHOW_DOCS
if "%choice%"=="5" goto CHECK_STATUS
if "%choice%"=="0" goto EXIT
echo.
echo ❌ 无效选择，请重新输入
echo.
goto MENU

:FULL_START
echo.
echo 🚀 启动完整系统...
echo.
python quick_start_learning.py
goto END

:BACKEND_ONLY
echo.
echo 🔧 启动后端服务...
echo.
cd backend
echo 📍 API地址: http://localhost:8000
echo 📚 API文档: http://localhost:8000/docs
echo.
python main.py
cd ..
goto END

:FRONTEND_ONLY
echo.
echo 🎨 启动前端服务...
echo.
cd frontend
echo 检查依赖...
if not exist node_modules (
    echo 📦 安装依赖...
    npm install
)
echo.
echo 📍 前端地址: http://localhost:5173
echo 🎓 学习平台: http://localhost:5173/learning
echo.
npm run dev
cd ..
goto END

:SHOW_DOCS
echo.
echo 📚 打开API文档...
echo.
start http://localhost:8000/docs
echo.
echo 如果浏览器没有自动打开，请手动访问:
echo http://localhost:8000/docs
echo.
pause
goto MENU

:CHECK_STATUS
echo.
echo 🔍 检查系统状态...
echo.
echo 检查Python环境...
python --version
echo.
echo 检查依赖包...
python -c "import fastapi, uvicorn, websockets; print('✅ 基础依赖OK')" 2>nul || echo "❌ 缺少依赖包"
echo.
echo 检查目录结构...
if exist backend echo ✅ backend目录存在
if not exist backend echo ❌ backend目录不存在
if exist frontend echo ✅ frontend目录存在  
if not exist frontend echo ❌ frontend目录不存在
echo.
echo 检查服务端口...
netstat -an | findstr :8000 >nul && echo ✅ 端口8000已被占用 || echo ℹ️ 端口8000可用
netstat -an | findstr :5173 >nul && echo ✅ 端口5173已被占用 || echo ℹ️ 端口5173可用
echo.
pause
goto MENU

:EXIT
echo.
echo 👋 再见！感谢使用手语学习训练系统
echo.
goto END

:END
echo.
echo 按任意键退出...
pause >nul
