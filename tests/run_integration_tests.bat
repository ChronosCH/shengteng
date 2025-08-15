@echo off
REM TFNet集成测试运行脚本 (Windows版本)

setlocal enabledelayedexpansion

echo ==========================================
echo TFNet MindSpore 集成测试
echo ==========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python 未安装或未添加到PATH
    pause
    exit /b 1
)

REM 获取项目根目录
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo 项目根目录: %PROJECT_ROOT%

REM 设置Python路径
set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%\training;%PROJECT_ROOT%\backend;%PYTHONPATH%"

REM 检查依赖
echo 检查依赖包...

REM 检查基础依赖
set "packages=numpy opencv-python tqdm"

for %%p in (%packages%) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        echo ✗ %%p 未安装
        echo 请运行: pip install %%p
    ) else (
        echo ✓ %%p 已安装
    )
)

REM 检查MindSpore
python -c "import mindspore" >nul 2>&1
if errorlevel 1 (
    echo ✗ MindSpore 未安装
    echo 请运行: pip install mindspore
    echo 或访问 https://www.mindspore.cn/install 获取安装指南
) else (
    for /f %%i in ('python -c "import mindspore; print(mindspore.__version__)"') do set "MINDSPORE_VERSION=%%i"
    echo ✓ MindSpore !MINDSPORE_VERSION! 已安装
)

REM 创建测试目录
if not exist "temp" mkdir "temp"

echo.
echo 开始运行集成测试...
echo ==========================================

REM 运行测试
python tests\test_tfnet_integration.py

set "TEST_EXIT_CODE=%errorlevel%"

echo.
echo ==========================================

if %TEST_EXIT_CODE% equ 0 (
    echo 🎉 所有测试通过！
    echo TFNet MindSpore 集成验证成功
) else (
    echo ⚠ 部分测试失败
    echo 请检查上述错误信息并解决相关问题
)

echo ==========================================

pause
exit /b %TEST_EXIT_CODE%
