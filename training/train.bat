@echo off
:: CE-CSL手语识别训练启动脚本
:: 使用方法: train.bat [参数]
:: 例如: train.bat --epochs 20 --batch_size 8

echo 🚀 CE-CSL手语识别训练系统
echo ================================

:: 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python环境，请先安装Python
    pause
    exit /b 1
)

:: 检查数据目录
if not exist "..\data\CE-CSL" (
    echo ❌ 数据目录不存在: ..\data\CE-CSL
    echo 请确保CE-CSL数据集已正确放置
    pause
    exit /b 1
)

:: 检查训练器文件
if not exist "cecsl_real_trainer.py" (
    echo ❌ 找不到训练器文件: cecsl_real_trainer.py
    echo 请确保在正确的目录下运行此脚本
    pause
    exit /b 1
)

echo ✅ 环境检查通过
echo.

:: 运行训练
echo 🎯 开始训练...
if "%~1"=="" (
    :: 无参数，使用默认配置
    python train.py
) else (
    :: 有参数，传递所有参数
    python train.py %*
)

:: 检查训练结果
if errorlevel 1 (
    echo.
    echo ❌ 训练失败，请检查错误信息
    pause
    exit /b 1
) else (
    echo.
    echo 🎉 训练成功完成！
    echo 📁 模型已保存到 output 目录
)

pause
