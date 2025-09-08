@echo off
echo ========================================
echo TFNet连续手语识别训练系统
echo ========================================

echo 激活conda环境...
call conda activate shengteng

echo 切换到训练目录...
cd /d "%~dp0training"

echo 开始训练...
python train_tfnet.py

echo 训练完成!
pause
