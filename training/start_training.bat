@echo off
setlocal

REM TFNet Training Startup Script for Windows
REM Optimized for CPU execution with MindSpore

echo ============================================================
echo TFNet Training System
echo Continuous Sign Language Recognition
echo ============================================================

REM Parse command line arguments first
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=train"

echo Action: %ACTION%

REM Initialize conda
echo Initializing conda...

REM Simple conda activation approach
call conda activate shengteng 2>nul
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'shengteng'
    echo.
    echo Please ensure:
    echo 1. Conda is installed and in PATH
    echo 2. Environment 'shengteng' exists
    echo.
    echo To create the environment:
    echo   conda create -n shengteng python=3.8
    echo   conda activate shengteng
    echo   pip install mindspore opencv-python numpy
    echo.
    pause
    exit /b 1
)

echo Success: Conda environment activated

REM Check project structure
echo Checking project structure...
if not exist "training" (
    echo Error: Please run this script from the project root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)
echo Success: Project structure OK

REM Route to appropriate action
if /i "%ACTION%"=="train" goto train
if /i "%ACTION%"=="eval" goto eval
if /i "%ACTION%"=="check" goto check
if /i "%ACTION%"=="help" goto help

echo Invalid action: %ACTION%
goto help

:train
echo.
echo Starting TFNet training...
python training\train_tfnet.py --config training\configs\tfnet_config.json
if errorlevel 1 (
    echo Training failed!
    pause
    exit /b 1
)
echo Training completed successfully!
goto end

:eval
echo.
echo Starting TFNet evaluation...
python training\evaluator.py --config training\configs\tfnet_config.json
if errorlevel 1 (
    echo Evaluation failed!
    pause
    exit /b 1
)
echo Evaluation completed successfully!
goto end

:check
echo.
echo Running environment checks...
python training\start_training.py check --config training\configs\tfnet_config.json
if errorlevel 1 (
    echo Environment check failed!
    pause
    exit /b 1
)
echo Environment check completed successfully!
goto end

:help
echo.
echo Usage: start_training.bat [ACTION]
echo.
echo Actions:
echo   train  - Start model training (default)
echo   eval   - Run model evaluation
echo   check  - Check environment setup
echo   help   - Show this help message
echo.
goto end

:end
echo.
echo Operation completed!
pause
