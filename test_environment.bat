@echo off
echo Testing MindSpore environment...

call conda activate shengteng
cd /d "%~dp0training"
python test_mindspore.py > test_result.txt 2>&1

echo Test completed. Check test_result.txt for results.
type test_result.txt
pause
