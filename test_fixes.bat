@echo off
echo ========================================
echo 视频处理和置信度问题修复测试
echo ========================================
echo.

echo 1. 检查后端服务状态...
python -c "
import requests
try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    if response.status_code == 200:
        print('✅ 后端服务运行正常')
    else:
        print('❌ 后端服务响应异常')
        exit(1)
except Exception as e:
    print(f'❌ 无法连接后端服务: {e}')
    print('请先启动后端服务: python backend/main.py')
    pause
    exit(1)
"

if %errorlevel% neq 0 (
    pause
    exit /b 1
)

echo.
echo 2. 测试模型预测（修复后的置信度）...
python -c "
import requests
import json
import numpy as np

# 生成测试数据
def generate_test_landmarks(frames=60):
    landmarks = []
    for _ in range(frames):
        frame_data = [float(np.random.rand()) for _ in range(63)]
        landmarks.append(frame_data)
    return landmarks

# 发送预测请求
try:
    test_landmarks = generate_test_landmarks(60)
    payload = {
        'landmarks': test_landmarks,
        'description': '测试置信度修复'
    }
    
    print(f'发送测试数据: {len(test_landmarks)}帧')
    response = requests.post(
        'http://localhost:8000/api/enhanced-cecsl/test',
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            prediction = data.get('prediction', {})
            confidence = prediction.get('confidence', 0)
            
            print(f'✅ 预测成功:')
            print(f'   文本: {prediction.get(\"text\", \"N/A\")}')
            print(f'   置信度: {confidence:.2%}')
            print(f'   手势序列: {prediction.get(\"gloss_sequence\", [])}')
            print(f'   推理时间: {prediction.get(\"inference_time\", 0)*1000:.1f} ms')
            
            if confidence > 0.1:
                print(f'✅ 置信度正常 (>10%): {confidence:.2%}')
            else:
                print(f'⚠️ 置信度仍然偏低: {confidence:.2%}')
        else:
            print(f'❌ 预测失败: {data.get(\"message\")}')
    else:
        print(f'❌ 请求失败: {response.status_code}')
        
except Exception as e:
    print(f'❌ 测试失败: {e}')
"

echo.
echo 3. 检查服务统计信息...
python -c "
import requests

try:
    response = requests.get('http://localhost:8000/api/enhanced-cecsl/stats', timeout=10)
    if response.status_code == 200:
        stats = response.json()
        print('✅ 服务统计信息:')
        print(f'   词汇表大小: {stats.get(\"vocab_size\", 0)}')
        print(f'   总预测次数: {stats.get(\"total_predictions\", 0)}')
        print(f'   平均推理时间: {stats.get(\"average_inference_time\", 0)*1000:.1f} ms')
        print(f'   模型状态: {\"已加载\" if stats.get(\"is_loaded\") else \"未加载\"}')
    else:
        print(f'❌ 获取统计失败: {response.status_code}')
except Exception as e:
    print(f'❌ 统计测试失败: {e}')
"

echo.
echo ========================================
echo 测试完成！
echo.
echo 如果置信度仍然很低，请检查:
echo 1. 模型是否正确加载
echo 2. 预测逻辑是否正确实现
echo 3. 是否需要加载真实的训练模型
echo ========================================
echo.
pause
