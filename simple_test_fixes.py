#!/usr/bin/env python3
"""
简单的修复验证测试
"""

import requests
import json
import numpy as np
import time

def test_backend_health():
    """测试后端健康状态"""
    try:
        response = requests.get('http://localhost:8000/api/health', timeout=5)
        if response.status_code == 200:
            print('✅ 后端服务运行正常')
            return True
        else:
            print('❌ 后端服务响应异常')
            return False
    except Exception as e:
        print(f'❌ 无法连接后端服务: {e}')
        print('请先启动后端服务: python backend/main.py')
        return False

def test_model_prediction():
    """测试模型预测（修复后的置信度）"""
    print("测试模型预测（修复后的置信度）...")
    
    # 生成测试数据
    def generate_test_landmarks(frames=60):
        landmarks = []
        for _ in range(frames):
            frame_data = [float(np.random.rand()) for _ in range(63)]
            landmarks.append(frame_data)
        return landmarks

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
                print(f'   文本: {prediction.get("text", "N/A")}')
                print(f'   置信度: {confidence:.2%}')
                print(f'   手势序列: {prediction.get("gloss_sequence", [])}')
                print(f'   推理时间: {prediction.get("inference_time", 0)*1000:.1f} ms')
                
                if confidence > 0.1:
                    print(f'✅ 置信度正常 (>10%): {confidence:.2%}')
                    return True
                else:
                    print(f'⚠️ 置信度仍然偏低: {confidence:.2%}')
                    return False
            else:
                print(f'❌ 预测失败: {data.get("message")}')
                return False
        else:
            print(f'❌ 请求失败: {response.status_code}')
            print(f'   响应内容: {response.text}')
            return False
            
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        return False

def test_service_stats():
    """测试服务统计信息"""
    print("测试服务统计信息...")
    
    try:
        response = requests.get('http://localhost:8000/api/enhanced-cecsl/stats', timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print('✅ 服务统计信息:')
            print(f'   词汇表大小: {stats.get("vocab_size", 0)}')
            print(f'   总预测次数: {stats.get("total_predictions", 0)}')
            print(f'   平均推理时间: {stats.get("average_inference_time", 0)*1000:.1f} ms')
            print(f'   模型状态: {"已加载" if stats.get("is_loaded") else "未加载"}')
            return True
        else:
            print(f'❌ 获取统计失败: {response.status_code}')
            return False
    except Exception as e:
        print(f'❌ 统计测试失败: {e}')
        return False

def main():
    print("=" * 60)
    print("🚀 视频处理和模型推理修复测试")
    print("=" * 60)
    print()
    
    # 1. 测试后端健康状态
    print("1. 检查后端服务状态...")
    if not test_backend_health():
        return
    print()
    
    # 2. 测试服务统计
    print("2. 测试服务统计...")
    stats_ok = test_service_stats()
    print()
    
    # 3. 测试模型预测
    print("3. 测试模型预测...")
    prediction_ok = test_model_prediction()
    print()
    
    # 总结
    print("=" * 60)
    print("📋 测试结果总结:")
    print(f"   服务统计: {'✅ 通过' if stats_ok else '❌ 失败'}")
    print(f"   模型预测: {'✅ 通过' if prediction_ok else '❌ 失败'}")
    
    if stats_ok and prediction_ok:
        print("\n🎉 测试通过！置信度问题已修复")
    else:
        print("\n⚠️ 部分测试失败，需要进一步检查")
    
    print("\n💡 下一步:")
    print("   1. 如果置信度正常，可以测试视频上传功能")
    print("   2. 考虑集成真实的训练模型替换模拟推理")
    print("   3. 添加MediaPipe关键点提取功能")
    print("=" * 60)

if __name__ == "__main__":
    main()
