#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版CE-CSL手语识别模型集成
"""

import asyncio
import json
import numpy as np
import requests
import time
from pathlib import Path

# 测试配置
BASE_URL = "http://localhost:8000"
TEST_LANDMARKS_SIZE = 543 * 3  # MediaPipe 543个关键点，每个点3个坐标(x,y,z)


def generate_test_landmarks(num_frames: int = 30) -> list:
    """生成测试用的关键点数据"""
    # 模拟MediaPipe输出的关键点数据
    landmarks = []
    for frame in range(num_frames):
        frame_landmarks = []
        for point in range(TEST_LANDMARKS_SIZE):
            # 生成正则化的随机坐标
            x = np.random.random() * 0.8 + 0.1  # 0.1-0.9范围
            frame_landmarks.append(x)
        landmarks.append(frame_landmarks)
    return landmarks


def test_health_check():
    """测试健康检查"""
    print("🏥 测试健康检查...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data['status']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False


def test_enhanced_cecsl_stats():
    """测试获取增强版CE-CSL统计信息"""
    print("📊 测试获取增强版CE-CSL统计信息...")
    try:
        response = requests.get(f"{BASE_URL}/api/enhanced-cecsl/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 统计信息获取成功:")
            print(f"   - 服务状态: {'已加载' if data.get('model_info', {}).get('is_loaded') else '未加载'}")
            print(f"   - 词汇表大小: {data.get('model_info', {}).get('vocab_size', 0)}")
            print(f"   - 模型路径: {data.get('model_info', {}).get('model_path', 'N/A')}")
            print(f"   - 预测次数: {data.get('stats', {}).get('predictions', 0)}")
            return True
        else:
            print(f"❌ 统计信息获取失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 统计信息获取异常: {e}")
        return False


def test_enhanced_cecsl_prediction():
    """测试增强版CE-CSL模型预测"""
    print("🤖 测试增强版CE-CSL模型预测...")
    try:
        # 生成测试数据
        test_landmarks = generate_test_landmarks(30)
        
        # 发送预测请求
        payload = {
            "landmarks": test_landmarks,
            "description": "测试手语识别"
        }
        
        print(f"   发送数据: {len(test_landmarks)}帧, 每帧{len(test_landmarks[0])}个特征")
        
        response = requests.post(
            f"{BASE_URL}/api/enhanced-cecsl/test",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                prediction = data.get("prediction", {})
                stats = data.get("stats", {})
                
                print(f"✅ 预测成功:")
                print(f"   - 预测文本: {prediction.get('text', 'N/A')}")
                print(f"   - 置信度: {prediction.get('confidence', 0):.4f}")
                print(f"   - 手势序列: {prediction.get('gloss_sequence', [])}")
                print(f"   - 推理时间: {prediction.get('inference_time', 0):.4f}秒")
                print(f"   - 状态: {prediction.get('status', 'N/A')}")
                
                if prediction.get('error'):
                    print(f"   - 错误信息: {prediction['error']}")
                
                print(f"   服务统计:")
                print(f"   - 总预测次数: {stats.get('predictions', 0)}")
                print(f"   - 错误次数: {stats.get('errors', 0)}")
                print(f"   - 平均推理时间: {stats.get('avg_inference_time', 0):.4f}秒")
                
                return True
            else:
                print(f"❌ 预测失败: {data.get('message')}")
                return False
        else:
            print(f"❌ 预测请求失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 预测测试异常: {e}")
        return False


def test_model_files():
    """检查模型文件是否存在"""
    print("📁 检查模型文件...")
    
    model_files = [
        "training/output/enhanced_cecsl_final_model.ckpt",
        "training/output/enhanced_vocab.json",
        "training/output/enhanced_training_history.json"
    ]
    
    all_exist = True
    for file_path in model_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} - 存在 ({path.stat().st_size} 字节)")
        else:
            print(f"❌ {file_path} - 不存在")
            all_exist = False
    
    return all_exist


def main():
    """主测试函数"""
    print("🚀 开始测试增强版CE-CSL手语识别模型集成")
    print("=" * 50)
    
    # 测试步骤
    tests = [
        ("模型文件检查", test_model_files),
        ("后端健康检查", test_health_check),
        ("服务统计信息", test_enhanced_cecsl_stats),
        ("模型预测测试", test_enhanced_cecsl_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        start_time = time.time()
        success = test_func()
        duration = time.time() - start_time
        results.append((test_name, success, duration))
        print(f"   耗时: {duration:.2f}秒")
    
    # 测试结果汇总
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = 0
    for test_name, success, duration in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status} ({duration:.2f}秒)")
        if success:
            passed += 1
    
    print(f"\n总结: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！增强版CE-CSL模型集成成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit(main())
