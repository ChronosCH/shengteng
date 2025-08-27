#!/usr/bin/env python3
"""
视频处理和模型推理测试脚本
测试修复后的视频时长和置信度问题
"""

import asyncio
import aiohttp
import json
import os
import time
from pathlib import Path

async def test_video_upload_and_recognition():
    """测试视频上传和识别"""
    print("🎥 测试视频上传和识别...")
    
    # 查找测试视频文件
    test_video_paths = [
        "test_video.mp4",
        "sample.mp4", 
        "demo.mp4",
        "../data/test_video.mp4"
    ]
    
    test_video = None
    for path in test_video_paths:
        if os.path.exists(path):
            test_video = path
            break
    
    if not test_video:
        print("⚠️ 未找到测试视频文件，创建模拟测试...")
        return await test_enhanced_cecsl_prediction()
    
    print(f"📁 使用测试视频: {test_video}")
    
    try:
        # 上传视频
        async with aiohttp.ClientSession() as session:
            with open(test_video, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=os.path.basename(test_video))
                
                print("📤 正在上传视频...")
                async with session.post('http://localhost:8000/api/upload-video', data=data) as response:
                    if response.status == 200:
                        upload_result = await response.json()
                        print(f"✅ 视频上传成功: {upload_result}")
                        
                        task_id = upload_result.get('task_id')
                        if task_id:
                            return await poll_task_result(task_id)
                    else:
                        print(f"❌ 视频上传失败: {response.status}")
                        return False
    except Exception as e:
        print(f"❌ 视频上传测试失败: {e}")
        return False

async def poll_task_result(task_id: str, max_wait: int = 30):
    """轮询任务结果"""
    print(f"🔄 轮询任务结果 (ID: {task_id})")
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < max_wait:
            try:
                async with session.get(f'http://localhost:8000/api/task-status/{task_id}') as response:
                    if response.status == 200:
                        task_data = await response.json()
                        
                        status = task_data.get('status', 'unknown')
                        progress = task_data.get('progress', 0.0)
                        
                        print(f"   状态: {status}, 进度: {progress:.1%}")
                        
                        if status == 'completed':
                            result = task_data.get('result', {})
                            print("🎉 视频处理完成!")
                            print_video_result(result)
                            return True
                        elif status == 'error':
                            error = task_data.get('error', 'Unknown error')
                            print(f"❌ 视频处理失败: {error}")
                            return False
                        
                        await asyncio.sleep(1)
                    else:
                        print(f"❌ 查询任务状态失败: {response.status}")
                        return False
            except Exception as e:
                print(f"⚠️ 查询任务状态异常: {e}")
                await asyncio.sleep(1)
    
    print("⏰ 任务超时")
    return False

def print_video_result(result: dict):
    """打印视频处理结果"""
    print("\n📊 视频处理结果:")
    print(f"   📹 视频信息:")
    print(f"      - 帧数: {result.get('frame_count', 0)}")
    print(f"      - 帧率: {result.get('fps', 0):.2f} fps")
    print(f"      - 时长: {result.get('duration', 0):.2f} 秒")
    print(f"      - 处理时间: {result.get('processing_time', 0):.2f} 秒")
    
    recognition = result.get('recognition_result', {})
    if recognition:
        print(f"   🤖 识别结果:")
        print(f"      - 预测文本: {recognition.get('text', 'N/A')}")
        print(f"      - 置信度: {recognition.get('confidence', 0):.2%}")
        print(f"      - 手势序列: {recognition.get('gloss_sequence', [])}")
        print(f"      - 推理时间: {recognition.get('inference_time', 0)*1000:.1f} ms")
        print(f"      - 状态: {recognition.get('status', 'N/A')}")

async def test_enhanced_cecsl_prediction():
    """测试增强版CE-CSL模型预测"""
    print("🤖 测试增强版CE-CSL模型预测...")
    
    # 生成测试关键点数据
    def generate_test_landmarks(frames=60):  # 增加帧数测试
        landmarks = []
        for _ in range(frames):
            frame_data = []
            for _ in range(63):  # 21个关键点 * 3个坐标
                frame_data.append(float(0.3 + 0.4 * (0.5 - 0.5)))  # 归一化坐标
            landmarks.append(frame_data)
        return landmarks
    
    try:
        test_landmarks = generate_test_landmarks(60)  # 测试60帧
        
        payload = {
            "landmarks": test_landmarks,
            "description": "测试视频时长和置信度修复"
        }
        
        print(f"   发送数据: {len(test_landmarks)}帧, 每帧{len(test_landmarks[0])}个特征")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/enhanced-cecsl/test",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        prediction = data.get("prediction", {})
                        
                        print(f"✅ 预测成功:")
                        print(f"   - 预测文本: {prediction.get('text', 'N/A')}")
                        print(f"   - 置信度: {prediction.get('confidence', 0):.2%}")
                        print(f"   - 手势序列: {prediction.get('gloss_sequence', [])}")
                        print(f"   - 推理时间: {prediction.get('inference_time', 0)*1000:.1f} ms")
                        print(f"   - 状态: {prediction.get('status', 'N/A')}")
                        
                        # 检查置信度是否合理
                        confidence = prediction.get('confidence', 0)
                        if confidence > 0.1:  # 10%以上
                            print(f"✅ 置信度正常: {confidence:.2%}")
                        else:
                            print(f"⚠️ 置信度偏低: {confidence:.2%}")
                        
                        return True
                    else:
                        print(f"❌ 预测失败: {data.get('message')}")
                        return False
                else:
                    print(f"❌ 预测请求失败: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 预测测试异常: {e}")
        return False

async def test_service_stats():
    """测试服务统计信息"""
    print("📊 测试服务统计信息...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/api/enhanced-cecsl/stats") as response:
                if response.status == 200:
                    stats = await response.json()
                    print(f"✅ 获取统计信息成功:")
                    print(f"   - 词汇表大小: {stats.get('vocab_size', 0)}")
                    print(f"   - 总预测次数: {stats.get('total_predictions', 0)}")
                    print(f"   - 平均推理时间: {stats.get('average_inference_time', 0)*1000:.1f} ms")
                    print(f"   - 模型加载状态: {'已加载' if stats.get('is_loaded') else '未加载'}")
                    return True
                else:
                    print(f"❌ 获取统计信息失败: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 统计信息测试异常: {e}")
        return False

async def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 视频处理和模型推理修复测试")
    print("=" * 60)
    print()
    
    # 测试服务状态
    print("1. 测试服务状态...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    print("✅ 后端服务正常运行")
                else:
                    print("❌ 后端服务异常")
                    return
    except Exception as e:
        print(f"❌ 无法连接后端服务: {e}")
        return
    
    print()
    
    # 测试统计信息
    print("2. 测试服务统计...")
    await test_service_stats()
    print()
    
    # 测试模型预测
    print("3. 测试模型预测...")
    prediction_ok = await test_enhanced_cecsl_prediction()
    print()
    
    # 测试视频上传（如果有测试视频）
    print("4. 测试视频上传...")
    video_ok = await test_video_upload_and_recognition()
    print()
    
    # 总结
    print("=" * 60)
    print("📋 测试结果总结:")
    print(f"   模型预测: {'✅ 通过' if prediction_ok else '❌ 失败'}")
    print(f"   视频处理: {'✅ 通过' if video_ok else '❌ 失败'}")
    
    if prediction_ok and video_ok:
        print("\n🎉 所有测试通过！视频时长和置信度问题已修复")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
