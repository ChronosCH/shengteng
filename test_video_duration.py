#!/usr/bin/env python3
"""
创建测试视频并测试视频时长修复
"""

import cv2
import numpy as np
import requests
import json
import time
import os

def create_test_video(filename="test_video.mp4", duration_seconds=5, fps=30):
    """创建一个测试视频文件"""
    try:
        # 视频参数
        width, height = 640, 480
        total_frames = int(duration_seconds * fps)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        print(f"正在创建测试视频: {filename}")
        print(f"   时长: {duration_seconds}秒")
        print(f"   帧率: {fps} fps")
        print(f"   总帧数: {total_frames}")
        
        # 生成帧
        for frame_num in range(total_frames):
            # 创建彩色背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 添加渐变背景
            color = int(255 * (frame_num / total_frames))
            frame[:, :, 1] = color  # 绿色通道
            
            # 添加文本
            text = f"Frame {frame_num + 1}/{total_frames}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 添加时间戳
            timestamp = f"Time: {frame_num/fps:.2f}s"
            cv2.putText(frame, timestamp, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 添加一个移动的圆形（模拟手势）
            center_x = int(width/2 + 100 * np.sin(2 * np.pi * frame_num / 60))
            center_y = int(height/2 + 50 * np.cos(2 * np.pi * frame_num / 60))
            cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), -1)
            
            out.write(frame)
        
        out.release()
        
        print(f"✅ 测试视频创建完成: {filename}")
        
        # 验证视频信息
        cap = cv2.VideoCapture(filename)
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_duration = actual_frames / actual_fps if actual_fps > 0 else 0
        cap.release()
        
        print(f"   验证信息:")
        print(f"   - 实际帧数: {actual_frames}")
        print(f"   - 实际帧率: {actual_fps:.2f}")
        print(f"   - 实际时长: {actual_duration:.2f}秒")
        
        return True, {
            "filename": filename,
            "expected_duration": duration_seconds,
            "actual_duration": actual_duration,
            "expected_frames": total_frames,
            "actual_frames": actual_frames,
            "fps": actual_fps
        }
        
    except Exception as e:
        print(f"❌ 创建测试视频失败: {e}")
        return False, None

def test_video_upload_and_duration(video_file):
    """测试视频上传和时长识别"""
    print(f"\n📤 测试视频上传: {video_file}")
    
    try:
        # 上传视频
        with open(video_file, 'rb') as f:
            files = {'file': (video_file, f, 'video/mp4')}
            response = requests.post('http://localhost:8000/api/enhanced-cecsl/upload-video', files=files, timeout=30)
        
        if response.status_code == 200:
            upload_result = response.json()
            print(f"✅ 视频上传成功: {upload_result}")
            
            task_id = upload_result.get('task_id')
            if task_id:
                print(f"🔄 等待处理完成 (任务ID: {task_id})")
                return poll_task_result(task_id)
        else:
            print(f"❌ 视频上传失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ 视频上传测试失败: {e}")
        return False, None

def poll_task_result(task_id, max_wait=60):
    """轮询任务结果"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f'http://localhost:8000/api/enhanced-cecsl/video-status/{task_id}', timeout=10)
            
            if response.status_code == 200:
                task_data = response.json()
                
                status = task_data.get('status', 'unknown')
                progress = task_data.get('progress', 0.0)
                
                print(f"   状态: {status}, 进度: {progress:.1%}")
                
                if status == 'completed':
                    result = task_data.get('result', {})
                    print("🎉 视频处理完成!")
                    print_video_result_detailed(result)
                    return True, result
                elif status == 'error':
                    error = task_data.get('error', 'Unknown error')
                    print(f"❌ 视频处理失败: {error}")
                    return False, None
                
                time.sleep(2)  # 等待2秒后再查询
            else:
                print(f"❌ 查询任务状态失败: {response.status_code}")
                print(f"   响应内容: {response.text}")
                return False, None
                
        except Exception as e:
            print(f"⚠️ 查询任务状态异常: {e}")
            time.sleep(2)
    
    print("⏰ 任务超时")
    return False, None

def print_video_result_detailed(result):
    """详细打印视频处理结果"""
    print("\n📊 详细视频处理结果:")
    print("=" * 50)
    
    # 视频信息
    print("📹 视频信息:")
    frame_count = result.get('frame_count', 0)
    fps = result.get('fps', 0)
    duration = result.get('duration', 0)
    processing_time = result.get('processing_time', 0)
    
    print(f"   帧数: {frame_count}")
    print(f"   帧率: {fps:.2f} fps")
    print(f"   时长: {duration:.2f} 秒")
    print(f"   处理时间: {processing_time:.2f} 秒")
    print(f"   关键点提取: {'成功' if result.get('landmarks_extracted') else '失败'}")
    
    # 识别结果
    recognition = result.get('recognition_result', {})
    if recognition:
        print("\n🤖 识别结果:")
        print(f"   预测文本: {recognition.get('text', 'N/A')}")
        print(f"   置信度: {recognition.get('confidence', 0):.2%}")
        print(f"   手势序列: {recognition.get('gloss_sequence', [])}")
        print(f"   推理时间: {recognition.get('inference_time', 0)*1000:.1f} ms")
        print(f"   状态: {recognition.get('status', 'N/A')}")
        
        if recognition.get('error'):
            print(f"   错误: {recognition['error']}")
    
    print("=" * 50)

def main():
    print("🎬 视频时长修复测试")
    print("=" * 60)
    
    # 1. 创建不同时长的测试视频
    test_videos = []
    
    for duration in [3, 5, 10]:  # 创建3秒、5秒、10秒的测试视频
        filename = f"test_video_{duration}s.mp4"
        print(f"\n创建 {duration} 秒测试视频...")
        
        success, video_info = create_test_video(filename, duration)
        if success:
            test_videos.append((filename, video_info))
        
    if not test_videos:
        print("❌ 无法创建测试视频")
        return
    
    print(f"\n✅ 创建了 {len(test_videos)} 个测试视频")
    
    # 2. 测试视频上传和时长识别
    print("\n" + "=" * 60)
    print("📤 开始视频上传和处理测试")
    print("=" * 60)
    
    results = []
    
    for filename, expected_info in test_videos:
        print(f"\n测试视频: {filename}")
        print(f"期望时长: {expected_info['expected_duration']}秒")
        print(f"实际时长: {expected_info['actual_duration']:.2f}秒")
        
        success, result = test_video_upload_and_duration(filename)
        
        if success and result:
            detected_duration = result.get('duration', 0)
            expected_duration = expected_info['expected_duration']
            
            # 检查时长识别准确性
            duration_error = abs(detected_duration - expected_duration)
            accuracy = (1 - duration_error / expected_duration) * 100 if expected_duration > 0 else 0
            
            print(f"\n📏 时长识别准确性:")
            print(f"   期望时长: {expected_duration:.2f}秒")
            print(f"   识别时长: {detected_duration:.2f}秒")
            print(f"   误差: {duration_error:.2f}秒")
            print(f"   准确率: {accuracy:.1f}%")
            
            results.append({
                'filename': filename,
                'expected': expected_duration,
                'detected': detected_duration,
                'accuracy': accuracy,
                'success': True
            })
        else:
            results.append({
                'filename': filename,
                'expected': expected_info['expected_duration'],
                'detected': 0,
                'accuracy': 0,
                'success': False
            })
    
    # 3. 总结测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果总结")
    print("=" * 60)
    
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        avg_accuracy = sum(r['accuracy'] for r in successful_tests) / len(successful_tests)
        print(f"✅ 成功测试: {len(successful_tests)}/{len(results)}")
        print(f"📊 平均时长识别准确率: {avg_accuracy:.1f}%")
        
        print("\n详细结果:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['filename']}: {result['expected']:.1f}s → {result['detected']:.1f}s ({result['accuracy']:.1f}%)")
        
        if avg_accuracy > 90:
            print("\n🎉 时长识别修复成功！准确率 > 90%")
        elif avg_accuracy > 70:
            print("\n✅ 时长识别基本正常，准确率 > 70%")
        else:
            print("\n⚠️ 时长识别可能还需要进一步优化")
    else:
        print("❌ 所有测试都失败了")
    
    # 4. 清理测试文件
    print(f"\n🧹 清理测试文件...")
    for filename, _ in test_videos:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"   删除: {filename}")
        except Exception as e:
            print(f"   删除失败 {filename}: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
