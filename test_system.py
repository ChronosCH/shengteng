#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语学习训练系统测试脚本
验证所有组件功能是否正常
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path

async def test_backend_health():
    """测试后端健康状态"""
    print("🔍 测试后端健康状态...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 后端健康状态: {data['status']}")
                    print(f"   服务状态: {data['services']}")
                    return True
                else:
                    print(f"❌ 后端健康检查失败: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 无法连接到后端: {e}")
        return False

async def test_learning_api():
    """测试学习API"""
    print("📚 测试学习API...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # 测试获取学习模块
            async with session.get("http://localhost:8000/api/learning/modules") as response:
                if response.status == 200:
                    modules = await response.json()
                    print(f"✅ 获取学习模块成功: {len(modules)}个模块")
                    return True
                else:
                    print(f"❌ 获取学习模块失败: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 学习API测试失败: {e}")
        return False

async def test_recognition_api():
    """测试手语识别API"""
    print("🤖 测试手语识别API...")
    
    try:
        # 模拟关键点数据
        landmarks = [[0.1, 0.2, 0.3] * 21 for _ in range(10)]  # 10帧数据
        
        test_data = {
            "landmarks": landmarks,
            "description": "测试数据"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/enhanced-cecsl/test",
                json=test_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        print("✅ 手语识别API测试成功")
                        prediction = result.get("prediction", {})
                        print(f"   识别结果: {prediction.get('text', 'N/A')}")
                        print(f"   置信度: {prediction.get('confidence', 0):.3f}")
                        return True
                    else:
                        print(f"❌ 手语识别失败: {result.get('message')}")
                        return False
                else:
                    print(f"❌ 手语识别API请求失败: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 手语识别API测试失败: {e}")
        return False

async def test_websocket():
    """测试WebSocket连接"""
    print("🔗 测试WebSocket连接...")
    
    try:
        import websockets
        
        async with websockets.connect("ws://localhost:8000/ws/sign-recognition") as websocket:
            # 接收连接确认消息
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "connection_established":
                print("✅ WebSocket连接成功")
                
                # 发送测试数据
                test_message = {
                    "type": "landmarks",
                    "payload": {
                        "landmarks": [[0.1, 0.2, 0.3] * 21],
                        "frameId": 1
                    }
                }
                
                await websocket.send(json.dumps(test_message))
                
                # 接收响应
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if response_data.get("type") == "recognition_result":
                    print("✅ WebSocket手语识别测试成功")
                    return True
                else:
                    print(f"❌ WebSocket响应异常: {response_data}")
                    return False
            else:
                print(f"❌ WebSocket连接确认失败: {data}")
                return False
                
    except ImportError:
        print("⚠️ websockets包未安装，跳过WebSocket测试")
        print("   安装命令: pip install websockets")
        return None
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("📁 检查文件结构...")
    
    required_files = [
        "backend/main.py",
        "backend/services/learning_training_service.py", 
        "backend/api/learning_routes.py",
        "frontend/package.json",
        "frontend/src/pages/LearningPage.tsx"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少以下文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ 所有必要文件都存在")
        return True

def test_dependencies():
    """测试Python依赖"""
    print("📦 检查Python依赖...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "aiosqlite",
        "pydantic"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下Python包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("   安装命令: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ 所有Python依赖都已安装")
        return True

async def main():
    """主测试函数"""
    print("🎓 手语学习训练系统测试")
    print("=" * 50)
    print()
    
    # 运行所有测试
    tests = [
        ("文件结构", test_file_structure()),
        ("Python依赖", test_dependencies()),
        ("后端健康", await test_backend_health()),
        ("学习API", await test_learning_api()),
        ("识别API", await test_recognition_api()),
        ("WebSocket", await test_websocket())
    ]
    
    print()
    print("📊 测试结果汇总:")
    print("-" * 30)
    
    passed = 0
    total = 0
    
    for test_name, result in tests:
        total += 1
        if result is True:
            print(f"✅ {test_name}: 通过")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}: 失败")
        else:
            print(f"⚠️ {test_name}: 跳过")
    
    print()
    print(f"📈 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常")
    elif passed > total * 0.7:
        print("⚠️ 大部分测试通过，系统基本可用")
    else:
        print("❌ 多项测试失败，请检查系统配置")
    
    print()
    print("💡 使用提示:")
    print("1. 如果后端测试失败，请先启动后端服务：python backend/main.py")
    print("2. 如果前端需要测试，请启动前端服务：cd frontend && npm run dev")
    print("3. 完整系统启动：python quick_start_learning.py")

if __name__ == "__main__":
    asyncio.run(main())
