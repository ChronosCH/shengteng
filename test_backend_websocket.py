#!/usr/bin/env python3
"""
测试后端WebSocket连接
"""

import asyncio
import websockets
import json
import sys

async def test_websocket():
    """测试WebSocket连接"""
    uri = "ws://localhost:8001/ws/sign-recognition"
    
    try:
        print(f"正在连接到 {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket连接成功！")
            
            # 等待连接确认消息
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📨 收到服务器消息: {data}")
            
            # 发送测试关键点数据
            test_landmarks = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ]
            
            test_message = {
                "type": "landmarks",
                "payload": {
                    "landmarks": test_landmarks,
                    "timestamp": 1234567890,
                    "frameId": 1
                }
            }
            
            print("📤 发送测试关键点数据...")
            await websocket.send(json.dumps(test_message))
            
            # 等待识别结果
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📨 收到识别结果: {result}")
            
            print("✅ WebSocket功能测试完成！")
            
    except ConnectionRefusedError:
        print("❌ 连接被拒绝，请确保后端服务器正在运行在端口8001")
        return False
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")
        return False
    
    return True

async def test_http_health():
    """测试HTTP健康检查"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8001/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ HTTP健康检查成功: {data}")
                    return True
                else:
                    print(f"❌ HTTP健康检查失败，状态码: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ HTTP连接失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🔍 开始测试后端服务...")
    print()
    
    # 测试HTTP健康检查
    print("1. 测试HTTP健康检查...")
    http_ok = await test_http_health()
    print()
    
    if not http_ok:
        print("❌ HTTP服务不可用，请先启动后端服务:")
        print("   cd backend")
        print("   python main_simple.py")
        return
    
    # 测试WebSocket连接
    print("2. 测试WebSocket连接...")
    websocket_ok = await test_websocket()
    print()
    
    if http_ok and websocket_ok:
        print("🎉 所有测试通过！后端服务运行正常。")
        print()
        print("📝 接下来可以:")
        print("   1. 启动前端服务: cd frontend && npm run dev")
        print("   2. 访问页面: http://localhost:5173")
        print("   3. 测试实时识别和视频上传功能")
    else:
        print("⚠️  部分测试失败，请检查服务器配置。")

if __name__ == "__main__":
    # 安装依赖
    try:
        import aiohttp
        import websockets
    except ImportError:
        print("📦 安装必要的依赖包...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "websockets"])
        import aiohttp
        import websockets
    
    asyncio.run(main())
