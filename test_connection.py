#!/usr/bin/env python3
"""
前后端连接测试脚本
测试后端服务是否正常启动并可以连接
"""

import asyncio
import aiohttp
import websockets
import json
import time

async def test_backend_api():
    """测试后端API连接"""
    print("🔍 测试后端API连接...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # 测试健康检查端点
            async with session.get('http://localhost:8000/health') as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 后端API连接成功: {data}")
                    return True
                else:
                    print(f"❌ 后端API响应异常: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 后端API连接失败: {e}")
        return False

async def test_websocket():
    """测试WebSocket连接"""
    print("🔍 测试WebSocket连接...")
    
    try:
        uri = "ws://localhost:8000/ws/sign-recognition"
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送测试消息
            test_message = {
                "type": "test",
                "data": "connection_test"
            }
            await websocket.send(json.dumps(test_message))
            print("📤 发送测试消息")
            
            # 等待响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 收到响应: {response}")
                return True
            except asyncio.TimeoutError:
                print("⚠️ WebSocket响应超时，但连接正常")
                return True
                
    except Exception as e:
        print(f"❌ WebSocket连接失败: {e}")
        return False

async def check_backend_process():
    """检查后端进程是否运行"""
    print("🔍 检查后端进程...")
    
    import subprocess
    try:
        # 检查端口8000是否被占用
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True, shell=True)
        if ':8000' in result.stdout:
            print("✅ 端口8000已被占用，后端服务可能正在运行")
            return True
        else:
            print("❌ 端口8000未被占用，后端服务可能未启动")
            return False
    except Exception as e:
        print(f"⚠️ 无法检查端口状态: {e}")
        return None

async def main():
    """主测试函数"""
    print("=" * 50)
    print("🚀 前后端连接测试开始")
    print("=" * 50)
    
    # 检查后端进程
    await check_backend_process()
    print()
    
    # 测试API连接
    api_ok = await test_backend_api()
    print()
    
    # 测试WebSocket连接
    ws_ok = await test_websocket()
    print()
    
    # 总结测试结果
    print("=" * 50)
    print("📊 测试结果总结:")
    print(f"   API连接: {'✅ 正常' if api_ok else '❌ 失败'}")
    print(f"   WebSocket连接: {'✅ 正常' if ws_ok else '❌ 失败'}")
    
    if api_ok and ws_ok:
        print("\n🎉 前后端连接测试通过！")
        print("💡 现在可以启动前端服务进行测试")
    else:
        print("\n⚠️ 存在连接问题，请检查:")
        if not api_ok:
            print("   - 后端服务是否正常启动")
            print("   - 端口8000是否被正确监听")
        if not ws_ok:
            print("   - WebSocket服务是否正常运行")
            print("   - 防火墙是否阻止连接")
    
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
