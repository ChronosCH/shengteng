#!/usr/bin/env python3
"""
快速测试后端服务状态
"""

import sys
import requests
import json
import time

def test_http_service():
    """测试HTTP服务"""
    try:
        print("🔍 测试HTTP服务...")
        response = requests.get("http://localhost:8001/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ HTTP服务正常:")
            print(f"   状态: {data.get('status', 'unknown')}")
            print(f"   消息: {data.get('message', 'no message')}")
            return True
        else:
            print(f"❌ HTTP服务异常，状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ HTTP连接失败 - 后端服务可能未启动")
        print("   请确保后端服务正在运行在端口8001")
        return False
    except Exception as e:
        print(f"❌ HTTP测试失败: {e}")
        return False

def test_websocket_service():
    """测试WebSocket服务 - 使用websockets库"""
    try:
        import websockets
        import asyncio
        
        async def test_ws():
            uri = "ws://localhost:8001/ws/sign-recognition"
            print("🔍 测试WebSocket服务...")
            print(f"   连接地址: {uri}")
            
            try:
                async with websockets.connect(uri, timeout=5) as websocket:
                    print("✅ WebSocket连接成功!")
                    
                    # 等待服务器确认消息
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=3)
                        data = json.loads(message)
                        print("✅ 收到服务器确认:")
                        print(f"   类型: {data.get('type', 'unknown')}")
                        print(f"   消息: {data.get('payload', {}).get('message', 'no message')}")
                        return True
                    except asyncio.TimeoutError:
                        print("⚠️  WebSocket连接成功但未收到确认消息")
                        return True
                        
            except Exception as e:
                if "403" in str(e):
                    print("❌ WebSocket连接被拒绝 (HTTP 403)")
                    print("   可能原因:")
                    print("   1. WebSocket端点配置错误")
                    print("   2. CORS配置问题")
                    print("   3. FastAPI WebSocket实现问题")
                else:
                    print(f"❌ WebSocket连接失败: {e}")
                return False
        
        return asyncio.run(test_ws())
        
    except ImportError:
        print("📦 安装websockets库...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
        return test_websocket_service()

def test_simple_websocket():
    """测试简单的WebSocket端点"""
    try:
        import websockets
        import asyncio
        
        async def test_simple_ws():
            uri = "ws://localhost:8001/ws/test"
            print("🔍 测试简单WebSocket端点...")
            print(f"   连接地址: {uri}")
            
            try:
                async with websockets.connect(uri, timeout=5) as websocket:
                    print("✅ 简单WebSocket连接成功!")
                    
                    # 等待服务器消息
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=3)
                        print(f"✅ 收到服务器消息: {message}")
                        
                        # 发送测试消息
                        await websocket.send("Hello Server!")
                        response = await asyncio.wait_for(websocket.recv(), timeout=3)
                        print(f"✅ 收到回复: {response}")
                        return True
                        
                    except asyncio.TimeoutError:
                        print("⚠️  简单WebSocket连接成功但通信超时")
                        return True
                        
            except Exception as e:
                if "403" in str(e):
                    print("❌ 简单WebSocket连接被拒绝 (HTTP 403)")
                else:
                    print(f"❌ 简单WebSocket连接失败: {e}")
                return False
        
        return asyncio.run(test_simple_ws())
        
    except ImportError:
        print("📦 websockets库未安装")
        return False

def check_port_status():
    """检查端口占用情况"""
    try:
        import subprocess
        print("🔍 检查端口8001占用情况...")
        result = subprocess.run(
            ["netstat", "-an"], 
            capture_output=True, 
            text=True,
            shell=True
        )
        
        lines = result.stdout.split('\n')
        port_8001_lines = [line for line in lines if ":8001" in line]
        
        if port_8001_lines:
            print("📋 端口8001使用情况:")
            for line in port_8001_lines:
                print(f"   {line.strip()}")
        else:
            print("⚠️  端口8001未被占用 - 后端服务可能未启动")
            
    except Exception as e:
        print(f"❌ 检查端口状态失败: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("   SignAvatar 后端服务状态检查")
    print("=" * 50)
    print()
    
    # 检查端口状态
    check_port_status()
    print()
    
    # 测试HTTP服务
    http_ok = test_http_service()
    print()
    
    if not http_ok:
        print("🚨 后端服务未运行！")
        print()
        print("请按以下步骤启动后端服务:")
        print("1. 打开新的终端窗口")
        print("2. 运行以下命令:")
        print("   cd d:\\shengteng\\backend")
        print("   conda activate shengteng")
        print("   python main_simple.py")
        print()
        print("或者使用一键启动脚本:")
        print("   双击运行: start_enhanced_server.bat")
        return
    
    # 测试WebSocket服务
    websocket_ok = test_websocket_service()
    print()
    
    # 测试简单WebSocket端点
    simple_websocket_ok = test_simple_websocket()
    print()
    
    if http_ok and websocket_ok and simple_websocket_ok:
        print("🎉 所有服务正常运行!")
        print()
        print("现在可以:")
        print("1. 访问前端页面: http://localhost:5173")
        print("2. 点击'连接服务器'按钮")
        print("3. 开始使用手语识别功能")
    elif http_ok and not websocket_ok:
        print("⚠️  HTTP服务正常，但WebSocket连接失败")
        print()
        print("可能的解决方案:")
        print("1. 重启后端服务")
        print("2. 检查防火墙设置")
        print("3. 查看后端服务日志是否有错误")
    else:
        print("❌ 服务状态异常，请检查后端服务")

if __name__ == "__main__":
    # 安装必要依赖
    try:
        import requests
    except ImportError:
        print("📦 安装requests库...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
    
    main()
