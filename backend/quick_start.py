#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 测试集成后的 main.py 服务
"""

import subprocess
import sys
import time
import requests
from threading import Thread

def test_server():
    """测试服务器响应"""
    print("等待服务器启动...")
    time.sleep(3)
    
    try:
        # 测试健康检查
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务器响应正常!")
            print(f"   状态: {data.get('status')}")
            print(f"   消息: {data.get('message')}")
            print(f"   服务: {data.get('services')}")
        else:
            print(f"❌ 服务器响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
    
    print("\n🌐 访问链接:")
    print("   - 主页: http://localhost:8000/")
    print("   - API文档: http://localhost:8000/api/docs")
    print("   - 健康检查: http://localhost:8000/api/health")
    print("\n按 Ctrl+C 停止服务器")

def main():
    """主函数"""
    print("🚀 启动集成版 SignAvatar 后端服务")
    print("=" * 50)
    
    # 在后台线程中启动测试
    test_thread = Thread(target=test_server, daemon=True)
    test_thread.start()
    
    try:
        # 启动服务器
        result = subprocess.run([
            sys.executable, "main.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 服务器启动失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
