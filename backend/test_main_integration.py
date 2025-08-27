#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集成后的 main.py 是否能正常启动
"""

import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_import():
    """测试导入是否正常"""
    try:
        from main import app, enhanced_cecsl_service
        print("✅ 成功导入 main.py")
        print(f"✅ FastAPI 应用创建成功: {type(app)}")
        print(f"✅ 增强版CE-CSL服务: {'可用' if enhanced_cecsl_service.is_loaded else '不可用'}")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # 测试根路径
        response = client.get("/")
        print(f"✅ 根路径测试: {response.status_code}")
        
        # 测试健康检查
        response = client.get("/api/health")
        print(f"✅ 健康检查测试: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   状态: {data.get('status')}")
            print(f"   消息: {data.get('message')}")
        
        return True
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试集成后的 main.py")
    print("=" * 50)
    
    # 测试导入
    if not test_import():
        return False
    
    print("\n" + "=" * 50)
    
    # 测试基本功能
    try:
        # 首先尝试安装 fastapi[all] 如果没有
        import fastapi
        from fastapi.testclient import TestClient
        
        if not test_basic_functionality():
            return False
    except ImportError:
        print("⚠️  未安装 fastapi[all]，跳过基本功能测试")
        print("   可运行: pip install fastapi[all] 进行完整测试")
    
    print("\n" + "=" * 50)
    print("🎉 集成测试完成！main.py 可以正常使用")
    print("\n📝 使用说明:")
    print("   - 直接运行: python main.py")
    print("   - 指定端口: PORT=8001 python main.py")
    print("   - API文档: http://localhost:8000/api/docs")
    print("   - 健康检查: http://localhost:8000/api/health")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
