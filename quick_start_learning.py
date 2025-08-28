#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语学习训练系统启动器
快速启动整个学习训练系统
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    return True

def check_dependencies():
    """检查依赖"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "websockets",
        "aiosqlite"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    return True

def start_backend():
    """启动后端服务"""
    print("🚀 启动后端服务...")
    backend_dir = Path("backend")
    
    if not backend_dir.exists():
        print("❌ backend目录不存在")
        return None
    
    try:
        # 启动后端服务
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待服务启动
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ 后端服务启动成功")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ 后端服务启动失败:")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ 启动后端服务时出错: {e}")
        return None

def start_frontend():
    """启动前端服务"""
    print("🎨 启动前端服务...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ frontend目录不存在")
        return None
    
    try:
        # 检查是否需要安装依赖
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            print("📦 安装前端依赖...")
            install_process = subprocess.run(
                ["npm", "install"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True
            )
            
            if install_process.returncode != 0:
                print(f"❌ 安装依赖失败: {install_process.stderr}")
                return None
            print("✅ 依赖安装完成")
        
        # 启动前端开发服务器
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待服务启动
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ 前端服务启动成功")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ 前端服务启动失败:")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except FileNotFoundError:
        print("❌ 未找到npm命令，请安装Node.js")
        return None
    except Exception as e:
        print(f"❌ 启动前端服务时出错: {e}")
        return None

def open_browser():
    """打开浏览器"""
    print("🌐 打开浏览器...")
    try:
        webbrowser.open("http://localhost:5173/learning")
        print("✅ 浏览器已打开")
    except Exception as e:
        print(f"⚠️ 无法自动打开浏览器: {e}")
        print("请手动访问: http://localhost:5173/learning")

def main():
    """主函数"""
    print("🎓 手语学习训练系统启动器")
    print("=" * 50)
    
    # 检查环境
    if not check_python_version():
        return
    
    if not check_dependencies():
        return
    
    backend_process = None
    frontend_process = None
    
    try:
        # 启动后端
        backend_process = start_backend()
        if not backend_process:
            print("❌ 后端服务启动失败，退出")
            return
        
        # 启动前端
        frontend_process = start_frontend()
        if not frontend_process:
            print("❌ 前端服务启动失败，但后端仍在运行")
            print("可以通过 http://localhost:8000 访问API")
        else:
            # 打开浏览器
            open_browser()
        
        print()
        print("🎉 系统启动完成！")
        print("-" * 30)
        print("📍 访问地址:")
        print("  • 学习平台: http://localhost:5173/learning")
        print("  • API文档: http://localhost:8000/docs")
        print("  • 后端状态: http://localhost:8000/health")
        print()
        print("⌨️ 按 Ctrl+C 停止服务")
        print()
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程状态
                if backend_process and backend_process.poll() is not None:
                    print("⚠️ 后端服务已停止")
                    break
                if frontend_process and frontend_process.poll() is not None:
                    print("⚠️ 前端服务已停止")
                    break
        except KeyboardInterrupt:
            print("\n🛑 用户中断服务")
    
    finally:
        # 清理进程
        print("🧹 正在清理...")
        
        if backend_process and backend_process.poll() is None:
            print("停止后端服务...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        
        if frontend_process and frontend_process.poll() is None:
            print("停止前端服务...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        print("✅ 清理完成")
        print("👋 感谢使用手语学习训练系统！")

if __name__ == "__main__":
    main()
