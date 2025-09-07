"""
调试后端启动问题
"""

import sys
import traceback
from pathlib import Path

print("🔍 开始调试后端启动问题...")

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

print(f"📁 项目根目录: {PROJECT_ROOT}")
print(f"📁 后端目录: {BACKEND_DIR}")

# 添加到Python路径
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
        print(f"✅ 已添加路径: {p}")

print("\n🧪 测试基础导入...")

try:
    import uvicorn
    print("✅ uvicorn 导入成功")
except Exception as e:
    print(f"❌ uvicorn 导入失败: {e}")

try:
    import fastapi
    print("✅ fastapi 导入成功")
except Exception as e:
    print(f"❌ fastapi 导入失败: {e}")

try:
    from fastapi import FastAPI
    app = FastAPI()
    print("✅ FastAPI 应用创建成功")
except Exception as e:
    print(f"❌ FastAPI 应用创建失败: {e}")

print("\n🧪 测试后端模块导入...")

try:
    import backend
    print("✅ backend 模块导入成功")
except Exception as e:
    print(f"❌ backend 模块导入失败: {e}")
    traceback.print_exc()

try:
    from backend import main
    print("✅ backend.main 导入成功")
except Exception as e:
    print(f"❌ backend.main 导入失败: {e}")
    traceback.print_exc()

try:
    from backend.main import app
    print("✅ backend.main.app 导入成功")
    print(f"📋 应用类型: {type(app)}")
except Exception as e:
    print(f"❌ backend.main.app 导入失败: {e}")
    traceback.print_exc()

print("\n🧪 测试认证路由导入...")

try:
    from backend.api.auth_routes import router
    print("✅ 认证路由导入成功")
except Exception as e:
    print(f"❌ 认证路由导入失败: {e}")
    traceback.print_exc()

print("\n🚀 尝试启动简单服务器...")

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # 创建简单应用
    simple_app = FastAPI(title="Debug Server")
    
    # 添加CORS
    simple_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @simple_app.get("/")
    async def root():
        return {"message": "Debug server is running"}
    
    @simple_app.get("/test")
    async def test():
        return {"status": "ok", "message": "Test endpoint working"}
    
    print("✅ 简单应用创建成功")
    
    # 尝试启动
    print("🚀 启动服务器在端口 8000...")
    uvicorn.run(simple_app, host="127.0.0.1", port=8000, log_level="info")
    
except Exception as e:
    print(f"❌ 服务器启动失败: {e}")
    traceback.print_exc()
