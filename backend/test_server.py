"""
简单的测试服务器，用于验证认证路由
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建简单的FastAPI应用
app = FastAPI(title="Sign Language Learning API Test", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加基本的健康检查路由
@app.get("/")
async def root():
    return {"message": "Sign Language Learning API is running", "status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sign-language-api"}

# 尝试导入和注册认证路由
try:
    from backend.api.auth_routes import router as auth_router
    app.include_router(auth_router, tags=["认证"])
    print("✅ 认证路由已注册")
except ImportError as e:
    print(f"⚠️ 认证路由注册失败: {e}")
    
    # 创建一个简单的测试认证路由
    from fastapi import APIRouter
    from pydantic import BaseModel
    
    test_auth_router = APIRouter(prefix="/api/auth", tags=["测试认证"])
    
    class TestRegisterRequest(BaseModel):
        username: str
        email: str
        password: str
        full_name: str = None
    
    @test_auth_router.post("/register")
    async def test_register(user_data: TestRegisterRequest):
        return {
            "success": True,
            "message": "测试注册成功",
            "data": {
                "user_id": 1,
                "username": user_data.username,
                "email": user_data.email
            }
        }
    
    @test_auth_router.post("/login")
    async def test_login(credentials: dict):
        return {
            "success": True,
            "message": "测试登录成功",
            "data": {
                "access_token": "test_token_123",
                "token_type": "bearer",
                "expires_in": 3600,
                "user_info": {
                    "id": 1,
                    "username": credentials.get("username", "test_user"),
                    "email": "test@example.com"
                }
            }
        }
    
    app.include_router(test_auth_router, tags=["测试认证"])
    print("✅ 测试认证路由已注册")

if __name__ == "__main__":
    print("🚀 启动测试服务器...")
    print("📍 服务器地址: http://127.0.0.1:8000")
    print("📋 API文档: http://127.0.0.1:8000/docs")
    print("🔧 健康检查: http://127.0.0.1:8000/health")
    
    uvicorn.run(
        "test_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
