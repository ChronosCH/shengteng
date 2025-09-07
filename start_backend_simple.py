"""
简单的后端启动脚本
"""

import sys
import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("🚀 开始启动后端服务器...")
        
        # 添加项目路径
        project_root = Path(__file__).resolve().parent
        backend_dir = project_root / "backend"
        
        logger.info(f"📁 项目根目录: {project_root}")
        logger.info(f"📁 后端目录: {backend_dir}")
        
        # 添加到Python路径
        for p in (str(project_root), str(backend_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
                logger.info(f"✅ 已添加路径: {p}")
        
        # 导入必要的模块
        logger.info("📦 导入模块...")
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        # 创建FastAPI应用
        logger.info("🏗️ 创建FastAPI应用...")
        app = FastAPI(
            title="Sign Language Learning API",
            description="手语学习训练系统API",
            version="1.0.0"
        )
        
        # 添加CORS中间件
        logger.info("🔧 配置CORS中间件...")
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
        
        # 添加基本路由
        @app.get("/")
        async def root():
            return {
                "message": "Sign Language Learning API is running",
                "status": "ok",
                "version": "1.0.0"
            }
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "sign-language-api"}
        
        # 尝试添加认证路由
        try:
            logger.info("🔐 注册认证路由...")
            from backend.api.auth_routes import router as auth_router
            app.include_router(auth_router, tags=["认证"])
            logger.info("✅ 认证路由注册成功")
        except Exception as e:
            logger.warning(f"⚠️ 认证路由注册失败: {e}")
            
            # 创建简单的测试认证路由
            from fastapi import APIRouter
            from pydantic import BaseModel
            
            test_auth_router = APIRouter(prefix="/api/auth", tags=["测试认证"])
            
            class TestRegisterRequest(BaseModel):
                username: str
                email: str
                password: str
                full_name: str = None
            
            class TestLoginRequest(BaseModel):
                username: str
                password: str
                remember_me: bool = False
            
            @test_auth_router.post("/register")
            async def test_register(user_data: TestRegisterRequest):
                logger.info(f"📝 测试注册请求: {user_data.username}")
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
            async def test_login(credentials: TestLoginRequest):
                logger.info(f"🔑 测试登录请求: {credentials.username}")
                return {
                    "success": True,
                    "message": "测试登录成功",
                    "data": {
                        "access_token": "test_token_123",
                        "token_type": "bearer",
                        "expires_in": 3600,
                        "refresh_token": "test_refresh_123",
                        "user_info": {
                            "id": 1,
                            "username": credentials.username,
                            "email": "test@example.com",
                            "full_name": "测试用户",
                            "is_active": True,
                            "is_admin": False,
                            "preferences": {},
                            "accessibility_settings": {}
                        }
                    }
                }
            
            app.include_router(test_auth_router, tags=["测试认证"])
            logger.info("✅ 测试认证路由注册成功")
        
        # 启动服务器
        logger.info("🚀 启动服务器...")
        logger.info("📍 服务器地址: http://127.0.0.1:8000")
        logger.info("📋 API文档: http://127.0.0.1:8000/docs")
        logger.info("🔧 健康检查: http://127.0.0.1:8000/health")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
