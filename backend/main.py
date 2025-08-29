"""
手语学习训练系统 - 集成版主应用
整合手语识别与学习训练功能的完整后端服务
"""

import asyncio
import logging
import os
import json
import time
import uuid
import cv2
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, status, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 新增导入
from utils.file_manager import FileManager

# 导入学习训练服务
try:
    from services.learning_training_service import LearningTrainingService
    from api.learning_routes import router as learning_router
    LEARNING_AVAILABLE = True
    logger.info("✅ 学习训练功能已导入")
except ImportError as e:
    logger.warning(f"⚠️ 学习训练功能导入失败: {e}")
    LEARNING_AVAILABLE = False

# 导入连续手语识别服务
try:
    from services.sign_recognition_service import SignRecognitionService
    from services.mediapipe_service import MediaPipeService
    from services.cslr_service import CSLRService
    SIGN_RECOGNITION_AVAILABLE = True
    logger.info("✅ 连续手语识别功能已导入")
except ImportError as e:
    logger.warning(f"⚠️ 连续手语识别功能导入失败: {e}")
    SIGN_RECOGNITION_AVAILABLE = False

# 全局服务实例
# enhanced_cecsl_service = SimpleEnhancedCECSLService()
file_manager = None
learning_service = None
sign_recognition_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global learning_service, sign_recognition_service, file_manager
     
    logger.info("🚀 启动手语学习训练系统...")
    
    try:
        # 初始化文件管理器
        file_manager = FileManager()
        app.state.file_manager = file_manager

        # 初始化连续手语识别服务
        if SIGN_RECOGNITION_AVAILABLE:
            try:
                mediapipe_service = MediaPipeService()
                cslr_service = CSLRService()
                await cslr_service.load_model()
                sign_recognition_service = SignRecognitionService(mediapipe_service, cslr_service)
                app.state.sign_recognition_service = sign_recognition_service
                logger.info("✅ 连续手语识别服务初始化完成")
            except Exception as e:
                logger.error(f"❌ 连续手语识别服务初始化失败: {e}")
                sign_recognition_service = None
        else:
            logger.warning("⚠️ 连续手语识别服务不可用")
            sign_recognition_service = None
        
        # 初始化学习训练服务
        if LEARNING_AVAILABLE:
            learning_service = LearningTrainingService()
            await learning_service.initialize()
            app.state.learning_service = learning_service
            logger.info("✅ 学习训练服务初始化完成")
        else:
            logger.warning("⚠️ 学习训练服务不可用")
        
        logger.info("✅ 系统初始化完成")
        yield
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        raise
    finally:
        # 清理资源
        logger.info("🔄 正在关闭服务...")
        if learning_service:
            await learning_service.close()
        logger.info("✅ 服务关闭完成")

# 创建FastAPI应用
app = FastAPI(
    title="手语学习训练系统",
    description="集成手语识别与学习训练功能的完整系统",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册学习训练路由
if LEARNING_AVAILABLE:
    app.include_router(learning_router, prefix="/api/learning", tags=["学习训练"])

# 数据模型
class HealthResponse(BaseModel):
    status: str
    message: str
    services: Dict[str, str]

class LandmarkData(BaseModel):
    landmarks: List[List[float]]
    timestamp: float
    frame_id: int

class EnhancedCECSLTestRequest(BaseModel):
    landmarks: List[List[float]]
    description: Optional[str] = None

class EnhancedCECSLTestResponse(BaseModel):
    success: bool
    message: str
    prediction: Optional[Dict] = None
    stats: Optional[Dict] = None

class VideoUploadResponse(BaseModel):
    success: bool
    task_id: str
    message: str
    status: str = "uploaded"

class VideoStatusResponse(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "error"
    progress: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None

class UpdateCSLRConfig(BaseModel):
    confidence_threshold: Optional[float] = None
    ctc_config: Optional[Dict] = None
    cache_size: Optional[int] = None

# API路由
@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 返回系统状态页面"""
    learning_status = "✅ 可用" if LEARNING_AVAILABLE and learning_service else "❌ 不可用"
    recognition_status = "✅ 可用" if SIGN_RECOGNITION_AVAILABLE and sign_recognition_service else "❌ 不可用"
    
    return f"""
    <html>
        <head>
            <title>手语学习训练系统</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .status {{ color: #4CAF50; font-weight: bold; font-size: 18px; }}
                .info {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; }}
                .feature {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; }}
                h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
                h3 {{ color: #555; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                a {{ color: #667eea; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎓 手语学习训练系统</h1>
                <p class="status">🌟 服务运行正常</p>
                
                <div class="info">
                    <h3>🔧 系统状态</h3>
                    <div class="grid">
                        <div>
                            <strong>学习训练服务:</strong> {learning_status}<br>
                            <strong>连续手语识别:</strong> {recognition_status}
                        </div>
                        <div>
                            <strong>版本:</strong> 2.0.0
                        </div>
                    </div>
                </div>
                
                <div class="feature">
                    <h3>🎯 核心功能</h3>
                    <div class="grid">
                        <div>
                            • 系统化学习路径<br>
                            • 互动式手语练习<br>
                            • 实时进度跟踪
                        </div>
                        <div>
                            • 成就系统激励<br>
                            • 个性化推荐<br>
                            • 连续手语识别
                        </div>
                    </div>
                </div>
                
                <div class="info">
                    <h3>🌐 可用端点</h3>
                    <ul>
                        <li><a href="/api/docs">📚 API 文档 (Swagger)</a></li>
                        <li><a href="/api/health">💓 健康检查</a></li>
                        <li><a href="/api/learning/modules">📖 学习模块</a></li>
                        <li><a href="/ws/sign-recognition">🔗 WebSocket 连接</a></li>
                    </ul>
                </div>
                
                <div class="info">
                    <h3>🚀 快速开始</h3>
                    <p>1. 访问 <a href="http://localhost:5173/learning">学习平台</a> 开始学习</p>
                    <p>2. 使用 <code>/api/sign-recognition/upload-video</code> 上传视频进行识别</p>
                    <p>3. 通过 <code>/api/sign-recognition/status/任务ID</code> 查询结果</p>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    services_status = {
        "learning_training": "ready" if LEARNING_AVAILABLE and learning_service else "not_available",
        "sign_recognition": "ready" if SIGN_RECOGNITION_AVAILABLE and sign_recognition_service else "not_available",
        "file_manager": "ready",
    }

    all_ready = all(status == "ready" for status in services_status.values())
    partial_ready = any(status == "ready" for status in services_status.values())

    return HealthResponse(
        status="healthy" if all_ready else "partial" if partial_ready else "unhealthy",
        message="所有服务正常运行" if all_ready else "部分服务可用" if partial_ready else "服务异常",
        services=services_status
)

@app.get("/api/status")
async def api_status():
    """API状态检查"""
    try:
        status_info = {
            "status": "active",
            "timestamp": time.time(),
            "services": {
                "learning_training": LEARNING_AVAILABLE and learning_service is not None,
                "sign_recognition": SIGN_RECOGNITION_AVAILABLE and sign_recognition_service is not None,
                "file_manager": True
            }
        }
        
        # 添加学习服务统计
        if LEARNING_AVAILABLE and learning_service:
            try:
                learning_stats = await learning_service.get_system_stats()
                status_info["learning_stats"] = learning_stats
            except Exception as e:
                logger.warning(f"获取学习统计失败: {e}")
        
        return status_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"状态检查失败: {str(e)}")

# 下面四个旧的增强版CE-CSL接口已下线，统一提示迁移到新的连续识别接口
@app.post("/api/enhanced-cecsl/test")
async def deprecated_enhanced_test():
    raise HTTPException(status_code=410, detail="该接口已移除，请使用 /api/sign-recognition/upload-video 与 /api/sign-recognition/status/{task_id}")

@app.get("/api/enhanced-cecsl/stats")
async def deprecated_enhanced_stats():
    raise HTTPException(status_code=410, detail="该接口已移除，请使用 /api/sign-recognition/stats")

@app.post("/api/enhanced-cecsl/upload-video")
async def deprecated_enhanced_upload_video():
    raise HTTPException(status_code=410, detail="该接口已移除，请使用 /api/sign-recognition/upload-video")

@app.get("/api/enhanced-cecsl/video-status/{task_id}")
async def deprecated_enhanced_video_status(task_id: str):
    raise HTTPException(status_code=410, detail="该接口已移除，请使用 /api/sign-recognition/status/{task_id}")

# WebSocket端点 - 实时手语识别
@app.websocket("/ws/sign-recognition")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点用于实时手语识别"""
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    
    try:
        # 发送连接确认消息
        await websocket.send_json({
            "type": "connection_established",
            "payload": {
                "message": "连接成功",
                "server": "手语学习训练系统",
                "version": "2.0.0",
                "timestamp": time.time()
            }
        })
        
        while True:
            try:
                # 接收客户端消息
                data = await websocket.receive_json()
                message_type = data.get("type")
                payload = data.get("payload", {})
                
                if message_type == "landmarks":
                    # 实时关键点识别在当前版本未开放，提示使用视频上传接口
                    await websocket.send_json({
                        "type": "error",
                        "payload": {
                            "message": "实时关键点识别暂未开放，请使用 /api/sign-recognition/upload-video 进行连续句子识别",
                            "timestamp": time.time()
                        }
                    })
                elif message_type == "learning_progress":
                    # 处理学习进度更新
                    if LEARNING_AVAILABLE and learning_service:
                        try:
                            user_id = payload.get("user_id", "default")
                            progress_data = payload.get("progress", {})
                            
                            # 更新学习进度
                            await learning_service.update_user_progress(
                                user_id, 
                                progress_data.get("module_id"),
                                progress_data.get("lesson_id"), 
                                progress_data
                            )
                            
                            await websocket.send_json({
                                "type": "progress_updated",
                                "payload": {
                                    "message": "学习进度已更新",
                                    "timestamp": time.time()
                                }
                            })
                        except Exception as e:
                            logger.error(f"学习进度更新失败: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "payload": {
                                    "message": f"进度更新失败: {str(e)}",
                                    "timestamp": time.time()
                                }
                            })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "payload": {
                                "message": "学习服务不可用",
                                "timestamp": time.time()
                            }
                        })
                
                elif message_type == "config":
                    # 处理配置更新
                    logger.info(f"收到配置更新: {payload}")
                    await websocket.send_json({
                        "type": "config_updated",
                        "payload": {
                            "message": "配置已更新",
                            "timestamp": time.time()
                        }
                    })
                else:
                    logger.warning(f"未知消息类型: {message_type}")
                    
            except WebSocketDisconnect:
                logger.info("WebSocket客户端断开连接")
                break
            except Exception as e:
                logger.error(f"WebSocket处理消息错误: {e}")
                await websocket.send_json({
                    "type": "error",
                    "payload": {
                        "message": f"处理消息时发生错误: {str(e)}",
                        "timestamp": time.time()
                    }
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket连接断开")
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        logger.info("WebSocket连接已关闭")

# 简单的WebSocket测试端点
@app.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """简单的WebSocket测试端点"""
    try:
        await websocket.accept()
        logger.info("WebSocket测试连接已建立")
        
        await websocket.send_text("Hello from WebSocket!")
        
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"收到WebSocket消息: {data}")
                await websocket.send_text(f"Echo: {data}")
            except WebSocketDisconnect:
                logger.info("WebSocket测试连接断开")
                break
                
    except Exception as e:
        logger.error(f"WebSocket测试连接错误: {e}")

# 挂载静态文件目录
if not os.path.exists("uploads"):
    os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 允许直接运行该文件以启动服务
if __name__ == "__main__":
    import os
    
    # 使用环境变量 PORT 可覆盖默认端口
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    logger.info(f"启动服务器: http://{host}:{port}")
    logger.info(f"调试模式: {debug}")
    logger.info(f"连续手语识别: {'可用' if (SIGN_RECOGNITION_AVAILABLE and sign_recognition_service) else '不可用'}")
    
    # 运行 Uvicorn 服务器
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
        access_log=debug,
    )
