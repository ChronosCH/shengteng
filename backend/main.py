"""
手语学习训练系统 - 集成版主应用
整合手语识别与学习训练功能的完整后端服务
"""

import logging
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

# 将仓库根目录加入 sys.path，避免相对导入问题
import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 新增导入
from backend.utils.file_manager import FileManager

# 导入学习训练服务
try:
    from backend.services.learning_training_service import LearningTrainingService
    from backend.api.learning_routes import router as learning_router
    from backend.services import IsolatedSignService
    LEARNING_AVAILABLE = True
    logger.info("✅ 学习训练功能已导入")
except ImportError as e:
    logger.info(f"ℹ️ 学习训练模块未启用: {e}")
    LEARNING_AVAILABLE = False

# 导入连续手语识别服务
MediaPipeService = None
try:
    from backend.services.sign_recognition_service import SignRecognitionService
    from backend.services.cslr_service import CSLRService

    try:
        from backend.services.mediapipe_service import MediaPipeService as _MediaPipeService
        MediaPipeService = _MediaPipeService
    except ImportError:
        logger.info("ℹ️ 未检测到 mediapipe，连续手语识别将使用 Mind-VAC 管线")

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
    """改进的应用生命周期管理"""
    from backend.core.service_manager import service_manager, default_health_check
    from backend.core.config_manager import get_config
    from backend.utils.file_manager import FileManager

    logger.info("🚀 启动手语学习训练系统...")

    try:
        # 获取配置
        config = get_config()

        # 注册文件管理器
        service_manager.register_service(
            "file_manager",
            FileManager,
            health_check=default_health_check
        )

        # 注册MediaPipe服务
        if SIGN_RECOGNITION_AVAILABLE:
            if MediaPipeService is not None:
                service_manager.register_service(
                    "mediapipe_service",
                    MediaPipeService,
                    health_check=default_health_check
                )

            # 注册CSLR服务
            service_manager.register_service(
                "cslr_service",
                CSLRService,
                health_check=default_health_check
            )

            # 注册手语识别服务
            def create_sign_recognition_service():
                mediapipe_svc = None
                if MediaPipeService is not None:
                    try:
                        mediapipe_svc = service_manager.get_service("mediapipe_service")
                    except Exception:
                        mediapipe_svc = None
                cslr_svc = service_manager.get_service("cslr_service")
                return SignRecognitionService(mediapipe_svc, cslr_svc)

            dependencies = ["cslr_service"]
            if MediaPipeService is not None:
                dependencies.append("mediapipe_service")

            service_manager.register_service(
                "sign_recognition_service",
                lambda: create_sign_recognition_service(),
                dependencies=dependencies,
                health_check=default_health_check
            )

        # 注册学习训练服务
        if LEARNING_AVAILABLE:
            service_manager.register_service(
                "learning_service",
                LearningTrainingService,
                health_check=default_health_check
            )

            iso_cfg = config.isolated_sign
            if getattr(iso_cfg, "enabled", False):
                try:
                    iso_config_dict = {
                        "sequence_length": iso_cfg.sequence_length,
                        "use_gpu": iso_cfg.use_gpu,
                        "device_id": iso_cfg.device_id,
                        "class_mapping_path": getattr(iso_cfg, "class_mapping_path", None),
                    }

                    service_manager.register_service(
                        "isolated_sign_service",
                        lambda: IsolatedSignService(
                            model_checkpoint=iso_cfg.model_path,
                            train_csv_path=iso_cfg.train_csv,
                            config=iso_config_dict,
                        ),
                        health_check=default_health_check,
                    )
                except Exception as iso_exc:
                    logger.warning(f"⚠️ 孤立手语服务注册失败: {iso_exc}")

        # 启动所有服务
        success = await service_manager.start_all_services()
        if not success:
            raise Exception("部分服务启动失败")

        # 将服务注册到app.state
        app.state.service_manager = service_manager
        app.state.config = config

        # 为了向后兼容，保留原有的访问方式
        try:
            app.state.file_manager = service_manager.get_service("file_manager")
        except:
            app.state.file_manager = None

        try:
            app.state.sign_recognition_service = service_manager.get_service("sign_recognition_service")
        except:
            app.state.sign_recognition_service = None

        try:
            app.state.learning_service = service_manager.get_service("learning_service")
        except:
            app.state.learning_service = None

        try:
            app.state.isolated_sign_service = service_manager.get_service("isolated_sign_service")
        except:
            app.state.isolated_sign_service = None

        logger.info("✅ 系统初始化完成")
        yield

    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        raise
    finally:
        # 清理资源
        logger.info("🔄 正在关闭服务...")
        try:
            await service_manager.stop_all_services()
        except Exception as e:
                logger.warning(f"学习训练服务关闭钩子执行失败: {e}")
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

# 确保字幕输出目录存在并挂载为静态资源
try:
    import os as _os
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    _os.makedirs("temp/sign_results", exist_ok=True)
    app.mount("/sign_results", _StaticFiles(directory="temp/sign_results"), name="sign_results")
except Exception as _e:
    logger.warning(f"挂载字幕静态目录失败: {_e}")

# 安全中间件
try:
    from backend.middleware.security_headers import security_headers_middleware
    from backend.middleware.rate_limiting import rate_limit_middleware

    # 添加安全头中间件
    app.middleware("http")(security_headers_middleware)

    # 添加速率限制中间件
    app.middleware("http")(rate_limit_middleware)

    logger.info("✅ 安全中间件已加载")
except ImportError as e:
    logger.warning(f"⚠️ 安全中间件加载失败: {e}")

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册认证路由
try:
    from backend.api.auth_routes import router as auth_router
    app.include_router(auth_router, tags=["认证"])
    logger.info("✅ 认证路由已注册")
except ImportError as e:
    logger.info(f"ℹ️ 认证路由未启用: {e}")

# 注册 Mind-VAC 实时识别路由
try:
    from backend.api.mindvac_routes import router as mindvac_router
    app.include_router(mindvac_router)
    logger.info("✅ Mind-VAC 路由已注册")
except ImportError as e:
    logger.warning(f"⚠️ Mind-VAC 路由注册失败: {e}")

# 注册学习训练路由
if LEARNING_AVAILABLE:
    app.include_router(learning_router, prefix="/api/learning", tags=["学习训练"])

# 注册系统管理路由
try:
    from backend.api.system_routes import router as system_router
    app.include_router(system_router)
    logger.info("✅ 系统管理路由已注册")
except ImportError as e:
    logger.warning(f"⚠️ 系统管理路由注册失败: {e}")

# 注册LLM对话路由
try:
    from backend.api.llm_routes import router as llm_router
    app.include_router(llm_router)
    logger.info("✅ LLM对话路由已注册")
except ImportError as e:
    logger.warning(f"⚠️ LLM对话路由注册失败: {e}")

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
    learning_svc = getattr(app.state, "learning_service", None)
    recognition_svc = getattr(app.state, "sign_recognition_service", None)
    
    learning_status = "✅ 可用" if LEARNING_AVAILABLE and learning_svc else "❌ 不可用"
    recognition_status = "✅ 可用" if SIGN_RECOGNITION_AVAILABLE and recognition_svc else "❌ 不可用"
    
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
    learning_svc = getattr(app.state, "learning_service", None)
    recognition_svc = getattr(app.state, "sign_recognition_service", None)
    
    services_status = {
        "learning_training": "ready" if LEARNING_AVAILABLE and learning_svc else "not_available",
        "sign_recognition": "ready" if SIGN_RECOGNITION_AVAILABLE and recognition_svc else "not_available",
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
        learning_svc = getattr(app.state, "learning_service", None)
        recognition_svc = getattr(app.state, "sign_recognition_service", None)
        
        status_info = {
            "status": "active",
            "timestamp": time.time(),
            "services": {
                "learning_training": LEARNING_AVAILABLE and learning_svc is not None,
                "sign_recognition": SIGN_RECOGNITION_AVAILABLE and recognition_svc is not None,
                "file_manager": True
            }
        }
        
        # 添加学习服务统计
        if LEARNING_AVAILABLE and learning_svc:
            try:
                learning_stats = await learning_svc.get_system_stats()
                status_info["learning_stats"] = learning_stats
            except Exception as e:
                logger.warning(f"获取学习统计失败: {e}")
        
        return status_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"状态检查失败: {str(e)}")

# 新增：WebSocket 实时识别端点，供前端 websocketService 连接
@app.websocket("/ws/sign-recognition")
async def ws_sign_recognition(websocket: WebSocket):
    """实时手语识别 WebSocket
    接收消息格式:
    - {"type":"landmarks","payload":{"landmarks": number[][], "timestamp": number, "frameId": number}}
    - {"type":"batch","payload":{"messages": WebSocketMessage[] }}
    - {"type":"config","payload": { 配置项 }}
    响应消息:
    - {"type":"connection_established", payload }
    - {"type":"recognition_result","payload": { text, confidence, glossSequence, timestamp, frameId }}
    - {"type":"config_updated","payload": {}}
    - {"type":"error","payload": { message }}
    """
    await websocket.accept()

    # 发送连接确认
    try:
        await websocket.send_json({
            "type": "connection_established",
            "payload": {"timestamp": time.time()}
        })
    except Exception:
        pass

    # 获取推理服务
    cslr = None
    try:
        if getattr(app.state, "sign_recognition_service", None):
            cslr = app.state.sign_recognition_service.cslr_service
    except Exception:
        cslr = None

    if not (SIGN_RECOGNITION_AVAILABLE and cslr):
        await websocket.send_json({"type": "error", "payload": {"message": "连续手语识别服务不可用"}})
        await websocket.close()
        return

    last_pred_ts = 0.0
    min_interval = 0.3  # 最小推理间隔 (秒)
    min_frames = max(8, min(32, getattr(cslr.config, "max_sequence_length", 64) // 4))

    def _to_vec(points: List[List[float]]) -> List[float]:
        """将关键点数组转换为固定长度向量(543*3)，不足则零填充，超出则截断"""
        try:
            flat: List[float] = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    flat.extend([float(p[0]), float(p[1]), float(p[2])])
            # 统一到 543*3 维
            target = 543 * 3
            if len(flat) < target:
                flat.extend([0.0] * (target - len(flat)))
            elif len(flat) > target:
                flat = flat[:target]
            return flat
        except Exception:
            return []

    async def _handle_landmarks(payload: Dict):
        nonlocal last_pred_ts
        points = payload.get("landmarks")
        if not isinstance(points, list):
            return
        vec = _to_vec(points)
        if not vec:
            return
        # 追加到序列缓冲
        try:
            with cslr._buffer_lock:
                cslr.sequence_buffer.append(vec)
        except Exception:
            # 回退：直接维护本地缓冲（不建议，尽量使用服务内缓冲）
            pass
        now = time.time()
        # 满足帧数且到达推理间隔再进行推理
        if len(cslr.sequence_buffer) >= min_frames and (now - last_pred_ts) >= min_interval:
            seq = list(cslr.sequence_buffer)
            try:
                pred = await cslr.predict(seq)
                last_pred_ts = now
                if getattr(pred, "status", "success") == "success":
                    await websocket.send_json({
                        "type": "recognition_result",
                        "payload": {
                            "text": pred.text,
                            "confidence": float(pred.confidence),
                            "glossSequence": pred.gloss_sequence,
                            "timestamp": now,
                            "frameId": payload.get("frameId")
                        }
                    })
            except Exception as e:
                logger.warning(f"实时推理失败: {e}")

    async def _handle_config(cfg: Dict):
        # 支持动态更新部分配置
        try:
            if not cfg:
                return
            if "confidence_threshold" in cfg and isinstance(cfg["confidence_threshold"], (int, float)):
                cslr.config.confidence_threshold = float(cfg["confidence_threshold"])
            if "cache_size" in cfg and isinstance(cfg["cache_size"], int) and cslr.cache:
                cslr.cache.max_size = int(cfg["cache_size"])
            if "ctc_config" in cfg and isinstance(cfg["ctc_config"], dict):
                cslr.ctc_config.update({k: v for k, v in cfg["ctc_config"].items() if k in {"blank_id", "beam_width", "alpha", "beta"}})
            await websocket.send_json({"type": "config_updated", "payload": {}})
        except Exception as e:
            await websocket.send_json({"type": "error", "payload": {"message": f"配置更新失败: {e}"}})

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                continue
            mtype = data.get("type")
            payload = data.get("payload", {})
            if mtype == "landmarks":
                await _handle_landmarks(payload)
            elif mtype == "batch" and isinstance(payload, dict) and isinstance(payload.get("messages"), list):
                for m in payload.get("messages"):
                    if isinstance(m, dict) and m.get("type") == "landmarks":
                        await _handle_landmarks(m.get("payload", {}))
            elif mtype == "config":
                await _handle_config(payload)
            elif mtype == "ping":
                await websocket.send_json({"type": "pong", "payload": {"timestamp": time.time()}})
            else:
                # 忽略未知消息类型
                pass
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        try:
            await websocket.send_json({"type": "error", "payload": {"message": str(e)}})
        except Exception:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

# 新增：连续手语识别上传与状态查询端点
@app.post("/api/sign-recognition/upload-video", response_model=VideoUploadResponse)
async def sign_recognition_upload_video(file: UploadFile = File(...)):
    # 从 app.state 获取服务实例
    sign_recognition_svc = getattr(app.state, "sign_recognition_service", None)
    file_mgr = getattr(app.state, "file_manager", None)
    
    if not (SIGN_RECOGNITION_AVAILABLE and sign_recognition_svc):
        raise HTTPException(status_code=503, detail="连续手语识别服务不可用")
    if not file_mgr:
        raise HTTPException(status_code=503, detail="文件管理器未初始化")
    
    file_info = await file_mgr.save_file(file)
    if file_info.get("file_type") != "video":
        raise HTTPException(status_code=400, detail="请上传视频文件")
    
    task_id = await sign_recognition_svc.start_video_recognition(file_info["file_path"])
    return VideoUploadResponse(success=True, task_id=task_id, message="上传成功，任务已开始", status="uploaded")

@app.get("/api/sign-recognition/status/{task_id}", response_model=VideoStatusResponse)
async def sign_recognition_status(task_id: str):
    # 从 app.state 获取服务实例
    sign_recognition_svc = getattr(app.state, "sign_recognition_service", None)
    
    if not (SIGN_RECOGNITION_AVAILABLE and sign_recognition_svc):
        raise HTTPException(status_code=503, detail="连续手语识别服务不可用")
    
    task = await sign_recognition_svc.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return VideoStatusResponse(
        task_id=task_id,
        status=str(task.get("status", "processing")),
        progress=float(task.get("progress", 0.0)) if task.get("progress") is not None else None,
        result=task.get("result"),
        error=task.get("error"),
    )

if __name__ == "__main__":
    import socket
    import platform
    # 使用8000端口以匹配前端配置
    host = "127.0.0.1"
    port = 8000
    # Windows 下关闭 reload
    is_windows = platform.system().lower().startswith("win")
    reload_flag = False if is_windows else False

    def _run_uvicorn(h: str, p: int, reload_: bool):
        uvicorn.run(
            "backend.main:app",
            host=h,
            port=p,
            reload=reload_,
        )

    chosen_port = port
    _run_uvicorn(host, chosen_port, reload_flag)
