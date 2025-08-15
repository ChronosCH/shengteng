"""
SignAvatar Web Backend - Main FastAPI Application
实时手语识别与虚拟人播报系统后端服务
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api.websocket import WebSocketManager
from services.mediapipe_service import MediaPipeService
from services.cslr_service import CSLRService
from services.diffusion_slp_service import DiffusionSLPService, DiffusionConfig, EmotionType, SigningSpeed
from services.privacy_service import PrivacyService, AnonymizationConfig, AnonymizationLevel, DataType
from services.multimodal_sensor_service import MultimodalSensorService, SensorConfig, FusionMode
from services.haptic_service import HapticService, HapticPattern, HapticIntensity
from services.federated_learning_service import FederatedLearningService, ClientRole, AggregationMethod
from services.sign_recognition_service import SignRecognitionService
from utils.config import Settings
from utils.logger import setup_logger
from utils.database import db_manager
from utils.security import security_manager, UserToken
from utils.cache import cache_manager
from utils.monitoring import performance_monitor
from utils.file_manager import file_manager

# 配置日志
logger = setup_logger(__name__)

# 全局设置
settings = Settings()

# 全局服务实例
mediapipe_service: MediaPipeService = None
cslr_service: CSLRService = None
diffusion_slp_service: DiffusionSLPService = None
privacy_service: PrivacyService = None
multimodal_sensor_service: MultimodalSensorService = None
haptic_service: HapticService = None
federated_learning_service: FederatedLearningService = None
websocket_manager: WebSocketManager = None
default_sign_recognition_service: Optional[SignRecognitionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global mediapipe_service, cslr_service, diffusion_slp_service, privacy_service, multimodal_sensor_service, haptic_service, federated_learning_service, websocket_manager, default_sign_recognition_service

    logger.info("正在启动 SignAvatar Web 后端服务...")

    try:
        # 初始化基础设施服务
        await db_manager.initialize()
        await cache_manager.initialize()
        await performance_monitor.initialize()

        # 初始化AI服务
        mediapipe_service = MediaPipeService()
        cslr_service = CSLRService()
        diffusion_slp_service = DiffusionSLPService()
        privacy_service = PrivacyService()
        multimodal_sensor_service = MultimodalSensorService()
        haptic_service = HapticService()
        federated_learning_service = FederatedLearningService()
        websocket_manager = WebSocketManager()

        # 预加载模型
        await cslr_service.load_model()
        await diffusion_slp_service.initialize()
        await privacy_service.initialize()
        await multimodal_sensor_service.initialize()
        await haptic_service.initialize()
        await federated_learning_service.initialize()

        # 初始化手语识别服务
        sign_recognition_service = SignRecognitionService(mediapipe_service, cslr_service)
        default_sign_recognition_service = sign_recognition_service

        logger.info("所有服务初始化完成")
        yield

    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        raise
    finally:
        # 清理资源
        logger.info("正在关闭服务...")
        
        # 清理AI服务
        if cslr_service:
            await cslr_service.cleanup()
        if diffusion_slp_service:
            await diffusion_slp_service.cleanup()
        if privacy_service:
            await privacy_service.cleanup()
        if multimodal_sensor_service:
            await multimodal_sensor_service.cleanup()
        if haptic_service:
            await haptic_service.cleanup()
        if federated_learning_service:
            await federated_learning_service.cleanup()
        if mediapipe_service:
            await mediapipe_service.cleanup()
        if default_sign_recognition_service:
            await default_sign_recognition_service.cleanup()
        
        # 清理基础设施服务
        await performance_monitor.cleanup()
        await cache_manager.cleanup()
        await db_manager.cleanup()


# 创建FastAPI应用
app = FastAPI(
    title="SignAvatar Web API",
    description="实时手语识别与虚拟人播报系统 API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class HealthResponse(BaseModel):
    status: str
    message: str
    services: Dict[str, str]


class LandmarkData(BaseModel):
    landmarks: List[List[float]]
    timestamp: float
    frame_id: int


class DiffusionGenerationRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    speed: str = "normal"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None


class DiffusionGenerationResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


class AnonymizationRequest(BaseModel):
    data_type: str  # "video", "image", "landmarks"
    level: str = "medium"  # "low", "medium", "high"
    preserve_gesture: bool = True
    preserve_expression: bool = False
    blur_background: bool = True
    add_noise: bool = True
    seed: Optional[int] = None


class AnonymizationResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
    metrics: Optional[Dict] = None


class SensorConfigRequest(BaseModel):
    emg_enabled: bool = True
    imu_enabled: bool = True
    visual_enabled: bool = True
    fusion_mode: str = "early"  # "early", "late", "hybrid"


class MultimodalPredictionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


class HapticMessageRequest(BaseModel):
    text: str
    use_braille: bool = True
    use_haptic: bool = True


class HapticSemanticRequest(BaseModel):
    semantic_type: str
    intensity: str = "medium"  # "low", "medium", "high"


class HapticResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


class FederatedTrainingRequest(BaseModel):
    config: Optional[Dict] = None


class ExplanationRequest(BaseModel):
    input_data: List[List[List[float]]]  # 关键点数据
    prediction: Dict


class FederatedResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


class UserRegistrationRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    preferences: Optional[Dict] = None
    accessibility_settings: Optional[Dict] = None


class UserLoginRequest(BaseModel):
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserFeedbackRequest(BaseModel):
    feedback_type: str
    content: str
    rating: Optional[int] = None
    recognition_accuracy: Optional[float] = None
    suggestions: Optional[str] = None


class UserStatsResponse(BaseModel):
    success: bool
    data: Optional[Dict] = None


class FileUploadResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


class FileDeleteRequest(BaseModel):
    file_hash: str


class VideoRecognitionStartResponse(BaseModel):
    success: bool
    task_id: Optional[str] = None
    message: str

class VideoRecognitionStatusResponse(BaseModel):
    status: str
    progress: Optional[float] = None
    error: Optional[str] = None

class VideoRecognitionResultResponse(BaseModel):
    status: str
    result: Optional[Dict] = None


# API路由
@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 返回简单的状态页面"""
    return """
    <html>
        <head>
            <title>SignAvatar Web API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .status { color: #4CAF50; font-weight: bold; }
                .info { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <h1>🤖 SignAvatar Web API</h1>
            <p class="status">✅ 服务运行正常</p>
            <div class="info">
                <h3>可用端点:</h3>
                <ul>
                    <li><a href="/api/docs">API 文档 (Swagger)</a></li>
                    <li><a href="/api/health">健康检查</a></li>
                    <li><a href="/ws/sign-recognition">WebSocket 连接</a></li>
                </ul>
            </div>
        </body>
    </html>
    """


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    services_status = {
        "mediapipe": "ready" if mediapipe_service else "not_initialized",
        "cslr": "ready" if cslr_service and cslr_service.is_loaded else "not_loaded",
        "diffusion_slp": "ready" if diffusion_slp_service and diffusion_slp_service.is_loaded else "not_loaded",
        "privacy": "ready" if privacy_service and privacy_service.is_loaded else "not_loaded",
        "multimodal_sensor": "ready" if multimodal_sensor_service and multimodal_sensor_service.is_loaded else "not_loaded",
        "haptic": "ready" if haptic_service and haptic_service.is_loaded else "not_loaded",
        "federated_learning": "ready" if federated_learning_service and federated_learning_service.is_loaded else "not_loaded",
        "websocket": "ready" if websocket_manager else "not_initialized",
    }

    all_ready = all(status == "ready" for status in services_status.values())

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        message="所有服务正常运行" if all_ready else "部分服务未就绪",
        services=services_status
    )


@app.post("/api/diffusion/generate", response_model=DiffusionGenerationResponse)
async def generate_sign_sequence(request: DiffusionGenerationRequest):
    """生成手语序列端点"""
    if not diffusion_slp_service or not diffusion_slp_service.is_loaded:
        raise HTTPException(status_code=503, detail="Diffusion SLP 服务未就绪")

    try:
        # 解析参数
        emotion = EmotionType(request.emotion)
        speed = SigningSpeed(request.speed)

        # 创建配置
        config = DiffusionConfig(
            emotion=emotion,
            speed=speed,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )

        # 生成手语序列
        result = await diffusion_slp_service.generate_sign_sequence(request.text, config)

        # 转换为可序列化的格式
        response_data = {
            "keypoints": result.keypoints.tolist(),
            "timestamps": result.timestamps.tolist(),
            "confidence": result.confidence,
            "emotion": result.emotion.value,
            "speed": result.speed.value,
            "text": result.text,
            "duration": result.duration,
            "num_frames": len(result.keypoints)
        }

        return DiffusionGenerationResponse(
            success=True,
            message="手语序列生成成功",
            data=response_data
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        logger.error(f"手语序列生成失败: {e}")
        return DiffusionGenerationResponse(
            success=False,
            message=f"生成失败: {str(e)}"
        )


@app.get("/api/diffusion/stats")
async def get_diffusion_stats():
    """获取 Diffusion 服务统计信息"""
    if not diffusion_slp_service:
        raise HTTPException(status_code=503, detail="Diffusion SLP 服务未初始化")

    stats = await diffusion_slp_service.get_stats()
    return {"success": True, "data": stats}


@app.post("/api/diffusion/clear-cache")
async def clear_diffusion_cache():
    """清空 Diffusion 服务缓存"""
    if not diffusion_slp_service:
        raise HTTPException(status_code=503, detail="Diffusion SLP 服务未初始化")

    await diffusion_slp_service.clear_cache()
    return {"success": True, "message": "缓存已清空"}


@app.post("/api/privacy/anonymize-image", response_model=AnonymizationResponse)
async def anonymize_image_data(request: AnonymizationRequest):
    """匿名化图像数据端点"""
    if not privacy_service or not privacy_service.is_loaded:
        raise HTTPException(status_code=503, detail="隐私保护服务未就绪")

    try:
        # 解析参数
        level = AnonymizationLevel(request.level)
        data_type = DataType(request.data_type)

        # 创建配置
        config = AnonymizationConfig(
            level=level,
            preserve_gesture=request.preserve_gesture,
            preserve_expression=request.preserve_expression,
            blur_background=request.blur_background,
            add_noise=request.add_noise,
            seed=request.seed
        )

        # 注意：这里需要从请求中获取实际的图像数据
        # 为了演示，我们创建一个模拟图像
        import numpy as np
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 执行匿名化
        anonymized_data, metrics = await privacy_service.anonymize_data(
            mock_image, data_type, config
        )

        # 转换为可序列化的格式
        response_data = {
            "anonymized_shape": anonymized_data.shape,
            "original_shape": mock_image.shape,
            "processing_completed": True
        }

        metrics_data = {
            "anonymization_score": metrics.anonymization_score,
            "utility_score": metrics.utility_score,
            "processing_time": metrics.processing_time,
            "data_size_reduction": metrics.data_size_reduction
        }

        return AnonymizationResponse(
            success=True,
            message="数据匿名化成功",
            data=response_data,
            metrics=metrics_data
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        logger.error(f"数据匿名化失败: {e}")
        return AnonymizationResponse(
            success=False,
            message=f"匿名化失败: {str(e)}"
        )


@app.get("/api/privacy/stats")
async def get_privacy_stats():
    """获取隐私保护服务统计信息"""
    if not privacy_service:
        raise HTTPException(status_code=503, detail="隐私保护服务未初始化")

    stats = await privacy_service.get_stats()
    return {"success": True, "data": stats}


@app.post("/api/privacy/clear-cache")
async def clear_privacy_cache():
    """清空隐私保护服务缓存"""
    if not privacy_service:
        raise HTTPException(status_code=503, detail="隐私保护服务未初始化")

    await privacy_service.clear_cache()
    return {"success": True, "message": "缓存已清空"}


@app.post("/api/multimodal/start-collection")
async def start_sensor_collection():
    """开始多模态传感器数据收集"""
    if not multimodal_sensor_service or not multimodal_sensor_service.is_loaded:
        raise HTTPException(status_code=503, detail="多模态传感器服务未就绪")

    try:
        await multimodal_sensor_service.start_collection()
        return {"success": True, "message": "数据收集已开始"}
    except Exception as e:
        logger.error(f"启动数据收集失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@app.post("/api/multimodal/stop-collection")
async def stop_sensor_collection():
    """停止多模态传感器数据收集"""
    if not multimodal_sensor_service:
        raise HTTPException(status_code=503, detail="多模态传感器服务未初始化")

    try:
        await multimodal_sensor_service.stop_collection()
        return {"success": True, "message": "数据收集已停止"}
    except Exception as e:
        logger.error(f"停止数据收集失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")


@app.post("/api/multimodal/predict", response_model=MultimodalPredictionResponse)
async def multimodal_predict():
    """多模态融合预测"""
    if not multimodal_sensor_service or not multimodal_sensor_service.is_loaded:
        raise HTTPException(status_code=503, detail="多模态传感器服务未就绪")

    try:
        result = await multimodal_sensor_service.predict_multimodal()

        return MultimodalPredictionResponse(
            success=True,
            message="多模态预测成功",
            data=result
        )

    except Exception as e:
        logger.error(f"多模态预测失败: {e}")
        return MultimodalPredictionResponse(
            success=False,
            message=f"预测失败: {str(e)}"
        )


@app.post("/api/multimodal/config")
async def update_sensor_config(request: SensorConfigRequest):
    """更新传感器配置"""
    if not multimodal_sensor_service:
        raise HTTPException(status_code=503, detail="多模态传感器服务未初始化")

    try:
        config_dict = {
            "emg_enabled": request.emg_enabled,
            "imu_enabled": request.imu_enabled,
            "visual_enabled": request.visual_enabled,
            "fusion_mode": request.fusion_mode
        }

        await multimodal_sensor_service.update_config(config_dict)
        return {"success": True, "message": "配置更新成功"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"配置错误: {str(e)}")
    except Exception as e:
        logger.error(f"配置更新失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@app.get("/api/multimodal/stats")
async def get_multimodal_stats():
    """获取多模态传感器统计信息"""
    if not multimodal_sensor_service:
        raise HTTPException(status_code=503, detail="多模态传感器服务未初始化")

    stats = await multimodal_sensor_service.get_stats()
    return {"success": True, "data": stats}


@app.post("/api/haptic/send-message", response_model=HapticResponse)
async def send_haptic_message(request: HapticMessageRequest):
    """发送触觉消息"""
    if not haptic_service or not haptic_service.is_loaded:
        raise HTTPException(status_code=503, detail="触觉反馈服务未就绪")

    try:
        message = await haptic_service.send_haptic_message(
            text=request.text,
            use_braille=request.use_braille,
            use_haptic=request.use_haptic
        )

        response_data = {
            "message_id": id(message),
            "text": message.text,
            "total_duration": message.total_duration,
            "haptic_commands": len(message.commands),
            "braille_cells": len(message.braille_cells),
            "timestamp": message.timestamp
        }

        return HapticResponse(
            success=True,
            message="触觉消息发送成功",
            data=response_data
        )

    except Exception as e:
        logger.error(f"发送触觉消息失败: {e}")
        return HapticResponse(
            success=False,
            message=f"发送失败: {str(e)}"
        )


@app.post("/api/haptic/send-semantic", response_model=HapticResponse)
async def send_semantic_feedback(request: HapticSemanticRequest):
    """发送语义触觉反馈"""
    if not haptic_service or not haptic_service.is_loaded:
        raise HTTPException(status_code=503, detail="触觉反馈服务未就绪")

    try:
        await haptic_service.send_semantic_feedback(
            semantic_type=request.semantic_type,
            intensity=request.intensity
        )

        return HapticResponse(
            success=True,
            message="语义反馈发送成功",
            data={"semantic_type": request.semantic_type, "intensity": request.intensity}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        logger.error(f"发送语义反馈失败: {e}")
        return HapticResponse(
            success=False,
            message=f"发送失败: {str(e)}"
        )


@app.post("/api/haptic/emergency-alert")
async def send_emergency_alert():
    """发送紧急警报"""
    if not haptic_service or not haptic_service.is_loaded:
        raise HTTPException(status_code=503, detail="触觉反馈服务未就绪")

    try:
        await haptic_service.send_emergency_alert()
        return {"success": True, "message": "紧急警报已发送"}

    except Exception as e:
        logger.error(f"发送紧急警报失败: {e}")
        raise HTTPException(status_code=500, detail=f"发送失败: {str(e)}")


@app.post("/api/haptic/stop-playback")
async def stop_haptic_playback():
    """停止触觉播放"""
    if not haptic_service:
        raise HTTPException(status_code=503, detail="触觉反馈服务未初始化")

    try:
        await haptic_service.stop_playback()
        return {"success": True, "message": "触觉播放已停止"}

    except Exception as e:
        logger.error(f"停止触觉播放失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")


@app.get("/api/haptic/stats")
async def get_haptic_stats():
    """获取触觉反馈统计信息"""
    if not haptic_service:
        raise HTTPException(status_code=503, detail="触觉反馈服务未初始化")

    stats = await haptic_service.get_stats()
    return {"success": True, "data": stats}


@app.post("/api/haptic/test-devices")
async def test_haptic_devices():
    """测试触觉设备"""
    if not haptic_service or not haptic_service.is_loaded:
        raise HTTPException(status_code=503, detail="触觉反馈服务未就绪")

    try:
        results = await haptic_service.test_devices()
        return {"success": True, "data": results}

    except Exception as e:
        logger.error(f"设备测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"测试失败: {str(e)}")


@app.post("/api/federated/start-training", response_model=FederatedResponse)
async def start_federated_training(request: FederatedTrainingRequest):
    """开始联邦学习训练"""
    if not federated_learning_service or not federated_learning_service.is_loaded:
        raise HTTPException(status_code=503, detail="联邦学习服务未就绪")

    try:
        success = await federated_learning_service.start_federated_training(request.config)

        if success:
            # 获取最新更新
            latest_update = await federated_learning_service.get_latest_update()

            response_data = {
                "training_started": True,
                "round_number": federated_learning_service.current_round,
                "latest_update": {
                    "loss": latest_update.loss if latest_update else None,
                    "accuracy": latest_update.accuracy if latest_update else None,
                    "privacy_noise": latest_update.privacy_noise if latest_update else None
                } if latest_update else None
            }

            return FederatedResponse(
                success=True,
                message="联邦学习训练已开始",
                data=response_data
            )
        else:
            return FederatedResponse(
                success=False,
                message="联邦学习训练启动失败"
            )

    except Exception as e:
        logger.error(f"启动联邦学习训练失败: {e}")
        return FederatedResponse(
            success=False,
            message=f"启动失败: {str(e)}"
        )


@app.post("/api/federated/generate-explanation", response_model=FederatedResponse)
async def generate_model_explanation(request: ExplanationRequest):
    """生成模型解释"""
    if not federated_learning_service or not federated_learning_service.is_loaded:
        raise HTTPException(status_code=503, detail="联邦学习服务未就绪")

    try:
        # 转换输入数据
        input_data = np.array(request.input_data, dtype=np.float32)

        # 生成解释
        explanation = await federated_learning_service.generate_explanation(
            input_data, request.prediction
        )

        # 转换为可序列化的格式
        response_data = {
            "saliency_maps": {k: v.tolist() for k, v in explanation.saliency_maps.items()},
            "attention_weights": {k: v.tolist() for k, v in explanation.attention_weights.items()},
            "feature_importance": explanation.feature_importance,
            "prediction_confidence": explanation.prediction_confidence,
            "explanation_confidence": explanation.explanation_confidence
        }

        return FederatedResponse(
            success=True,
            message="模型解释生成成功",
            data=response_data
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        logger.error(f"生成模型解释失败: {e}")
        return FederatedResponse(
            success=False,
            message=f"生成失败: {str(e)}"
        )


@app.get("/api/federated/stats")
async def get_federated_stats():
    """获取联邦学习统计信息"""
    if not federated_learning_service:
        raise HTTPException(status_code=503, detail="联邦学习服务未初始化")

    stats = await federated_learning_service.get_federated_stats()
    return {"success": True, "data": stats}


@app.get("/api/federated/explanation-summary")
async def get_explanation_summary():
    """获取解释摘要"""
    if not federated_learning_service:
        raise HTTPException(status_code=503, detail="联邦学习服务未初始化")

    summary = await federated_learning_service.get_explanation_summary()
    return {"success": True, "data": summary}


@app.websocket("/ws/sign-recognition")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点 - 处理实时手语识别"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # 接收客户端数据
            data = await websocket.receive_json()
            
            # 处理不同类型的消息
            if data.get("type") == "landmarks":
                await handle_landmarks_message(websocket, data)
            elif data.get("type") == "config":
                await handle_config_message(websocket, data)
            elif data.get("type") == "multimodal_predict":
                await handle_multimodal_predict_message(websocket, data)
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"未知消息类型: {data.get('type')}"
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"服务器错误: {str(e)}"
        })
    finally:
        websocket_manager.disconnect(websocket)


async def handle_landmarks_message(websocket: WebSocket, data: dict):
    """处理关键点数据消息"""
    try:
        landmark_data = LandmarkData(**data["payload"])
        
        # 使用CSLR服务进行推理
        result = await cslr_service.predict(landmark_data.landmarks)

        # 如果识别到文本且触觉服务可用，发送触觉反馈
        recognized_text = result.get("text", "")
        if recognized_text and haptic_service and haptic_service.is_loaded:
            try:
                await haptic_service.send_haptic_message(recognized_text)
            except Exception as e:
                logger.warning(f"发送触觉反馈失败: {e}")

        # 发送识别结果
        await websocket.send_json({
            "type": "recognition_result",
            "payload": {
                "text": recognized_text,
                "confidence": result.get("confidence", 0.0),
                "gloss_sequence": result.get("gloss_sequence", []),
                "timestamp": landmark_data.timestamp,
                "frame_id": landmark_data.frame_id
            }
        })
        
    except Exception as e:
        logger.error(f"处理关键点数据失败: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"处理关键点数据失败: {str(e)}"
        })


async def handle_multimodal_predict_message(websocket: WebSocket, data: dict):
    """处理多模态预测消息"""
    try:
        # 如果有视觉关键点数据，添加到多模态服务
        if "landmarks" in data.get("payload", {}):
            landmarks = np.array(data["payload"]["landmarks"])
            await multimodal_sensor_service.add_visual_landmarks(landmarks)

        # 执行多模态预测
        result = await multimodal_sensor_service.predict_multimodal()

        # 发送预测结果
        await websocket.send_json({
            "type": "multimodal_prediction",
            "payload": result
        })

    except Exception as e:
        logger.error(f"多模态预测失败: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"多模态预测失败: {str(e)}"
        })


async def handle_config_message(websocket: WebSocket, data: dict):
    """处理配置消息"""
    try:
        config = data.get("payload", {})
        
        # 更新服务配置
        if "model_config" in config:
            await cslr_service.update_config(config["model_config"])
        
        await websocket.send_json({
            "type": "config_updated",
            "payload": {"status": "success"}
        })
        
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"更新配置失败: {str(e)}"
        })


# 用户认证和管理API路由
@app.post("/api/auth/register", response_model=UserToken)
async def register_user(request: UserRegistrationRequest, client_request: Request):
    """用户注册"""
    try:
        # 检查速率限制
        await security_manager.check_rate_limit(client_request, limit=5, window=300)  # 5次/5分钟
        
        # 创建用户
        user_id = await db_manager.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            preferences=request.preferences or {},
            accessibility_settings=request.accessibility_settings or {}
        )
        
        # 自动登录
        device_info = {
            "user_agent": client_request.headers.get("user-agent", ""),
            "platform": "web"
        }
        
        token_response = await security_manager.login_user(
            username=request.username,
            password=request.password,
            device_info=device_info,
            ip_address=client_request.client.host
        )
        
        logger.info(f"新用户注册成功: {request.username}")
        return token_response
        
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"注册失败: {str(e)}"
        )


@app.post("/api/auth/login", response_model=UserToken)
async def login_user(request: UserLoginRequest, client_request: Request):
    """用户登录"""
    try:
        # 检查速率限制
        await security_manager.check_rate_limit(client_request, limit=10, window=300)  # 10次/5分钟
        
        device_info = {
            "user_agent": client_request.headers.get("user-agent", ""),
            "platform": "web"
        }
        
        token_response = await security_manager.login_user(
            username=request.username,
            password=request.password,
            device_info=device_info,
            ip_address=client_request.client.host
        )
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录过程中发生错误"
        )


@app.post("/api/auth/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """刷新访问令牌"""
    try:
        return await security_manager.refresh_access_token(request.refresh_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刷新令牌失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="刷新令牌失败"
        )


@app.post("/api/auth/logout")
async def logout_user(current_user: dict = Depends(security_manager.get_current_user)):
    """用户登出"""
    try:
        # 从令牌中获取会话ID（这需要在令牌验证时设置）
        session_id = current_user.get("session_id")
        if session_id:
            await security_manager.logout_user(session_id)
        
        return {"success": True, "message": "登出成功"}
        
    except Exception as e:
        logger.error(f"用户登出失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出失败"
        )


@app.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(security_manager.get_current_active_user)):
    """获取当前用户信息"""
    return {
        "success": True,
        "data": {
            "id": current_user["id"],
            "username": current_user["username"],
            "email": current_user["email"],
            "full_name": current_user["full_name"],
            "is_admin": current_user["is_admin"],
            "preferences": current_user["preferences"],
            "accessibility_settings": current_user["accessibility_settings"]
        }
    }


@app.get("/api/user/stats", response_model=UserStatsResponse)
async def get_user_statistics(
    days: int = 30,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """获取用户统计信息"""
    try:
        stats = await db_manager.get_user_statistics(current_user["id"], days)
        return UserStatsResponse(success=True, data=stats)
        
    except Exception as e:
        logger.error(f"获取用户统计失败: {e}")
        return UserStatsResponse(success=False, data={"error": str(e)})


@app.post("/api/user/feedback")
async def submit_user_feedback(
    request: UserFeedbackRequest,
    session_id: str = None,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """提交用户反馈"""
    try:
        await db_manager.save_user_feedback(
            session_id=session_id or "unknown",
            user_id=current_user["id"],
            feedback_type=request.feedback_type,
            content=request.content,
            rating=request.rating,
            recognition_accuracy=request.recognition_accuracy,
            suggestions=request.suggestions
        )
        
        return {"success": True, "message": "反馈提交成功"}
        
    except Exception as e:
        logger.error(f"提交用户反馈失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="提交反馈失败"
        )


# 系统监控和管理API路由
@app.get("/api/admin/metrics")
async def get_system_metrics(current_user: dict = Depends(security_manager.require_admin)):
    """获取系统指标（管理员）"""
    try:
        metrics = performance_monitor.get_comprehensive_report()
        cache_stats = await cache_manager.get_stats()
        
        return {
            "success": True,
            "data": {
                "performance": metrics,
                "cache": cache_stats,
                "active_sessions": security_manager.get_active_sessions()
            }
        }
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统指标失败"
        )


@app.post("/api/admin/cache/clear")
async def clear_cache(
    namespace: Optional[str] = None,
    current_user: dict = Depends(security_manager.require_admin)
):
    """清理缓存（管理员）"""
    try:
        if namespace:
            deleted = await cache_manager.clear_namespace(namespace)
            message = f"清理命名空间 {namespace} 下的 {deleted} 个缓存项"
        else:
            # 清理所有缓存（需要实现）
            message = "清理所有缓存"
        
        return {"success": True, "message": message}
        
    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清理缓存失败"
        )


# 文件上传和管理API路由
@app.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """上传文件"""
    try:
        file_info = await file_manager.save_file(
            file=file,
            user_id=current_user["id"],
            metadata={"uploaded_by": current_user["username"]}
        )
        
        return FileUploadResponse(
            success=True,
            message="文件上传成功",
            data=file_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return FileUploadResponse(
            success=False,
            message=f"上传失败: {str(e)}"
        )


@app.post("/api/files/upload-video-for-recognition")
async def upload_video_for_recognition(
    file: UploadFile,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """上传视频文件进行手语识别"""
    try:
        # 验证是视频文件
        if not file.filename or not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="请上传视频文件 (mp4, avi, mov, mkv)"
            )
        
        # 保存文件
        file_info = await file_manager.save_file(
            file=file,
            user_id=current_user["id"],
            metadata={
                "purpose": "sign_recognition",
                "uploaded_by": current_user["username"]
            }
        )
        
        # 这里可以添加异步视频处理逻辑
        # 例如：提取关键点、进行识别等
        
        return {
            "success": True,
            "message": "视频上传成功，正在处理中",
            "data": {
                "file_hash": file_info["file_hash"],
                "file_path": file_info["file_path"],
                "processing_status": "queued"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"视频上传失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传失败: {str(e)}"
        )


@app.get("/api/files/{file_hash}")
async def get_file_info(
    file_hash: str,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """获取文件信息"""
    try:
        file_info = await file_manager.get_file_info(file_hash)
        
        if not file_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在"
            )
        
        # 检查权限（用户只能查看自己的文件或公开文件）
        if (file_info.get("user_id") != current_user["id"] and 
            not current_user.get("is_admin")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权访问此文件"
            )
        
        return {"success": True, "data": file_info}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文件信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文件信息失败"
        )


@app.delete("/api/files/{file_hash}")
async def delete_file(
    file_hash: str,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """删除文件"""
    try:
        success = await file_manager.delete_file(file_hash, current_user["id"])
        
        if success:
            return {"success": True, "message": "文件删除成功"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在或已被删除"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除文件失败"
        )


@app.get("/api/files/user/list")
async def list_user_files(
    current_user: dict = Depends(security_manager.get_current_active_user),
    file_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """获取用户文件列表"""
    try:
        # 这里应该从数据库查询用户文件
        # 暂时返回存储统计信息
        storage_stats = file_manager.get_storage_stats()
        
        return {
            "success": True,
            "data": {
                "files": [],  # 实际文件列表
                "storage_stats": storage_stats,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 0
                }
            }
        }
        
    except Exception as e:
        logger.error(f"获取用户文件列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文件列表失败"
        )


@app.get("/api/admin/files/stats")
async def get_file_storage_stats(
    current_user: dict = Depends(security_manager.require_admin)
):
    """获取文件存储统计（管理员）"""
    try:
        stats = file_manager.get_storage_stats()
        return {"success": True, "data": stats}
        
    except Exception as e:
        logger.error(f"获取存储统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取存储统计失败"
        )


@app.post("/api/admin/files/cleanup")
async def cleanup_temp_files(
    current_user: dict = Depends(security_manager.require_admin),
    max_age_hours: int = 24
):
    """清理临时文件（管理员）"""
    try:
        deleted_count = await file_manager.cleanup_temp_files(max_age_hours)
        return {
            "success": True,
            "message": f"清理了 {deleted_count} 个临时文件"
        }
        
    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清理临时文件失败"
        )


@app.post("/api/sign/start", response_model=VideoRecognitionStartResponse)
async def start_sign_recognition(
    file_hash: str,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    if not default_sign_recognition_service:
        raise HTTPException(status_code=503, detail="Sign recognition service unavailable")
    file_info = await file_manager.get_file_info(file_hash)
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    if file_info.get("user_id") != current_user["id"] and not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Permission denied")
    if file_info.get("file_type") != "video":
        raise HTTPException(status_code=400, detail="File is not a video")
    task_id = await default_sign_recognition_service.start_video_recognition(file_info["file_path"])
    return VideoRecognitionStartResponse(success=True, task_id=task_id, message="任务已启动")

@app.get("/api/sign/status/{task_id}", response_model=VideoRecognitionStatusResponse)
async def get_sign_status(task_id: str, current_user: dict = Depends(security_manager.get_current_active_user)):
    if not default_sign_recognition_service:
        raise HTTPException(status_code=503, detail="Sign recognition service unavailable")
    data = await default_sign_recognition_service.get_status(task_id)
    if data.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    return VideoRecognitionStatusResponse(status=data.get("status"), progress=data.get("progress"), error=data.get("error"))

@app.get("/api/sign/result/{task_id}", response_model=VideoRecognitionResultResponse)
async def get_sign_result(task_id: str, current_user: dict = Depends(security_manager.get_current_active_user)):
    if not default_sign_recognition_service:
        raise HTTPException(status_code=503, detail="Sign recognition service unavailable")
    result = await default_sign_recognition_service.get_result(task_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    return VideoRecognitionResultResponse(status=result.get("status"), result=result.get("result"))


# 挂载识别结果静态目录 (SRT 等)
if not os.path.exists("temp/sign_results"):
    os.makedirs("temp/sign_results", exist_ok=True)
app.mount("/sign_results", StaticFiles(directory="temp/sign_results"), name="sign_results")


if __name__ == "__main__":
    # 开发环境运行
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG,
    )
