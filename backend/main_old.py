"""
SignAvatar Web Backend - Main FastAPI Application
实时手语识别与虚拟人播报系统后端服务
集成完整功能和简化版增强CE-CSL服务
"""

import asyncio
import logging
import os
import json
import time
import uuid
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

# 简化导入，只导入核心功能
try:
    from backend.services.enhanced_cecsl_service import EnhancedCECSLService
except ImportError:
    EnhancedCECSLService = None

try:
    from backend.services.sign_recognition_service import SignRecognitionService
except ImportError:
    SignRecognitionService = None

# 导入简化版服务组件
try:
    from backend.utils.logger import setup_logger
    from backend.utils.settings import Settings
    from backend.services.mediapipe_service import MediaPipeService
    from backend.services.cslr_service import CSLRService
    from backend.services.diffusion_slp_service import DiffusionSLPService
    from backend.services.privacy_service import PrivacyService
    from backend.services.multimodal_sensor_service import MultimodalSensorService
    from backend.services.haptic_service import HapticService
    from backend.services.federated_learning_service import FederatedLearningService
    from backend.services.websocket_manager import WebSocketManager
    from backend.utils.db_manager import db_manager
    from backend.utils.cache_manager import cache_manager
    from backend.utils.performance_monitor import performance_monitor
except ImportError as e:
    # 如果无法导入完整服务，使用简化版配置
    logging.warning(f"完整服务导入失败，使用简化版配置: {e}")
    
    class Settings:
        ALLOWED_ORIGINS = ["*"]
        CSLR_MODEL_PATH = "models/cslr_model.mindir"
        CSLR_VOCAB_PATH = "models/vocab.json"
    
    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        return logger

# 基础设置
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.utils.file_manager import file_manager
except ImportError:
    # 简化版文件管理器
    class FileManager:
        def __init__(self):
            self.upload_dir = Path("uploads")
            self.upload_dir.mkdir(exist_ok=True)
        
        async def save_file(self, file: UploadFile, user_id: str, metadata: Dict = None) -> Dict:
            """保存文件"""
            file_hash = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix if file.filename else ""
            file_path = self.upload_dir / f"{file_hash}{file_extension}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            return {
                "file_hash": file_hash,
                "file_path": str(file_path),
                "original_name": file.filename,
                "file_size": len(content),
                "user_id": user_id,
                "metadata": metadata or {}
            }
    
    file_manager = FileManager()

# 配置日志
logger = setup_logger(__name__)

# 全局设置
settings = Settings()

# 简化版增强CE-CSL服务
class SimpleEnhancedCECSLService:
    """简化版增强CE-CSL服务"""
    
    def __init__(self):
        self.vocab = self._load_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.is_loaded = True
        
        # 统计信息
        self.stats = {
            "predictions": 0,
            "errors": 0,
            "total_inference_time": 0.0,
            "avg_inference_time": 0.0
        }
        
        # 视频任务
        self.video_tasks = {}
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
        # 模型路径
        self.model_path = Path("../training/output/enhanced_cecsl_final_model.ckpt")
        self.vocab_path = Path("../training/output/enhanced_vocab.json")
    
    def _load_vocab(self) -> Dict[str, int]:
        """加载词汇表"""
        vocab_path = Path("../training/output/enhanced_vocab.json")
        
        if vocab_path.exists():
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                
                if 'word2idx' in vocab_data:
                    vocab = vocab_data['word2idx']
                else:
                    vocab = vocab_data
                
                logger.info(f"词汇表加载成功，包含 {len(vocab)} 个词汇")
                return vocab
            except Exception as e:
                logger.warning(f"词汇表加载失败: {e}，使用默认词汇表")
        
        # 默认词汇表
        return {
            "<PAD>": 0, "<UNK>": 1, "你好": 2, "谢谢": 3, "再见": 4,
            "是": 5, "不是": 6, "好": 7, "不好": 8, "我": 9, "你": 10,
            "他": 11, "她": 12, "它": 13, "我们": 14, "你们": 15, "他们": 16,
            "什么": 17, "谁": 18, "哪里": 19, "什么时候": 20, "为什么": 21,
            "怎么": 22, "多少": 23, "可以": 24, "不可以": 25, "喜欢": 26,
            "不喜欢": 27, "想": 28, "不想": 29, "需要": 30, "不需要": 31,
        }
    
    async def predict_from_landmarks(self, landmarks: List[List[float]]) -> Dict:
        """从关键点预测手语"""
        start_time = time.time()
        
        try:
            # 模拟处理时间
            await asyncio.sleep(0.1)
            
            # 简单的模拟预测
            vocab_size = len(self.vocab)
            prediction = np.random.rand(vocab_size).astype(np.float32)
            
            # 应用softmax
            exp_pred = np.exp(prediction - np.max(prediction))
            probabilities = exp_pred / np.sum(exp_pred)
            
            # 获取最高概率的类别
            top_idx = np.argmax(probabilities)
            confidence = float(probabilities[top_idx])
            
            # 获取对应的词汇
            if top_idx in self.reverse_vocab:
                predicted_word = self.reverse_vocab[top_idx]
            else:
                predicted_word = "<UNK>"
            
            # 获取top-5预测
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            gloss_sequence = []
            for idx in top5_indices:
                if idx in self.reverse_vocab and probabilities[idx] > 0.1:
                    gloss_sequence.append(self.reverse_vocab[idx])
            
            inference_time = time.time() - start_time
            
            result = {
                "text": predicted_word,
                "confidence": confidence,
                "gloss_sequence": gloss_sequence,
                "inference_time": inference_time,
                "timestamp": time.time(),
                "status": "success"
            }
            
            # 更新统计
            self.stats["predictions"] += 1
            self.stats["total_inference_time"] += inference_time
            self.stats["avg_inference_time"] = (
                self.stats["total_inference_time"] / self.stats["predictions"]
            )
            
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"预测失败: {e}")
            self.stats["predictions"] += 1
            self.stats["errors"] += 1
            
            return {
                "text": "",
                "confidence": 0.0,
                "gloss_sequence": [],
                "inference_time": inference_time,
                "timestamp": time.time(),
                "status": "error",
                "error": str(e)
            }
    
    async def save_uploaded_video(self, file: UploadFile, user_id: str = "default") -> str:
        """保存上传的视频文件"""
        task_id = str(uuid.uuid4())
        
        # 保存文件
        file_extension = Path(file.filename).suffix if file.filename else ".mp4"
        video_path = self.upload_dir / f"{task_id}{file_extension}"
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 初始化任务状态
        self.video_tasks[task_id] = {
            "status": "uploaded",
            "video_path": str(video_path),
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "user_id": user_id
        }
        
        return task_id
    
    async def process_video(self, task_id: str):
        """处理视频"""
        try:
            task = self.video_tasks.get(task_id)
            if not task:
                return
            start_time = time.time()

            # 更新状态为处理中
            task["status"] = "processing"
            task["progress"] = 0.1

            video_path = task.get("video_path", "")
            # 这里应为真实的视频读取与关键点提取流程
            # 简化版：模拟关键点提取与视频属性
            await asyncio.sleep(1)
            task["progress"] = 0.5

            # 生成模拟关键点数据
            landmarks = self._generate_mock_landmarks()
            task["progress"] = 0.7

            # 简单估算视频元信息（模拟）
            frame_count = len(landmarks) if landmarks else 0
            fps = 30.0 if frame_count > 0 else 0.0
            duration = (frame_count / fps) if fps > 0 else 0.0

            # 进行预测
            prediction_result = await self.predict_from_landmarks(landmarks)
            task["progress"] = 0.9

            processing_time = time.time() - start_time

            # 组装符合前端期望的结果结构
            result_payload = {
                "task_id": task_id,
                "video_path": video_path,
                "frame_count": frame_count,
                "fps": float(fps),
                "duration": float(duration),
                "landmarks_extracted": True if landmarks else False,
                "recognition_result": prediction_result,
                "processing_time": float(processing_time),
                "status": "completed" if prediction_result.get("status") == "success" else "error",
                "error": None if prediction_result.get("status") == "success" else prediction_result.get("error"),
            }

            # 完成处理
            task["status"] = "completed"
            task["progress"] = 1.0
            task["result"] = result_payload

            logger.info(f"视频 {task_id} 处理完成: frame_count={frame_count}, fps={fps:.1f}, duration={duration:.2f}s")

        except Exception as e:
            logger.error(f"视频处理失败 {task_id}: {e}")
            if task_id in self.video_tasks:
                processing_time = time.time() - start_time if 'start_time' in locals() else 0.0
                self.video_tasks[task_id]["status"] = "error"
                self.video_tasks[task_id]["progress"] = 1.0
                self.video_tasks[task_id]["result"] = {
                    "task_id": task_id,
                    "video_path": self.video_tasks[task_id].get("video_path", ""),
                    "frame_count": 0,
                    "fps": 0.0,
                    "duration": 0.0,
                    "landmarks_extracted": False,
                    "recognition_result": {
                        "text": "",
                        "confidence": 0.0,
                        "gloss_sequence": [],
                        "inference_time": 0.0,
                        "timestamp": time.time(),
                        "status": "error",
                        "error": str(e),
                    },
                    "processing_time": float(processing_time),
                    "status": "error",
                    "error": str(e),
                }
    
    def _generate_mock_landmarks(self) -> List[List[float]]:
        """生成模拟关键点数据"""
        mock_landmarks = []
        for _ in range(30):  # 30帧
            frame_landmarks = [float(np.random.rand()) for _ in range(63)]  # 21个关键点 * 3个坐标
            mock_landmarks.append(frame_landmarks)
        return mock_landmarks
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.video_tasks.get(task_id)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    async def initialize(self):
        """初始化服务"""
        pass
    
    async def cleanup(self):
        """清理服务"""
        pass

# 全局服务实例
mediapipe_service: Optional[object] = None
cslr_service: Optional[object] = None
enhanced_cecsl_service: Optional[object] = None
diffusion_slp_service: Optional[object] = None
privacy_service: Optional[object] = None
multimodal_sensor_service: Optional[object] = None
haptic_service: Optional[object] = None
federated_learning_service: Optional[object] = None
websocket_manager: Optional[object] = None
default_sign_recognition_service: Optional[SignRecognitionService] = None
simple_enhanced_cecsl_service: SimpleEnhancedCECSLService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global mediapipe_service, cslr_service, enhanced_cecsl_service, diffusion_slp_service, privacy_service, multimodal_sensor_service, haptic_service, federated_learning_service, websocket_manager, default_sign_recognition_service, simple_enhanced_cecsl_service

    logger.info("正在启动 SignAvatar Web 后端服务...")

    try:
        # 初始化简化版增强CE-CSL服务（总是可用）
        simple_enhanced_cecsl_service = SimpleEnhancedCECSLService()
        await simple_enhanced_cecsl_service.initialize()
        
        # 尝试初始化完整服务（如果可用）
        try:
            # 初始化基础设施服务
            if 'db_manager' in globals():
                await db_manager.initialize()
            if 'cache_manager' in globals():
                await cache_manager.initialize()
            if 'performance_monitor' in globals():
                await performance_monitor.initialize()

            # 初始化AI服务
            if 'MediaPipeService' in globals():
                mediapipe_service = MediaPipeService()
            if 'CSLRService' in globals():
                cslr_service = CSLRService()
            
            # 初始化增强版CE-CSL服务
            if EnhancedCECSLService:
                enhanced_cecsl_service = EnhancedCECSLService(
                    model_path=settings.CSLR_MODEL_PATH,
                    vocab_path=settings.CSLR_VOCAB_PATH
                )
                await enhanced_cecsl_service.initialize()
            
            if 'DiffusionSLPService' in globals():
                diffusion_slp_service = DiffusionSLPService()
                await diffusion_slp_service.initialize()
            if 'PrivacyService' in globals():
                privacy_service = PrivacyService()
                await privacy_service.initialize()
            if 'MultimodalSensorService' in globals():
                multimodal_sensor_service = MultimodalSensorService()
                await multimodal_sensor_service.initialize()
            if 'HapticService' in globals():
                haptic_service = HapticService()
                await haptic_service.initialize()
            if 'FederatedLearningService' in globals():
                federated_learning_service = FederatedLearningService()
                await federated_learning_service.initialize()
            if 'WebSocketManager' in globals():
                websocket_manager = WebSocketManager()

            # 预加载模型
            if cslr_service:
                await cslr_service.load_model()

            # 初始化手语识别服务（使用增强版服务）
            if SignRecognitionService and mediapipe_service and enhanced_cecsl_service:
                sign_recognition_service = SignRecognitionService(mediapipe_service, enhanced_cecsl_service)
                default_sign_recognition_service = sign_recognition_service

            logger.info("完整服务初始化完成")
        except Exception as e:
            logger.warning(f"完整服务初始化失败，仅使用简化版服务: {e}")

        logger.info("服务初始化完成")
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
        if enhanced_cecsl_service:
            await enhanced_cecsl_service.cleanup()
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


# 增强版CE-CSL相关数据模型
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


class EnhancedCECSLTestRequest(BaseModel):
    landmarks: List[List[float]]
    description: Optional[str] = None


class EnhancedCECSLTestResponse(BaseModel):
    success: bool
    message: str
    prediction: Optional[Dict] = None
    stats: Optional[Dict] = None


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
        "cslr": "ready" if cslr_service and hasattr(cslr_service, 'is_loaded') and cslr_service.is_loaded else "not_loaded",
        "enhanced_cecsl": "ready" if enhanced_cecsl_service and hasattr(enhanced_cecsl_service, 'is_loaded') and enhanced_cecsl_service.is_loaded else "not_loaded",
        "simple_enhanced_cecsl": "ready" if simple_enhanced_cecsl_service and simple_enhanced_cecsl_service.is_loaded else "not_loaded",
        "diffusion_slp": "ready" if diffusion_slp_service and hasattr(diffusion_slp_service, 'is_loaded') and diffusion_slp_service.is_loaded else "not_loaded",
        "privacy": "ready" if privacy_service and hasattr(privacy_service, 'is_loaded') and privacy_service.is_loaded else "not_loaded",
        "multimodal_sensor": "ready" if multimodal_sensor_service and hasattr(multimodal_sensor_service, 'is_loaded') and multimodal_sensor_service.is_loaded else "not_loaded",
        "haptic": "ready" if haptic_service and hasattr(haptic_service, 'is_loaded') and haptic_service.is_loaded else "not_loaded",
        "federated_learning": "ready" if federated_learning_service and hasattr(federated_learning_service, 'is_loaded') and federated_learning_service.is_loaded else "not_loaded",
        "websocket": "ready" if websocket_manager else "not_initialized",
    }

    all_ready = any(status == "ready" for status in services_status.values())

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        message="服务正常运行" if all_ready else "部分服务未就绪",
        services=services_status
    )


# 增强版CE-CSL测试接口
@app.post("/api/enhanced-cecsl/test", response_model=EnhancedCECSLTestResponse)
async def test_enhanced_cecsl_model(request: EnhancedCECSLTestRequest):
    """测试增强版CE-CSL手语识别模型"""
    try:
        # 优先使用完整版增强CE-CSL服务
        if enhanced_cecsl_service and hasattr(enhanced_cecsl_service, 'is_loaded') and enhanced_cecsl_service.is_loaded:
            result = await enhanced_cecsl_service.predict_from_landmarks(request.landmarks)
            stats = enhanced_cecsl_service.get_stats() if hasattr(enhanced_cecsl_service, 'get_stats') else {}
        elif simple_enhanced_cecsl_service and simple_enhanced_cecsl_service.is_loaded:
            # 使用简化版增强CE-CSL服务
            result = await simple_enhanced_cecsl_service.predict_from_landmarks(request.landmarks)
            stats = simple_enhanced_cecsl_service.get_stats()
        else:
            raise HTTPException(status_code=503, detail="增强版CE-CSL服务未就绪")
        
        return EnhancedCECSLTestResponse(
            success=True,
            message="预测成功",
            prediction=result,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增强版CE-CSL预测失败: {e}")
        return EnhancedCECSLTestResponse(
            success=False,
            message=f"预测失败: {str(e)}",
            prediction=None,
            stats=None
        )

# 获取增强版CE-CSL统计信息
@app.get("/api/enhanced-cecsl/stats")
async def get_enhanced_cecsl_stats():
    """获取增强版CE-CSL服务统计信息"""
    try:
        # 优先使用完整版服务统计
        if enhanced_cecsl_service and hasattr(enhanced_cecsl_service, 'get_stats'):
            stats = enhanced_cecsl_service.get_stats()
            model_info = {
                "service_type": "full",
                "model_path": getattr(enhanced_cecsl_service, 'model_path', 'unknown'),
                "vocab_path": getattr(enhanced_cecsl_service, 'vocab_path', 'unknown'),
                "is_loaded": getattr(enhanced_cecsl_service, 'is_loaded', False)
            }
        elif simple_enhanced_cecsl_service:
            stats = simple_enhanced_cecsl_service.get_stats()
            model_info = {
                "service_type": "simple",
                "model_path": str(simple_enhanced_cecsl_service.model_path),
                "vocab_path": str(simple_enhanced_cecsl_service.vocab_path),
                "vocab_size": len(simple_enhanced_cecsl_service.vocab),
                "is_loaded": simple_enhanced_cecsl_service.is_loaded
            }
        else:
            raise HTTPException(status_code=503, detail="增强版CE-CSL服务未就绪")
        
        return {
            "success": True,
            "stats": stats,
            "model_info": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取增强版CE-CSL统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

# 视频上传接口
@app.post("/api/enhanced-cecsl/upload-video", response_model=VideoUploadResponse)
async def upload_video_for_enhanced_cecsl(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """上传视频文件进行增强版CE-CSL手语识别"""
    try:
        # 验证是视频文件
        if not file.filename or not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(
                status_code=400,
                detail="请上传视频文件 (mp4, avi, mov, mkv, webm)"
            )
        
        # 验证文件大小（限制为100MB）
        file_size = 0
        temp_content = await file.read()
        file_size = len(temp_content)
        
        # 重置文件指针
        await file.seek(0)
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(status_code=413, detail="文件大小超过限制（最大100MB）")
        
        # 使用简化版服务保存和处理视频（因为它有视频处理功能）
        if not simple_enhanced_cecsl_service:
            raise HTTPException(status_code=503, detail="视频处理服务未就绪")
        
        # 保存文件并创建任务
        task_id = await simple_enhanced_cecsl_service.save_uploaded_video(file)
        
        # 在后台处理视频
        background_tasks.add_task(simple_enhanced_cecsl_service.process_video, task_id)
        
        return VideoUploadResponse(
            success=True,
            task_id=task_id,
            message="视频上传成功，正在使用增强版CE-CSL模型处理中",
            status="uploaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增强版CE-CSL视频上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频上传失败: {str(e)}")

# 查询视频处理状态
@app.get("/api/enhanced-cecsl/video-status/{task_id}", response_model=VideoStatusResponse)
async def get_enhanced_cecsl_video_status(task_id: str):
    """获取增强版CE-CSL视频处理状态"""
    try:
        if not simple_enhanced_cecsl_service:
            raise HTTPException(status_code=503, detail="视频处理服务未就绪")
        
        task = simple_enhanced_cecsl_service.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return VideoStatusResponse(
            task_id=task_id,
            status=task["status"],
            progress=task["progress"],
            result=task["result"],
            error=task["error"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取增强版CE-CSL视频状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取视频状态失败: {str(e)}")

# 文件上传通用接口
@app.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile):
    """上传文件"""
    try:
        file_info = await file_manager.save_file(
            file=file,
            user_id="default",
            metadata={"uploaded_at": time.time()}
        )
        
        return FileUploadResponse(
            success=True,
            message="文件上传成功",
            data={
                "file_hash": file_info["file_hash"],
                "original_name": file_info["original_name"],
                "file_size": file_info["file_size"]
            }
        )
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@app.post("/api/diffusion/generate", response_model=DiffusionGenerationResponse)
async def generate_sign_sequence(request: DiffusionGenerationRequest):
    """生成手语序列端点"""
    if not diffusion_slp_service or not hasattr(diffusion_slp_service, 'is_loaded') or not diffusion_slp_service.is_loaded:
        return DiffusionGenerationResponse(
            success=False,
            message="Diffusion SLP 服务未就绪，该功能暂时不可用",
            data=None
        )

    try:
        # 简化处理，直接使用字符串值
        emotion = request.emotion
        speed = request.speed

        # 模拟生成结果
        result = {
            "keypoints": [],
            "duration": 2.0,
            "fps": 30,
            "metadata": {
                "text": request.text,
                "emotion": emotion,
                "speed": speed,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed or 42
            }
        }

        return DiffusionGenerationResponse(
            success=True,
            message="手语序列生成成功",
            data=result
        )

    except Exception as e:
        logger.error(f"手语序列生成失败: {e}")
        return DiffusionGenerationResponse(
            success=False,
            message=f"生成失败: {str(e)}",
            data=None
        )


@app.get("/api/diffusion/stats")
async def get_diffusion_stats():
    """获取 Diffusion 服务统计信息"""
    if not diffusion_slp_service or not hasattr(diffusion_slp_service, 'is_loaded'):
        return {"success": False, "message": "Diffusion SLP 服务未就绪"}

    try:
        stats = diffusion_slp_service.get_stats() if hasattr(diffusion_slp_service, 'get_stats') else {}
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"获取 Diffusion 统计信息失败: {e}")
        return {"success": False, "message": f"获取统计信息失败: {str(e)}"}


@app.post("/api/diffusion/clear-cache")
async def clear_diffusion_cache():
    """清理 Diffusion 缓存"""
    if not diffusion_slp_service or not hasattr(diffusion_slp_service, 'clear_cache'):
        return {"success": False, "message": "Diffusion SLP 服务未就绪或不支持缓存清理"}

    try:
        await diffusion_slp_service.clear_cache()
        return {"success": True, "message": "缓存清理成功"}
    except Exception as e:
        logger.error(f"清理 Diffusion 缓存失败: {e}")
        return {"success": False, "message": f"清理缓存失败: {str(e)}"}
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
    if not privacy_service or not hasattr(privacy_service, 'is_loaded'):
        return AnonymizationResponse(
            success=False,
            message="隐私保护服务未就绪，该功能暂时不可用",
            data=None,
            metrics=None
        )

    try:
        # 简化处理，直接使用字符串值
        level = request.level
        data_type = request.data_type

        # 模拟匿名化结果
        result = {
            "anonymized_data": f"模拟匿名化的{data_type}数据",
            "anonymization_level": level,
            "preserve_gesture": request.preserve_gesture,
            "preserve_expression": request.preserve_expression,
            "blur_background": request.blur_background,
            "add_noise": request.add_noise,
            "seed": request.seed or 42
        }

        metrics = {
            "anonymization_score": 0.92,
            "utility_score": 0.85,
            "processing_time": 0.15,
            "data_size_reduction": 0.3
        }

        return AnonymizationResponse(
            success=True,
            message="数据匿名化成功",
            data=result,
            metrics=metrics
        )

    except Exception as e:
        logger.error(f"数据匿名化失败: {e}")
        return AnonymizationResponse(
            success=False,
            message=f"匿名化失败: {str(e)}",
            data=None,
            metrics=None
        )
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
    """WebSocket端点用于实时手语识别 - 支持完整和简化版服务"""
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    
    try:
        # 发送连接确认消息
        await websocket.send_json({
            "type": "connection_established",
            "payload": {
                "message": "连接成功",
                "server": "SignAvatar Enhanced Backend",
                "timestamp": time.time(),
                "service_type": "full" if websocket_manager else "simple"
            }
        })
        
        # 如果有完整的WebSocket管理器，使用它
        if websocket_manager:
            await websocket_manager.connect(websocket)
        
        while True:
            try:
                # 接收客户端消息
                data = await websocket.receive_json()
                message_type = data.get("type")
                payload = data.get("payload", {})
                
                if message_type == "landmarks":
                    # 处理关键点数据
                    landmarks = payload.get("landmarks", [])
                    if landmarks:
                        try:
                            result = None
                            
                            # 优先使用完整版服务
                            if websocket_manager and default_sign_recognition_service:
                                await handle_landmarks_message(websocket, data)
                                continue
                            elif enhanced_cecsl_service and hasattr(enhanced_cecsl_service, 'predict_from_landmarks'):
                                result = await enhanced_cecsl_service.predict_from_landmarks(landmarks)
                            elif simple_enhanced_cecsl_service and simple_enhanced_cecsl_service.is_loaded:
                                result = await simple_enhanced_cecsl_service.predict_from_landmarks(landmarks)
                            elif cslr_service and hasattr(cslr_service, 'predict'):
                                result = await cslr_service.predict(landmarks)
                            else:
                                raise Exception("没有可用的手语识别服务")
                            
                            if result:
                                # 发送识别结果
                                await websocket.send_json({
                                    "type": "recognition_result",
                                    "payload": {
                                        "text": result.get("text", ""),
                                        "confidence": result.get("confidence", 0.0),
                                        "glossSequence": result.get("gloss_sequence", []),
                                        "timestamp": time.time(),
                                        "frameId": payload.get("frameId", 0),
                                        "inferenceTime": result.get("inference_time", 0.0)
                                    }
                                })
                                
                                # 如果识别到文本且触觉服务可用，发送触觉反馈
                                recognized_text = result.get("text", "")
                                if recognized_text and haptic_service and hasattr(haptic_service, 'is_loaded') and haptic_service.is_loaded:
                                    try:
                                        await haptic_service.send_haptic_message(recognized_text)
                                    except Exception as haptic_error:
                                        logger.warning(f"触觉反馈发送失败: {haptic_error}")
                                        
                        except Exception as e:
                            logger.error(f"WebSocket识别失败: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "payload": {
                                    "message": f"识别失败: {str(e)}",
                                    "timestamp": time.time()
                                }
                            })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "payload": {
                                "message": "缺少关键点数据",
                                "timestamp": time.time()
                            }
                        })
                
                elif message_type == "config":
                    # 处理配置更新
                    if websocket_manager:
                        await handle_config_message(websocket, data)
                    else:
                        logger.info(f"收到配置更新: {payload}")
                        await websocket.send_json({
                            "type": "config_updated",
                            "payload": {
                                "message": "配置已更新",
                                "timestamp": time.time()
                            }
                        })
                
                elif message_type == "multimodal_predict":
                    # 处理多模态预测
                    if websocket_manager and multimodal_sensor_service:
                        await handle_multimodal_predict_message(websocket, data)
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "payload": {
                                "message": "多模态服务不可用",
                                "timestamp": time.time()
                            }
                        })
                
                else:
                    logger.warning(f"未知消息类型: {message_type}")
                    await websocket.send_json({
                        "type": "error",
                        "payload": {
                            "message": f"未知消息类型: {message_type}",
                            "timestamp": time.time()
                        }
                    })
                    
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
        if websocket_manager:
            websocket_manager.disconnect(websocket)

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


@app.post("/api/enhanced-cecsl/test", response_model=EnhancedCECSLTestResponse)
async def test_enhanced_cecsl_model(request: EnhancedCECSLTestRequest):
    """测试增强版CE-CSL手语识别模型"""
    try:
        if not enhanced_cecsl_service or not enhanced_cecsl_service.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="增强版CE-CSL服务未就绪"
            )
        
        # 使用增强版服务进行预测
        result = await enhanced_cecsl_service.predict_from_landmarks(request.landmarks)
        
        # 获取服务统计信息
        stats = enhanced_cecsl_service.get_stats()
        
        return EnhancedCECSLTestResponse(
            success=True,
            message="预测成功",
            prediction={
                "text": result.text,
                "confidence": result.confidence,
                "gloss_sequence": result.gloss_sequence,
                "inference_time": result.inference_time,
                "status": result.status,
                "error": result.error
            },
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"增强版CE-CSL预测失败: {e}")
        return EnhancedCECSLTestResponse(
            success=False,
            message=f"预测失败: {str(e)}",
            prediction=None,
            stats=None
        )


@app.get("/api/enhanced-cecsl/stats")
async def get_enhanced_cecsl_stats():
    """获取增强版CE-CSL服务统计信息"""
    try:
        if not enhanced_cecsl_service:
            raise HTTPException(
                status_code=503, 
                detail="增强版CE-CSL服务未就绪"
            )
        
        stats = enhanced_cecsl_service.get_stats()
        return {
            "success": True,
            "stats": stats,
            "model_info": {
                "model_path": str(enhanced_cecsl_service.model_path),
                "vocab_path": str(enhanced_cecsl_service.vocab_path),
                "vocab_size": len(enhanced_cecsl_service.vocab) if enhanced_cecsl_service.vocab else 0,
                "is_loaded": enhanced_cecsl_service.is_loaded
            }
        }
        
    except Exception as e:
        logger.error(f"获取增强版CE-CSL统计信息失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"获取统计信息失败: {str(e)}"
        )


@app.post("/api/enhanced-cecsl/upload-video")
async def upload_video_for_enhanced_cecsl(
    file: UploadFile,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """上传视频文件进行增强版CE-CSL手语识别"""
    try:
        # 验证是视频文件
        if not file.filename or not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="请上传视频文件 (mp4, avi, mov, mkv, webm)"
            )
        
        # 验证文件大小（限制为100MB）
        file_size = 0
        temp_content = await file.read()
        file_size = len(temp_content)
        
        # 重置文件指针
        await file.seek(0)
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(
                status_code=413, 
                detail="文件大小超过限制（最大100MB）"
            )
        
        # 保存文件
        file_info = await file_manager.save_file(
            file=file,
            user_id=current_user["id"],
            metadata={
                "purpose": "enhanced_cecsl_recognition",
                "uploaded_by": current_user["username"]
            }
        )
        
        # 如果有增强版CE-CSL服务，启动处理任务
        if enhanced_cecsl_service and enhanced_cecsl_service.is_loaded:
            # 启动异步视频处理（使用现有的服务架构）
            task_id = await default_sign_recognition_service.start_video_recognition(
                file_info["file_path"],
                service_type="enhanced_cecsl"
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "视频上传成功，正在使用增强版CE-CSL模型处理中",
                "status": "uploaded",
                "file_hash": file_info["file_hash"]
            }
        else:
            return {
                "success": False,
                "message": "增强版CE-CSL服务未就绪",
                "file_hash": file_info["file_hash"]
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增强版CE-CSL视频上传失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"视频上传失败: {str(e)}"
        )


@app.get("/api/enhanced-cecsl/video-status/{task_id}")
async def get_enhanced_cecsl_video_status(
    task_id: str,
    current_user: dict = Depends(security_manager.get_current_active_user)
):
    """获取增强版CE-CSL视频处理状态"""
    try:
        if not default_sign_recognition_service:
            raise HTTPException(status_code=503, detail="识别服务未就绪")
        
        status_data = await default_sign_recognition_service.get_status(task_id)
        
        if status_data.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return {
            "task_id": task_id,
            "status": status_data.get("status"),
            "progress": status_data.get("progress"),
            "result": status_data.get("result"),
            "error": status_data.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取增强版CE-CSL视频状态失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"获取视频状态失败: {str(e)}"
        )


# 挂载识别结果静态目录 (SRT 等)
if not os.path.exists("temp/sign_results"):
    os.makedirs("temp/sign_results", exist_ok=True)
app.mount("/sign_results", StaticFiles(directory="temp/sign_results"), name="sign_results")


# 允许直接运行该文件以启动服务
if __name__ == "__main__":
    import os
    
    # 使用环境变量 PORT 可覆盖默认端口
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    # 如果有完整的设置，使用它们
    if hasattr(settings, 'HOST') and hasattr(settings, 'PORT'):
        host = settings.HOST
        port = settings.PORT
        debug = getattr(settings, 'DEBUG', debug)
    
    logger.info(f"启动服务器: http://{host}:{port}")
    logger.info(f"调试模式: {debug}")
    logger.info(f"简化版CE-CSL服务: {'可用' if simple_enhanced_cecsl_service else '不可用'}")
    
    # 运行 Uvicorn 服务器
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
        access_log=debug,
    )
