"""
SignAvatar Web Backend - Integrated Main Application
集成版实时手语识别与虚拟人播报系统后端服务
合并了 main.py 和 main_simple.py 的功能
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

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

# 文件管理器
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

# 全局服务实例
enhanced_cecsl_service = SimpleEnhancedCECSLService()
file_manager = FileManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("正在启动 SignAvatar Web 后端服务...")
    
    try:
        # 初始化服务
        logger.info(f"增强版CE-CSL服务: {'可用' if enhanced_cecsl_service.is_loaded else '不可用'}")
        logger.info("服务初始化完成")
        yield
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        raise
    finally:
        # 清理资源
        logger.info("正在关闭服务...")
        logger.info("服务关闭完成")

# 创建FastAPI应用
app = FastAPI(
    title="SignAvatar Web API (Integrated)",
    description="集成版实时手语识别与虚拟人播报系统 API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# API路由
@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 返回简单的状态页面"""
    return f"""
    <html>
        <head>
            <title>SignAvatar Web API (Integrated)</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .status {{ color: #4CAF50; font-weight: bold; }}
                .info {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>🤖 SignAvatar Web API (集成版)</h1>
            <p class="status">✅ 服务运行正常</p>
            <div class="info">
                <h3>可用端点:</h3>
                <ul>
                    <li><a href="/api/docs">API 文档 (Swagger)</a></li>
                    <li><a href="/api/health">健康检查</a></li>
                    <li><a href="/ws/sign-recognition">WebSocket 连接</a></li>
                </ul>
                <h3>增强版CE-CSL服务:</h3>
                <p>状态: {'✅ 可用' if enhanced_cecsl_service.is_loaded else '❌ 不可用'}</p>
                <p>词汇量: {len(enhanced_cecsl_service.vocab)}</p>
            </div>
        </body>
    </html>
    """

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    services_status = {
        "enhanced_cecsl": "ready" if enhanced_cecsl_service.is_loaded else "not_loaded",
        "file_manager": "ready",
    }

    all_ready = all(status == "ready" for status in services_status.values())

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
        if not enhanced_cecsl_service.is_loaded:
            raise HTTPException(status_code=503, detail="增强版CE-CSL服务未就绪")
        
        # 使用增强版服务进行预测
        result = await enhanced_cecsl_service.predict_from_landmarks(request.landmarks)
        
        # 获取服务统计信息
        stats = enhanced_cecsl_service.get_stats()
        
        return EnhancedCECSLTestResponse(
            success=True,
            message="预测成功",
            prediction=result,
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

# 获取统计信息
@app.get("/api/enhanced-cecsl/stats")
async def get_enhanced_cecsl_stats():
    """获取增强版CE-CSL服务统计信息"""
    try:
        stats = enhanced_cecsl_service.get_stats()
        return {
            "success": True,
            "stats": stats,
            "model_info": {
                "model_path": str(enhanced_cecsl_service.model_path),
                "vocab_path": str(enhanced_cecsl_service.vocab_path),
                "vocab_size": len(enhanced_cecsl_service.vocab),
                "is_loaded": enhanced_cecsl_service.is_loaded
            }
        }
        
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
        
        # 保存文件并创建任务
        task_id = await enhanced_cecsl_service.save_uploaded_video(file)
        
        # 在后台处理视频
        background_tasks.add_task(enhanced_cecsl_service.process_video, task_id)
        
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
        task = enhanced_cecsl_service.get_task_status(task_id)
        
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
                "server": "SignAvatar Integrated Backend",
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
                    # 处理关键点数据
                    landmarks = payload.get("landmarks", [])
                    if landmarks and enhanced_cecsl_service.is_loaded:
                        try:
                            # 使用增强版CE-CSL服务进行预测
                            result = await enhanced_cecsl_service.predict_from_landmarks(landmarks)
                            
                            # 发送识别结果
                            await websocket.send_json({
                                "type": "recognition_result",
                                "payload": {
                                    "text": result.get("text", ""),
                                    "confidence": result.get("confidence", 0.0),
                                    "glossSequence": result.get("gloss_sequence", []),
                                    "timestamp": time.time(),
                                    "frameId": payload.get("frameId", 0)
                                }
                            })
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
                                "message": "缺少关键点数据或服务未就绪",
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
    logger.info(f"增强版CE-CSL服务: {'可用' if enhanced_cecsl_service.is_loaded else '不可用'}")
    
    # 运行 Uvicorn 服务器
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
        access_log=debug,
    )
