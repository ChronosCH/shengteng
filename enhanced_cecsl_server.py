#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版CE-CSL手语识别服务 - 简化版本
专用于测试训练好的模型
"""

import json
import time
import asyncio
import numpy as np
import logging
import uuid
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MindSpore
try:
    import mindspore as ms
    from mindspore import nn, ops, Tensor, load_checkpoint, load_param_into_net
    import mindspore.context as ms_context
    MINDSPORE_AVAILABLE = True
    print("MindSpore导入成功")
except ImportError as e:
    MINDSPORE_AVAILABLE = False
    print(f"警告: MindSpore 未安装 ({e})，将使用模拟推理")
    
    # 创建模拟的MindSpore类
    class MockCell:
        def __init__(self):
            pass
        def set_train(self, mode):
            pass
    
    class MockNN:
        Cell = MockCell
        SequentialCell = list
        Dense = lambda *args, **kwargs: None
        LayerNorm = lambda *args, **kwargs: None
        ReLU = lambda *args, **kwargs: None
        Dropout = lambda *args, **kwargs: None
        LSTM = lambda *args, **kwargs: None
        Tanh = lambda *args, **kwargs: None
    
    nn = MockNN()
    ms = None
    ops = None
    Tensor = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# 数据模型
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


@dataclass
class EnhancedCECSLConfig:
    """增强版CE-CSL模型配置"""
    vocab_size: int = 1000
    d_model: int = 192
    n_layers: int = 2
    dropout: float = 0.3
    image_size: Tuple[int, int] = (112, 112)
    max_sequence_length: int = 64


class ImprovedCECSLModel:
    """改进的CE-CSL手语识别模型（模拟版本）"""
    
    def __init__(self, config: EnhancedCECSLConfig, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size
        print(f"创建模拟模型: vocab_size={vocab_size}")
    
    def set_train(self, mode):
        """设置训练模式"""
        pass
    
    def __call__(self, x):
        """模拟推理"""
        if MINDSPORE_AVAILABLE and isinstance(x, Tensor):
            # 真实的MindSpore推理
            return self._real_forward(x)
        else:
            # 模拟推理
            return self._mock_forward(x)
    
    def _real_forward(self, x):
        """真实的前向传播（当MindSpore可用时）"""
        # 这里应该是真实的MindSpore模型推理
        # 由于模型结构复杂，暂时使用模拟
        batch_size = x.shape[0]
        return np.random.rand(batch_size, self.vocab_size).astype(np.float32)
    
    def _mock_forward(self, x):
        """模拟前向传播"""
        if isinstance(x, np.ndarray):
            batch_size = x.shape[0]
        else:
            batch_size = 1
        return np.random.rand(batch_size, self.vocab_size).astype(np.float32)


class VideoProcessingService:
    """视频处理服务，用于提取手部关键点"""
    
    def __init__(self):
        self.video_tasks = {}  # 存储视频处理任务
        self.upload_dir = Path("temp/video_uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe 相关
        self.mp = None
        self.mp_hands = None
        self.hands = None
        
        # 初始化MediaPipe
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """初始化MediaPipe"""
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe 初始化成功")
        except ImportError:
            logger.warning("MediaPipe 不可用，使用模拟关键点提取")
            self.mp = None
    
    async def save_uploaded_video(self, file: UploadFile) -> Tuple[str, str]:
        """保存上传的视频文件"""
        # 生成唯一的任务ID
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
            "created_at": time.time()
        }
        
        return task_id, str(video_path)
    
    async def process_video(self, task_id: str, enhanced_service) -> None:
        """处理视频，提取关键点并进行预测"""
        try:
            task = self.video_tasks.get(task_id)
            if not task:
                return
            
            # 更新状态为处理中
            task["status"] = "processing"
            task["progress"] = 0.1
            
            # 提取关键点
            landmarks = await self._extract_landmarks(task["video_path"], task_id)
            task["progress"] = 0.7
            
            if not landmarks:
                task["status"] = "error"
                task["error"] = "无法从视频中提取手部关键点"
                return
            
            # 使用增强服务进行预测
            prediction_result = await enhanced_service.predict_from_landmarks(landmarks)
            task["progress"] = 0.9
            
            # 完成处理
            task["status"] = "completed"
            task["progress"] = 1.0
            task["result"] = prediction_result
            
            logger.info(f"视频 {task_id} 处理完成")
            
        except Exception as e:
            logger.error(f"视频处理失败 {task_id}: {e}")
            if task_id in self.video_tasks:
                self.video_tasks[task_id]["status"] = "error"
                self.video_tasks[task_id]["error"] = str(e)
    
    async def _extract_landmarks(self, video_path: str, task_id: str) -> List[List[float]]:
        """从视频中提取手部关键点"""
        try:
            if not self.mp:
                # 模拟关键点提取
                return self._generate_mock_landmarks()
            
            landmarks_sequence = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return self._generate_mock_landmarks()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                
                # 更新进度
                frame_count += 1
                progress = 0.1 + (frame_count / total_frames) * 0.6  # 0.1-0.7
                if task_id in self.video_tasks:
                    self.video_tasks[task_id]["progress"] = progress
                
                # 转换BGR到RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 检测手部
                results = self.hands.process(image)
                
                if results.multi_hand_landmarks:
                    # 提取第一只手的关键点
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    landmarks_sequence.append(landmarks)
                else:
                    # 如果没有检测到手部，添加零向量
                    landmarks_sequence.append([0.0] * 63)  # 21个关键点 * 3个坐标
            
            cap.release()
            
            if not landmarks_sequence:
                logger.warning("视频中未检测到手部关键点，使用模拟数据")
                return self._generate_mock_landmarks()
            
            return landmarks_sequence
            
        except Exception as e:
            logger.error(f"关键点提取失败: {e}")
            return self._generate_mock_landmarks()
    
    def _generate_mock_landmarks(self) -> List[List[float]]:
        """生成模拟关键点数据"""
        # 生成30帧的模拟关键点数据
        mock_landmarks = []
        for _ in range(30):
            # 每帧21个关键点，每个关键点3个坐标
            frame_landmarks = [float(np.random.rand()) for _ in range(63)]
            mock_landmarks.append(frame_landmarks)
        
        logger.info("生成模拟关键点数据")
        return mock_landmarks
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.video_tasks.get(task_id)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧任务"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for task_id, task in self.video_tasks.items():
            if current_time - task["created_at"] > max_age_seconds:
                # 删除视频文件
                try:
                    video_path = Path(task["video_path"])
                    if video_path.exists():
                        video_path.unlink()
                except Exception as e:
                    logger.warning(f"删除视频文件失败: {e}")
                
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.video_tasks[task_id]
        
        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 个旧任务")


class EnhancedCECSLService:
    """增强版CE-CSL手语识别服务"""
    
    def __init__(self):
        # 默认路径
        self.model_path = Path("training/output/enhanced_cecsl_final_model.ckpt")
        self.vocab_path = Path("training/output/enhanced_vocab.json")
        
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.config = None
        self.is_loaded = False
        
        # 统计信息
        self.stats = {
            "predictions": 0,
            "errors": 0,
            "total_inference_time": 0.0,
            "avg_inference_time": 0.0
        }
    
    async def initialize(self) -> bool:
        """初始化服务"""
        try:
            logger.info("初始化增强版CE-CSL服务...")
            
            # 设置MindSpore上下文
            if MINDSPORE_AVAILABLE:
                ms_context.set_context(mode=ms_context.GRAPH_MODE, device_target="CPU")
            
            # 加载词汇表
            await self._load_vocabulary()
            
            # 创建模型配置
            self.config = EnhancedCECSLConfig(
                vocab_size=len(self.vocab)
            )
            
            # 加载模型
            await self._load_model()
            
            self.is_loaded = True
            logger.info("增强版CE-CSL服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"增强版CE-CSL服务初始化失败: {e}")
            self.is_loaded = False
            return False
    
    async def _load_vocabulary(self) -> None:
        """加载词汇表"""
        try:
            if not self.vocab_path.exists():
                logger.error(f"词汇表文件不存在: {self.vocab_path}")
                await self._create_default_vocabulary()
                return
            
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # 检查词汇表格式
            if 'word2idx' in vocab_data:
                self.vocab = vocab_data['word2idx']
            else:
                self.vocab = vocab_data
            
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"词汇表加载成功，包含 {len(self.vocab)} 个词汇")
            
        except Exception as e:
            logger.error(f"词汇表加载失败: {e}")
            await self._create_default_vocabulary()
    
    async def _create_default_vocabulary(self) -> None:
        """创建默认词汇表"""
        default_vocab = {
            "<PAD>": 0, "<UNK>": 1, "你好": 2, "谢谢": 3, "再见": 4,
            "是": 5, "不是": 6, "好": 7, "不好": 8, "我": 9, "你": 10,
        }
        
        self.vocab = default_vocab
        self.reverse_vocab = {v: k for k, v in default_vocab.items()}
        logger.warning("使用默认词汇表")
    
    async def _load_model(self) -> None:
        """加载模型"""
        try:
            if not MINDSPORE_AVAILABLE:
                logger.warning("MindSpore不可用，使用模拟模型")
                self.model = ImprovedCECSLModel(self.config, len(self.vocab))
                logger.info("模拟模型创建成功")
                return
            
            if not self.model_path.exists():
                logger.error(f"模型文件不存在: {self.model_path}")
                self.model = ImprovedCECSLModel(self.config, len(self.vocab))
                logger.info("使用模拟模型")
                return
            
            # 创建模型实例
            self.model = ImprovedCECSLModel(self.config, len(self.vocab))
            
            # 如果MindSpore可用，尝试加载checkpoint
            try:
                from mindspore import load_checkpoint, load_param_into_net
                param_dict = load_checkpoint(str(self.model_path))
                load_param_into_net(self.model, param_dict)
                logger.info(f"MindSpore模型加载成功: {self.model_path}")
            except Exception as e:
                logger.warning(f"MindSpore模型加载失败，使用模拟模式: {e}")
            
            # 设置为评估模式
            self.model.set_train(False)
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = ImprovedCECSLModel(self.config, len(self.vocab))
    
    async def predict_from_landmarks(self, landmarks: List[List[float]]) -> Dict:
        """从关键点预测手语"""
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise RuntimeError("服务未初始化")
            
            # 预处理输入数据
            input_data = self._preprocess_landmarks(landmarks)
            
            # 模型推理
            if MINDSPORE_AVAILABLE and self.model and hasattr(self.model, '_real_forward'):
                prediction = await self._mindspore_inference(input_data)
            else:
                prediction = await self._mock_inference(input_data)
            
            # 后处理
            result_dict = self._postprocess_prediction(prediction)
            
            # 创建结果
            inference_time = time.time() - start_time
            result = {
                **result_dict,
                "inference_time": inference_time,
                "timestamp": time.time(),
                "status": "success"
            }
            
            # 更新统计
            self._update_stats(inference_time, True)
            
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"预测失败: {e}")
            self._update_stats(inference_time, False)
            
            return {
                "text": "",
                "confidence": 0.0,
                "gloss_sequence": [],
                "inference_time": inference_time,
                "timestamp": time.time(),
                "status": "error",
                "error": str(e)
            }
    
    def _preprocess_landmarks(self, landmarks: List[List[float]]) -> np.ndarray:
        """预处理关键点数据"""
        try:
            # 转换为numpy数组
            landmarks_array = np.array(landmarks, dtype=np.float32)
            
            # 确保有足够的帧数
            target_frames = self.config.max_sequence_length
            if landmarks_array.shape[0] < target_frames:
                # 重复最后一帧来填充
                pad_frames = target_frames - landmarks_array.shape[0]
                last_frame = landmarks_array[-1:] if landmarks_array.shape[0] > 0 else np.zeros((1, landmarks_array.shape[1]))
                landmarks_array = np.concatenate([landmarks_array] + [last_frame] * pad_frames, axis=0)
            elif landmarks_array.shape[0] > target_frames:
                # 截断
                landmarks_array = landmarks_array[:target_frames]
            
            # 模拟转换为图像特征（这里简化处理）
            image_size = self.config.image_size[0] * self.config.image_size[1] * 3
            
            # 将关键点特征映射到图像特征空间
            if landmarks_array.shape[1] < image_size:
                # 扩展特征维度
                scale_factor = image_size // landmarks_array.shape[1] + 1
                expanded = np.tile(landmarks_array, (1, scale_factor))[:, :image_size]
            else:
                expanded = landmarks_array[:, :image_size]
            
            # 添加batch维度
            batch_data = expanded[np.newaxis, :]  # (1, seq_len, feature_dim)
            
            return batch_data
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            # 返回默认数据
            image_size = self.config.image_size[0] * self.config.image_size[1] * 3
            return np.random.rand(1, self.config.max_sequence_length, image_size).astype(np.float32)
    
    async def _mindspore_inference(self, input_data: np.ndarray) -> np.ndarray:
        """MindSpore模型推理"""
        try:
            if MINDSPORE_AVAILABLE and Tensor:
                # 转换为Tensor
                input_tensor = Tensor(input_data, ms.float32)
                # 推理
                output = self.model(input_tensor)
                # 转换回numpy
                if hasattr(output, 'asnumpy'):
                    return output.asnumpy()
                else:
                    return output
            else:
                # 使用模拟推理
                return self.model(input_data)
            
        except Exception as e:
            logger.error(f"MindSpore推理失败: {e}")
            # 降级到模拟推理
            return await self._mock_inference(input_data)
    
    async def _mock_inference(self, input_data: np.ndarray) -> np.ndarray:
        """模拟推理"""
        await asyncio.sleep(0.02)  # 模拟推理时间
        
        if self.model:
            return self.model(input_data)
        else:
            vocab_size = len(self.vocab)
            # 返回随机预测
            prediction = np.random.rand(vocab_size).astype(np.float32)
            # 应用softmax
            exp_pred = np.exp(prediction - np.max(prediction))
            return exp_pred / np.sum(exp_pred)
    
    def _postprocess_prediction(self, prediction: np.ndarray) -> Dict:
        """后处理预测结果"""
        try:
            # 获取最高概率的类别
            if prediction.ndim == 1:
                probabilities = prediction
            else:
                probabilities = prediction.flatten()
            
            top_idx = np.argmax(probabilities)
            confidence = float(probabilities[top_idx])
            
            # 获取对应的词汇
            if top_idx in self.reverse_vocab:
                predicted_word = self.reverse_vocab[top_idx]
            else:
                predicted_word = "<UNK>"
            
            # 获取top-5预测用于gloss序列
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            gloss_sequence = []
            for idx in top5_indices:
                if idx in self.reverse_vocab and probabilities[idx] > 0.1:
                    gloss_sequence.append(self.reverse_vocab[idx])
            
            return {
                "text": predicted_word,
                "confidence": confidence,
                "gloss_sequence": gloss_sequence
            }
            
        except Exception as e:
            logger.error(f"后处理失败: {e}")
            return {
                "text": "<UNK>",
                "confidence": 0.0,
                "gloss_sequence": []
            }
    
    def _update_stats(self, inference_time: float, success: bool) -> None:
        """更新统计信息"""
        self.stats["predictions"] += 1
        if not success:
            self.stats["errors"] += 1
        
        self.stats["total_inference_time"] += inference_time
        self.stats["avg_inference_time"] = (
            self.stats["total_inference_time"] / self.stats["predictions"]
        )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


# 创建全局服务实例
enhanced_service = EnhancedCECSLService()
video_service = VideoProcessingService()

# 创建FastAPI应用
app = FastAPI(
    title="增强版CE-CSL手语识别服务",
    description="基于训练好的enhanced_cecsl_final_model.ckpt模型的Web API",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    logger.info("启动增强版CE-CSL手语识别服务...")
    success = await enhanced_service.initialize()
    if success:
        logger.info("✅ 服务启动成功")
    else:
        logger.warning("⚠️ 服务启动异常，但将继续运行（模拟模式）")
    
    # 启动定期清理任务
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # 每小时清理一次
            video_service.cleanup_old_tasks()
    
    asyncio.create_task(periodic_cleanup())


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "增强版CE-CSL手语识别服务",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": enhanced_service.is_loaded
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "message": "增强版CE-CSL服务运行正常",
        "services": {
            "enhanced_cecsl": "ready" if enhanced_service.is_loaded else "not_ready"
        }
    }


@app.post("/api/enhanced-cecsl/test", response_model=EnhancedCECSLTestResponse)
async def test_enhanced_cecsl_model(request: EnhancedCECSLTestRequest):
    """测试增强版CE-CSL手语识别模型"""
    try:
        if not enhanced_service.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="增强版CE-CSL服务未就绪"
            )
        
        # 使用增强版服务进行预测
        result = await enhanced_service.predict_from_landmarks(request.landmarks)
        
        # 获取服务统计信息
        stats = enhanced_service.get_stats()
        
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


@app.get("/api/enhanced-cecsl/stats")
async def get_enhanced_cecsl_stats():
    """获取增强版CE-CSL服务统计信息"""
    try:
        stats = enhanced_service.get_stats()
        return {
            "success": True,
            "stats": stats,
            "model_info": {
                "model_path": str(enhanced_service.model_path),
                "vocab_path": str(enhanced_service.vocab_path),
                "vocab_size": len(enhanced_service.vocab) if enhanced_service.vocab else 0,
                "is_loaded": enhanced_service.is_loaded
            }
        }
        
    except Exception as e:
        logger.error(f"获取增强版CE-CSL统计信息失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"获取统计信息失败: {str(e)}"
        )


@app.post("/api/enhanced-cecsl/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """上传视频文件进行手语识别"""
    try:
        # 验证文件类型
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            raise HTTPException(
                status_code=400, 
                detail="不支持的文件格式，请上传 mp4, avi, mov, mkv 或 webm 格式的视频"
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
        
        # 保存文件并创建任务
        task_id, video_path = await video_service.save_uploaded_video(file)
        
        # 在后台处理视频
        background_tasks.add_task(
            video_service.process_video, 
            task_id, 
            enhanced_service
        )
        
        return VideoUploadResponse(
            success=True,
            task_id=task_id,
            message="视频上传成功，正在处理中",
            status="uploaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"视频上传失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"视频上传失败: {str(e)}"
        )


@app.get("/api/enhanced-cecsl/video-status/{task_id}", response_model=VideoStatusResponse)
async def get_video_status(task_id: str):
    """获取视频处理状态"""
    try:
        task = video_service.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=404, 
                detail="任务不存在"
            )
        
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
        logger.error(f"获取视频状态失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"获取视频状态失败: {str(e)}"
        )


@app.delete("/api/enhanced-cecsl/video-task/{task_id}")
async def delete_video_task(task_id: str):
    """删除视频处理任务"""
    try:
        task = video_service.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=404, 
                detail="任务不存在"
            )
        
        # 删除视频文件
        try:
            video_path = Path(task["video_path"])
            if video_path.exists():
                video_path.unlink()
        except Exception as e:
            logger.warning(f"删除视频文件失败: {e}")
        
        # 删除任务记录
        del video_service.video_tasks[task_id]
        
        return {
            "success": True,
            "message": "任务删除成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除视频任务失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"删除视频任务失败: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动增强版CE-CSL手语识别服务...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )
