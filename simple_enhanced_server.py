#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版增强CE-CSL手语识别服务器
用于测试集成，不依赖复杂的模块
"""

import json
import time
import asyncio
import logging
import uuid
import os
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


class SimpleEnhancedCECSLService:
    """简化版增强CE-CSL服务"""
    
    def __init__(self):
        self.vocab = self._load_default_vocab()
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
        self.upload_dir = Path("temp/video_uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_default_vocab(self) -> Dict[str, int]:
        """加载默认词汇表"""
        vocab_path = Path("training/output/enhanced_vocab.json")
        
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
    
    async def save_uploaded_video(self, file: UploadFile) -> str:
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
            "created_at": time.time()
        }
        
        return task_id
    
    async def process_video(self, task_id: str):
        """处理视频"""
        try:
            task = self.video_tasks.get(task_id)
            if not task:
                return
            
            # 更新状态为处理中
            task["status"] = "processing"
            task["progress"] = 0.1
            
            # 模拟关键点提取
            await asyncio.sleep(1)
            task["progress"] = 0.5
            
            # 生成模拟关键点数据
            landmarks = self._generate_mock_landmarks()
            task["progress"] = 0.7
            
            # 进行预测
            prediction_result = await self.predict_from_landmarks(landmarks)
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


# 创建全局服务实例
enhanced_service = SimpleEnhancedCECSLService()

# 创建FastAPI应用
app = FastAPI(
    title="简化版增强CE-CSL手语识别服务",
    description="用于测试的简化版API服务",
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


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "简化版增强CE-CSL手语识别服务",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": enhanced_service.is_loaded
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "message": "简化版增强CE-CSL服务运行正常",
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
                "model_path": "简化版模拟模型",
                "vocab_path": "training/output/enhanced_vocab.json",
                "vocab_size": len(enhanced_service.vocab),
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
        task_id = await enhanced_service.save_uploaded_video(file)
        
        # 在后台处理视频
        background_tasks.add_task(enhanced_service.process_video, task_id)
        
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
        task = enhanced_service.get_task_status(task_id)
        
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


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动简化版增强CE-CSL手语识别服务...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        access_log=True
    )
