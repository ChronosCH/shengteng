"""
实时手语识别服务
Real-time Sign Language Recognition Service
提供WebSocket实时手语识别功能
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from backend.core.websocket_manager import websocket_manager, WebSocketMessage, ConnectionInfo

logger = logging.getLogger(__name__)

@dataclass
class RecognitionSession:
    """识别会话"""
    session_id: str
    connection_id: str
    user_id: Optional[str]
    start_time: float
    last_activity: float
    frame_count: int = 0
    recognition_count: int = 0
    total_confidence: float = 0.0
    sequence_buffer: List[List[float]] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sequence_buffer is None:
            self.sequence_buffer = []
        if self.config is None:
            self.config = {}
    
    def update_activity(self):
        """更新活动时间"""
        self.last_activity = time.time()
    
    def add_recognition_result(self, confidence: float):
        """添加识别结果"""
        self.recognition_count += 1
        self.total_confidence += confidence
    
    def get_average_confidence(self) -> float:
        """获取平均置信度"""
        if self.recognition_count == 0:
            return 0.0
        return self.total_confidence / self.recognition_count

class RealtimeRecognitionService:
    """实时手语识别服务"""
    
    def __init__(self, cslr_service=None, mediapipe_service=None):
        self.cslr_service = cslr_service
        self.mediapipe_service = mediapipe_service
        self.sessions: Dict[str, RecognitionSession] = {}
        
        # 配置参数
        self.min_frames = 8
        self.max_frames = 64
        self.min_interval = 0.3  # 最小推理间隔（秒）
        self.session_timeout = 300  # 会话超时（秒）
        self.target_vector_size = 543 * 3  # MediaPipe关键点向量大小
        
        # 注册WebSocket处理器
        self._register_handlers()
        
        # 启动清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动服务"""
        try:
            if self.cleanup_task is None or self.cleanup_task.done():
                self.cleanup_task = asyncio.create_task(self._cleanup_sessions())
            
            logger.info("✅ 实时手语识别服务已启动")
            
        except Exception as e:
            logger.error(f"❌ 实时手语识别服务启动失败: {e}")
            raise
    
    async def stop(self):
        """停止服务"""
        try:
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # 清理所有会话
            self.sessions.clear()
            
            logger.info("✅ 实时手语识别服务已停止")
            
        except Exception as e:
            logger.error(f"❌ 实时手语识别服务停止失败: {e}")
    
    def _register_handlers(self):
        """注册WebSocket消息处理器"""
        websocket_manager.register_handler("start_recognition", self._handle_start_recognition)
        websocket_manager.register_handler("stop_recognition", self._handle_stop_recognition)
        websocket_manager.register_handler("landmarks", self._handle_landmarks)
        websocket_manager.register_handler("batch_landmarks", self._handle_batch_landmarks)
        websocket_manager.register_handler("config", self._handle_config)
        websocket_manager.register_handler("get_stats", self._handle_get_stats)
    
    async def _handle_start_recognition(self, connection_info: ConnectionInfo, payload: Dict[str, Any]):
        """处理开始识别请求"""
        try:
            session_id = payload.get("session_id", f"session_{connection_info.connection_id}_{int(time.time())}")
            
            # 创建识别会话
            session = RecognitionSession(
                session_id=session_id,
                connection_id=connection_info.connection_id,
                user_id=connection_info.user_id,
                start_time=time.time(),
                last_activity=time.time(),
                config=payload.get("config", {})
            )
            
            self.sessions[session_id] = session
            
            # 发送确认消息
            await websocket_manager.send_message(
                connection_info.connection_id,
                WebSocketMessage(
                    type="recognition_started",
                    payload={
                        "session_id": session_id,
                        "config": session.config,
                        "timestamp": session.start_time
                    }
                )
            )
            
            logger.info(f"✅ 开始识别会话: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ 开始识别失败: {e}")
            await websocket_manager.send_error(connection_info.connection_id, f"开始识别失败: {str(e)}")
    
    async def _handle_stop_recognition(self, connection_info: ConnectionInfo, payload: Dict[str, Any]):
        """处理停止识别请求"""
        try:
            session_id = payload.get("session_id")
            
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                
                # 生成会话统计
                duration = time.time() - session.start_time
                stats = {
                    "session_id": session_id,
                    "duration": duration,
                    "frame_count": session.frame_count,
                    "recognition_count": session.recognition_count,
                    "average_confidence": session.get_average_confidence()
                }
                
                # 删除会话
                del self.sessions[session_id]
                
                # 发送确认消息
                await websocket_manager.send_message(
                    connection_info.connection_id,
                    WebSocketMessage(
                        type="recognition_stopped",
                        payload=stats
                    )
                )
                
                logger.info(f"✅ 停止识别会话: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ 停止识别失败: {e}")
            await websocket_manager.send_error(connection_info.connection_id, f"停止识别失败: {str(e)}")
    
    async def _handle_landmarks(self, connection_info: ConnectionInfo, payload: Dict[str, Any]):
        """处理关键点数据"""
        try:
            # 查找活跃会话
            session = self._find_active_session(connection_info.connection_id)
            if not session:
                await websocket_manager.send_error(connection_info.connection_id, "没有活跃的识别会话")
                return
            
            # 提取关键点数据
            landmarks = payload.get("landmarks", [])
            timestamp = payload.get("timestamp", time.time())
            frame_id = payload.get("frame_id", session.frame_count)
            
            # 转换关键点为向量
            vector = self._landmarks_to_vector(landmarks)
            if not vector:
                return
            
            # 添加到序列缓冲区
            session.sequence_buffer.append(vector)
            session.frame_count += 1
            session.update_activity()
            
            # 限制缓冲区大小
            if len(session.sequence_buffer) > self.max_frames:
                session.sequence_buffer = session.sequence_buffer[-self.max_frames:]
            
            # 检查是否可以进行识别
            if await self._should_recognize(session):
                await self._perform_recognition(session, frame_id, timestamp)
            
        except Exception as e:
            logger.error(f"❌ 处理关键点数据失败: {e}")
            await websocket_manager.send_error(connection_info.connection_id, f"处理关键点失败: {str(e)}")
    
    async def _handle_batch_landmarks(self, connection_info: ConnectionInfo, payload: Dict[str, Any]):
        """处理批量关键点数据"""
        try:
            messages = payload.get("messages", [])
            
            for message in messages:
                if message.get("type") == "landmarks":
                    await self._handle_landmarks(connection_info, message.get("payload", {}))
            
        except Exception as e:
            logger.error(f"❌ 处理批量关键点失败: {e}")
            await websocket_manager.send_error(connection_info.connection_id, f"处理批量关键点失败: {str(e)}")
    
    async def _handle_config(self, connection_info: ConnectionInfo, payload: Dict[str, Any]):
        """处理配置更新"""
        try:
            session = self._find_active_session(connection_info.connection_id)
            if not session:
                await websocket_manager.send_error(connection_info.connection_id, "没有活跃的识别会话")
                return
            
            # 更新配置
            session.config.update(payload)
            session.update_activity()
            
            # 发送确认
            await websocket_manager.send_message(
                connection_info.connection_id,
                WebSocketMessage(
                    type="config_updated",
                    payload=session.config
                )
            )
            
        except Exception as e:
            logger.error(f"❌ 更新配置失败: {e}")
            await websocket_manager.send_error(connection_info.connection_id, f"更新配置失败: {str(e)}")
    
    async def _handle_get_stats(self, connection_info: ConnectionInfo, payload: Dict[str, Any]):
        """处理获取统计信息请求"""
        try:
            session = self._find_active_session(connection_info.connection_id)
            if not session:
                stats = {"error": "没有活跃的识别会话"}
            else:
                stats = {
                    "session_id": session.session_id,
                    "duration": time.time() - session.start_time,
                    "frame_count": session.frame_count,
                    "recognition_count": session.recognition_count,
                    "average_confidence": session.get_average_confidence(),
                    "buffer_size": len(session.sequence_buffer)
                }
            
            await websocket_manager.send_message(
                connection_info.connection_id,
                WebSocketMessage(
                    type="stats",
                    payload=stats
                )
            )
            
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            await websocket_manager.send_error(connection_info.connection_id, f"获取统计信息失败: {str(e)}")
    
    def _find_active_session(self, connection_id: str) -> Optional[RecognitionSession]:
        """查找活跃会话"""
        for session in self.sessions.values():
            if session.connection_id == connection_id:
                return session
        return None
    
    def _landmarks_to_vector(self, landmarks: List[List[float]]) -> Optional[List[float]]:
        """将关键点转换为固定长度向量"""
        try:
            if not landmarks:
                return None
            
            flat_vector = []
            for point in landmarks:
                if isinstance(point, (list, tuple)) and len(point) >= 3:
                    flat_vector.extend([float(point[0]), float(point[1]), float(point[2])])
            
            # 统一到目标维度
            if len(flat_vector) < self.target_vector_size:
                flat_vector.extend([0.0] * (self.target_vector_size - len(flat_vector)))
            elif len(flat_vector) > self.target_vector_size:
                flat_vector = flat_vector[:self.target_vector_size]
            
            return flat_vector
            
        except Exception as e:
            logger.error(f"❌ 关键点转换失败: {e}")
            return None
    
    async def _should_recognize(self, session: RecognitionSession) -> bool:
        """判断是否应该进行识别"""
        # 检查帧数
        if len(session.sequence_buffer) < self.min_frames:
            return False
        
        # 检查时间间隔
        current_time = time.time()
        if hasattr(session, 'last_recognition_time'):
            if current_time - session.last_recognition_time < self.min_interval:
                return False
        
        return True
    
    async def _perform_recognition(self, session: RecognitionSession, frame_id: int, timestamp: float):
        """执行识别"""
        try:
            if not self.cslr_service:
                return
            
            # 记录识别时间
            session.last_recognition_time = time.time()
            
            # 执行识别
            sequence = list(session.sequence_buffer)
            result = await self.cslr_service.predict(sequence)
            
            if result and hasattr(result, 'status') and result.status == "success":
                # 更新会话统计
                session.add_recognition_result(result.confidence)
                
                # 发送识别结果
                await websocket_manager.send_message(
                    session.connection_id,
                    WebSocketMessage(
                        type="recognition_result",
                        payload={
                            "text": result.text,
                            "confidence": result.confidence,
                            "gloss_sequence": result.gloss_sequence,
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "processing_time": getattr(result, 'processing_time', 0.0)
                        }
                    )
                )
            
        except Exception as e:
            logger.error(f"❌ 执行识别失败: {e}")
            await websocket_manager.send_error(session.connection_id, f"识别失败: {str(e)}")
    
    async def _cleanup_sessions(self):
        """清理超时会话"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                
                # 清理过期会话
                for session_id in expired_sessions:
                    session = self.sessions[session_id]
                    
                    # 通知客户端会话过期
                    try:
                        await websocket_manager.send_message(
                            session.connection_id,
                            WebSocketMessage(
                                type="session_expired",
                                payload={"session_id": session_id}
                            )
                        )
                    except:
                        pass
                    
                    del self.sessions[session_id]
                
                if expired_sessions:
                    logger.info(f"🧹 清理了 {len(expired_sessions)} 个过期识别会话")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 会话清理出错: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        current_time = time.time()
        
        total_sessions = len(self.sessions)
        total_frames = sum(session.frame_count for session in self.sessions.values())
        total_recognitions = sum(session.recognition_count for session in self.sessions.values())
        
        if self.sessions:
            avg_session_duration = sum(
                current_time - session.start_time 
                for session in self.sessions.values()
            ) / total_sessions
            
            avg_confidence = sum(
                session.get_average_confidence() 
                for session in self.sessions.values()
            ) / total_sessions
        else:
            avg_session_duration = 0
            avg_confidence = 0
        
        return {
            "active_sessions": total_sessions,
            "total_frames_processed": total_frames,
            "total_recognitions": total_recognitions,
            "average_session_duration": avg_session_duration,
            "average_confidence": avg_confidence,
            "service_available": self.cslr_service is not None
        }
