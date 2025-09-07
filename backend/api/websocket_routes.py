"""
WebSocket API路由
WebSocket API Routes
提供改进的WebSocket端点和实时通信功能
"""

import logging
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Optional

from backend.core.websocket_manager import websocket_manager, WebSocketMessage
from backend.services.realtime_recognition_service import RealtimeRecognitionService

logger = logging.getLogger(__name__)

router = APIRouter()

# 全局实时识别服务实例
realtime_recognition_service: Optional[RealtimeRecognitionService] = None

def get_realtime_service():
    """获取实时识别服务"""
    global realtime_recognition_service
    return realtime_recognition_service

def init_realtime_service(cslr_service=None, mediapipe_service=None):
    """初始化实时识别服务"""
    global realtime_recognition_service
    realtime_recognition_service = RealtimeRecognitionService(cslr_service, mediapipe_service)
    return realtime_recognition_service

@router.websocket("/ws/sign-recognition")
async def websocket_sign_recognition(websocket: WebSocket):
    """
    改进的实时手语识别WebSocket端点
    
    支持的消息类型:
    - start_recognition: 开始识别会话
    - stop_recognition: 停止识别会话
    - landmarks: 发送关键点数据
    - batch_landmarks: 批量发送关键点数据
    - config: 更新配置
    - get_stats: 获取统计信息
    - ping: 心跳检测
    
    响应消息类型:
    - connect: 连接确认
    - recognition_started: 识别会话开始
    - recognition_stopped: 识别会话停止
    - recognition_result: 识别结果
    - config_updated: 配置更新确认
    - stats: 统计信息
    - pong: 心跳响应
    - error: 错误信息
    """
    connection_id = None
    
    try:
        # 建立WebSocket连接
        connection_id = await websocket_manager.connect(websocket)
        
        # 检查实时识别服务是否可用
        service = get_realtime_service()
        if not service:
            await websocket_manager.send_message(
                connection_id,
                WebSocketMessage(
                    type="error",
                    payload={"message": "实时识别服务不可用"}
                )
            )
            await websocket_manager.disconnect(connection_id, reason="Service unavailable")
            return
        
        # 消息处理循环
        while True:
            try:
                # 接收消息
                raw_message = await websocket.receive_text()
                
                # 处理消息
                await websocket_manager.handle_message(connection_id, raw_message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket客户端断开连接: {connection_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket消息处理错误: {e}")
                await websocket_manager.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="error",
                        payload={"message": f"消息处理错误: {str(e)}"}
                    )
                )
    
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    
    finally:
        # 清理连接
        if connection_id:
            await websocket_manager.disconnect(connection_id)

@router.websocket("/ws/learning")
async def websocket_learning(websocket: WebSocket):
    """
    学习训练WebSocket端点
    
    支持的消息类型:
    - start_session: 开始学习会话
    - end_session: 结束学习会话
    - lesson_progress: 课程进度更新
    - exercise_result: 练习结果
    - achievement_check: 成就检查
    """
    connection_id = None
    
    try:
        # 建立WebSocket连接
        connection_id = await websocket_manager.connect(websocket)
        
        # 注册学习相关的消息处理器
        await _register_learning_handlers(connection_id)
        
        # 消息处理循环
        while True:
            try:
                raw_message = await websocket.receive_text()
                await websocket_manager.handle_message(connection_id, raw_message)
                
            except WebSocketDisconnect:
                logger.info(f"学习WebSocket客户端断开连接: {connection_id}")
                break
            except Exception as e:
                logger.error(f"学习WebSocket消息处理错误: {e}")
                await websocket_manager.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="error",
                        payload={"message": f"消息处理错误: {str(e)}"}
                    )
                )
    
    except Exception as e:
        logger.error(f"学习WebSocket连接错误: {e}")
    
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)

async def _register_learning_handlers(connection_id: str):
    """注册学习相关的消息处理器"""
    
    async def handle_start_session(connection_info, payload):
        """处理开始学习会话"""
        try:
            session_data = {
                "session_id": f"learning_{connection_id}_{int(time.time())}",
                "user_id": connection_info.user_id,
                "start_time": time.time(),
                "course_id": payload.get("course_id"),
                "lesson_id": payload.get("lesson_id")
            }
            
            # 保存会话数据到连接信息
            connection_info.session_data.update(session_data)
            
            await websocket_manager.send_message(
                connection_id,
                WebSocketMessage(
                    type="session_started",
                    payload=session_data
                )
            )
            
        except Exception as e:
            logger.error(f"开始学习会话失败: {e}")
            await websocket_manager.send_error(connection_id, f"开始学习会话失败: {str(e)}")
    
    async def handle_end_session(connection_info, payload):
        """处理结束学习会话"""
        try:
            session_data = connection_info.session_data
            if not session_data.get("session_id"):
                await websocket_manager.send_error(connection_id, "没有活跃的学习会话")
                return
            
            # 计算会话统计
            duration = time.time() - session_data.get("start_time", 0)
            stats = {
                "session_id": session_data["session_id"],
                "duration": duration,
                "completed_lessons": payload.get("completed_lessons", []),
                "total_score": payload.get("total_score", 0),
                "achievements": payload.get("achievements", [])
            }
            
            await websocket_manager.send_message(
                connection_id,
                WebSocketMessage(
                    type="session_ended",
                    payload=stats
                )
            )
            
            # 清理会话数据
            connection_info.session_data.clear()
            
        except Exception as e:
            logger.error(f"结束学习会话失败: {e}")
            await websocket_manager.send_error(connection_id, f"结束学习会话失败: {str(e)}")
    
    async def handle_lesson_progress(connection_info, payload):
        """处理课程进度更新"""
        try:
            progress_data = {
                "lesson_id": payload.get("lesson_id"),
                "progress": payload.get("progress", 0),
                "score": payload.get("score", 0),
                "timestamp": time.time()
            }
            
            # 这里可以调用学习服务保存进度
            # learning_service.update_progress(connection_info.user_id, progress_data)
            
            await websocket_manager.send_message(
                connection_id,
                WebSocketMessage(
                    type="progress_updated",
                    payload=progress_data
                )
            )
            
        except Exception as e:
            logger.error(f"更新课程进度失败: {e}")
            await websocket_manager.send_error(connection_id, f"更新课程进度失败: {str(e)}")
    
    async def handle_exercise_result(connection_info, payload):
        """处理练习结果"""
        try:
            result_data = {
                "exercise_id": payload.get("exercise_id"),
                "score": payload.get("score", 0),
                "accuracy": payload.get("accuracy", 0),
                "time_spent": payload.get("time_spent", 0),
                "mistakes": payload.get("mistakes", []),
                "timestamp": time.time()
            }
            
            # 这里可以调用学习服务保存结果
            # learning_service.save_exercise_result(connection_info.user_id, result_data)
            
            await websocket_manager.send_message(
                connection_id,
                WebSocketMessage(
                    type="exercise_completed",
                    payload=result_data
                )
            )
            
        except Exception as e:
            logger.error(f"保存练习结果失败: {e}")
            await websocket_manager.send_error(connection_id, f"保存练习结果失败: {str(e)}")
    
    async def handle_achievement_check(connection_info, payload):
        """处理成就检查"""
        try:
            event_data = payload.get("event_data", {})
            event_type = payload.get("event_type", "")
            
            # 这里可以调用成就服务检查成就
            # unlocked_achievements = await achievement_service.trigger_achievement_check(
            #     connection_info.user_id, event_type, event_data
            # )
            
            unlocked_achievements = []  # 模拟数据
            
            if unlocked_achievements:
                await websocket_manager.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="achievements_unlocked",
                        payload={"achievements": unlocked_achievements}
                    )
                )
            
        except Exception as e:
            logger.error(f"成就检查失败: {e}")
            await websocket_manager.send_error(connection_id, f"成就检查失败: {str(e)}")
    
    # 临时注册处理器（仅对当前连接有效）
    # 注意：这是一个简化的实现，实际应用中可能需要更复杂的处理器管理
    import time
    
    websocket_manager.register_handler("start_session", handle_start_session)
    websocket_manager.register_handler("end_session", handle_end_session)
    websocket_manager.register_handler("lesson_progress", handle_lesson_progress)
    websocket_manager.register_handler("exercise_result", handle_exercise_result)
    websocket_manager.register_handler("achievement_check", handle_achievement_check)

@router.websocket("/ws/system")
async def websocket_system(websocket: WebSocket):
    """
    系统监控WebSocket端点
    
    支持的消息类型:
    - get_system_stats: 获取系统统计
    - get_service_status: 获取服务状态
    - subscribe_updates: 订阅系统更新
    """
    connection_id = None
    
    try:
        connection_id = await websocket_manager.connect(websocket)
        
        # 注册系统监控处理器
        async def handle_get_system_stats(connection_info, payload):
            """获取系统统计"""
            try:
                # 获取WebSocket统计
                ws_stats = websocket_manager.get_stats()
                
                # 获取实时识别服务统计
                recognition_stats = {}
                service = get_realtime_service()
                if service:
                    recognition_stats = service.get_service_stats()
                
                stats = {
                    "websocket": ws_stats,
                    "recognition": recognition_stats,
                    "timestamp": time.time()
                }
                
                await websocket_manager.send_message(
                    connection_id,
                    WebSocketMessage(
                        type="system_stats",
                        payload=stats
                    )
                )
                
            except Exception as e:
                logger.error(f"获取系统统计失败: {e}")
                await websocket_manager.send_error(connection_id, f"获取系统统计失败: {str(e)}")
        
        websocket_manager.register_handler("get_system_stats", handle_get_system_stats)
        
        # 消息处理循环
        while True:
            try:
                raw_message = await websocket.receive_text()
                await websocket_manager.handle_message(connection_id, raw_message)
                
            except WebSocketDisconnect:
                logger.info(f"系统监控WebSocket客户端断开连接: {connection_id}")
                break
            except Exception as e:
                logger.error(f"系统监控WebSocket消息处理错误: {e}")
    
    except Exception as e:
        logger.error(f"系统监控WebSocket连接错误: {e}")
    
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)
