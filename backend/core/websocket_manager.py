"""
WebSocket连接管理器
WebSocket Connection Manager
提供稳定的WebSocket连接管理、消息路由、连接池等功能
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import weakref

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """连接状态"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MessageType(Enum):
    """消息类型"""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    DATA = "data"
    ERROR = "error"
    SYSTEM = "system"

@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    type: str
    payload: Any
    timestamp: float = None
    message_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }

@dataclass
class ConnectionInfo:
    """连接信息"""
    connection_id: str
    websocket: WebSocket
    state: ConnectionState
    connected_at: float
    last_ping: float
    last_pong: float
    user_id: Optional[str] = None
    session_data: Dict[str, Any] = None
    message_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.session_data is None:
            self.session_data = {}

class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.connections: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.message_handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        # 配置
        self.ping_interval = 30  # 心跳间隔（秒）
        self.ping_timeout = 10   # 心跳超时（秒）
        self.max_message_size = 1024 * 1024  # 最大消息大小（1MB）
        self.max_connections_per_user = 5
        
        # 统计信息
        self.total_connections = 0
        self.total_messages = 0
        self.total_errors = 0
        
        # 后台任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self.ping_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动WebSocket管理器"""
        try:
            # 启动清理任务
            if self.cleanup_task is None or self.cleanup_task.done():
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # 启动心跳任务
            if self.ping_task is None or self.ping_task.done():
                self.ping_task = asyncio.create_task(self._ping_loop())
            
            logger.info("✅ WebSocket管理器已启动")
            
        except Exception as e:
            logger.error(f"❌ WebSocket管理器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止WebSocket管理器"""
        try:
            # 停止后台任务
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self.ping_task and not self.ping_task.done():
                self.ping_task.cancel()
                try:
                    await self.ping_task
                except asyncio.CancelledError:
                    pass
            
            # 关闭所有连接
            await self.disconnect_all()
            
            logger.info("✅ WebSocket管理器已停止")
            
        except Exception as e:
            logger.error(f"❌ WebSocket管理器停止失败: {e}")
    
    async def connect(self, websocket: WebSocket, user_id: str = None) -> str:
        """建立WebSocket连接"""
        try:
            await websocket.accept()
            
            connection_id = str(uuid.uuid4())
            current_time = time.time()
            
            # 检查用户连接数限制
            if user_id:
                user_connection_count = len(self.user_connections.get(user_id, set()))
                if user_connection_count >= self.max_connections_per_user:
                    await websocket.close(code=1008, reason="Too many connections")
                    raise Exception(f"用户 {user_id} 连接数超过限制")
            
            # 创建连接信息
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                websocket=websocket,
                state=ConnectionState.CONNECTED,
                connected_at=current_time,
                last_ping=current_time,
                last_pong=current_time,
                user_id=user_id
            )
            
            # 保存连接
            self.connections[connection_id] = connection_info
            
            # 更新用户连接映射
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            # 更新统计
            self.total_connections += 1
            
            # 发送连接确认消息
            await self.send_message(connection_id, WebSocketMessage(
                type=MessageType.CONNECT.value,
                payload={
                    "connection_id": connection_id,
                    "connected_at": current_time,
                    "server_time": current_time
                }
            ))
            
            logger.info(f"✅ WebSocket连接建立: {connection_id}, 用户: {user_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"❌ WebSocket连接建立失败: {e}")
            raise
    
    async def disconnect(self, connection_id: str, code: int = 1000, reason: str = "Normal closure"):
        """断开WebSocket连接"""
        try:
            if connection_id not in self.connections:
                return
            
            connection_info = self.connections[connection_id]
            connection_info.state = ConnectionState.DISCONNECTING
            
            # 发送断开连接消息
            try:
                await self.send_message(connection_id, WebSocketMessage(
                    type=MessageType.DISCONNECT.value,
                    payload={"reason": reason}
                ))
            except:
                pass  # 忽略发送失败
            
            # 关闭WebSocket连接
            try:
                await connection_info.websocket.close(code=code, reason=reason)
            except:
                pass  # 忽略关闭失败
            
            # 清理连接信息
            self._cleanup_connection(connection_id)
            
            logger.info(f"✅ WebSocket连接断开: {connection_id}")
            
        except Exception as e:
            logger.error(f"❌ WebSocket连接断开失败: {e}")
    
    async def disconnect_all(self):
        """断开所有连接"""
        connection_ids = list(self.connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id, reason="Server shutdown")
    
    def _cleanup_connection(self, connection_id: str):
        """清理连接信息"""
        if connection_id not in self.connections:
            return
        
        connection_info = self.connections[connection_id]
        
        # 从用户连接映射中移除
        if connection_info.user_id:
            user_connections = self.user_connections.get(connection_info.user_id)
            if user_connections:
                user_connections.discard(connection_id)
                if not user_connections:
                    del self.user_connections[connection_info.user_id]
        
        # 移除连接
        del self.connections[connection_id]
        connection_info.state = ConnectionState.DISCONNECTED
    
    async def send_message(self, connection_id: str, message: WebSocketMessage) -> bool:
        """发送消息到指定连接"""
        try:
            if connection_id not in self.connections:
                return False
            
            connection_info = self.connections[connection_id]
            
            if connection_info.state != ConnectionState.CONNECTED:
                return False
            
            # 应用中间件
            for middleware in self.middleware:
                message = await middleware(connection_info, message)
                if message is None:
                    return False
            
            # 检查消息大小
            message_data = message.to_dict()
            message_json = json.dumps(message_data)
            
            if len(message_json.encode('utf-8')) > self.max_message_size:
                logger.warning(f"⚠️ 消息过大，跳过发送: {connection_id}")
                return False
            
            # 发送消息
            await connection_info.websocket.send_text(message_json)
            
            # 更新统计
            connection_info.message_count += 1
            self.total_messages += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 发送消息失败: {connection_id}, {e}")
            connection_info = self.connections.get(connection_id)
            if connection_info:
                connection_info.error_count += 1
                self.total_errors += 1
            return False
    
    async def broadcast_message(self, message: WebSocketMessage, user_ids: List[str] = None) -> int:
        """广播消息"""
        sent_count = 0
        
        if user_ids:
            # 发送给指定用户
            for user_id in user_ids:
                connection_ids = self.user_connections.get(user_id, set())
                for connection_id in connection_ids:
                    if await self.send_message(connection_id, message):
                        sent_count += 1
        else:
            # 发送给所有连接
            for connection_id in list(self.connections.keys()):
                if await self.send_message(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """处理接收到的消息"""
        try:
            if connection_id not in self.connections:
                return
            
            connection_info = self.connections[connection_id]
            
            # 解析消息
            try:
                message_data = json.loads(raw_message)
                message_type = message_data.get("type", "")
                payload = message_data.get("payload", {})
            except json.JSONDecodeError:
                await self.send_error(connection_id, "Invalid JSON format")
                return
            
            # 处理系统消息
            if message_type == MessageType.PING.value:
                await self.send_message(connection_id, WebSocketMessage(
                    type=MessageType.PONG.value,
                    payload={"timestamp": time.time()}
                ))
                connection_info.last_ping = time.time()
                return
            
            elif message_type == MessageType.PONG.value:
                connection_info.last_pong = time.time()
                return
            
            # 调用注册的处理器
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                try:
                    await handler(connection_info, payload)
                except Exception as e:
                    logger.error(f"❌ 消息处理器执行失败: {message_type}, {e}")
                    await self.send_error(connection_id, f"Handler error: {str(e)}")
            else:
                logger.warning(f"⚠️ 未知消息类型: {message_type}")
                await self.send_error(connection_id, f"Unknown message type: {message_type}")
            
        except Exception as e:
            logger.error(f"❌ 处理消息失败: {connection_id}, {e}")
            await self.send_error(connection_id, "Message processing error")
    
    async def send_error(self, connection_id: str, error_message: str):
        """发送错误消息"""
        await self.send_message(connection_id, WebSocketMessage(
            type=MessageType.ERROR.value,
            payload={"message": error_message}
        ))
    
    def register_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        logger.info(f"✅ 注册消息处理器: {message_type}")
    
    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middleware.append(middleware)
        logger.info(f"✅ 添加WebSocket中间件")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                current_time = time.time()
                disconnected_connections = []
                
                for connection_id, connection_info in self.connections.items():
                    # 检查连接超时
                    if current_time - connection_info.last_pong > self.ping_timeout * 3:
                        disconnected_connections.append(connection_id)
                
                # 清理超时连接
                for connection_id in disconnected_connections:
                    await self.disconnect(connection_id, reason="Connection timeout")
                
                if disconnected_connections:
                    logger.info(f"🧹 清理了 {len(disconnected_connections)} 个超时连接")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 连接清理出错: {e}")
    
    async def _ping_loop(self):
        """心跳循环"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                current_time = time.time()
                ping_message = WebSocketMessage(
                    type=MessageType.PING.value,
                    payload={"timestamp": current_time}
                )
                
                # 发送心跳到所有连接
                for connection_id in list(self.connections.keys()):
                    await self.send_message(connection_id, ping_message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 心跳发送出错: {e}")
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """获取连接信息"""
        return self.connections.get(connection_id)
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """获取用户的所有连接"""
        return list(self.user_connections.get(user_id, set()))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        
        # 计算连接统计
        active_connections = len(self.connections)
        user_count = len(self.user_connections)
        
        # 计算平均连接时间
        if self.connections:
            avg_connection_time = sum(
                current_time - conn.connected_at 
                for conn in self.connections.values()
            ) / len(self.connections)
        else:
            avg_connection_time = 0
        
        return {
            "active_connections": active_connections,
            "total_connections": self.total_connections,
            "active_users": user_count,
            "total_messages": self.total_messages,
            "total_errors": self.total_errors,
            "avg_connection_time": avg_connection_time,
            "error_rate": self.total_errors / max(1, self.total_messages),
            "connections_per_user": active_connections / max(1, user_count)
        }

# 全局WebSocket管理器实例
websocket_manager = WebSocketManager()

# 便捷函数
async def start_websocket_manager():
    """启动WebSocket管理器"""
    await websocket_manager.start()

async def stop_websocket_manager():
    """停止WebSocket管理器"""
    await websocket_manager.stop()
