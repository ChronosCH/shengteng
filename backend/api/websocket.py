"""
WebSocket连接管理器
处理实时通信、连接池管理、消息广播等功能
"""

import asyncio
import json
import time
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from utils.logger import setup_logger
from utils.monitoring import performance_monitor

logger = setup_logger(__name__)


class ConnectionInfo(BaseModel):
    """连接信息"""
    connection_id: str
    user_id: Optional[int] = None
    ip_address: str
    user_agent: str
    connected_at: float
    last_activity: float
    room: Optional[str] = None
    metadata: Dict = {}


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活跃连接
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        
        # 房间管理
        self.rooms: Dict[str, Set[str]] = defaultdict(set)
        
        # 消息统计
        self.message_stats = {
            "total_messages": 0,
            "messages_by_type": defaultdict(int),
            "errors": 0,
            "disconnections": 0
        }
        
        # 心跳检测
        self.heartbeat_interval = 30  # 30秒
        self.heartbeat_task = None
        
        logger.info("WebSocket管理器初始化完成")
    
    async def connect(self, websocket: WebSocket, user_id: int = None, 
                     room: str = None, metadata: Dict = None):
        """建立WebSocket连接"""
        try:
            await websocket.accept()
            
            # 生成连接ID
            connection_id = str(uuid.uuid4())
            
            # 存储连接
            self.active_connections[connection_id] = websocket
            
            # 创建连接信息
            current_time = time.time()
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                user_id=user_id,
                ip_address=getattr(websocket.client, 'host', 'unknown'),
                user_agent=websocket.headers.get('user-agent', 'unknown'),
                connected_at=current_time,
                last_activity=current_time,
                room=room,
                metadata=metadata or {}
            )
            
            self.connection_info[connection_id] = connection_info
            
            # 加入房间
            if room:
                self.rooms[room].add(connection_id)
            
            # 更新监控指标
            performance_monitor.metrics_collector.set_active_connections(
                len(self.active_connections)
            )
            
            # 启动心跳检测（如果还没启动）
            if not self.heartbeat_task:
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info(f"WebSocket连接建立: {connection_id}, 用户: {user_id}, 房间: {room}")
            
            # 发送连接确认
            await self.send_to_connection(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": current_time
            })
            
            return connection_id
            
        except Exception as e:
            logger.error(f"建立WebSocket连接失败: {e}")
            raise
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        connection_id = None
        
        # 查找连接ID
        for conn_id, ws in self.active_connections.items():
            if ws == websocket:
                connection_id = conn_id
                break
        
        if connection_id:
            self._remove_connection(connection_id)
    
    def disconnect_by_id(self, connection_id: str):
        """根据连接ID断开连接"""
        self._remove_connection(connection_id)
    
    def _remove_connection(self, connection_id: str):
        """移除连接"""
        try:
            # 移除连接
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            # 获取连接信息
            connection_info = self.connection_info.get(connection_id)
            
            # 从房间移除
            if connection_info and connection_info.room:
                self.rooms[connection_info.room].discard(connection_id)
                
                # 如果房间为空，删除房间
                if not self.rooms[connection_info.room]:
                    del self.rooms[connection_info.room]
            
            # 移除连接信息
            if connection_id in self.connection_info:
                del self.connection_info[connection_id]
            
            # 更新统计
            self.message_stats["disconnections"] += 1
            
            # 更新监控指标
            performance_monitor.metrics_collector.set_active_connections(
                len(self.active_connections)
            )
            
            logger.info(f"WebSocket连接已断开: {connection_id}")
            
        except Exception as e:
            logger.error(f"移除WebSocket连接失败: {e}")
    
    async def send_to_connection(self, connection_id: str, message: Dict):
        """向指定连接发送消息"""
        try:
            websocket = self.active_connections.get(connection_id)
            if websocket:
                await websocket.send_json(message)
                
                # 更新活动时间
                if connection_id in self.connection_info:
                    self.connection_info[connection_id].last_activity = time.time()
                
                # 更新统计
                self.message_stats["total_messages"] += 1
                self.message_stats["messages_by_type"][message.get("type", "unknown")] += 1
                
                return True
            else:
                logger.warning(f"连接不存在: {connection_id}")
                return False
                
        except Exception as e:
            logger.error(f"发送WebSocket消息失败: {e}")
            self.message_stats["errors"] += 1
            
            # 连接可能已断开，移除它
            self._remove_connection(connection_id)
            return False
    
    async def send_to_user(self, user_id: int, message: Dict):
        """向指定用户的所有连接发送消息"""
        sent_count = 0
        
        for connection_id, connection_info in self.connection_info.items():
            if connection_info.user_id == user_id:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def send_to_room(self, room: str, message: Dict, exclude_connection: str = None):
        """向房间内所有连接发送消息"""
        sent_count = 0
        
        if room in self.rooms:
            for connection_id in self.rooms[room].copy():  # 使用copy避免迭代时修改
                if connection_id != exclude_connection:
                    if await self.send_to_connection(connection_id, message):
                        sent_count += 1
        
        return sent_count
    
    async def broadcast(self, message: Dict, exclude_connection: str = None):
        """向所有连接广播消息"""
        sent_count = 0
        
        for connection_id in list(self.active_connections.keys()):  # 避免迭代时修改
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def join_room(self, connection_id: str, room: str):
        """加入房间"""
        if connection_id in self.connection_info:
            # 离开之前的房间
            old_room = self.connection_info[connection_id].room
            if old_room:
                self.rooms[old_room].discard(connection_id)
                if not self.rooms[old_room]:
                    del self.rooms[old_room]
            
            # 加入新房间
            self.rooms[room].add(connection_id)
            self.connection_info[connection_id].room = room
            
            logger.info(f"连接 {connection_id} 加入房间 {room}")
            return True
        
        return False
    
    async def leave_room(self, connection_id: str):
        """离开房间"""
        if connection_id in self.connection_info:
            room = self.connection_info[connection_id].room
            if room:
                self.rooms[room].discard(connection_id)
                if not self.rooms[room]:
                    del self.rooms[room]
                
                self.connection_info[connection_id].room = None
                
                logger.info(f"连接 {connection_id} 离开房间 {room}")
                return True
        
        return False
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """获取连接信息"""
        return self.connection_info.get(connection_id)
    
    def get_user_connections(self, user_id: int) -> List[str]:
        """获取用户的所有连接"""
        return [
            connection_id for connection_id, info in self.connection_info.items()
            if info.user_id == user_id
        ]
    
    def get_room_connections(self, room: str) -> Set[str]:
        """获取房间内的所有连接"""
        return self.rooms.get(room, set()).copy()
    
    def get_stats(self) -> Dict:
        """获取WebSocket统计信息"""
        current_time = time.time()
        
        # 计算连接持续时间统计
        connection_durations = [
            current_time - info.connected_at
            for info in self.connection_info.values()
        ]
        
        avg_duration = sum(connection_durations) / len(connection_durations) if connection_durations else 0
        
        # 用户分布统计
        users_online = len(set(
            info.user_id for info in self.connection_info.values()
            if info.user_id is not None
        ))
        
        return {
            "active_connections": len(self.active_connections),
            "total_rooms": len(self.rooms),
            "users_online": users_online,
            "message_stats": dict(self.message_stats),
            "average_connection_duration": avg_duration,
            "rooms": {room: len(connections) for room, connections in self.rooms.items()},
            "connections_by_user": len([
                info for info in self.connection_info.values()
                if info.user_id is not None
            ])
        }
    
    async def _heartbeat_loop(self):
        """心跳检测循环"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                current_time = time.time()
                timeout_threshold = current_time - (self.heartbeat_interval * 2)
                
                # 检查超时连接
                timeout_connections = []
                for connection_id, info in self.connection_info.items():
                    if info.last_activity < timeout_threshold:
                        timeout_connections.append(connection_id)
                
                # 移除超时连接
                for connection_id in timeout_connections:
                    logger.info(f"移除超时连接: {connection_id}")
                    self._remove_connection(connection_id)
                
                # 发送心跳消息给活跃连接
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": current_time
                }
                
                failed_connections = []
                for connection_id in list(self.active_connections.keys()):
                    try:
                        websocket = self.active_connections[connection_id]
                        await websocket.send_json(heartbeat_message)
                    except Exception as e:
                        logger.warning(f"心跳发送失败: {connection_id}, {e}")
                        failed_connections.append(connection_id)
                
                # 移除失败的连接
                for connection_id in failed_connections:
                    self._remove_connection(connection_id)
                
            except Exception as e:
                logger.error(f"心跳检测循环错误: {e}")
                await asyncio.sleep(5)  # 出错时短暂休息
    
    async def cleanup(self):
        """清理资源"""
        # 停止心跳检测
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        for connection_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"关闭WebSocket连接失败: {e}")
        
        # 清理数据
        self.active_connections.clear()
        self.connection_info.clear()
        self.rooms.clear()
        
        logger.info("WebSocket管理器已清理")
