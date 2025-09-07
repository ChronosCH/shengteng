"""
数据库管理模块
提供用户管理、数据存储、模型版本管理等功能
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import sqlite3
import aiosqlite
import hashlib
import uuid
from pathlib import Path

from utils.logger import setup_logger
from utils.config import settings

logger = setup_logger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "data/signavatar.db"):
        self.db_path = db_path
        self.connection = None
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """确保数据库目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """初始化数据库"""
        try:
            await self._create_tables()
            logger.info("数据库初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    async def _create_tables(self):
        """创建数据库表"""
        async with aiosqlite.connect(self.db_path) as db:
            # 用户表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    avatar_url TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    preferences JSON,
                    accessibility_settings JSON
                )
            """)
            
            # 手语识别历史表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT NOT NULL,
                    recognized_text TEXT,
                    confidence REAL,
                    gloss_sequence JSON,
                    landmarks_data JSON,
                    processing_time REAL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # 手语生成历史表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS generation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    generated_sequence JSON,
                    emotion TEXT,
                    speed TEXT,
                    duration REAL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # 用户反馈表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT,
                    feedback_type TEXT NOT NULL,
                    content TEXT,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    recognition_accuracy REAL,
                    suggestions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # 系统日志表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details JSON,
                    user_id INTEGER,
                    ip_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # 模型版本表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    md5_hash TEXT,
                    accuracy_metrics JSON,
                    training_info JSON,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, version)
                )
            """)
            
            # 会话表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    duration INTEGER,
                    recognition_count INTEGER DEFAULT 0,
                    generation_count INTEGER DEFAULT 0,
                    device_info JSON,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # 设备配置表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS device_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    device_type TEXT NOT NULL,
                    config_data JSON NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            await db.commit()
    
    # 用户管理方法
    async def create_user(self, username: str, email: str, password: str, 
                         full_name: str = None, **kwargs) -> int:
        """创建新用户"""
        password_hash = self._hash_password(password)
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO users (username, email, password_hash, full_name, preferences, accessibility_settings)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, email, password_hash, full_name, 
                  json.dumps(kwargs.get('preferences', {})),
                  json.dumps(kwargs.get('accessibility_settings', {}))))
            
            await db.commit()
            return cursor.lastrowid
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """验证用户登录"""
        password_hash = self._hash_password(password)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, full_name, is_active, is_admin, preferences, accessibility_settings
                FROM users WHERE username = ? AND password_hash = ? AND is_active = TRUE
            """, (username, password_hash)) as cursor:
                
                row = await cursor.fetchone()
                if row:
                    # 更新最后登录时间
                    await db.execute("""
                        UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                    """, (row[0],))
                    await db.commit()
                    
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[3],
                        'is_active': bool(row[4]),
                        'is_admin': bool(row[5]),
                        'preferences': json.loads(row[6] or '{}'),
                        'accessibility_settings': json.loads(row[7] or '{}')
                    }
                return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """根据ID获取用户信息"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, full_name, avatar_url, is_active, is_admin,
                       created_at, last_login, preferences, accessibility_settings
                FROM users WHERE id = ?
            """, (user_id,)) as cursor:
                
                row = await cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[3],
                        'avatar_url': row[4],
                        'is_active': bool(row[5]),
                        'is_admin': bool(row[6]),
                        'created_at': row[7],
                        'last_login': row[8],
                        'preferences': json.loads(row[9] or '{}'),
                        'accessibility_settings': json.loads(row[10] or '{}')
                    }
                return None

    async def get_user_by_username(self, username: str) -> Optional[Dict]:
        """根据用户名获取用户信息"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, full_name, avatar_url, is_active, is_admin,
                       created_at, last_login, preferences, accessibility_settings
                FROM users WHERE username = ?
            """, (username,)) as cursor:

                row = await cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[3],
                        'avatar_url': row[4],
                        'is_active': bool(row[5]),
                        'is_admin': bool(row[6]),
                        'created_at': row[7],
                        'last_login': row[8],
                        'preferences': json.loads(row[9] or '{}'),
                        'accessibility_settings': json.loads(row[10] or '{}')
                    }
                return None

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """根据邮箱获取用户信息"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, full_name, avatar_url, is_active, is_admin,
                       created_at, last_login, preferences, accessibility_settings
                FROM users WHERE email = ?
            """, (email,)) as cursor:

                row = await cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[3],
                        'avatar_url': row[4],
                        'is_active': bool(row[5]),
                        'is_admin': bool(row[6]),
                        'created_at': row[7],
                        'last_login': row[8],
                        'preferences': json.loads(row[9] or '{}'),
                        'accessibility_settings': json.loads(row[10] or '{}')
                    }
                return None

    async def update_user_profile(self, user_id: int, full_name: str = None,
                                email: str = None, preferences: Dict = None,
                                accessibility_settings: Dict = None) -> bool:
        """更新用户个人资料"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 构建动态更新语句
                update_fields = []
                params = []

                if full_name is not None:
                    update_fields.append("full_name = ?")
                    params.append(full_name)

                if email is not None:
                    update_fields.append("email = ?")
                    params.append(email)

                if preferences is not None:
                    update_fields.append("preferences = ?")
                    params.append(json.dumps(preferences))

                if accessibility_settings is not None:
                    update_fields.append("accessibility_settings = ?")
                    params.append(json.dumps(accessibility_settings))

                if not update_fields:
                    return True  # 没有需要更新的字段

                # 添加更新时间
                update_fields.append("updated_at = CURRENT_TIMESTAMP")
                params.append(user_id)

                query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
                await db.execute(query, params)
                await db.commit()

                return True
        except Exception as e:
            logger.error(f"更新用户个人资料失败: {e}")
            return False

    async def update_user_password(self, user_id: int, new_password: str) -> bool:
        """更新用户密码"""
        try:
            password_hash = self._hash_password(new_password)

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE users
                    SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (password_hash, user_id))
                await db.commit()

                return True
        except Exception as e:
            logger.error(f"更新用户密码失败: {e}")
            return False

    # 会话管理方法
    async def create_session(self, user_id: int = None, device_info: Dict = None, 
                           ip_address: str = None, user_agent: str = None) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO sessions (id, user_id, device_info, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, json.dumps(device_info or {}), ip_address, user_agent))
            await db.commit()
        
        return session_id
    
    async def end_session(self, session_id: str):
        """结束会话"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE sessions 
                SET end_time = CURRENT_TIMESTAMP,
                    duration = (julianday(CURRENT_TIMESTAMP) - julianday(start_time)) * 86400
                WHERE id = ?
            """, (session_id,))
            await db.commit()

    async def get_user_sessions(self, user_id: int) -> List[Dict]:
        """获取用户的活跃会话"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, device_info, ip_address, user_agent, start_time, end_time
                FROM sessions
                WHERE user_id = ? AND end_time IS NULL
                ORDER BY start_time DESC
            """, (user_id,)) as cursor:

                sessions = []
                async for row in cursor:
                    sessions.append({
                        'id': row[0],
                        'device_info': json.loads(row[1] or '{}'),
                        'ip_address': row[2],
                        'user_agent': row[3],
                        'start_time': row[4],
                        'end_time': row[5]
                    })
                return sessions

    async def get_session_by_id(self, session_id: str) -> Optional[Dict]:
        """根据ID获取会话信息"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, user_id, device_info, ip_address, user_agent, start_time, end_time
                FROM sessions WHERE id = ?
            """, (session_id,)) as cursor:

                row = await cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'user_id': row[1],
                        'device_info': json.loads(row[2] or '{}'),
                        'ip_address': row[3],
                        'user_agent': row[4],
                        'start_time': row[5],
                        'end_time': row[6]
                    }
                return None

    # 识别历史记录方法
    async def save_recognition_result(self, session_id: str, user_id: int = None, 
                                    recognized_text: str = "", confidence: float = 0.0,
                                    gloss_sequence: List = None, landmarks_data: List = None,
                                    processing_time: float = 0.0, model_version: str = ""):
        """保存识别结果"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO recognition_history 
                (user_id, session_id, recognized_text, confidence, gloss_sequence, 
                 landmarks_data, processing_time, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, recognized_text, confidence,
                  json.dumps(gloss_sequence or []), json.dumps(landmarks_data or []),
                  processing_time, model_version))
            
            # 更新会话统计
            await db.execute("""
                UPDATE sessions SET recognition_count = recognition_count + 1 WHERE id = ?
            """, (session_id,))
            
            await db.commit()
    
    # 生成历史记录方法
    async def save_generation_result(self, session_id: str, user_id: int = None,
                                   input_text: str = "", generated_sequence: Dict = None,
                                   emotion: str = "", speed: str = "", duration: float = 0.0,
                                   model_version: str = ""):
        """保存生成结果"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO generation_history 
                (user_id, session_id, input_text, generated_sequence, emotion, speed, duration, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, input_text, json.dumps(generated_sequence or {}),
                  emotion, speed, duration, model_version))
            
            # 更新会话统计
            await db.execute("""
                UPDATE sessions SET generation_count = generation_count + 1 WHERE id = ?
            """, (session_id,))
            
            await db.commit()
    
    # 用户反馈方法
    async def save_user_feedback(self, session_id: str, user_id: int = None,
                               feedback_type: str = "", content: str = "",
                               rating: int = None, recognition_accuracy: float = None,
                               suggestions: str = ""):
        """保存用户反馈"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO user_feedback 
                (user_id, session_id, feedback_type, content, rating, recognition_accuracy, suggestions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, feedback_type, content, rating, recognition_accuracy, suggestions))
            await db.commit()
    
    # 统计查询方法
    async def get_user_statistics(self, user_id: int, days: int = 30) -> Dict:
        """获取用户统计信息"""
        since_date = datetime.now() - timedelta(days=days)
        
        async with aiosqlite.connect(self.db_path) as db:
            # 识别统计
            async with db.execute("""
                SELECT COUNT(*), AVG(confidence), AVG(processing_time)
                FROM recognition_history 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, since_date)) as cursor:
                recognition_stats = await cursor.fetchone()
            
            # 生成统计
            async with db.execute("""
                SELECT COUNT(*), AVG(duration)
                FROM generation_history 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, since_date)) as cursor:
                generation_stats = await cursor.fetchone()
            
            # 会话统计
            async with db.execute("""
                SELECT COUNT(*), AVG(duration), SUM(recognition_count), SUM(generation_count)
                FROM sessions 
                WHERE user_id = ? AND start_time >= ?
            """, (user_id, since_date)) as cursor:
                session_stats = await cursor.fetchone()
            
            return {
                'recognition': {
                    'count': recognition_stats[0] or 0,
                    'avg_confidence': recognition_stats[1] or 0.0,
                    'avg_processing_time': recognition_stats[2] or 0.0
                },
                'generation': {
                    'count': generation_stats[0] or 0,
                    'avg_duration': generation_stats[1] or 0.0
                },
                'sessions': {
                    'count': session_stats[0] or 0,
                    'avg_duration': session_stats[1] or 0.0,
                    'total_recognitions': session_stats[2] or 0,
                    'total_generations': session_stats[3] or 0
                }
            }
    
    # 模型版本管理方法
    async def register_model_version(self, model_name: str, version: str, 
                                   file_path: str, accuracy_metrics: Dict = None,
                                   training_info: Dict = None) -> int:
        """注册新的模型版本"""
        # 计算文件哈希和大小
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        md5_hash = self._calculate_file_hash(file_path) if Path(file_path).exists() else ""
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO model_versions 
                (model_name, version, file_path, file_size, md5_hash, accuracy_metrics, training_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model_name, version, file_path, file_size, md5_hash,
                  json.dumps(accuracy_metrics or {}), json.dumps(training_info or {})))
            
            await db.commit()
            return cursor.lastrowid
    
    async def set_active_model_version(self, model_name: str, version: str):
        """设置活跃的模型版本"""
        async with aiosqlite.connect(self.db_path) as db:
            # 首先取消所有该模型的活跃状态
            await db.execute("""
                UPDATE model_versions SET is_active = FALSE WHERE model_name = ?
            """, (model_name,))
            
            # 设置指定版本为活跃
            await db.execute("""
                UPDATE model_versions SET is_active = TRUE 
                WHERE model_name = ? AND version = ?
            """, (model_name, version))
            
            await db.commit()
    
    async def get_active_model_version(self, model_name: str) -> Optional[Dict]:
        """获取活跃的模型版本"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT version, file_path, accuracy_metrics, training_info, created_at
                FROM model_versions 
                WHERE model_name = ? AND is_active = TRUE
            """, (model_name,)) as cursor:
                
                row = await cursor.fetchone()
                if row:
                    return {
                        'version': row[0],
                        'file_path': row[1],
                        'accuracy_metrics': json.loads(row[2] or '{}'),
                        'training_info': json.loads(row[3] or '{}'),
                        'created_at': row[4]
                    }
                return None
    
    async def get_model_versions(self, model_name: str) -> List[Dict]:
        """获取模型的所有版本"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT version, file_path, file_size, accuracy_metrics, 
                       training_info, is_active, created_at
                FROM model_versions 
                WHERE model_name = ? 
                ORDER BY created_at DESC
            """, (model_name,)) as cursor:
                
                rows = await cursor.fetchall()
                return [{
                    'version': row[0],
                    'file_path': row[1],
                    'file_size': row[2],
                    'accuracy_metrics': json.loads(row[3] or '{}'),
                    'training_info': json.loads(row[4] or '{}'),
                    'is_active': bool(row[5]),
                    'created_at': row[6]
                } for row in rows]
    
    # 设备配置管理方法
    async def save_device_config(self, user_id: int, device_type: str, 
                               config_data: Dict) -> int:
        """保存设备配置"""
        async with aiosqlite.connect(self.db_path) as db:
            # 首先停用该用户该设备类型的所有配置
            await db.execute("""
                UPDATE device_configs SET is_active = FALSE 
                WHERE user_id = ? AND device_type = ?
            """, (user_id, device_type))
            
            # 插入新配置
            cursor = await db.execute("""
                INSERT INTO device_configs (user_id, device_type, config_data)
                VALUES (?, ?, ?)
            """, (user_id, device_type, json.dumps(config_data)))
            
            await db.commit()
            return cursor.lastrowid
    
    async def get_device_config(self, user_id: int, device_type: str) -> Optional[Dict]:
        """获取设备配置"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT config_data, created_at, updated_at
                FROM device_configs 
                WHERE user_id = ? AND device_type = ? AND is_active = TRUE
                ORDER BY updated_at DESC LIMIT 1
            """, (user_id, device_type)) as cursor:
                
                row = await cursor.fetchone()
                if row:
                    return {
                        'config_data': json.loads(row[0]),
                        'created_at': row[1],
                        'updated_at': row[2]
                    }
                return None
    
    # 系统日志方法
    async def log_system_event(self, level: str, module: str, message: str,
                             details: Dict = None, user_id: int = None,
                             ip_address: str = None):
        """记录系统事件"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO system_logs (level, module, message, details, user_id, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (level, module, message, json.dumps(details or {}), user_id, ip_address))
            await db.commit()
    
    async def get_system_logs(self, level: str = None, module: str = None,
                            user_id: int = None, limit: int = 100,
                            offset: int = 0) -> List[Dict]:
        """获取系统日志"""
        query = "SELECT level, module, message, details, user_id, ip_address, created_at FROM system_logs"
        params = []
        conditions = []
        
        if level:
            conditions.append("level = ?")
            params.append(level)
        
        if module:
            conditions.append("module = ?")
            params.append(module)
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [{
                    'level': row[0],
                    'module': row[1],
                    'message': row[2],
                    'details': json.loads(row[3] or '{}'),
                    'user_id': row[4],
                    'ip_address': row[5],
                    'created_at': row[6]
                } for row in rows]
    
    # 数据分析方法
    async def get_recognition_trends(self, days: int = 30) -> Dict:
        """获取识别趋势分析"""
        since_date = datetime.now() - timedelta(days=days)
        
        async with aiosqlite.connect(self.db_path) as db:
            # 按日期分组的识别次数
            async with db.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM recognition_history 
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date
            """, (since_date,)) as cursor:
                daily_stats = await cursor.fetchall()
            
            # 识别文本长度分析
            async with db.execute("""
                SELECT AVG(LENGTH(recognized_text)) as avg_length, 
                       MIN(LENGTH(recognized_text)) as min_length,
                       MAX(LENGTH(recognized_text)) as max_length
                FROM recognition_history 
                WHERE created_at >= ? AND recognized_text IS NOT NULL
            """, (since_date,)) as cursor:
                length_stats = await cursor.fetchone()
            
            # 处理时间分析
            async with db.execute("""
                SELECT AVG(processing_time) as avg_time, 
                       MIN(processing_time) as min_time,
                       MAX(processing_time) as max_time
                FROM recognition_history 
                WHERE created_at >= ? AND processing_time > 0
            """, (since_date,)) as cursor:
                time_stats = await cursor.fetchone()
            
            return {
                'daily_recognition': [
                    {'date': row[0], 'count': row[1], 'avg_confidence': row[2]}
                    for row in daily_stats
                ],
                'text_length': {
                    'avg_length': length_stats[0] or 0,
                    'min_length': length_stats[1] or 0,
                    'max_length': length_stats[2] or 0
                } if length_stats else {},
                'processing_time': {
                    'avg_time': time_stats[0] or 0,
                    'min_time': time_stats[1] or 0,
                    'max_time': time_stats[2] or 0
                } if time_stats else {}
            }
    
    async def get_user_activity_analysis(self, user_id: int, days: int = 30) -> Dict:
        """获取用户活动分析"""
        since_date = datetime.now() - timedelta(days=days)
        
        async with aiosqlite.connect(self.db_path) as db:
            # 按小时分组的活动分析
            async with db.execute("""
                SELECT CAST(strftime('%H', created_at) AS INTEGER) as hour, 
                       COUNT(*) as activity_count
                FROM recognition_history 
                WHERE user_id = ? AND created_at >= ?
                GROUP BY CAST(strftime('%H', created_at) AS INTEGER)
                ORDER BY hour
            """, (user_id, since_date)) as cursor:
                hourly_activity = await cursor.fetchall()
            
            # 会话时长分析
            async with db.execute("""
                SELECT AVG(duration) as avg_duration, 
                       MIN(duration) as min_duration,
                       MAX(duration) as max_duration,
                       COUNT(*) as session_count
                FROM sessions 
                WHERE user_id = ? AND start_time >= ? AND duration IS NOT NULL
            """, (user_id, since_date)) as cursor:
                session_stats = await cursor.fetchone()
            
            # 使用频率分析
            async with db.execute("""
                SELECT COUNT(DISTINCT DATE(created_at)) as active_days
                FROM recognition_history 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, since_date)) as cursor:
                active_days = await cursor.fetchone()
            
            return {
                'hourly_activity': [
                    {'hour': row[0], 'count': row[1]}
                    for row in hourly_activity
                ],
                'session_analysis': {
                    'avg_duration': session_stats[0] or 0,
                    'min_duration': session_stats[1] or 0,
                    'max_duration': session_stats[2] or 0,
                    'session_count': session_stats[3] or 0
                } if session_stats else {},
                'usage_frequency': {
                    'active_days': active_days[0] or 0,
                    'total_days': days,
                    'activity_rate': (active_days[0] or 0) / days if days > 0 else 0
                } if active_days else {}
            }
    
    async def get_system_health_metrics(self) -> Dict:
        """获取系统健康指标"""
        async with aiosqlite.connect(self.db_path) as db:
            # 用户活跃度
            async with db.execute("""
                SELECT COUNT(DISTINCT user_id) as active_users_today
                FROM recognition_history 
                WHERE DATE(created_at) = DATE('now')
            """) as cursor:
                active_users_today = await cursor.fetchone()
            
            async with db.execute("""
                SELECT COUNT(DISTINCT user_id) as active_users_week
                FROM recognition_history 
                WHERE created_at >= datetime('now', '-7 days')
            """) as cursor:
                active_users_week = await cursor.fetchone()
            
            # 系统负载
            async with db.execute("""
                SELECT COUNT(*) as total_recognitions_today,
                       AVG(processing_time) as avg_processing_time
                FROM recognition_history 
                WHERE DATE(created_at) = DATE('now')
            """) as cursor:
                load_stats = await cursor.fetchone()
            
            # 错误率
            async with db.execute("""
                SELECT COUNT(*) as error_count
                FROM system_logs 
                WHERE level = 'ERROR' AND DATE(created_at) = DATE('now')
            """) as cursor:
                error_count = await cursor.fetchone()
            
            # 数据库大小
            async with db.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()") as cursor:
                db_size = await cursor.fetchone()
            
            return {
                'user_activity': {
                    'active_users_today': active_users_today[0] if active_users_today else 0,
                    'active_users_week': active_users_week[0] if active_users_week else 0
                },
                'system_load': {
                    'recognitions_today': load_stats[0] if load_stats else 0,
                    'avg_processing_time': load_stats[1] if load_stats else 0
                },
                'error_metrics': {
                    'errors_today': error_count[0] if error_count else 0
                },
                'database': {
                    'size_bytes': db_size[0] if db_size else 0,
                    'size_mb': (db_size[0] / 1024 / 1024) if db_size else 0
                }
            }
    
    # 数据清理方法
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with aiosqlite.connect(self.db_path) as db:
            # 清理旧的识别历史
            result1 = await db.execute("""
                DELETE FROM recognition_history WHERE created_at < ?
            """, (cutoff_date,))
            
            # 清理旧的生成历史
            result2 = await db.execute("""
                DELETE FROM generation_history WHERE created_at < ?
            """, (cutoff_date,))
            
            # 清理旧的系统日志
            result3 = await db.execute("""
                DELETE FROM system_logs WHERE created_at < ?
            """, (cutoff_date,))
            
            # 清理已结束的旧会话
            result4 = await db.execute("""
                DELETE FROM sessions WHERE end_time IS NOT NULL AND end_time < ?
            """, (cutoff_date,))
            
            await db.commit()
            
            # 执行VACUUM以回收空间
            await db.execute("VACUUM")
            
            deleted_counts = {
                'recognition_history': result1.rowcount,
                'generation_history': result2.rowcount,
                'system_logs': result3.rowcount,
                'sessions': result4.rowcount
            }
            
            logger.info(f"数据清理完成: {deleted_counts}")
            return deleted_counts
    
    async def backup_database(self, backup_path: str = None) -> str:
        """备份数据库"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backup/signavatar_backup_{timestamp}.db"
        
        # 确保备份目录存在
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as source:
            async with aiosqlite.connect(backup_path) as backup:
                await source.backup(backup)
        
        logger.info(f"数据库备份完成: {backup_path}")
        return backup_path
    
    async def restore_database(self, backup_path: str):
        """恢复数据库"""
        if not Path(backup_path).exists():
            raise FileNotFoundError(f"备份文件不存在: {backup_path}")
        
        # 创建当前数据库的备份
        current_backup = await self.backup_database()
        logger.info(f"当前数据库已备份到: {current_backup}")
        
        try:
            # 恢复数据库
            async with aiosqlite.connect(backup_path) as source:
                async with aiosqlite.connect(self.db_path) as target:
                    await source.backup(target)
            
            logger.info(f"数据库恢复完成: {backup_path}")
            
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            # 尝试恢复原始数据库
            try:
                async with aiosqlite.connect(current_backup) as source:
                    async with aiosqlite.connect(self.db_path) as target:
                        await source.backup(target)
                logger.info("已恢复到原始状态")
            except Exception as restore_error:
                logger.error(f"恢复原始状态失败: {restore_error}")
            
            raise
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256((password + settings.SECRET_KEY).encode()).hexdigest()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            return ""
    
    async def cleanup(self):
        """清理资源"""
        if self.connection:
            await self.connection.close()
        logger.info("数据库管理器已清理")
    
    async def optimize_database(self):
        """优化数据库性能"""
        async with aiosqlite.connect(self.db_path) as db:
            # 更新统计信息
            await db.execute("ANALYZE")
            
            # 重建索引
            await db.execute("REINDEX")
            
            # 清理碎片
            await db.execute("VACUUM")
            
            await db.commit()
        
        logger.info("数据库优化完成")
    
    async def create_indexes(self):
        """创建数据库索引以提高查询性能"""
        async with aiosqlite.connect(self.db_path) as db:
            # 用户表索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)")
            
            # 识别历史表索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_recognition_user_id ON recognition_history(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_recognition_session_id ON recognition_history(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_recognition_created_at ON recognition_history(created_at)")
            
            # 生成历史表索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_generation_user_id ON generation_history(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_generation_session_id ON generation_history(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_generation_created_at ON generation_history(created_at)")
            
            # 会话表索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time)")
            
            # 系统日志表索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_module ON system_logs(module)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_created_at ON system_logs(created_at)")
            
            # 模型版本表索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_name ON model_versions(model_name)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active)")
            
            # 设备配置表索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_device_configs_user_id ON device_configs(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_device_configs_type ON device_configs(device_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_device_configs_active ON device_configs(is_active)")
            
            await db.commit()
        
        logger.info("数据库索引创建完成")


# 全局数据库管理器实例
db_manager = DatabaseManager()