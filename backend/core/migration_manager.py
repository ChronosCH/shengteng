"""
数据迁移管理器
Migration Manager
提供数据库版本管理和数据迁移功能
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class Migration:
    """数据迁移定义"""
    version: int
    name: str
    description: str
    up_sql: str
    down_sql: str = ""
    python_up: Optional[Callable] = None
    python_down: Optional[Callable] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class MigrationManager:
    """数据迁移管理器"""
    
    def __init__(self, db_path: str, migrations_dir: str = "migrations"):
        self.db_path = db_path
        self.migrations_dir = Path(migrations_dir)
        self.migrations: Dict[int, Migration] = {}
        self.current_version = 0
        
        # 确保迁移目录存在
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化迁移表
        self._init_migration_table()
        
        # 加载迁移
        self._load_migrations()
        
        # 获取当前版本
        self._get_current_version()
    
    def _init_migration_table(self):
        """初始化迁移表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS _migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        applied_at TEXT NOT NULL,
                        execution_time REAL
                    )
                ''')
                
                conn.commit()
                logger.info("✅ 迁移表初始化完成")
                
        except Exception as e:
            logger.error(f"❌ 迁移表初始化失败: {e}")
            raise
    
    def _load_migrations(self):
        """加载迁移文件"""
        try:
            # 加载内置迁移
            self._load_builtin_migrations()
            
            # 加载文件迁移
            self._load_file_migrations()
            
            logger.info(f"✅ 加载了 {len(self.migrations)} 个迁移")
            
        except Exception as e:
            logger.error(f"❌ 加载迁移失败: {e}")
    
    def _load_builtin_migrations(self):
        """加载内置迁移"""
        # 基础表结构迁移
        self.add_migration(Migration(
            version=1,
            name="create_basic_tables",
            description="创建基础表结构",
            up_sql='''
                -- 用户表
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                );
                
                -- 用户配置表
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    settings TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                );
                
                -- 文件表
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    content_type TEXT,
                    uploaded_by TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (uploaded_by) REFERENCES users (user_id)
                );
            ''',
            down_sql='''
                DROP TABLE IF EXISTS files;
                DROP TABLE IF EXISTS user_settings;
                DROP TABLE IF EXISTS users;
            '''
        ))
        
        # 索引优化迁移
        self.add_migration(Migration(
            version=2,
            name="add_indexes",
            description="添加性能优化索引",
            up_sql='''
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
                CREATE INDEX IF NOT EXISTS idx_files_uploaded_by ON files(uploaded_by);
                CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at);
            ''',
            down_sql='''
                DROP INDEX IF EXISTS idx_files_created_at;
                DROP INDEX IF EXISTS idx_files_uploaded_by;
                DROP INDEX IF EXISTS idx_users_created_at;
                DROP INDEX IF EXISTS idx_users_username;
                DROP INDEX IF EXISTS idx_users_email;
            '''
        ))
        
        # 学习数据表迁移
        self.add_migration(Migration(
            version=3,
            name="create_learning_tables",
            description="创建学习相关表",
            up_sql='''
                -- 学习会话表
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_minutes INTEGER DEFAULT 0,
                    lessons_completed TEXT,
                    accuracy_score REAL DEFAULT 0.0,
                    xp_earned INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                );
                
                -- 识别历史表
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_id TEXT,
                    input_type TEXT,
                    input_data TEXT,
                    result_text TEXT,
                    confidence REAL,
                    processing_time REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_learning_sessions_user_id ON learning_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_learning_sessions_start_time ON learning_sessions(start_time);
                CREATE INDEX IF NOT EXISTS idx_recognition_history_user_id ON recognition_history(user_id);
                CREATE INDEX IF NOT EXISTS idx_recognition_history_created_at ON recognition_history(created_at);
            ''',
            down_sql='''
                DROP INDEX IF EXISTS idx_recognition_history_created_at;
                DROP INDEX IF EXISTS idx_recognition_history_user_id;
                DROP INDEX IF EXISTS idx_learning_sessions_start_time;
                DROP INDEX IF EXISTS idx_learning_sessions_user_id;
                DROP TABLE IF EXISTS recognition_history;
                DROP TABLE IF EXISTS learning_sessions;
            '''
        ))
    
    def _load_file_migrations(self):
        """从文件加载迁移"""
        try:
            for migration_file in self.migrations_dir.glob("*.sql"):
                self._load_migration_file(migration_file)
                
        except Exception as e:
            logger.error(f"❌ 加载文件迁移失败: {e}")
    
    def _load_migration_file(self, file_path: Path):
        """加载单个迁移文件"""
        try:
            # 解析文件名获取版本号
            # 格式: 001_migration_name.sql
            filename = file_path.stem
            parts = filename.split('_', 1)
            
            if len(parts) < 2:
                logger.warning(f"⚠️ 跳过格式不正确的迁移文件: {file_path}")
                return
            
            try:
                version = int(parts[0])
                name = parts[1]
            except ValueError:
                logger.warning(f"⚠️ 跳过版本号无效的迁移文件: {file_path}")
                return
            
            # 读取文件内容
            content = file_path.read_text(encoding='utf-8')
            
            # 解析UP和DOWN部分
            up_sql = ""
            down_sql = ""
            
            if "-- UP" in content and "-- DOWN" in content:
                parts = content.split("-- DOWN")
                up_part = parts[0].replace("-- UP", "").strip()
                down_part = parts[1].strip() if len(parts) > 1 else ""
                up_sql = up_part
                down_sql = down_part
            else:
                up_sql = content
            
            # 创建迁移对象
            migration = Migration(
                version=version,
                name=name,
                description=f"从文件加载: {file_path.name}",
                up_sql=up_sql,
                down_sql=down_sql
            )
            
            self.add_migration(migration)
            
        except Exception as e:
            logger.error(f"❌ 加载迁移文件失败 {file_path}: {e}")
    
    def _get_current_version(self):
        """获取当前数据库版本"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT MAX(version) FROM _migrations")
                result = cursor.fetchone()
                
                self.current_version = result[0] if result[0] is not None else 0
                logger.info(f"✅ 当前数据库版本: {self.current_version}")
                
        except Exception as e:
            logger.error(f"❌ 获取数据库版本失败: {e}")
            self.current_version = 0
    
    def add_migration(self, migration: Migration):
        """添加迁移"""
        if migration.version in self.migrations:
            logger.warning(f"⚠️ 迁移版本 {migration.version} 已存在，将被覆盖")
        
        self.migrations[migration.version] = migration
    
    def migrate_to_latest(self) -> bool:
        """迁移到最新版本"""
        try:
            if not self.migrations:
                logger.info("ℹ️ 没有可用的迁移")
                return True
            
            latest_version = max(self.migrations.keys())
            return self.migrate_to_version(latest_version)
            
        except Exception as e:
            logger.error(f"❌ 迁移到最新版本失败: {e}")
            return False
    
    def migrate_to_version(self, target_version: int) -> bool:
        """迁移到指定版本"""
        try:
            if target_version == self.current_version:
                logger.info(f"ℹ️ 已经是目标版本 {target_version}")
                return True
            
            if target_version > self.current_version:
                # 向上迁移
                return self._migrate_up(target_version)
            else:
                # 向下迁移
                return self._migrate_down(target_version)
                
        except Exception as e:
            logger.error(f"❌ 迁移到版本 {target_version} 失败: {e}")
            return False
    
    def _migrate_up(self, target_version: int) -> bool:
        """向上迁移"""
        try:
            versions_to_apply = sorted([
                v for v in self.migrations.keys() 
                if self.current_version < v <= target_version
            ])
            
            if not versions_to_apply:
                logger.info("ℹ️ 没有需要应用的迁移")
                return True
            
            logger.info(f"🔄 开始向上迁移，目标版本: {target_version}")
            
            for version in versions_to_apply:
                if not self._apply_migration(version):
                    logger.error(f"❌ 迁移版本 {version} 失败，停止迁移")
                    return False
            
            logger.info(f"✅ 向上迁移完成，当前版本: {self.current_version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 向上迁移失败: {e}")
            return False
    
    def _migrate_down(self, target_version: int) -> bool:
        """向下迁移"""
        try:
            versions_to_rollback = sorted([
                v for v in self.migrations.keys() 
                if target_version < v <= self.current_version
            ], reverse=True)
            
            if not versions_to_rollback:
                logger.info("ℹ️ 没有需要回滚的迁移")
                return True
            
            logger.info(f"🔄 开始向下迁移，目标版本: {target_version}")
            
            for version in versions_to_rollback:
                if not self._rollback_migration(version):
                    logger.error(f"❌ 回滚版本 {version} 失败，停止迁移")
                    return False
            
            logger.info(f"✅ 向下迁移完成，当前版本: {self.current_version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 向下迁移失败: {e}")
            return False
    
    def _apply_migration(self, version: int) -> bool:
        """应用迁移"""
        try:
            migration = self.migrations[version]
            start_time = time.time()
            
            logger.info(f"🔄 应用迁移 {version}: {migration.name}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 开始事务
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    # 执行SQL迁移
                    if migration.up_sql:
                        cursor.executescript(migration.up_sql)
                    
                    # 执行Python迁移
                    if migration.python_up:
                        migration.python_up(cursor)
                    
                    # 记录迁移
                    execution_time = time.time() - start_time
                    cursor.execute('''
                        INSERT INTO _migrations (version, name, description, applied_at, execution_time)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        version,
                        migration.name,
                        migration.description,
                        datetime.now().isoformat(),
                        execution_time
                    ))
                    
                    # 提交事务
                    cursor.execute("COMMIT")
                    
                    self.current_version = version
                    logger.info(f"✅ 迁移 {version} 应用成功，耗时: {execution_time:.2f}s")
                    return True
                    
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    logger.error(f"❌ 迁移 {version} 应用失败: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 应用迁移 {version} 失败: {e}")
            return False
    
    def _rollback_migration(self, version: int) -> bool:
        """回滚迁移"""
        try:
            migration = self.migrations[version]
            start_time = time.time()
            
            logger.info(f"🔄 回滚迁移 {version}: {migration.name}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 开始事务
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    # 执行SQL回滚
                    if migration.down_sql:
                        cursor.executescript(migration.down_sql)
                    
                    # 执行Python回滚
                    if migration.python_down:
                        migration.python_down(cursor)
                    
                    # 删除迁移记录
                    cursor.execute("DELETE FROM _migrations WHERE version = ?", (version,))
                    
                    # 提交事务
                    cursor.execute("COMMIT")
                    
                    # 更新当前版本
                    cursor.execute("SELECT MAX(version) FROM _migrations")
                    result = cursor.fetchone()
                    self.current_version = result[0] if result[0] is not None else 0
                    
                    execution_time = time.time() - start_time
                    logger.info(f"✅ 迁移 {version} 回滚成功，耗时: {execution_time:.2f}s")
                    return True
                    
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    logger.error(f"❌ 迁移 {version} 回滚失败: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 回滚迁移 {version} 失败: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取已应用的迁移
                cursor.execute("SELECT version, name, applied_at, execution_time FROM _migrations ORDER BY version")
                applied_migrations = [
                    {
                        "version": row[0],
                        "name": row[1],
                        "applied_at": row[2],
                        "execution_time": row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                # 获取待应用的迁移
                applied_versions = {m["version"] for m in applied_migrations}
                pending_migrations = [
                    {
                        "version": version,
                        "name": migration.name,
                        "description": migration.description
                    }
                    for version, migration in self.migrations.items()
                    if version not in applied_versions
                ]
                
                return {
                    "current_version": self.current_version,
                    "latest_version": max(self.migrations.keys()) if self.migrations else 0,
                    "applied_migrations": applied_migrations,
                    "pending_migrations": sorted(pending_migrations, key=lambda x: x["version"]),
                    "total_migrations": len(self.migrations)
                }
                
        except Exception as e:
            logger.error(f"❌ 获取迁移状态失败: {e}")
            return {}
    
    def create_migration_file(self, name: str, up_sql: str, down_sql: str = "") -> Path:
        """创建迁移文件"""
        try:
            # 生成版本号
            next_version = max(self.migrations.keys()) + 1 if self.migrations else 1
            
            # 生成文件名
            filename = f"{next_version:03d}_{name}.sql"
            file_path = self.migrations_dir / filename
            
            # 生成文件内容
            content = f"""-- Migration: {name}
-- Version: {next_version}
-- Created: {datetime.now().isoformat()}

-- UP
{up_sql}

-- DOWN
{down_sql}
"""
            
            # 写入文件
            file_path.write_text(content, encoding='utf-8')
            
            logger.info(f"✅ 创建迁移文件: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ 创建迁移文件失败: {e}")
            raise
