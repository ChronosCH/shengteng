"""
数据库管理器
Database Manager
提供统一的数据库连接管理、查询优化、数据迁移等功能
"""

import asyncio
import logging
import sqlite3
import json
import time
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from datetime import datetime
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """数据库配置"""
    db_path: str
    max_connections: int = 10
    timeout: float = 30.0
    check_same_thread: bool = False
    enable_wal: bool = True
    cache_size: int = -64000  # 64MB
    temp_store: str = "memory"
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"

class ConnectionPool:
    """数据库连接池"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connections: List[sqlite3.Connection] = []
        self.available_connections: List[sqlite3.Connection] = []
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化连接池"""
        try:
            # 确保数据库目录存在
            Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            for _ in range(self.config.max_connections):
                conn = self._create_connection()
                self.connections.append(conn)
                self.available_connections.append(conn)
            
            logger.info(f"✅ 数据库连接池初始化完成，连接数: {len(self.connections)}")
            
        except Exception as e:
            logger.error(f"❌ 数据库连接池初始化失败: {e}")
            raise
    
    def _create_connection(self) -> sqlite3.Connection:
        """创建数据库连接"""
        try:
            conn = sqlite3.connect(
                self.config.db_path,
                timeout=self.config.timeout,
                check_same_thread=self.config.check_same_thread
            )
            
            # 优化设置
            conn.execute(f"PRAGMA cache_size = {self.config.cache_size}")
            conn.execute(f"PRAGMA temp_store = {self.config.temp_store}")
            conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            conn.execute("PRAGMA foreign_keys = ON")
            
            # 设置行工厂
            conn.row_factory = sqlite3.Row
            
            return conn
            
        except Exception as e:
            logger.error(f"❌ 创建数据库连接失败: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = None
        try:
            with self.lock:
                if self.available_connections:
                    conn = self.available_connections.pop()
                else:
                    # 如果没有可用连接，创建新连接
                    conn = self._create_connection()
            
            yield conn
            
        except Exception as e:
            logger.error(f"❌ 数据库操作失败: {e}")
            raise
        finally:
            if conn:
                with self.lock:
                    self.available_connections.append(conn)
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"⚠️ 关闭数据库连接失败: {e}")
            
            self.connections.clear()
            self.available_connections.clear()
            logger.info("✅ 所有数据库连接已关闭")

class QueryBuilder:
    """SQL查询构建器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置查询构建器"""
        self._select_fields = []
        self._from_table = ""
        self._joins = []
        self._where_conditions = []
        self._group_by = []
        self._having_conditions = []
        self._order_by = []
        self._limit_value = None
        self._offset_value = None
        self._parameters = []
        return self
    
    def select(self, *fields):
        """SELECT子句"""
        self._select_fields.extend(fields)
        return self
    
    def from_table(self, table):
        """FROM子句"""
        self._from_table = table
        return self
    
    def join(self, table, condition):
        """JOIN子句"""
        self._joins.append(f"JOIN {table} ON {condition}")
        return self
    
    def left_join(self, table, condition):
        """LEFT JOIN子句"""
        self._joins.append(f"LEFT JOIN {table} ON {condition}")
        return self
    
    def where(self, condition, *params):
        """WHERE子句"""
        self._where_conditions.append(condition)
        self._parameters.extend(params)
        return self
    
    def group_by(self, *fields):
        """GROUP BY子句"""
        self._group_by.extend(fields)
        return self
    
    def having(self, condition, *params):
        """HAVING子句"""
        self._having_conditions.append(condition)
        self._parameters.extend(params)
        return self
    
    def order_by(self, field, direction="ASC"):
        """ORDER BY子句"""
        self._order_by.append(f"{field} {direction}")
        return self
    
    def limit(self, count):
        """LIMIT子句"""
        self._limit_value = count
        return self
    
    def offset(self, count):
        """OFFSET子句"""
        self._offset_value = count
        return self
    
    def build(self):
        """构建SQL查询"""
        if not self._from_table:
            raise ValueError("FROM table is required")
        
        # SELECT
        if self._select_fields:
            select_clause = f"SELECT {', '.join(self._select_fields)}"
        else:
            select_clause = "SELECT *"
        
        # FROM
        from_clause = f"FROM {self._from_table}"
        
        # JOIN
        join_clause = " ".join(self._joins) if self._joins else ""
        
        # WHERE
        where_clause = f"WHERE {' AND '.join(self._where_conditions)}" if self._where_conditions else ""
        
        # GROUP BY
        group_by_clause = f"GROUP BY {', '.join(self._group_by)}" if self._group_by else ""
        
        # HAVING
        having_clause = f"HAVING {' AND '.join(self._having_conditions)}" if self._having_conditions else ""
        
        # ORDER BY
        order_by_clause = f"ORDER BY {', '.join(self._order_by)}" if self._order_by else ""
        
        # LIMIT
        limit_clause = f"LIMIT {self._limit_value}" if self._limit_value is not None else ""
        
        # OFFSET
        offset_clause = f"OFFSET {self._offset_value}" if self._offset_value is not None else ""
        
        # 组合查询
        query_parts = [
            select_clause,
            from_clause,
            join_clause,
            where_clause,
            group_by_clause,
            having_clause,
            order_by_clause,
            limit_clause,
            offset_clause
        ]
        
        query = " ".join(part for part in query_parts if part)
        
        return query, self._parameters

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = ConnectionPool(config)
        self.query_builder = QueryBuilder()
        self._migration_version = 0
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """初始化元数据表"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 创建元数据表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS _database_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                ''')
                
                # 检查版本
                cursor.execute("SELECT value FROM _database_metadata WHERE key = 'version'")
                result = cursor.fetchone()
                
                if result:
                    self._migration_version = int(result[0])
                else:
                    # 初始化版本
                    cursor.execute('''
                        INSERT INTO _database_metadata (key, value, created_at, updated_at)
                        VALUES ('version', '0', ?, ?)
                    ''', (datetime.now().isoformat(), datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"✅ 数据库元数据初始化完成，当前版本: {self._migration_version}")
                
        except Exception as e:
            logger.error(f"❌ 数据库元数据初始化失败: {e}")
            raise
    
    def execute_query(self, query: str, parameters: tuple = ()) -> List[sqlite3.Row]:
        """执行查询"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, parameters)
                return cursor.fetchall()
                
        except Exception as e:
            logger.error(f"❌ 查询执行失败: {query}, 参数: {parameters}, 错误: {e}")
            raise
    
    def execute_update(self, query: str, parameters: tuple = ()) -> int:
        """执行更新操作"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, parameters)
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"❌ 更新执行失败: {query}, 参数: {parameters}, 错误: {e}")
            raise
    
    def execute_batch(self, query: str, parameters_list: List[tuple]) -> int:
        """批量执行操作"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, parameters_list)
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"❌ 批量执行失败: {query}, 错误: {e}")
            raise
    
    def execute_transaction(self, operations: List[tuple]) -> bool:
        """执行事务"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    for query, parameters in operations:
                        cursor.execute(query, parameters)
                    
                    cursor.execute("COMMIT")
                    return True
                    
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    logger.error(f"❌ 事务回滚: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"❌ 事务执行失败: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表信息"""
        try:
            query = f"PRAGMA table_info({table_name})"
            rows = self.execute_query(query)
            
            return [
                {
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default_value": row[4],
                    "pk": bool(row[5])
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"❌ 获取表信息失败: {e}")
            return []
    
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            result = self.execute_query(query, (table_name,))
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"❌ 检查表存在性失败: {e}")
            return False
    
    def get_query_builder(self) -> QueryBuilder:
        """获取查询构建器"""
        return self.query_builder.reset()
    
    def optimize_database(self):
        """优化数据库"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 分析数据库
                cursor.execute("ANALYZE")
                
                # 清理数据库
                cursor.execute("VACUUM")
                
                # 重建索引
                cursor.execute("REINDEX")
                
                conn.commit()
                logger.info("✅ 数据库优化完成")
                
        except Exception as e:
            logger.error(f"❌ 数据库优化失败: {e}")
            raise
    
    def backup_database(self, backup_path: str):
        """备份数据库"""
        try:
            with self.pool.get_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
                
                logger.info(f"✅ 数据库备份完成: {backup_path}")
                
        except Exception as e:
            logger.error(f"❌ 数据库备份失败: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 获取数据库大小
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # 获取表数量
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                # 获取索引数量
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
                index_count = cursor.fetchone()[0]
                
                # 获取各表的记录数
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                table_stats = {}
                for table in tables:
                    table_name = table[0]
                    if not table_name.startswith('sqlite_') and not table_name.startswith('_'):
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        table_stats[table_name] = count
                
                return {
                    "database_size": db_size,
                    "table_count": table_count,
                    "index_count": index_count,
                    "table_stats": table_stats,
                    "migration_version": self._migration_version,
                    "connection_pool_size": len(self.pool.connections),
                    "available_connections": len(self.pool.available_connections)
                }
                
        except Exception as e:
            logger.error(f"❌ 获取数据库统计失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库管理器"""
        try:
            self.pool.close_all()
            logger.info("✅ 数据库管理器已关闭")
            
        except Exception as e:
            logger.error(f"❌ 关闭数据库管理器失败: {e}")

# 全局数据库管理器实例
_database_managers: Dict[str, DatabaseManager] = {}

def get_database_manager(db_name: str = "main", db_path: str = None) -> DatabaseManager:
    """获取数据库管理器实例"""
    if db_name not in _database_managers:
        if not db_path:
            db_path = f"data/{db_name}.db"
        
        config = DatabaseConfig(db_path=db_path)
        _database_managers[db_name] = DatabaseManager(config)
    
    return _database_managers[db_name]

def close_all_databases():
    """关闭所有数据库连接"""
    for manager in _database_managers.values():
        manager.close()
    _database_managers.clear()
