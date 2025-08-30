"""
缓存管理模块
提供Redis缓存、内存缓存、缓存策略管理等功能
"""

import asyncio
import json
import time
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("警告: Redis 未安装，将使用内存缓存")

from utils.logger import setup_logger
from utils.config import settings

logger = setup_logger(__name__)


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        self.max_memory_cache_size = 1000
        self.default_ttl = 3600  # 1小时
        
        logger.info("缓存管理器初始化完成")

    def _evict_if_needed(self):
        """当内存缓存达到上限时进行清理/淘汰
        规则：
        1) 先移除已过期的条目
        2) 若仍超过阈值，按过期时间升序淘汰一定比例（10%）或至少1条
        """
        try:
            now = time.time()
            # 1) 清理过期
            expired_keys = [k for k, v in list(self.memory_cache.items()) if v.get('expire_time', now) < now]
            for k in expired_keys:
                self.memory_cache.pop(k, None)
            if expired_keys:
                self.cache_stats["evictions"] += len(expired_keys)
            
            # 2) 容量淘汰
            current_size = len(self.memory_cache)
            if current_size >= self.max_memory_cache_size:
                # 计算需要额外淘汰的数量（超出部分 + 额外10% 缓冲）
                buffer_evict = max(1, int(self.max_memory_cache_size * 0.1))
                to_evict = (current_size - self.max_memory_cache_size) + buffer_evict
                # 按过期时间排序（最早过期的优先淘汰）
                sorted_items = sorted(self.memory_cache.items(), key=lambda kv: kv[1].get('expire_time', now))
                evicted = 0
                for k, _ in sorted_items:
                    if evicted >= to_evict:
                        break
                    self.memory_cache.pop(k, None)
                    evicted += 1
                if evicted:
                    self.cache_stats["evictions"] += evicted
        except Exception as e:
            logger.error(f"内存缓存淘汰失败: {e}")

    async def initialize(self):
        """初始化缓存连接"""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=getattr(settings, 'REDIS_HOST', 'localhost'),
                    port=getattr(settings, 'REDIS_PORT', 6379),
                    db=getattr(settings, 'REDIS_DB', 0),
                    password=getattr(settings, 'REDIS_PASSWORD', None),
                    decode_responses=False
                )
                
                # 测试连接
                await self.redis_client.ping()
                logger.info("Redis 缓存连接成功")
                
            except Exception as e:
                logger.warning(f"Redis 连接失败，使用内存缓存: {e}")
                self.redis_client = None
        else:
            logger.info("使用内存缓存")
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """生成缓存键"""
        return f"{settings.APP_NAME}:{namespace}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """序列化值"""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """反序列化值"""
        return pickle.loads(data)
    
    async def set(self, namespace: str, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        try:
            if self.redis_client:
                # 使用Redis
                serialized_value = self._serialize_value(value)
                result = await self.redis_client.setex(cache_key, ttl, serialized_value)
                success = bool(result)
            else:
                # 使用内存缓存
                self._evict_if_needed()
                expire_time = time.time() + ttl
                self.memory_cache[cache_key] = {
                    'value': value,
                    'expire_time': expire_time
                }
                success = True
            
            if success:
                self.cache_stats["sets"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """获取缓存"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                # 使用Redis
                data = await self.redis_client.get(cache_key)
                if data:
                    self.cache_stats["hits"] += 1
                    return self._deserialize_value(data)
                else:
                    self.cache_stats["misses"] += 1
                    return None
            else:
                # 使用内存缓存
                if cache_key in self.memory_cache:
                    cache_item = self.memory_cache[cache_key]
                    if time.time() < cache_item['expire_time']:
                        self.cache_stats["hits"] += 1
                        return cache_item['value']
                    else:
                        # 过期删除
                        del self.memory_cache[cache_key]
                        self.cache_stats["evictions"] += 1
                
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    async def delete(self, namespace: str, key: str) -> bool:
        """删除缓存"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                # 使用Redis
                result = await self.redis_client.delete(cache_key)
                success = bool(result)
            else:
                # 使用内存缓存
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    success = True
                else:
                    success = False
            
            if success:
                self.cache_stats["deletes"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """检查缓存是否存在"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                return bool(await self.redis_client.exists(cache_key))
            else:
                if cache_key in self.memory_cache:
                    cache_item = self.memory_cache[cache_key]
                    if time.time() < cache_item['expire_time']:
                        return True
                    else:
                        del self.memory_cache[cache_key]
                        self.cache_stats["evictions"] += 1
                return False
                
        except Exception as e:
            logger.error(f"检查缓存存在失败: {e}")
            return False
    
    async def clear_namespace(self, namespace: str) -> int:
        """清空命名空间下的所有缓存"""
        pattern = self._generate_key(namespace, "*")
        
        try:
            if self.redis_client:
                # 使用Redis
                keys = await self.redis_client.keys(pattern)
                if keys:
                    deleted = await self.redis_client.delete(*keys)
                    return deleted
                return 0
            else:
                # 使用内存缓存
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(pattern[:-1])]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                return len(keys_to_delete)
                
        except Exception as e:
            logger.error(f"清空命名空间缓存失败: {e}")
            return 0
    
    async def increment(self, namespace: str, key: str, amount: int = 1) -> Optional[int]:
        """递增计数器"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                return await self.redis_client.incrby(cache_key, amount)
            else:
                current_value = await self.get(namespace, key) or 0
                new_value = current_value + amount
                await self.set(namespace, key, new_value)
                return new_value
                
        except Exception as e:
            logger.error(f"递增计数器失败: {e}")
            return None
    
    async def set_hash(self, namespace: str, key: str, field: str, value: Any) -> bool:
        """设置哈希字段"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                serialized_value = self._serialize_value(value)
                result = await self.redis_client.hset(cache_key, field, serialized_value)
                return bool(result)
            else:
                # 内存缓存模拟哈希
                if cache_key not in self.memory_cache:
                    self.memory_cache[cache_key] = {
                        'value': {},
                        'expire_time': time.time() + self.default_ttl
                    }
                self.memory_cache[cache_key]['value'][field] = value
                return True
                
        except Exception as e:
            logger.error(f"设置哈希字段失败: {e}")
            return False
    
    async def get_hash(self, namespace: str, key: str, field: str) -> Optional[Any]:
        """获取哈希字段"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                data = await self.redis_client.hget(cache_key, field)
                if data:
                    return self._deserialize_value(data)
                return None
            else:
                # 内存缓存模拟哈希
                if cache_key in self.memory_cache:
                    cache_item = self.memory_cache[cache_key]
                    if time.time() < cache_item['expire_time']:
                        return cache_item['value'].get(field)
                    else:
                        del self.memory_cache[cache_key]
                return None
                
        except Exception as e:
            logger.error(f"获取哈希字段失败: {e}")
            return None
    
    async def set_list(self, namespace: str, key: str, values: List[Any], ttl: int = None) -> bool:
        """设置列表缓存"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        try:
            if self.redis_client:
                # 使用Redis列表
                pipe = self.redis_client.pipeline()
                await pipe.delete(cache_key)  # 清空现有列表
                for value in values:
                    serialized_value = self._serialize_value(value)
                    await pipe.rpush(cache_key, serialized_value)
                await pipe.expire(cache_key, ttl)
                await pipe.execute()
                return True
            else:
                # 内存缓存
                self._evict_if_needed()
                expire_time = time.time() + ttl
                self.memory_cache[cache_key] = {
                    'value': values,
                    'expire_time': expire_time,
                    'type': 'list'
                }
                return True
                
        except Exception as e:
            logger.error(f"设置列表缓存失败: {e}")
            return False
    
    async def get_list(self, namespace: str, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """获取列表缓存"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                data_list = await self.redis_client.lrange(cache_key, start, end)
                if data_list:
                    self.cache_stats["hits"] += 1
                    return [self._deserialize_value(data) for data in data_list]
                else:
                    self.cache_stats["misses"] += 1
                    return []
            else:
                if cache_key in self.memory_cache:
                    cache_item = self.memory_cache[cache_key]
                    if time.time() < cache_item['expire_time'] and cache_item.get('type') == 'list':
                        self.cache_stats["hits"] += 1
                        values = cache_item['value']
                        if end == -1:
                            return values[start:]
                        else:
                            return values[start:end+1]
                    else:
                        del self.memory_cache[cache_key]
                        
                self.cache_stats["misses"] += 1
                return []
                
        except Exception as e:
            logger.error(f"获取列表缓存失败: {e}")
            self.cache_stats["misses"] += 1
            return []
    
    async def push_to_list(self, namespace: str, key: str, value: Any, max_length: int = None) -> bool:
        """向列表添加元素"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                serialized_value = self._serialize_value(value)
                pipe = self.redis_client.pipeline()
                await pipe.rpush(cache_key, serialized_value)
                if max_length:
                    await pipe.ltrim(cache_key, -max_length, -1)
                await pipe.execute()
                return True
            else:
                if cache_key in self.memory_cache:
                    cache_item = self.memory_cache[cache_key]
                    if time.time() < cache_item['expire_time'] and cache_item.get('type') == 'list':
                        cache_item['value'].append(value)
                        if max_length and len(cache_item['value']) > max_length:
                            cache_item['value'] = cache_item['value'][-max_length:]
                        return True
                    else:
                        del self.memory_cache[cache_key]
                
                # 如果不存在，创建新列表
                expire_time = time.time() + self.default_ttl
                self.memory_cache[cache_key] = {
                    'value': [value],
                    'expire_time': expire_time,
                    'type': 'list'
                }
                return True
                
        except Exception as e:
            logger.error(f"向列表添加元素失败: {e}")
            return False
    
    async def set_with_tags(self, namespace: str, key: str, value: Any, tags: List[str], ttl: int = None) -> bool:
        """设置带标签的缓存"""
        success = await self.set(namespace, key, value, ttl)
        if success and tags:
            # 为每个标签维护一个键列表
            for tag in tags:
                tag_key = f"tag:{tag}"
                await self.push_to_list(namespace, tag_key, key, max_length=10000)
        return success
    
    async def invalidate_by_tag(self, namespace: str, tag: str) -> int:
        """根据标签失效缓存"""
        tag_key = f"tag:{tag}"
        keys = await self.get_list(namespace, tag_key)
        
        deleted_count = 0
        for key in keys:
            if await self.delete(namespace, key):
                deleted_count += 1
        
        # 清空标签列表
        await self.delete(namespace, tag_key)
        
        return deleted_count
    
    async def get_or_set(self, namespace: str, key: str, func, ttl: int = None, *args, **kwargs) -> Any:
        """获取缓存，如果不存在则执行函数并缓存结果"""
        cached_value = await self.get(namespace, key)
        if cached_value is not None:
            return cached_value
        
        # 防止缓存穿透：使用分布式锁
        lock_key = f"lock:{key}"
        if await self.set(namespace, lock_key, "locked", ttl=30):
            try:
                # 再次检查缓存（双重检查锁定）
                cached_value = await self.get(namespace, key)
                if cached_value is not None:
                    return cached_value
                
                # 执行函数
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 缓存结果
                await self.set(namespace, key, result, ttl)
                return result
                
            finally:
                await self.delete(namespace, lock_key)
        else:
            # 如果获取锁失败，等待一下再重试
            await asyncio.sleep(0.1)
            return await self.get(namespace, key)
    
    async def mget(self, namespace: str, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存"""
        cache_keys = [self._generate_key(namespace, key) for key in keys]
        result = {}
        
        try:
            if self.redis_client:
                # 使用Redis批量获取
                values = await self.redis_client.mget(cache_keys)
                for i, (original_key, value) in enumerate(zip(keys, values)):
                    if value:
                        result[original_key] = self._deserialize_value(value)
                        self.cache_stats["hits"] += 1
                    else:
                        self.cache_stats["misses"] += 1
            else:
                # 内存缓存批量获取
                current_time = time.time()
                for i, cache_key in enumerate(cache_keys):
                    original_key = keys[i]
                    if cache_key in self.memory_cache:
                        cache_item = self.memory_cache[cache_key]
                        if current_time < cache_item['expire_time']:
                            result[original_key] = cache_item['value']
                            self.cache_stats["hits"] += 1
                        else:
                            del self.memory_cache[cache_key]
                            self.cache_stats["evictions"] += 1
                            self.cache_stats["misses"] += 1
                    else:
                        self.cache_stats["misses"] += 1
                        
        except Exception as e:
            logger.error(f"批量获取缓存失败: {e}")
        
        return result
    
    async def mset(self, namespace: str, mapping: Dict[str, Any], ttl: int = None) -> bool:
        """批量设置缓存"""
        ttl = ttl or self.default_ttl
        
        try:
            if self.redis_client:
                # 使用Redis管道批量设置
                pipe = self.redis_client.pipeline()
                for key, value in mapping.items():
                    cache_key = self._generate_key(namespace, key)
                    serialized_value = self._serialize_value(value)
                    await pipe.setex(cache_key, ttl, serialized_value)
                await pipe.execute()
                self.cache_stats["sets"] += len(mapping)
                return True
            else:
                # 内存缓存批量设置
                current_time = time.time()
                expire_time = current_time + ttl
                
                for key, value in mapping.items():
                    self._evict_if_needed()
                    cache_key = self._generate_key(namespace, key)
                    self.memory_cache[cache_key] = {
                        'value': value,
                        'expire_time': expire_time
                    }
                
                self.cache_stats["sets"] += len(mapping)
                return True
                
        except Exception as e:
            logger.error(f"批量设置缓存失败: {e}")
            return False
    
    async def get_pattern(self, namespace: str, pattern: str) -> Dict[str, Any]:
        """根据模式获取缓存"""
        cache_pattern = self._generate_key(namespace, pattern)
        result = {}
        
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(cache_pattern)
                if keys:
                    values = await self.redis_client.mget(keys)
                    for key, value in zip(keys, values):
                        if value:
                            # 提取原始键名
                            original_key = key.decode() if isinstance(key, bytes) else key
                            original_key = original_key.split(':', 2)[-1]  # 移除namespace前缀
                            result[original_key] = self._deserialize_value(value)
            else:
                # 内存缓存模式匹配
                import fnmatch
                current_time = time.time()
                for cache_key in list(self.memory_cache.keys()):
                    if fnmatch.fnmatch(cache_key, cache_pattern):
                        cache_item = self.memory_cache[cache_key]
                        if current_time < cache_item['expire_time']:
                            # 提取原始键名
                            original_key = cache_key.split(':', 2)[-1]
                            result[original_key] = cache_item['value']
                        else:
                            del self.memory_cache[cache_key]
                            self.cache_stats["evictions"] += 1
                            
        except Exception as e:
            logger.error(f"根据模式获取缓存失败: {e}")
        
        return result
    
    async def touch(self, namespace: str, key: str, ttl: int = None) -> bool:
        """更新缓存过期时间"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        try:
            if self.redis_client:
                return bool(await self.redis_client.expire(cache_key, ttl))
            else:
                if cache_key in self.memory_cache:
                    self.memory_cache[cache_key]['expire_time'] = time.time() + ttl
                    return True
                return False
                
        except Exception as e:
            logger.error(f"更新缓存过期时间失败: {e}")
            return False
    
    async def get_ttl(self, namespace: str, key: str) -> Optional[int]:
        """获取缓存剩余过期时间"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                ttl = await self.redis_client.ttl(cache_key)
                return ttl if ttl > 0 else None
            else:
                if cache_key in self.memory_cache:
                    remaining = self.memory_cache[cache_key]['expire_time'] - time.time()
                    return int(remaining) if remaining > 0 else None
                return None
                
        except Exception as e:
            logger.error(f"获取缓存过期时间失败: {e}")
            return None
    
    async def scan_keys(self, namespace: str, pattern: str = "*", count: int = 100) -> List[str]:
        """扫描缓存键"""
        cache_pattern = self._generate_key(namespace, pattern)
        
        try:
            if self.redis_client:
                keys = []
                cursor = 0
                while True:
                    cursor, batch_keys = await self.redis_client.scan(
                        cursor=cursor, match=cache_pattern, count=count
                    )
                    keys.extend([key.decode() if isinstance(key, bytes) else key for key in batch_keys])
                    if cursor == 0:
                        break
                return [key.split(':', 2)[-1] for key in keys]  # 移除namespace前缀
            else:
                import fnmatch
                return [
                    key.split(':', 2)[-1] for key in self.memory_cache.keys()
                    if fnmatch.fnmatch(key, cache_pattern)
                ]
                
        except Exception as e:
            logger.error(f"扫描缓存键失败: {e}")
            return []
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        usage_info = {
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_max_size": self.max_memory_cache_size
        }
        
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info("memory")
                usage_info.update({
                    "redis_used_memory": redis_info.get("used_memory", 0),
                    "redis_used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "redis_max_memory": redis_info.get("maxmemory", 0),
                    "redis_memory_usage": redis_info.get("used_memory_percentage", "0%")
                })
            except Exception as e:
                logger.error(f"获取Redis内存信息失败: {e}")
        
        return usage_info
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            "status": "healthy",
            "backend": "redis" if self.redis_client else "memory",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if self.redis_client:
                # 测试Redis连接
                start_time = time.time()
                await self.redis_client.ping()
                response_time = (time.time() - start_time) * 1000
                
                health.update({
                    "redis_connected": True,
                    "redis_response_time_ms": response_time
                })
                
                # 获取Redis信息
                redis_info = await self.redis_client.info()
                health.update({
                    "redis_version": redis_info.get("redis_version"),
                    "redis_uptime_seconds": redis_info.get("uptime_in_seconds"),
                    "redis_connected_clients": redis_info.get("connected_clients")
                })
            else:
                health.update({
                    "memory_cache_size": len(self.memory_cache),
                    "memory_cache_usage_percentage": len(self.memory_cache) / self.max_memory_cache_size * 100
                })
                
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error(f"缓存健康检查失败: {e}")
        
        return health
    
    async def warm_up(self, warm_up_data: Dict[str, Dict[str, Any]]):
        """缓存预热"""
        logger.info("开始缓存预热")
        total_items = 0
        
        try:
            for namespace, items in warm_up_data.items():
                if items:
                    await self.mset(namespace, items)
                    total_items += len(items)
                    logger.info(f"预热命名空间 {namespace}: {len(items)} 项")
            
            logger.info(f"缓存预热完成，总计 {total_items} 项")
            
        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
    
    async def export_cache(self, namespace: str = None) -> Dict[str, Any]:
        """导出缓存数据"""
        export_data = {}
        
        try:
            if namespace:
                # 导出特定命名空间
                pattern = f"{settings.APP_NAME}:{namespace}:*"
                cache_data = await self.get_pattern(namespace, "*")
                export_data[namespace] = cache_data
            else:
                # 导出所有缓存
                if self.redis_client:
                    all_keys = await self.redis_client.keys(f"{settings.APP_NAME}:*")
                    for key in all_keys:
                        key_str = key.decode() if isinstance(key, bytes) else key
                        parts = key_str.split(':', 2)
                        if len(parts) >= 3:
                            ns, actual_key = parts[1], parts[2]
                            if ns not in export_data:
                                export_data[ns] = {}
                            value = await self.redis_client.get(key)
                            if value:
                                export_data[ns][actual_key] = self._deserialize_value(value)
                else:
                    current_time = time.time()
                    for cache_key, cache_item in self.memory_cache.items():
                        if current_time < cache_item['expire_time']:
                            parts = cache_key.split(':', 2)
                            if len(parts) >= 3:
                                ns, actual_key = parts[1], parts[2]
                                if ns not in export_data:
                                    export_data[ns] = {}
                                export_data[ns][actual_key] = cache_item['value']
            
            return export_data
            
        except Exception as e:
            logger.error(f"导出缓存数据失败: {e}")
            return {}
    
    async def import_cache(self, import_data: Dict[str, Dict[str, Any]], ttl: int = None):
        """导入缓存数据"""
        try:
            for namespace, items in import_data.items():
                if items:
                    await self.mset(namespace, items, ttl)
                    logger.info(f"导入命名空间 {namespace}: {len(items)} 项")
            
            logger.info("缓存数据导入完成")
            
        except Exception as e:
            logger.error(f"导入缓存数据失败: {e}")
    
    def reset_stats(self):
        """重置统计信息"""
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        logger.info("缓存统计信息已重置")
    
    async def cleanup(self):
        """清理缓存资源"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis 连接已关闭")
            
            self.memory_cache.clear()
            self.reset_stats()
            logger.info("缓存管理器清理完成")
            
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")


# 缓存装饰器
def cached(namespace: str, key_func=None, ttl: int = None):
    """缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 基于函数名和参数生成键
                args_str = str(args) + str(sorted(kwargs.items()))
                cache_key = hashlib.md5(args_str.encode()).hexdigest()
            
            # 尝试从缓存获取
            cached_result = await cache_manager.get(namespace, cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await cache_manager.set(namespace, cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# 全局缓存管理器实例
cache_manager = CacheManager()