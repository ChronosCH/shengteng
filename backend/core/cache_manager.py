"""
缓存管理器
Cache Manager
提供多层缓存策略，包括内存缓存、Redis缓存等
"""

import asyncio
import logging
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"              # 最近最少使用
    LFU = "lfu"              # 最少使用频率
    FIFO = "fifo"            # 先进先出
    TTL = "ttl"              # 基于时间过期

@dataclass
class CacheItem:
    """缓存项"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """更新访问时间和次数"""
        self.last_accessed = time.time()
        self.access_count += 1

class MemoryCache:
    """内存缓存"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheItem] = {}
        self.access_order: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        
        # 统计信息
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            item = self.cache[key]
            
            # 检查是否过期
            if item.is_expired():
                del self.cache[key]
                if key in self.access_order:
                    del self.access_order[key]
                self.misses += 1
                return None
            
            # 更新访问信息
            item.touch()
            self.hits += 1
            
            # 更新访问顺序（LRU策略）
            if self.strategy == CacheStrategy.LRU:
                self.access_order.move_to_end(key)
            
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        with self.lock:
            current_time = time.time()
            
            # 使用默认TTL
            if ttl is None:
                ttl = self.default_ttl
            
            # 创建缓存项
            item = CacheItem(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                ttl=ttl
            )
            
            # 如果键已存在，更新
            if key in self.cache:
                self.cache[key] = item
                if self.strategy == CacheStrategy.LRU:
                    self.access_order.move_to_end(key)
                return True
            
            # 检查是否需要清理空间
            if len(self.cache) >= self.max_size:
                self._evict()
            
            # 添加新项
            self.cache[key] = item
            if self.strategy == CacheStrategy.LRU:
                self.access_order[key] = None
            
            return True
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    del self.access_order[key]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def _evict(self):
        """根据策略清理缓存"""
        if not self.cache:
            return
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            # 最近最少使用
            key_to_evict = next(iter(self.access_order))
        
        elif self.strategy == CacheStrategy.LFU:
            # 最少使用频率
            min_access_count = min(item.access_count for item in self.cache.values())
            for key, item in self.cache.items():
                if item.access_count == min_access_count:
                    key_to_evict = key
                    break
        
        elif self.strategy == CacheStrategy.FIFO:
            # 先进先出
            oldest_time = min(item.created_at for item in self.cache.values())
            for key, item in self.cache.items():
                if item.created_at == oldest_time:
                    key_to_evict = key
                    break
        
        elif self.strategy == CacheStrategy.TTL:
            # 优先清理过期项
            current_time = time.time()
            for key, item in self.cache.items():
                if item.is_expired():
                    key_to_evict = key
                    break
            
            # 如果没有过期项，使用LRU策略
            if key_to_evict is None and self.access_order:
                key_to_evict = next(iter(self.access_order))
        
        # 执行清理
        if key_to_evict:
            del self.cache[key_to_evict]
            if key_to_evict in self.access_order:
                del self.access_order[key_to_evict]
            self.evictions += 1
    
    def cleanup_expired(self):
        """清理过期项"""
        with self.lock:
            expired_keys = []
            for key, item in self.cache.items():
                if item.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    del self.access_order[key]
                self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "strategy": self.strategy.value
            }

class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.caches: Dict[str, MemoryCache] = {}
        self.default_cache = MemoryCache()
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300  # 5分钟
    
    def create_cache(
        self, 
        name: str, 
        max_size: int = 1000, 
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None
    ) -> MemoryCache:
        """创建命名缓存"""
        cache = MemoryCache(max_size, strategy, default_ttl)
        self.caches[name] = cache
        logger.info(f"✅ 创建缓存: {name}, 策略: {strategy.value}, 大小: {max_size}")
        return cache
    
    def get_cache(self, name: str = "default") -> MemoryCache:
        """获取缓存实例"""
        if name == "default":
            return self.default_cache
        return self.caches.get(name)
    
    def get(self, key: str, cache_name: str = "default") -> Optional[Any]:
        """从指定缓存获取值"""
        cache = self.get_cache(cache_name)
        if cache:
            return cache.get(key)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, cache_name: str = "default") -> bool:
        """向指定缓存设置值"""
        cache = self.get_cache(cache_name)
        if cache:
            return cache.set(key, value, ttl)
        return False
    
    def delete(self, key: str, cache_name: str = "default") -> bool:
        """从指定缓存删除值"""
        cache = self.get_cache(cache_name)
        if cache:
            return cache.delete(key)
        return False
    
    def clear(self, cache_name: str = "default"):
        """清空指定缓存"""
        cache = self.get_cache(cache_name)
        if cache:
            cache.clear()
    
    def clear_all(self):
        """清空所有缓存"""
        self.default_cache.clear()
        for cache in self.caches.values():
            cache.clear()
    
    async def start_cleanup_task(self):
        """启动清理任务"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("✅ 缓存清理任务已启动")
    
    async def stop_cleanup_task(self):
        """停止清理任务"""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ 缓存清理任务已停止")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # 清理默认缓存
                self.default_cache.cleanup_expired()
                
                # 清理所有命名缓存
                for cache in self.caches.values():
                    cache.cleanup_expired()
                
                logger.debug("🧹 缓存清理完成")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 缓存清理出错: {e}")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有缓存统计"""
        stats = {
            "default": self.default_cache.get_stats()
        }
        
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        
        return stats

# 装饰器支持
def cached(
    ttl: Optional[float] = None,
    cache_name: str = "default",
    key_func: Optional[Callable] = None
):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认键生成策略
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key, cache_name)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, cache_name)
            
            return result
        
        return wrapper
    return decorator

def async_cached(
    ttl: Optional[float] = None,
    cache_name: str = "default",
    key_func: Optional[Callable] = None
):
    """异步缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认键生成策略
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key, cache_name)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, cache_name)
            
            return result
        
        return wrapper
    return decorator

# 全局缓存管理器实例
cache_manager = CacheManager()

# 便捷函数
def get_cache(name: str = "default") -> MemoryCache:
    """获取缓存实例"""
    return cache_manager.get_cache(name)

def cache_get(key: str, cache_name: str = "default") -> Optional[Any]:
    """获取缓存值"""
    return cache_manager.get(key, cache_name)

def cache_set(key: str, value: Any, ttl: Optional[float] = None, cache_name: str = "default") -> bool:
    """设置缓存值"""
    return cache_manager.set(key, value, ttl, cache_name)

def cache_delete(key: str, cache_name: str = "default") -> bool:
    """删除缓存值"""
    return cache_manager.delete(key, cache_name)

def cache_clear(cache_name: str = "default"):
    """清空缓存"""
    cache_manager.clear(cache_name)
