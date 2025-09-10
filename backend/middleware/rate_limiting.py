"""
速率限制中间件
防止API滥用和DDoS攻击
"""

import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, float] = {}  # IP -> 解封时间
        self.cleanup_interval = 300  # 5分钟清理一次
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self):
        """清理过期的请求记录"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.window_seconds
        
        # 清理请求记录
        for ip in list(self.requests.keys()):
            request_times = self.requests[ip]
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()
            
            # 如果队列为空，删除该IP记录
            if not request_times:
                del self.requests[ip]
        
        # 清理已解封的IP
        for ip in list(self.blocked_ips.keys()):
            if current_time > self.blocked_ips[ip]:
                del self.blocked_ips[ip]
        
        self.last_cleanup = current_time
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, Optional[str]]:
        """检查是否允许请求"""
        current_time = time.time()
        
        # 清理过期记录
        self._cleanup_old_requests()
        
        # 检查是否被封禁
        if client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                remaining_time = int(self.blocked_ips[client_ip] - current_time)
                return False, f"IP被临时封禁，剩余时间: {remaining_time}秒"
            else:
                del self.blocked_ips[client_ip]
        
        # 检查请求频率
        request_times = self.requests[client_ip]
        cutoff_time = current_time - self.window_seconds
        
        # 移除过期的请求记录
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # 检查是否超过限制
        if len(request_times) >= self.max_requests:
            # 封禁IP 10分钟
            self.blocked_ips[client_ip] = current_time + 600
            logger.warning(f"IP {client_ip} 因请求过于频繁被临时封禁")
            return False, f"请求过于频繁，已被临时封禁10分钟"
        
        # 记录当前请求
        request_times.append(current_time)
        return True, None
    
    def get_stats(self) -> Dict:
        """获取速率限制统计信息"""
        self._cleanup_old_requests()
        return {
            "active_ips": len(self.requests),
            "blocked_ips": len(self.blocked_ips),
            "total_requests": sum(len(times) for times in self.requests.values())
        }


class RateLimitMiddleware:
    """速率限制中间件"""
    
    def __init__(self, 
                 default_max_requests: int = 100,
                 default_window_seconds: int = 60,
                 strict_endpoints: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        初始化速率限制中间件
        
        Args:
            default_max_requests: 默认最大请求数
            default_window_seconds: 默认时间窗口（秒）
            strict_endpoints: 严格限制的端点 {路径: (最大请求数, 时间窗口)}
        """
        self.default_limiter = RateLimiter(default_max_requests, default_window_seconds)
        self.strict_limiters = {}
        
        # 为严格限制的端点创建专门的限制器
        if strict_endpoints:
            for endpoint, (max_req, window) in strict_endpoints.items():
                self.strict_limiters[endpoint] = RateLimiter(max_req, window)
        
        # 默认严格限制的端点
        default_strict = {
            "/api/auth/login": (5, 300),  # 登录：5次/5分钟
            "/api/auth/register": (3, 3600),  # 注册：3次/小时
            "/api/auth/reset-password": (3, 3600),  # 重置密码：3次/小时
            "/api/sign-recognition/upload-video": (10, 60),  # 视频上传：10次/分钟
        }
        
        for endpoint, (max_req, window) in default_strict.items():
            if endpoint not in self.strict_limiters:
                self.strict_limiters[endpoint] = RateLimiter(max_req, window)
    
    def get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        # 检查代理头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # 回退到直接连接IP
        return request.client.host if request.client else "unknown"
    
    async def __call__(self, request: Request, call_next):
        """中间件处理函数"""
        client_ip = self.get_client_ip(request)
        path = request.url.path
        
        # 选择合适的限制器
        limiter = self.strict_limiters.get(path, self.default_limiter)
        
        # 检查是否允许请求
        allowed, message = limiter.is_allowed(client_ip)
        
        if not allowed:
            logger.warning(f"速率限制阻止请求: IP={client_ip}, Path={path}, Reason={message}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "请求过于频繁",
                    "message": message,
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # 处理请求
        response = await call_next(request)
        
        # 添加速率限制头
        remaining = limiter.max_requests - len(limiter.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + limiter.window_seconds))
        
        return response
    
    def get_stats(self) -> Dict:
        """获取所有限制器的统计信息"""
        stats = {
            "default": self.default_limiter.get_stats(),
            "strict_endpoints": {}
        }
        
        for endpoint, limiter in self.strict_limiters.items():
            stats["strict_endpoints"][endpoint] = limiter.get_stats()
        
        return stats


# 全局速率限制器实例
rate_limit_middleware = RateLimitMiddleware()


async def get_rate_limit_stats():
    """获取速率限制统计信息的依赖函数"""
    return rate_limit_middleware.get_stats()
