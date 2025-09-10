"""
安全头中间件
添加各种安全相关的HTTP头以提高应用安全性
"""

from fastapi import Request, Response
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware:
    """安全头中间件"""
    
    def __init__(self, 
                 enable_csp: bool = True,
                 enable_hsts: bool = True,
                 enable_frame_options: bool = True,
                 enable_content_type_options: bool = True,
                 enable_xss_protection: bool = True,
                 enable_referrer_policy: bool = True,
                 custom_headers: Optional[Dict[str, str]] = None):
        """
        初始化安全头中间件
        
        Args:
            enable_csp: 启用内容安全策略
            enable_hsts: 启用HTTP严格传输安全
            enable_frame_options: 启用X-Frame-Options
            enable_content_type_options: 启用X-Content-Type-Options
            enable_xss_protection: 启用X-XSS-Protection
            enable_referrer_policy: 启用Referrer-Policy
            custom_headers: 自定义安全头
        """
        self.enable_csp = enable_csp
        self.enable_hsts = enable_hsts
        self.enable_frame_options = enable_frame_options
        self.enable_content_type_options = enable_content_type_options
        self.enable_xss_protection = enable_xss_protection
        self.enable_referrer_policy = enable_referrer_policy
        self.custom_headers = custom_headers or {}
        
        # 默认安全头配置
        self.security_headers = self._get_default_headers()
    
    def _get_default_headers(self) -> Dict[str, str]:
        """获取默认安全头"""
        headers = {}
        
        if self.enable_csp:
            # 内容安全策略 - 防止XSS攻击
            csp_policy = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "font-src 'self' data:; "
                "connect-src 'self' ws: wss:; "
                "media-src 'self'; "
                "object-src 'none'; "
                "frame-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
            headers["Content-Security-Policy"] = csp_policy
        
        if self.enable_hsts:
            # HTTP严格传输安全 - 强制HTTPS
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        if self.enable_frame_options:
            # 防止点击劫持
            headers["X-Frame-Options"] = "DENY"
        
        if self.enable_content_type_options:
            # 防止MIME类型嗅探
            headers["X-Content-Type-Options"] = "nosniff"
        
        if self.enable_xss_protection:
            # XSS保护（虽然现代浏览器已内置，但仍建议设置）
            headers["X-XSS-Protection"] = "1; mode=block"
        
        if self.enable_referrer_policy:
            # 控制Referer头的发送
            headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # 其他安全头
        headers.update({
            # 防止缓存敏感信息
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            
            # 权限策略
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=()"
            ),
            
            # 跨域嵌入器策略
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
            
            # 服务器信息隐藏
            "Server": "SignAvatar-Web"
        })
        
        # 添加自定义头
        headers.update(self.custom_headers)
        
        return headers
    
    def _should_apply_headers(self, request: Request, response: Response) -> bool:
        """判断是否应该应用安全头"""
        # 跳过静态文件和某些API端点
        path = request.url.path
        
        # 静态文件路径
        static_paths = ["/static/", "/assets/", "/favicon.ico", "/robots.txt"]
        if any(path.startswith(static_path) for static_path in static_paths):
            return False
        
        # WebSocket连接
        if path.startswith("/ws/"):
            return False
        
        # 健康检查端点
        if path in ["/health", "/api/health", "/ping"]:
            return False
        
        return True
    
    def _apply_api_specific_headers(self, request: Request, response: Response):
        """应用API特定的安全头"""
        path = request.url.path
        
        # 认证相关端点的额外安全措施
        if path.startswith("/api/auth/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            
        # 文件上传端点
        if "upload" in path:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Content-Security-Policy"] = "default-src 'none'"
        
        # API端点的CORS预检响应
        if request.method == "OPTIONS":
            response.headers["Access-Control-Max-Age"] = "86400"  # 24小时
    
    async def __call__(self, request: Request, call_next):
        """中间件处理函数"""
        # 处理请求
        response = await call_next(request)
        
        # 检查是否应该应用安全头
        if not self._should_apply_headers(request, response):
            return response
        
        # 应用通用安全头
        for header_name, header_value in self.security_headers.items():
            # 避免覆盖已设置的头
            if header_name not in response.headers:
                response.headers[header_name] = header_value
        
        # 应用API特定的安全头
        self._apply_api_specific_headers(request, response)
        
        # 记录安全头应用情况（仅在调试模式下）
        if logger.isEnabledFor(logging.DEBUG):
            applied_headers = [name for name in self.security_headers.keys() 
                             if name in response.headers]
            logger.debug(f"应用安全头到 {request.url.path}: {applied_headers}")
        
        return response


class CSRFProtectionMiddleware:
    """CSRF保护中间件"""
    
    def __init__(self, secret_key: str, exempt_paths: Optional[list] = None):
        """
        初始化CSRF保护中间件
        
        Args:
            secret_key: 用于生成CSRF令牌的密钥
            exempt_paths: 免除CSRF检查的路径列表
        """
        self.secret_key = secret_key
        self.exempt_paths = exempt_paths or [
            "/api/auth/login",  # 登录页面需要获取CSRF令牌
            "/api/health",
            "/ws/",  # WebSocket连接
        ]
    
    def _generate_csrf_token(self, session_id: str) -> str:
        """生成CSRF令牌"""
        import hmac
        import hashlib
        import time
        
        timestamp = str(int(time.time()))
        message = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def _verify_csrf_token(self, token: str, session_id: str) -> bool:
        """验证CSRF令牌"""
        import hmac
        import hashlib
        import time
        
        try:
            timestamp, signature = token.split(":", 1)
            
            # 检查令牌是否过期（1小时）
            if int(time.time()) - int(timestamp) > 3600:
                return False
            
            # 验证签名
            message = f"{session_id}:{timestamp}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        
        except (ValueError, TypeError):
            return False
    
    async def __call__(self, request: Request, call_next):
        """中间件处理函数"""
        path = request.url.path
        method = request.method
        
        # 跳过GET、HEAD、OPTIONS请求和免除路径
        if method in ["GET", "HEAD", "OPTIONS"] or any(path.startswith(exempt) for exempt in self.exempt_paths):
            response = await call_next(request)
            
            # 为GET请求添加CSRF令牌到响应头
            if method == "GET" and not any(path.startswith(exempt) for exempt in self.exempt_paths):
                session_id = request.headers.get("X-Session-ID", "default")
                csrf_token = self._generate_csrf_token(session_id)
                response.headers["X-CSRF-Token"] = csrf_token
            
            return response
        
        # 验证CSRF令牌
        csrf_token = request.headers.get("X-CSRF-Token")
        session_id = request.headers.get("X-Session-ID", "default")
        
        if not csrf_token or not self._verify_csrf_token(csrf_token, session_id):
            logger.warning(f"CSRF令牌验证失败: IP={request.client.host}, Path={path}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"error": "CSRF令牌无效或缺失"}
            )
        
        return await call_next(request)


# 全局中间件实例
security_headers_middleware = SecurityHeadersMiddleware()
