"""
身份认证和安全管理模块
提供JWT令牌、权限控制、安全中间件等功能
"""

import jwt
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps
import hashlib
import secrets
import time
from pathlib import Path
from collections import deque, defaultdict

from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from utils.logger import setup_logger
from utils.config import settings
from utils.database import db_manager

logger = setup_logger(__name__)


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: List[str] = []


class UserToken(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str
    user_info: Dict


class SecurityAuditor:
    """安全审计器"""
    
    def __init__(self):
        self.audit_log = deque(maxlen=10000)
        self.suspicious_activities = defaultdict(list)
        
        logger.info("安全审计器初始化完成")
    
    async def log_activity(self, event_type: str, user_id: int = None, 
                          ip_address: str = None, details: Dict = None):
        """记录活动日志"""
        timestamp = time.time()
        audit_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details or {},
            'severity': self._get_event_severity(event_type)
        }
        
        self.audit_log.append(audit_entry)
        
        # 检测可疑活动
        await self._detect_suspicious_activity(audit_entry)
        
        # 记录到数据库
        await self._persist_audit_log(audit_entry)
    
    def _get_event_severity(self, event_type: str) -> str:
        """获取事件严重程度"""
        high_severity_events = [
            'login_failed_multiple', 'unauthorized_access_attempt',
            'privilege_escalation', 'data_breach_attempt',
            'malicious_request', 'account_lockout'
        ]
        
        medium_severity_events = [
            'login_failed', 'password_reset_request',
            'unusual_access_pattern', 'rate_limit_exceeded'
        ]
        
        if event_type in high_severity_events:
            return 'high'
        elif event_type in medium_severity_events:
            return 'medium'
        else:
            return 'low'
    
    async def _detect_suspicious_activity(self, audit_entry: Dict):
        """检测可疑活动"""
        ip_address = audit_entry.get('ip_address')
        user_id = audit_entry.get('user_id')
        event_type = audit_entry.get('event_type')
        
        if not ip_address and not user_id:
            return
        
        # 检测短时间内多次失败登录
        if event_type == 'login_failed':
            await self._check_brute_force_attack(ip_address, user_id)
        
        # 检测异常访问模式
        if user_id:
            await self._check_unusual_access_pattern(user_id, ip_address)
        
        # 检测可疑IP地址
        if ip_address:
            await self._check_suspicious_ip(ip_address)
    
    async def _check_brute_force_attack(self, ip_address: str, user_id: int):
        """检测暴力破解攻击"""
        current_time = time.time()
        time_window = 300  # 5分钟
        
        # 检查IP地址的失败尝试
        if ip_address:
            ip_failures = [
                entry for entry in self.audit_log
                if (entry.get('ip_address') == ip_address and
                    entry.get('event_type') == 'login_failed' and
                    current_time - entry.get('timestamp', 0) < time_window)
            ]
            
            if len(ip_failures) >= 5:
                await self._trigger_security_alert(
                    'brute_force_attack_ip',
                    f"IP地址 {ip_address} 在5分钟内尝试登录失败5次",
                    {'ip_address': ip_address, 'attempts': len(ip_failures)}
                )
        
        # 检查用户的失败尝试
        if user_id:
            user_failures = [
                entry for entry in self.audit_log
                if (entry.get('user_id') == user_id and
                    entry.get('event_type') == 'login_failed' and
                    current_time - entry.get('timestamp', 0) < time_window)
            ]
            
            if len(user_failures) >= 3:
                await self._trigger_security_alert(
                    'brute_force_attack_user',
                    f"用户 {user_id} 在5分钟内登录失败3次",
                    {'user_id': user_id, 'attempts': len(user_failures)}
                )
    
    async def _check_unusual_access_pattern(self, user_id: int, ip_address: str):
        """检测异常访问模式"""
        current_time = time.time()
        time_window = 3600  # 1小时
        
        # 获取用户最近的访问记录
        user_activities = [
            entry for entry in self.audit_log
            if (entry.get('user_id') == user_id and
                current_time - entry.get('timestamp', 0) < time_window)
        ]
        
        if len(user_activities) < 2:
            return
        
        # 检查地理位置异常（模拟）
        ip_locations = set(entry.get('ip_address') for entry in user_activities)
        if len(ip_locations) > 3:  # 1小时内从超过3个不同IP访问
            await self._trigger_security_alert(
                'unusual_access_pattern',
                f"用户 {user_id} 在1小时内从{len(ip_locations)}个不同IP地址访问",
                {'user_id': user_id, 'ip_addresses': list(ip_locations)}
            )
        
        # 检查访问频率异常
        if len(user_activities) > 100:  # 1小时内超过100次活动
            await self._trigger_security_alert(
                'high_frequency_access',
                f"用户 {user_id} 在1小时内活动{len(user_activities)}次",
                {'user_id': user_id, 'activity_count': len(user_activities)}
            )
    
    async def _check_suspicious_ip(self, ip_address: str):
        """检测可疑IP地址"""
        # 这里可以集成IP威胁情报数据库
        # 目前实现简单的黑名单检查
        
        known_bad_ips = [
            '192.168.1.100',  # 示例恶意IP
            '10.0.0.50'       # 示例恶意IP
        ]
        
        if ip_address in known_bad_ips:
            await self._trigger_security_alert(
                'suspicious_ip_detected',
                f"检测到可疑IP地址: {ip_address}",
                {'ip_address': ip_address}
            )
    
    async def _trigger_security_alert(self, alert_type: str, message: str, details: Dict):
        """触发安全告警"""
        alert = {
            'type': alert_type,
            'message': message,
            'details': details,
            'timestamp': time.time(),
            'severity': 'high'
        }
        
        logger.warning(f"安全告警: {message}")
        
        # 可以在这里添加邮件、短信等通知机制
        await self._send_security_notification(alert)
    
    async def _send_security_notification(self, alert: Dict):
        """发送安全通知"""
        # 这里可以实现邮件、短信、Webhook等通知方式
        logger.warning(f"安全通知: {alert['message']}")
    
    async def _persist_audit_log(self, audit_entry: Dict):
        """持久化审计日志"""
        try:
            # 将审计日志保存到数据库
            await db_manager.add_audit_log(audit_entry)
        except Exception as e:
            logger.error(f"保存审计日志失败: {e}")
    
    def get_security_report(self, hours: int = 24) -> Dict:
        """获取安全报告"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_activities = [
            entry for entry in self.audit_log
            if entry.get('timestamp', 0) > cutoff_time
        ]
        
        # 统计事件类型
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        ip_counts = defaultdict(int)
        
        for entry in recent_activities:
            event_counts[entry.get('event_type', 'unknown')] += 1
            severity_counts[entry.get('severity', 'unknown')] += 1
            if entry.get('ip_address'):
                ip_counts[entry.get('ip_address')] += 1
        
        return {
            'report_period_hours': hours,
            'total_activities': len(recent_activities),
            'event_types': dict(event_counts),
            'severity_distribution': dict(severity_counts),
            'top_ips': dict(sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'generated_at': datetime.utcnow().isoformat()
        }


class EncryptionManager:
    """加密管理器"""
    
    def __init__(self):
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        
        self.fernet = None
        self._initialize_encryption()
        
        logger.info("加密管理器初始化完成")
    
    def _initialize_encryption(self):
        """初始化加密"""
        try:
            from cryptography.fernet import Fernet
            
            # 使用配置中的密钥或生成新密钥
            if hasattr(settings, 'ENCRYPTION_KEY') and settings.ENCRYPTION_KEY:
                key = settings.ENCRYPTION_KEY.encode()
                if len(key) != 44:  # Fernet key must be 32 url-safe base64-encoded bytes
                    key = base64.urlsafe_b64encode(hashlib.sha256(key).digest())
            else:
                key = Fernet.generate_key()
                logger.warning("使用生成的临时加密密钥，建议在配置中设置固定密钥")
            
            self.fernet = Fernet(key)
            
        except ImportError:
            logger.warning("cryptography库未安装，加密功能将被禁用")
        except Exception as e:
            logger.error(f"初始化加密失败: {e}")
    
    def encrypt_data(self, data: str) -> str:
        """加密数据"""
        if not self.fernet:
            return data
        
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        if not self.fernet:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            return encrypted_data
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """密码哈希"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # 使用PBKDF2进行密码哈希
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 迭代次数
        )
        
        return base64.b64encode(password_hash).decode('utf-8'), salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """验证密码"""
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == hashed_password
    
    def generate_secure_token(self, length: int = 32) -> str:
        """生成安全令牌"""
        return secrets.token_urlsafe(length)


class SecurityHeadersMiddleware:
    """安全头中间件"""
    
    def __init__(self):
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    async def add_security_headers(self, request: Request, response):
        """添加安全头"""
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response


class InputValidator:
    """输入验证器"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_username(username: str) -> bool:
        """验证用户名格式"""
        import re
        # 用户名必须是3-20个字符，只能包含字母、数字、下划线
        pattern = r'^[a-zA-Z0-9_]{3,20}$'
        return re.match(pattern, username) is not None
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict:
        """验证密码强度"""
        import re
        
        result = {
            'is_valid': True,
            'score': 0,
            'issues': []
        }
        
        # 长度检查
        if len(password) < 8:
            result['issues'].append('密码长度至少8个字符')
            result['is_valid'] = False
        else:
            result['score'] += 1
        
        # 包含大写字母
        if re.search(r'[A-Z]', password):
            result['score'] += 1
        else:
            result['issues'].append('密码应包含大写字母')
        
        # 包含小写字母
        if re.search(r'[a-z]', password):
            result['score'] += 1
        else:
            result['issues'].append('密码应包含小写字母')
        
        # 包含数字
        if re.search(r'\d', password):
            result['score'] += 1
        else:
            result['issues'].append('密码应包含数字')
        
        # 包含特殊字符
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            result['score'] += 1
        else:
            result['issues'].append('密码应包含特殊字符')
        
        # 评估强度
        if result['score'] >= 4 and len(password) >= 12:
            result['strength'] = 'strong'
        elif result['score'] >= 3:
            result['strength'] = 'medium'
        else:
            result['strength'] = 'weak'
            result['is_valid'] = False
        
        return result
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """清理输入数据"""
        import html
        import re
        
        # HTML转义
        sanitized = html.escape(input_str)
        
        # 移除潜在的脚本标签
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除其他潜在危险的标签
        dangerous_tags = ['iframe', 'object', 'embed', 'form', 'input']
        for tag in dangerous_tags:
            pattern = f'<{tag}[^>]*>.*?</{tag}>'
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_file_upload(filename: str, content_type: str, file_size: int) -> Dict:
        """验证文件上传"""
        result = {
            'is_valid': True,
            'issues': []
        }
        
        # 文件大小检查 (10MB限制)
        max_size = 10 * 1024 * 1024
        if file_size > max_size:
            result['issues'].append(f'文件大小超过限制 ({max_size / 1024 / 1024:.1f}MB)')
            result['is_valid'] = False
        
        # 文件扩展名检查
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.pdf', '.txt', '.docx', '.xlsx'}
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            result['issues'].append(f'不支持的文件类型: {file_ext}')
            result['is_valid'] = False
        
        # MIME类型检查
        allowed_mimes = {
            'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
            'application/pdf', 'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        if content_type not in allowed_mimes:
            result['issues'].append(f'不支持的MIME类型: {content_type}')
            result['is_valid'] = False
        
        return result


# 更新SecurityManager类以包含新功能
class SecurityManager:
    """安全管理器 - 更新版本"""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = 7
        
        # 新增组件
        self.auditor = SecurityAuditor()
        self.encryption_manager = EncryptionManager()
        self.security_headers = SecurityHeadersMiddleware()
        self.input_validator = InputValidator()
        
        # 活跃会话跟踪
        self.active_sessions = {}
        
        # 速率限制
        self.rate_limit_cache = {}
        
        # 黑名单IP
        self.blacklisted_ips = set()
        
        logger.info("增强安全管理器初始化完成")
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """创建刷新令牌"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire, 
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # JWT ID for revocation
        })
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict:
        """验证令牌"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """验证用户身份"""
        user = await db_manager.authenticate_user(username, password)
        return user
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """获取当前用户"""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的令牌类型",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的用户信息",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = await db_manager.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
    
    async def get_current_active_user(self, current_user: dict = Depends(lambda: None)):
        """获取当前活跃用户"""
        if current_user is None:
            current_user = await self.get_current_user()
        
        if not current_user.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户账户已被禁用"
            )
        return current_user
    
    async def require_admin(self, current_user: dict = Depends(lambda: None)):
        """要求管理员权限"""
        if current_user is None:
            current_user = await self.get_current_active_user()
        
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="需要管理员权限"
            )
        return current_user
    
    async def login_user(self, username: str, password: str, 
                        device_info: Dict = None, ip_address: str = None) -> UserToken:
        """用户登录"""
        user = await self.authenticate_user(username, password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 创建会话
        session_id = await db_manager.create_session(
            user_id=user["id"],
            device_info=device_info,
            ip_address=ip_address
        )
        
        # 创建令牌
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"user_id": user["id"], "username": user["username"], "session_id": session_id},
            expires_delta=access_token_expires
        )
        
        refresh_token = self.create_refresh_token(
            data={"user_id": user["id"], "username": user["username"], "session_id": session_id}
        )
        
        # 记录活跃会话
        self.active_sessions[session_id] = {
            "user_id": user["id"],
            "username": user["username"],
            "login_time": datetime.utcnow(),
            "ip_address": ip_address,
            "device_info": device_info
        }
        
        return UserToken(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60,
            refresh_token=refresh_token,
            user_info={
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "full_name": user["full_name"],
                "is_admin": user["is_admin"],
                "preferences": user["preferences"],
                "accessibility_settings": user["accessibility_settings"]
            }
        )
    
    async def enhanced_login_user(self, username: str, password: str, 
                                device_info: Dict = None, ip_address: str = None) -> UserToken:
        """增强版用户登录"""
        # 检查IP黑名单
        if ip_address in self.blacklisted_ips:
            await self.auditor.log_activity(
                'blacklisted_ip_access', 
                ip_address=ip_address,
                details={'attempted_username': username}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="访问被拒绝"
            )
        
        # 记录登录尝试
        await self.auditor.log_activity(
            'login_attempt',
            ip_address=ip_address,
            details={'username': username, 'device_info': device_info}
        )
        
        try:
            user = await self.authenticate_user(username, password)
            
            if not user:
                # 记录失败的登录尝试
                await self.auditor.log_activity(
                    'login_failed',
                    ip_address=ip_address,
                    details={'username': username, 'reason': 'invalid_credentials'}
                )
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="用户名或密码错误",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # 记录成功登录
            await self.auditor.log_activity(
                'login_success',
                user_id=user["id"],
                ip_address=ip_address,
                details={'device_info': device_info}
            )
            
            # 继续原有的登录流程...
            return await self.login_user(username, password, device_info, ip_address)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"登录过程中发生错误: {e}")
            await self.auditor.log_activity(
                'login_error',
                ip_address=ip_address,
                details={'username': username, 'error': str(e)}
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="登录过程中发生错误"
            )
    
    async def refresh_access_token(self, refresh_token: str) -> Dict:
        """刷新访问令牌"""
        payload = self.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的刷新令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("user_id")
        username = payload.get("username")
        session_id = payload.get("session_id")
        
        # 验证用户仍然存在且活跃
        user = await db_manager.get_user_by_id(user_id)
        if not user or not user.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户账户无效或已被禁用"
            )
        
        # 创建新的访问令牌
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"user_id": user_id, "username": username, "session_id": session_id},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60
        }
    
    async def logout_user(self, session_id: str):
        """用户登出"""
        # 结束会话
        await db_manager.end_session(session_id)
        
        # 移除活跃会话
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        logger.info(f"用户会话 {session_id} 已登出")
    
    async def check_rate_limit(self, request: Request, limit: int = 100, window: int = 3600):
        """检查速率限制"""
        client_ip = request.client.host
        current_time = int(time.time())
        window_start = current_time - window
        
        # 清理过期的记录
        if client_ip in self.rate_limit_cache:
            self.rate_limit_cache[client_ip] = [
                timestamp for timestamp in self.rate_limit_cache[client_ip]
                if timestamp > window_start
            ]
        else:
            self.rate_limit_cache[client_ip] = []
        
        # 检查是否超过限制
        if len(self.rate_limit_cache[client_ip]) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，请在{window}秒后重试"
            )
        
        # 记录当前请求
        self.rate_limit_cache[client_ip].append(current_time)
    
    def generate_api_key(self, user_id: int, scopes: List[str] = None) -> str:
        """生成API密钥"""
        data = {
            "user_id": user_id,
            "scopes": scopes or [],
            "type": "api_key",
            "created_at": datetime.utcnow().isoformat()
        }
        
        # 创建永不过期的令牌（除非手动撤销）
        encoded_jwt = jwt.encode(data, settings.SECRET_KEY, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_api_key(self, api_key: str) -> Dict:
        """验证API密钥"""
        try:
            payload = jwt.decode(api_key, settings.SECRET_KEY, algorithms=[self.algorithm])
            
            if payload.get("type") != "api_key":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="无效的API密钥类型"
                )
            
            return payload
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的API密钥"
            )
    
    async def log_security_event(self, event_type: str, user_id: int = None, 
                                ip_address: str = None, details: Dict = None):
        """记录安全事件"""
        try:
            # 这里可以集成到数据库的系统日志表
            log_data = {
                "level": "SECURITY",
                "module": "security_manager",
                "message": event_type,
                "details": details or {},
                "user_id": user_id,
                "ip_address": ip_address
            }
            
            logger.warning(f"安全事件: {event_type} - 用户: {user_id} - IP: {ip_address}")
            
        except Exception as e:
            logger.error(f"记录安全事件失败: {e}")
    
    def get_active_sessions(self) -> Dict:
        """获取活跃会话信息"""
        return {
            "total_sessions": len(self.active_sessions),
            "sessions": list(self.active_sessions.values())
        }
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_info in self.active_sessions.items():
            login_time = session_info["login_time"]
            if current_time - login_time > timedelta(hours=24):  # 24小时自动过期
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.logout_user(session_id)
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")


# 权限装饰器
def require_auth(func):
    """要求身份认证的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 这里可以添加身份验证逻辑
        return await func(*args, **kwargs)
    return wrapper


def require_admin_auth(func):
    """要求管理员权限的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 这里可以添加管理员权限验证逻辑
        return await func(*args, **kwargs)
    return wrapper


def rate_limit(limit: int = 100, window: int = 3600):
    """速率限制装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 这里可以添加速率限制逻辑
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# 全局安全管理器实例
security_manager = SecurityManager()