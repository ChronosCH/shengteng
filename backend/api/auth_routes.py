"""
手语学习训练系统认证API路由
提供用户注册、登录、登出、个人资料管理等功能
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pydantic import BaseModel, EmailStr, validator

from ..utils.security import SecurityManager
from ..utils.database import DatabaseManager

logger = logging.getLogger(__name__)

# 初始化路由和服务
router = APIRouter(prefix="/api/auth", tags=["认证"])
security_manager = SecurityManager()
db_manager = DatabaseManager()

# 请求模型
class UserRegistration(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('用户名至少需要3个字符')
        if len(v) > 50:
            raise ValueError('用户名不能超过50个字符')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('密码至少需要6个字符')
        return v

class UserLogin(BaseModel):
    username: str
    password: str
    remember_me: bool = False

class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError('新密码至少需要6个字符')
        return v

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    preferences: Optional[Dict[str, Any]] = None
    accessibility_settings: Optional[Dict[str, Any]] = None

# 响应模型
class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: str
    last_login: Optional[str]
    preferences: Dict[str, Any]
    accessibility_settings: Dict[str, Any]

class AuthResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

def get_client_ip(request: Request) -> str:
    """获取客户端IP地址"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

@router.post("/register", response_model=AuthResponse)
async def register_user(user_data: UserRegistration, request: Request):
    """用户注册"""
    try:
        # 检查用户名是否已存在
        existing_user = await db_manager.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="用户名已存在"
            )
        
        # 检查邮箱是否已存在
        existing_email = await db_manager.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=400,
                detail="邮箱已被注册"
            )
        
        # 创建用户
        user_id = await db_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            preferences={
                "language": "zh-CN",
                "theme": "light",
                "notifications": True,
                "learning_reminders": True
            },
            accessibility_settings={
                "high_contrast": False,
                "large_text": False,
                "screen_reader": False,
                "reduced_motion": False
            }
        )
        
        logger.info(f"新用户注册成功: {user_data.username} (ID: {user_id})")
        
        return AuthResponse(
            success=True,
            message="注册成功，请登录",
            data={"user_id": user_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(status_code=500, detail="注册失败，请稍后重试")

@router.post("/login", response_model=AuthResponse)
async def login_user(login_data: UserLogin, request: Request):
    """用户登录"""
    try:
        client_ip = get_client_ip(request)
        device_info = {
            "user_agent": request.headers.get("User-Agent", ""),
            "platform": "web",
            "timestamp": datetime.now().isoformat()
        }
        
        # 使用增强版登录
        token_data = await security_manager.enhanced_login_user(
            username=login_data.username,
            password=login_data.password,
            device_info=device_info,
            ip_address=client_ip
        )
        
        return AuthResponse(
            success=True,
            message="登录成功",
            data={
                "access_token": token_data.access_token,
                "token_type": token_data.token_type,
                "expires_in": token_data.expires_in,
                "refresh_token": token_data.refresh_token,
                "user_info": token_data.user_info
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(status_code=500, detail="登录失败，请稍后重试")

@router.post("/logout", response_model=AuthResponse)
async def logout_user(current_user: dict = Depends(security_manager.get_current_user)):
    """用户登出"""
    try:
        # 获取会话ID（从令牌中）
        session_id = current_user.get("session_id")
        if session_id:
            await security_manager.logout_user(session_id)
        
        logger.info(f"用户登出: {current_user.get('username')}")
        
        return AuthResponse(
            success=True,
            message="登出成功"
        )
        
    except Exception as e:
        logger.error(f"用户登出失败: {e}")
        raise HTTPException(status_code=500, detail="登出失败")

@router.get("/profile", response_model=AuthResponse)
async def get_user_profile(current_user: dict = Depends(security_manager.get_current_user)):
    """获取用户个人资料"""
    try:
        # 获取完整的用户信息
        user_info = await db_manager.get_user_by_id(current_user["id"])
        if not user_info:
            raise HTTPException(status_code=404, detail="用户不存在")
        
        profile_data = UserProfile(
            id=user_info["id"],
            username=user_info["username"],
            email=user_info["email"],
            full_name=user_info.get("full_name"),
            is_active=user_info["is_active"],
            is_admin=user_info["is_admin"],
            created_at=user_info.get("created_at", ""),
            last_login=user_info.get("last_login"),
            preferences=user_info.get("preferences", {}),
            accessibility_settings=user_info.get("accessibility_settings", {})
        )
        
        return AuthResponse(
            success=True,
            message="获取个人资料成功",
            data=profile_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户个人资料失败: {e}")
        raise HTTPException(status_code=500, detail="获取个人资料失败")

@router.put("/profile", response_model=AuthResponse)
async def update_user_profile(
    profile_data: ProfileUpdate,
    current_user: dict = Depends(security_manager.get_current_user)
):
    """更新用户个人资料"""
    try:
        user_id = current_user["id"]
        
        # 如果要更新邮箱，检查是否已被其他用户使用
        if profile_data.email:
            existing_email = await db_manager.get_user_by_email(profile_data.email)
            if existing_email and existing_email["id"] != user_id:
                raise HTTPException(
                    status_code=400,
                    detail="邮箱已被其他用户使用"
                )
        
        # 更新用户信息
        success = await db_manager.update_user_profile(
            user_id=user_id,
            full_name=profile_data.full_name,
            email=profile_data.email,
            preferences=profile_data.preferences,
            accessibility_settings=profile_data.accessibility_settings
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="更新个人资料失败")
        
        logger.info(f"用户个人资料更新成功: {current_user['username']}")
        
        return AuthResponse(
            success=True,
            message="个人资料更新成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户个人资料失败: {e}")
        raise HTTPException(status_code=500, detail="更新个人资料失败")

@router.post("/change-password", response_model=AuthResponse)
async def change_password(
    password_data: PasswordChange,
    current_user: dict = Depends(security_manager.get_current_user)
):
    """修改密码"""
    try:
        user_id = current_user["id"]

        # 验证当前密码
        user = await security_manager.authenticate_user(
            current_user["username"],
            password_data.current_password
        )

        if not user:
            raise HTTPException(
                status_code=400,
                detail="当前密码错误"
            )

        # 更新密码
        success = await db_manager.update_user_password(
            user_id=user_id,
            new_password=password_data.new_password
        )

        if not success:
            raise HTTPException(status_code=500, detail="密码修改失败")

        logger.info(f"用户密码修改成功: {current_user['username']}")

        return AuthResponse(
            success=True,
            message="密码修改成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"修改密码失败: {e}")
        raise HTTPException(status_code=500, detail="密码修改失败")

@router.post("/refresh-token", response_model=AuthResponse)
async def refresh_access_token(refresh_token: str):
    """刷新访问令牌"""
    try:
        # 使用安全管理器的刷新令牌功能
        new_token_data = await security_manager.refresh_access_token(refresh_token)

        return AuthResponse(
            success=True,
            message="令牌刷新成功",
            data=new_token_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刷新令牌失败: {e}")
        raise HTTPException(status_code=500, detail="令牌刷新失败")

@router.get("/verify-token", response_model=AuthResponse)
async def verify_token(current_user: dict = Depends(security_manager.get_current_user)):
    """验证令牌有效性"""
    try:
        return AuthResponse(
            success=True,
            message="令牌有效",
            data={
                "user_id": current_user["id"],
                "username": current_user["username"],
                "is_active": current_user["is_active"]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证令牌失败: {e}")
        raise HTTPException(status_code=500, detail="令牌验证失败")

@router.get("/sessions", response_model=AuthResponse)
async def get_user_sessions(current_user: dict = Depends(security_manager.get_current_user)):
    """获取用户活跃会话"""
    try:
        user_id = current_user["id"]
        sessions = await db_manager.get_user_sessions(user_id)

        return AuthResponse(
            success=True,
            message="获取会话列表成功",
            data={"sessions": sessions}
        )

    except Exception as e:
        logger.error(f"获取用户会话失败: {e}")
        raise HTTPException(status_code=500, detail="获取会话列表失败")

@router.delete("/sessions/{session_id}", response_model=AuthResponse)
async def terminate_session(
    session_id: str,
    current_user: dict = Depends(security_manager.get_current_user)
):
    """终止指定会话"""
    try:
        # 验证会话属于当前用户
        session = await db_manager.get_session_by_id(session_id)
        if not session or session["user_id"] != current_user["id"]:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 终止会话
        await security_manager.logout_user(session_id)

        return AuthResponse(
            success=True,
            message="会话已终止"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"终止会话失败: {e}")
        raise HTTPException(status_code=500, detail="终止会话失败")
