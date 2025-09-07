"""
API响应模型 - 标准化API响应格式
提供统一的响应结构和错误处理
"""

from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

T = TypeVar('T')

class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ErrorCode(str, Enum):
    """错误代码枚举"""
    # 通用错误
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    
    # 服务错误
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_ERROR = "SERVICE_ERROR"
    
    # 文件错误
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    
    # 识别错误
    RECOGNITION_FAILED = "RECOGNITION_FAILED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INVALID_INPUT = "INVALID_INPUT"
    
    # 学习训练错误
    TRAINING_FAILED = "TRAINING_FAILED"
    COURSE_NOT_FOUND = "COURSE_NOT_FOUND"
    PROGRESS_ERROR = "PROGRESS_ERROR"

class BaseResponse(BaseModel, Generic[T]):
    """基础响应模型"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: str = "操作成功"
    data: Optional[T] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

class ErrorDetail(BaseModel):
    """错误详情"""
    code: ErrorCode
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """错误响应模型"""
    status: ResponseStatus = ResponseStatus.ERROR
    message: str
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

class PaginationInfo(BaseModel):
    """分页信息"""
    page: int = 1
    page_size: int = 20
    total: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_prev: bool = False

class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应模型"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: str = "查询成功"
    data: List[T] = []
    pagination: PaginationInfo
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

# 手语识别相关响应模型
class RecognitionResult(BaseModel):
    """识别结果"""
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    gloss_sequence: List[str] = []
    processing_time: float = 0.0
    video_info: Optional[Dict[str, Any]] = None

class RecognitionResponse(BaseResponse[RecognitionResult]):
    """识别响应"""
    pass

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskInfo(BaseModel):
    """任务信息"""
    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class TaskResponse(BaseResponse[TaskInfo]):
    """任务响应"""
    pass

# 学习训练相关响应模型
class CourseInfo(BaseModel):
    """课程信息"""
    id: str
    title: str
    description: str
    difficulty: str
    duration: int  # 分钟
    lessons_count: int
    completed: bool = False
    progress: float = Field(ge=0.0, le=1.0, default=0.0)

class LessonInfo(BaseModel):
    """课程信息"""
    id: str
    title: str
    content: str
    video_url: Optional[str] = None
    exercises: List[Dict[str, Any]] = []
    completed: bool = False

class UserProgress(BaseModel):
    """用户进度"""
    user_id: str
    course_id: str
    lesson_id: Optional[str] = None
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    score: Optional[float] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)

class Achievement(BaseModel):
    """成就"""
    id: str
    title: str
    description: str
    icon: str
    earned_at: datetime
    points: int = 0

# 系统状态相关响应模型
class ServiceStatus(BaseModel):
    """服务状态"""
    name: str
    status: str
    uptime: float = 0.0
    error_message: Optional[str] = None

class SystemHealth(BaseModel):
    """系统健康状态"""
    overall_status: str
    services: List[ServiceStatus]
    performance: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)

class SystemHealthResponse(BaseResponse[SystemHealth]):
    """系统健康响应"""
    pass

# 文件上传相关响应模型
class FileInfo(BaseModel):
    """文件信息"""
    filename: str
    file_path: str
    file_size: int
    content_type: str
    upload_time: datetime = Field(default_factory=datetime.now)
    file_id: Optional[str] = None

class FileUploadResponse(BaseResponse[FileInfo]):
    """文件上传响应"""
    pass

# WebSocket消息模型
class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    type: str
    data: Any
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: Optional[str] = None

class WebSocketResponse(BaseModel):
    """WebSocket响应"""
    type: str
    status: ResponseStatus
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# 响应构建器
class ResponseBuilder:
    """响应构建器"""
    
    @staticmethod
    def success(data: Any = None, message: str = "操作成功", request_id: str = None) -> BaseResponse:
        """构建成功响应"""
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data,
            request_id=request_id
        )
    
    @staticmethod
    def error(
        error_code: ErrorCode,
        message: str,
        field: str = None,
        details: Dict[str, Any] = None,
        request_id: str = None
    ) -> ErrorResponse:
        """构建错误响应"""
        return ErrorResponse(
            message=message,
            error=ErrorDetail(
                code=error_code,
                message=message,
                field=field,
                details=details
            ),
            request_id=request_id
        )
    
    @staticmethod
    def paginated(
        data: List[Any],
        page: int,
        page_size: int,
        total: int,
        message: str = "查询成功",
        request_id: str = None
    ) -> PaginatedResponse:
        """构建分页响应"""
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            message=message,
            data=data,
            pagination=PaginationInfo(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_prev=page > 1
            ),
            request_id=request_id
        )
    
    @staticmethod
    def task_created(task_id: str, message: str = "任务已创建") -> TaskResponse:
        """构建任务创建响应"""
        return TaskResponse(
            data=TaskInfo(
                task_id=task_id,
                status=TaskStatus.PENDING,
                message=message
            )
        )
    
    @staticmethod
    def websocket_message(msg_type: str, data: Any = None, status: ResponseStatus = ResponseStatus.SUCCESS) -> WebSocketResponse:
        """构建WebSocket响应"""
        return WebSocketResponse(
            type=msg_type,
            status=status,
            message="",
            data=data
        )
