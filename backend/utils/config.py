"""
配置管理模块 - 优化版本
支持动态配置、环境验证、性能优化等功能
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import validator, Field
from pydantic_settings import BaseSettings

# 使用标准库logging，避免循环导入
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """应用配置 - 增强版"""
    
    # 基本设置
    APP_NAME: str = "SignAvatar Web"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = Field(default=True)
    
    # 服务器设置
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=16)
    
    # CORS设置
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    
    # MediaPipe设置
    MEDIAPIPE_MODEL_COMPLEXITY: int = Field(default=1, ge=0, le=2)
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = Field(default=0.5, ge=0.1, le=1.0)
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = Field(default=0.5, ge=0.1, le=1.0)
    MEDIAPIPE_STATIC_IMAGE_MODE: bool = False
    MEDIAPIPE_MAX_NUM_HANDS: int = Field(default=2, ge=1, le=4)
    
    # CSLR模型设置
    CSLR_MODEL_PATH: str = "models/cslr_transformer.mindir"
    CSLR_VOCAB_PATH: str = "backend/models/vocab.json"
    CSLR_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.1, le=1.0)
    CSLR_MAX_SEQUENCE_LENGTH: int = Field(default=100, ge=10, le=500)
    CSLR_ENABLE_CACHE: bool = True
    CSLR_CACHE_SIZE: int = Field(default=1000, ge=100, le=10000)

    # Diffusion SLP模型设置
    DIFFUSION_MODEL_PATH: str = "models/diffusion_slp.mindir"
    DIFFUSION_TEXT_ENCODER_PATH: str = "models/text_encoder.mindir"
    DIFFUSION_MAX_SEQUENCE_LENGTH: int = Field(default=200, ge=50, le=1000)
    DIFFUSION_DEFAULT_STEPS: int = Field(default=20, ge=5, le=100)
    DIFFUSION_GUIDANCE_SCALE: float = Field(default=7.5, ge=1.0, le=20.0)
    DIFFUSION_ENABLE_CACHE: bool = True
    DIFFUSION_CACHE_SIZE: int = Field(default=500, ge=50, le=5000)
    USE_ASCEND: bool = False

    # 隐私保护设置
    DIFFUSION_ANONYMIZATION_MODEL_PATH: str = "models/diffusion_anonymization.mindir"
    PRIVACY_DEFAULT_LEVEL: str = Field(default="medium", pattern="^(low|medium|high)$")
    PRIVACY_CACHE_SIZE: int = Field(default=50, ge=10, le=500)
    PRIVACY_MAX_FILE_SIZE: int = Field(default=100, ge=1, le=1000)  # MB
    PRIVACY_ENABLE_DIFFERENTIAL: bool = True
    PRIVACY_NOISE_MULTIPLIER: float = Field(default=1.0, ge=0.1, le=10.0)

    # 多模态传感器设置
    MULTIMODAL_FUSION_MODEL_PATH: str = "models/multimodal_fusion.mindir"
    EMG_DEVICE_PORT: str = "/dev/ttyUSB0"
    IMU_DEVICE_PORT: str = "/dev/ttyUSB1"
    EMG_CHANNELS: int = Field(default=16, ge=8, le=64)
    EMG_SAMPLING_RATE: int = Field(default=1000, ge=200, le=5000)  # Hz
    IMU_SAMPLING_RATE: int = Field(default=100, ge=10, le=1000)   # Hz
    SENSOR_BUFFER_SIZE: int = Field(default=1000, ge=100, le=10000)
    SENSOR_TIMEOUT: float = Field(default=1.0, ge=0.1, le=10.0)  # 秒

    # 触觉反馈设置
    HAPTIC_DEVICE_PORT: str = "/dev/ttyUSB2"
    BRAILLE_DEVICE_PORT: str = "/dev/ttyUSB3"
    HAPTIC_ACTUATORS: int = Field(default=16, ge=8, le=64)
    BRAILLE_CELLS: int = Field(default=8, ge=4, le=40)
    HAPTIC_MESSAGE_QUEUE_SIZE: int = Field(default=100, ge=10, le=1000)
    HAPTIC_DEVICE_TIMEOUT: float = Field(default=2.0, ge=0.5, le=10.0)

    # 联邦学习设置
    FEDERATED_MODEL_PATH: str = "models/federated_slr.mindir"
    FL_DIFFERENTIAL_PRIVACY: bool = True
    FL_NOISE_MULTIPLIER: float = Field(default=1.0, ge=0.1, le=10.0)
    FL_MAX_GRAD_NORM: float = Field(default=1.0, ge=0.1, le=10.0)
    FL_PRIVACY_BUDGET: float = Field(default=1.0, ge=0.1, le=10.0)
    FL_EXPLANATION_CACHE_SIZE: int = Field(default=100, ge=10, le=1000)
    FL_MIN_CLIENTS: int = Field(default=2, ge=1, le=100)
    FL_MAX_ROUNDS: int = Field(default=100, ge=1, le=1000)
    
    # 性能设置
    MAX_WEBSOCKET_CONNECTIONS: int = Field(default=100, ge=10, le=1000)
    FRAME_BUFFER_SIZE: int = Field(default=30, ge=5, le=100)
    INFERENCE_BATCH_SIZE: int = Field(default=1, ge=1, le=32)
    REQUEST_TIMEOUT: float = Field(default=30.0, ge=5.0, le=300.0)
    WEBSOCKET_PING_INTERVAL: int = Field(default=20, ge=5, le=60)
    
    # 缓存设置
    CACHE_DEFAULT_TTL: int = Field(default=3600, ge=60, le=86400)  # 1小时
    CACHE_MAX_MEMORY: int = Field(default=256, ge=64, le=2048)  # MB
    CACHE_ENABLE_COMPRESSION: bool = True
    
    # 日志设置
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE: int = Field(default=10, ge=1, le=100)  # MB
    LOG_FILE_BACKUP_COUNT: int = Field(default=5, ge=1, le=20)
    
    # 监控设置
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = Field(default=9090, ge=1024, le=65535)
    HEALTH_CHECK_INTERVAL: int = Field(default=30, ge=5, le=300)
    PERFORMANCE_METRICS_RETENTION: int = Field(default=7, ge=1, le=30)  # 天
    
    # 安全设置
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=5, le=1440)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=30)
    RATE_LIMIT_REQUESTS: int = Field(default=100, ge=10, le=10000)
    RATE_LIMIT_WINDOW: int = Field(default=60, ge=10, le=3600)  # 秒
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    
    # 数据库设置
    DATABASE_URL: str = "sqlite:///./data/signavatar.db"
    DATABASE_POOL_SIZE: int = Field(default=5, ge=1, le=20)
    DATABASE_MAX_OVERFLOW: int = Field(default=10, ge=0, le=50)
    DATABASE_POOL_TIMEOUT: int = Field(default=30, ge=5, le=300)
    
    # Redis设置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_PASSWORD: Optional[str] = None
    REDIS_POOL_SIZE: int = Field(default=10, ge=1, le=50)
    REDIS_SOCKET_TIMEOUT: float = Field(default=5.0, ge=1.0, le=30.0)
    
    # 文件上传设置
    MAX_UPLOAD_SIZE: int = Field(default=100, ge=1, le=1000)  # MB
    UPLOAD_DIR: str = "uploads"
    TEMP_DIR: str = "temp"
    ALLOWED_FILE_TYPES: List[str] = ["image", "video", "audio", "document", "data"]
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".flac", ".ogg"]
    FILE_CLEANUP_INTERVAL: int = Field(default=24, ge=1, le=168)  # 小时
    TEMP_FILE_RETENTION: int = Field(default=24, ge=1, le=72)  # 小时
    
    # AI模型通用设置
    MODEL_INFERENCE_DEVICE: str = Field(default="cpu", pattern="^(cpu|gpu|ascend)$")
    MODEL_PRECISION: str = Field(default="fp32", pattern="^(fp16|fp32|int8)$")
    MODEL_BATCH_SIZE: int = Field(default=1, ge=1, le=32)
    MODEL_CACHE_ENABLED: bool = True
    MODEL_WARMUP_ITERATIONS: int = Field(default=3, ge=1, le=10)
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v):
        if v == "your-secret-key-here-change-in-production":
            logger.warning("使用默认密钥，生产环境中请更改！")
        if len(v) < 32:
            raise ValueError("密钥长度必须至少32个字符")
        return v
    
    @validator('ALLOWED_ORIGINS')
    def validate_origins(cls, v):
        # 在生产环境中验证CORS设置
        if any("*" in origin for origin in v):
            logger.warning("CORS配置包含通配符，可能存在安全风险")
        return v
    
    @validator('CSLR_MODEL_PATH', 'DIFFUSION_MODEL_PATH', 'FEDERATED_MODEL_PATH')
    def validate_model_paths(cls, v):
        if not Path(v).parent.exists():
            logger.warning(f"模型目录不存在: {Path(v).parent}")
            # 创建模型目录
            Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('UPLOAD_DIR', 'TEMP_DIR')
    def validate_directories(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.ENVIRONMENT == "development"
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """数据库配置"""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "pool_timeout": self.DATABASE_POOL_TIMEOUT,
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Redis配置"""
        config = {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DB,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": self.REDIS_SOCKET_TIMEOUT,
            "max_connections": self.REDIS_POOL_SIZE,
        }
        if self.REDIS_PASSWORD:
            config["password"] = self.REDIS_PASSWORD
        return config
    
    @property
    def model_device_config(self) -> Dict[str, Any]:
        """模型设备配置"""
        return {
            "device": self.MODEL_INFERENCE_DEVICE,
            "precision": self.MODEL_PRECISION,
            "batch_size": self.MODEL_BATCH_SIZE,
            "cache_enabled": self.MODEL_CACHE_ENABLED,
            "warmup_iterations": self.MODEL_WARMUP_ITERATIONS,
        }
    
    def get_file_extensions(self, file_type: str) -> List[str]:
        """获取指定文件类型的扩展名列表"""
        extension_map = {
            "image": self.ALLOWED_IMAGE_EXTENSIONS,
            "video": self.ALLOWED_VIDEO_EXTENSIONS,
            "audio": self.ALLOWED_AUDIO_EXTENSIONS,
        }
        return extension_map.get(file_type, [])
    
    def validate_file_size(self, file_size: int) -> bool:
        """验证文件大小"""
        max_size_bytes = self.MAX_UPLOAD_SIZE * 1024 * 1024
        return file_size <= max_size_bytes
    
    def get_sensor_config(self) -> Dict[str, Any]:
        """获取传感器配置"""
        return {
            "emg": {
                "port": self.EMG_DEVICE_PORT,
                "channels": self.EMG_CHANNELS,
                "sampling_rate": self.EMG_SAMPLING_RATE,
                "buffer_size": self.SENSOR_BUFFER_SIZE,
                "timeout": self.SENSOR_TIMEOUT,
            },
            "imu": {
                "port": self.IMU_DEVICE_PORT,
                "sampling_rate": self.IMU_SAMPLING_RATE,
                "buffer_size": self.SENSOR_BUFFER_SIZE,
                "timeout": self.SENSOR_TIMEOUT,
            }
        }
    
    def get_haptic_config(self) -> Dict[str, Any]:
        """获取触觉设备配置"""
        return {
            "haptic_device": {
                "port": self.HAPTIC_DEVICE_PORT,
                "actuators": self.HAPTIC_ACTUATORS,
                "timeout": self.HAPTIC_DEVICE_TIMEOUT,
            },
            "braille_device": {
                "port": self.BRAILLE_DEVICE_PORT,
                "cells": self.BRAILLE_CELLS,
                "timeout": self.HAPTIC_DEVICE_TIMEOUT,
            },
            "queue_size": self.HAPTIC_MESSAGE_QUEUE_SIZE,
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"配置已更新: {key} = {value}")
            else:
                logger.warning(f"未知配置项: {key}")
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置（脱敏）"""
        config = self.dict()
        
        # 脱敏敏感信息
        sensitive_keys = ["SECRET_KEY", "REDIS_PASSWORD", "DATABASE_URL"]
        for key in sensitive_keys:
            if key in config and config[key]:
                config[key] = "***"
        
        return config
    
    def validate_environment(self) -> List[str]:
        """验证环境配置"""
        warnings = []
        
        # 生产环境检查
        if self.is_production:
            if self.DEBUG:
                warnings.append("生产环境不应启用DEBUG模式")
            
            if self.SECRET_KEY == "your-secret-key-here-change-in-production":
                warnings.append("生产环境必须更改默认密钥")
            
            if not self.SESSION_COOKIE_SECURE:
                warnings.append("生产环境应启用安全Cookie")
        
        # 模型文件检查
        model_paths = [
            self.CSLR_MODEL_PATH,
            self.DIFFUSION_MODEL_PATH,
            self.FEDERATED_MODEL_PATH,
        ]
        
        for path in model_paths:
            if not Path(path).exists():
                warnings.append(f"模型文件不存在: {path}")
        
        # 设备端口检查 (Linux环境)
        if os.name == 'posix':
            device_ports = [
                self.EMG_DEVICE_PORT,
                self.IMU_DEVICE_PORT,
                self.HAPTIC_DEVICE_PORT,
                self.BRAILLE_DEVICE_PORT,
            ]
            
            for port in device_ports:
                if port.startswith('/dev/') and not Path(port).exists():
                    warnings.append(f"设备端口不存在: {port}")
        
        return warnings
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True
        extra = "ignore"  # 忽略未知的环境变量


# 配置管理器
class ConfigManager:
    """配置管理器 - 支持动态配置和热重载"""
    
    def __init__(self):
        self._settings = Settings()
        self._config_file_path = Path(".env")
        self._last_modified = None
        self._update_callbacks = []
        
        # 检查配置文件
        self._check_config_file()
        
        logger.info("配置管理器初始化完成")
    
    @property
    def settings(self) -> Settings:
        """获取当前设置"""
        return self._settings
    
    def _check_config_file(self):
        """检查配置文件"""
        if not self._config_file_path.exists():
            logger.warning(f"配置文件不存在: {self._config_file_path}")
            self._create_default_config()
        
        # 验证环境配置
        warnings = self._settings.validate_environment()
        for warning in warnings:
            logger.warning(f"配置警告: {warning}")
    
    def _create_default_config(self):
        """创建默认配置文件"""
        try:
            # 从 .env.example 复制
            example_path = Path(".env.example")
            if example_path.exists():
                self._config_file_path.write_text(example_path.read_text())
                logger.info("已从 .env.example 创建配置文件")
            else:
                # 创建基本配置
                default_config = """# SignAvatar Web 配置文件
DEBUG=true
SECRET_KEY=your-secret-key-here-change-in-production
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development

# 数据库配置
DATABASE_URL=sqlite:///./data/signavatar.db

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# 日志配置
LOG_LEVEL=INFO
"""
                self._config_file_path.write_text(default_config)
                logger.info("已创建默认配置文件")
        
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
    
    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            # 检查文件是否被修改
            if self._config_file_path.exists():
                current_modified = self._config_file_path.stat().st_mtime
                if self._last_modified is None or current_modified > self._last_modified:
                    # 重新创建设置对象
                    old_settings = self._settings
                    self._settings = Settings()
                    self._last_modified = current_modified
                    
                    # 调用更新回调
                    for callback in self._update_callbacks:
                        try:
                            callback(old_settings, self._settings)
                        except Exception as e:
                            logger.error(f"配置更新回调失败: {e}")
                    
                    logger.info("配置已重新加载")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"重新加载配置失败: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """动态更新配置"""
        try:
            # 更新内存中的配置
            old_settings = self._settings
            self._settings.update_from_dict(updates)
            
            # 调用更新回调
            for callback in self._update_callbacks:
                try:
                    callback(old_settings, self._settings)
                except Exception as e:
                    logger.error(f"配置更新回调失败: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False
    
    def add_update_callback(self, callback):
        """添加配置更新回调"""
        self._update_callbacks.append(callback)
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        return self._settings.export_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "app_name": self._settings.APP_NAME,
            "version": self._settings.VERSION,
            "environment": self._settings.ENVIRONMENT,
            "debug": self._settings.DEBUG,
            "host": self._settings.HOST,
            "port": self._settings.PORT,
            "database_type": "sqlite" if "sqlite" in self._settings.DATABASE_URL else "other",
            "redis_enabled": bool(self._settings.REDIS_HOST),
            "metrics_enabled": self._settings.ENABLE_METRICS,
            "models_configured": {
                "cslr": Path(self._settings.CSLR_MODEL_PATH).exists(),
                "diffusion": Path(self._settings.DIFFUSION_MODEL_PATH).exists(),
                "federated": Path(self._settings.FEDERATED_MODEL_PATH).exists(),
            }
        }


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例（单例模式）"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_settings() -> Settings:
    """获取当前设置的便捷函数"""
    return get_config_manager().settings

# 向后兼容的全局设置实例
settings = get_settings()

__all__ = ["Settings", "ConfigManager", "get_config_manager", "get_settings", "settings"]
