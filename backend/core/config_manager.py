"""
配置管理器 - 统一管理应用配置
支持环境变量、配置文件、默认值等多种配置源
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """数据库配置"""
    url: str = "sqlite:///./sign_language_learning.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = True

@dataclass
class CSLRConfig:
    """CSLR服务配置"""
    model_path: str = "models/cslr_model.ckpt"
    vocab_path: str = "data/vocab.json"
    confidence_threshold: float = 0.5
    max_sequence_length: int = 100
    batch_size: int = 1
    device: str = "cpu"

@dataclass
class MediaPipeConfig:
    """MediaPipe配置"""
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False

@dataclass
class FileConfig:
    """文件管理配置"""
    upload_dir: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = field(default_factory=lambda: ['.mp4', '.avi', '.mov', '.jpg', '.png'])
    cleanup_interval: int = 3600  # 1小时

@dataclass
class SecurityConfig:
    """安全配置"""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class PerformanceConfig:
    """性能配置"""
    max_workers: int = 4
    request_timeout: int = 30
    websocket_timeout: int = 300
    cache_ttl: int = 3600
    enable_compression: bool = True

@dataclass
class AppConfig:
    """应用主配置"""
    # 基本配置
    app_name: str = "手语学习训练系统"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 子配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    cslr: CSLRConfig = field(default_factory=CSLRConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    file: FileConfig = field(default_factory=FileConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = AppConfig()
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        # 1. 加载默认配置（已在AppConfig中定义）
        
        # 2. 加载配置文件
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_file(self.config_file)
        
        # 3. 加载环境变量
        self._load_from_env()
        
        # 4. 验证配置
        self._validate_config()
        
        logger.info(f"✅ 配置加载完成 - 环境: {self.config.environment.value}")
    
    def _load_from_file(self, file_path: str):
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    file_config = json.load(f)
                else:
                    # 支持其他格式（如YAML）
                    raise ValueError(f"不支持的配置文件格式: {file_path}")
            
            self._update_config_from_dict(file_config)
            logger.info(f"✅ 从文件加载配置: {file_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ 配置文件加载失败: {e}")
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        env_mappings = {
            # 基本配置
            'APP_NAME': ('app_name', str),
            'APP_VERSION': ('version', str),
            'APP_ENVIRONMENT': ('environment', lambda x: Environment(x)),
            'APP_DEBUG': ('debug', lambda x: x.lower() == 'true'),
            'APP_HOST': ('host', str),
            'APP_PORT': ('port', int),
            
            # 数据库配置
            'DATABASE_URL': ('database.url', str),
            'DATABASE_ECHO': ('database.echo', lambda x: x.lower() == 'true'),
            
            # Redis配置
            'REDIS_HOST': ('redis.host', str),
            'REDIS_PORT': ('redis.port', int),
            'REDIS_DB': ('redis.db', int),
            'REDIS_PASSWORD': ('redis.password', str),
            
            # CSLR配置
            'CSLR_MODEL_PATH': ('cslr.model_path', str),
            'CSLR_VOCAB_PATH': ('cslr.vocab_path', str),
            'CSLR_CONFIDENCE_THRESHOLD': ('cslr.confidence_threshold', float),
            
            # 文件配置
            'FILE_UPLOAD_DIR': ('file.upload_dir', str),
            'FILE_MAX_SIZE': ('file.max_file_size', int),
            
            # 安全配置
            'SECRET_KEY': ('security.secret_key', str),
            'ACCESS_TOKEN_EXPIRE_MINUTES': ('security.access_token_expire_minutes', int),
            
            # 日志配置
            'LOG_LEVEL': ('logging.level', str),
            'LOG_FILE': ('logging.file_path', str),
        }
        
        for env_key, (config_path, converter) in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    self._set_nested_config(config_path, converted_value)
                    logger.debug(f"从环境变量设置配置: {config_path} = {converted_value}")
                except Exception as e:
                    logger.warning(f"⚠️ 环境变量 {env_key} 转换失败: {e}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        def update_nested(obj, data):
            for key, value in data.items():
                if hasattr(obj, key):
                    attr = getattr(obj, key)
                    if isinstance(value, dict) and hasattr(attr, '__dict__'):
                        update_nested(attr, value)
                    else:
                        setattr(obj, key, value)
        
        update_nested(self.config, config_dict)
    
    def _set_nested_config(self, path: str, value: Any):
        """设置嵌套配置值"""
        parts = path.split('.')
        obj = self.config
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        setattr(obj, parts[-1], value)
    
    def _validate_config(self):
        """验证配置"""
        # 验证必要的目录
        upload_dir = Path(self.config.file.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证模型文件路径
        model_path = Path(self.config.cslr.model_path)
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"⚠️ 模型目录不存在，已创建: {model_path.parent}")
        
        # 验证词汇表路径
        vocab_path = Path(self.config.cslr.vocab_path)
        if not vocab_path.parent.exists():
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"⚠️ 词汇表目录不存在，已创建: {vocab_path.parent}")
        
        # 验证端口范围
        if not (1 <= self.config.port <= 65535):
            raise ValueError(f"无效的端口号: {self.config.port}")
        
        # 验证置信度阈值
        if not (0.0 <= self.config.cslr.confidence_threshold <= 1.0):
            raise ValueError(f"无效的置信度阈值: {self.config.cslr.confidence_threshold}")
    
    def get_config(self) -> AppConfig:
        """获取配置对象"""
        return self.config
    
    def get(self, path: str, default: Any = None) -> Any:
        """获取配置值"""
        try:
            parts = path.split('.')
            obj = self.config
            
            for part in parts:
                obj = getattr(obj, part)
            
            return obj
        except AttributeError:
            return default
    
    def set(self, path: str, value: Any):
        """设置配置值"""
        self._set_nested_config(path, value)
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        try:
            config_dict = self._config_to_dict(self.config)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 配置已保存到: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ 配置保存失败: {e}")
            raise
    
    def _config_to_dict(self, obj) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._config_to_dict(value)
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
            return result
        else:
            return obj

# 全局配置管理器实例
config_manager = ConfigManager()

# 便捷访问
def get_config() -> AppConfig:
    """获取应用配置"""
    return config_manager.get_config()

def get_setting(path: str, default: Any = None) -> Any:
    """获取配置项"""
    return config_manager.get(path, default)
