"""
日志配置模块
"""

import logging
import sys
from typing import Optional

import structlog
from .config import settings


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """设置结构化日志"""
    
    # 配置标准库日志
    logging.basicConfig(
        format=settings.LOG_FORMAT,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        stream=sys.stdout,
    )
    
    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not settings.DEBUG 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)
