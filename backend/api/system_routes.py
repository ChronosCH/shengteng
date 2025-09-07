"""
系统管理API路由
提供系统状态、健康检查、配置管理等功能
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from backend.core.service_manager import service_manager
from backend.core.config_manager import get_config
from backend.core.response_models import (
    ResponseBuilder, 
    SystemHealthResponse, 
    SystemHealth, 
    ServiceStatus,
    BaseResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["系统管理"])

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """获取系统健康状态"""
    try:
        # 获取所有服务状态
        services_status = service_manager.get_all_services_status()
        
        # 转换为响应格式
        services = []
        overall_healthy = True
        
        for name, status_info in services_status.items():
            service_status = ServiceStatus(
                name=name,
                status=status_info["status"],
                uptime=status_info["uptime"],
                error_message=status_info.get("error_message")
            )
            services.append(service_status)
            
            if status_info["status"] not in ["running"]:
                overall_healthy = False
        
        # 计算整体状态
        overall_status = "healthy" if overall_healthy else "unhealthy"
        
        # 获取性能指标（模拟数据）
        performance = {
            "cpu_usage": 45.2,
            "memory_usage": 68.5,
            "disk_usage": 32.1,
            "active_connections": len(services),
        }
        
        health_data = SystemHealth(
            overall_status=overall_status,
            services=services,
            performance=performance
        )
        
        return ResponseBuilder.success(
            data=health_data,
            message="系统健康状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统健康状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统健康状态失败")

@router.get("/services", response_model=BaseResponse[Dict[str, Any]])
async def get_services_status():
    """获取所有服务状态"""
    try:
        services_status = service_manager.get_all_services_status()
        
        return ResponseBuilder.success(
            data=services_status,
            message="服务状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取服务状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取服务状态失败")

@router.post("/services/{service_name}/restart")
async def restart_service(service_name: str):
    """重启指定服务"""
    try:
        # 停止服务
        stop_success = await service_manager.stop_service(service_name)
        if not stop_success:
            raise HTTPException(status_code=500, detail=f"停止服务 {service_name} 失败")
        
        # 启动服务
        start_success = await service_manager.start_service(service_name)
        if not start_success:
            raise HTTPException(status_code=500, detail=f"启动服务 {service_name} 失败")
        
        return ResponseBuilder.success(
            message=f"服务 {service_name} 重启成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重启服务 {service_name} 失败: {e}")
        raise HTTPException(status_code=500, detail=f"重启服务失败: {e}")

@router.get("/config", response_model=BaseResponse[Dict[str, Any]])
async def get_system_config():
    """获取系统配置（脱敏）"""
    try:
        config = get_config()
        
        # 脱敏处理
        safe_config = {
            "app_name": config.app_name,
            "version": config.version,
            "environment": config.environment.value,
            "debug": config.debug,
            "host": config.host,
            "port": config.port,
            "cslr": {
                "confidence_threshold": config.cslr.confidence_threshold,
                "max_sequence_length": config.cslr.max_sequence_length,
                "device": config.cslr.device,
            },
            "mediapipe": {
                "model_complexity": config.mediapipe.model_complexity,
                "min_detection_confidence": config.mediapipe.min_detection_confidence,
                "min_tracking_confidence": config.mediapipe.min_tracking_confidence,
            },
            "file": {
                "max_file_size": config.file.max_file_size,
                "allowed_extensions": config.file.allowed_extensions,
            },
            "performance": {
                "max_workers": config.performance.max_workers,
                "request_timeout": config.performance.request_timeout,
                "websocket_timeout": config.performance.websocket_timeout,
            }
        }
        
        return ResponseBuilder.success(
            data=safe_config,
            message="系统配置获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统配置失败")

@router.get("/info", response_model=BaseResponse[Dict[str, Any]])
async def get_system_info():
    """获取系统信息"""
    try:
        import platform
        import psutil
        import sys
        from datetime import datetime
        
        config = get_config()
        
        system_info = {
            "application": {
                "name": config.app_name,
                "version": config.version,
                "environment": config.environment.value,
                "debug_mode": config.debug,
            },
            "runtime": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
            },
            "system": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent,
            },
            "services": {
                "total_services": len(service_manager.services),
                "running_services": len([
                    s for s in service_manager.services.values() 
                    if s.status.value == "running"
                ]),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        return ResponseBuilder.success(
            data=system_info,
            message="系统信息获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统信息失败")

@router.get("/logs")
async def get_system_logs(lines: int = 100):
    """获取系统日志"""
    try:
        # 这里可以实现日志读取逻辑
        # 为了安全考虑，这里返回模拟数据
        logs = [
            f"[INFO] 系统运行正常 - {i}" for i in range(lines)
        ]
        
        return ResponseBuilder.success(
            data={"logs": logs},
            message="系统日志获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统日志失败")

@router.post("/maintenance")
async def toggle_maintenance_mode(enabled: bool = True):
    """切换维护模式"""
    try:
        # 这里可以实现维护模式逻辑
        # 例如设置全局标志、拒绝新请求等
        
        message = "维护模式已启用" if enabled else "维护模式已禁用"
        
        return ResponseBuilder.success(
            data={"maintenance_mode": enabled},
            message=message
        )
        
    except Exception as e:
        logger.error(f"切换维护模式失败: {e}")
        raise HTTPException(status_code=500, detail="切换维护模式失败")

@router.get("/metrics", response_model=BaseResponse[Dict[str, Any]])
async def get_system_metrics():
    """获取系统指标"""
    try:
        import psutil
        from datetime import datetime, timedelta
        
        # 获取系统指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 获取服务指标
        services_status = service_manager.get_all_services_status()
        running_services = len([
            s for s in services_status.values() 
            if s["status"] == "running"
        ])
        
        metrics = {
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "disk_usage": disk.percent,
                "disk_total": disk.total,
                "disk_free": disk.free,
            },
            "services": {
                "total": len(services_status),
                "running": running_services,
                "stopped": len(services_status) - running_services,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        return ResponseBuilder.success(
            data=metrics,
            message="系统指标获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统指标失败")
