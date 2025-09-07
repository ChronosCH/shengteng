"""
服务管理器 - 统一管理所有后端服务
提供服务的生命周期管理、健康检查、依赖注入等功能
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Type, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """服务状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    STOPPING = "stopping"

@dataclass
class ServiceInfo:
    """服务信息"""
    name: str
    instance: Any
    status: ServiceStatus
    start_time: Optional[float] = None
    error_message: Optional[str] = None
    health_check: Optional[Callable] = None
    dependencies: list = None

class ServiceManager:
    """服务管理器"""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self._startup_order: list = []
        self._shutdown_order: list = []
        self._health_check_interval = 30  # 健康检查间隔（秒）
        self._health_check_task: Optional[asyncio.Task] = None
        
    def register_service(
        self, 
        name: str, 
        service_class: Type, 
        dependencies: list = None,
        health_check: Callable = None,
        **kwargs
    ):
        """注册服务"""
        try:
            # 创建服务实例
            instance = service_class(**kwargs)
            
            service_info = ServiceInfo(
                name=name,
                instance=instance,
                status=ServiceStatus.STOPPED,
                dependencies=dependencies or [],
                health_check=health_check
            )
            
            self.services[name] = service_info
            logger.info(f"✅ 服务 {name} 注册成功")
            
        except Exception as e:
            logger.error(f"❌ 服务 {name} 注册失败: {e}")
            raise
    
    def get_service(self, name: str) -> Any:
        """获取服务实例"""
        if name not in self.services:
            raise ValueError(f"服务 {name} 未注册")
        
        service_info = self.services[name]
        if service_info.status != ServiceStatus.RUNNING:
            logger.warning(f"⚠️ 服务 {name} 状态为 {service_info.status.value}")
        
        return service_info.instance
    
    def get_service_status(self, name: str) -> ServiceStatus:
        """获取服务状态"""
        if name not in self.services:
            return ServiceStatus.STOPPED
        return self.services[name].status
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务状态"""
        status_info = {}
        for name, service_info in self.services.items():
            status_info[name] = {
                "status": service_info.status.value,
                "start_time": service_info.start_time,
                "error_message": service_info.error_message,
                "uptime": time.time() - service_info.start_time if service_info.start_time else 0
            }
        return status_info
    
    async def start_service(self, name: str) -> bool:
        """启动单个服务"""
        if name not in self.services:
            logger.error(f"❌ 服务 {name} 未注册")
            return False
        
        service_info = self.services[name]
        
        if service_info.status == ServiceStatus.RUNNING:
            logger.info(f"ℹ️ 服务 {name} 已在运行")
            return True
        
        try:
            logger.info(f"🚀 启动服务 {name}...")
            service_info.status = ServiceStatus.STARTING
            service_info.error_message = None
            
            # 检查依赖服务
            for dep_name in service_info.dependencies:
                if self.get_service_status(dep_name) != ServiceStatus.RUNNING:
                    logger.info(f"📦 启动依赖服务 {dep_name}")
                    if not await self.start_service(dep_name):
                        raise Exception(f"依赖服务 {dep_name} 启动失败")
            
            # 启动服务
            instance = service_info.instance
            if hasattr(instance, 'initialize') and callable(getattr(instance, 'initialize')):
                await instance.initialize()
            elif hasattr(instance, 'start') and callable(getattr(instance, 'start')):
                await instance.start()
            
            service_info.status = ServiceStatus.RUNNING
            service_info.start_time = time.time()
            
            logger.info(f"✅ 服务 {name} 启动成功")
            return True
            
        except Exception as e:
            service_info.status = ServiceStatus.ERROR
            service_info.error_message = str(e)
            logger.error(f"❌ 服务 {name} 启动失败: {e}")
            return False
    
    async def stop_service(self, name: str) -> bool:
        """停止单个服务"""
        if name not in self.services:
            logger.error(f"❌ 服务 {name} 未注册")
            return False
        
        service_info = self.services[name]
        
        if service_info.status == ServiceStatus.STOPPED:
            logger.info(f"ℹ️ 服务 {name} 已停止")
            return True
        
        try:
            logger.info(f"🛑 停止服务 {name}...")
            service_info.status = ServiceStatus.STOPPING
            
            # 停止服务
            instance = service_info.instance
            if hasattr(instance, 'close') and callable(getattr(instance, 'close')):
                await instance.close()
            elif hasattr(instance, 'stop') and callable(getattr(instance, 'stop')):
                await instance.stop()
            
            service_info.status = ServiceStatus.STOPPED
            service_info.start_time = None
            service_info.error_message = None
            
            logger.info(f"✅ 服务 {name} 停止成功")
            return True
            
        except Exception as e:
            service_info.status = ServiceStatus.ERROR
            service_info.error_message = str(e)
            logger.error(f"❌ 服务 {name} 停止失败: {e}")
            return False
    
    async def start_all_services(self) -> bool:
        """启动所有服务"""
        logger.info("🚀 启动所有服务...")
        
        # 计算启动顺序（基于依赖关系）
        startup_order = self._calculate_startup_order()
        
        success = True
        for service_name in startup_order:
            if not await self.start_service(service_name):
                success = False
                logger.error(f"❌ 服务启动序列中断，{service_name} 启动失败")
                break
        
        if success:
            logger.info("✅ 所有服务启动完成")
            # 启动健康检查
            await self._start_health_check()
        else:
            logger.error("❌ 部分服务启动失败")
        
        return success
    
    async def stop_all_services(self) -> bool:
        """停止所有服务"""
        logger.info("🛑 停止所有服务...")
        
        # 停止健康检查
        await self._stop_health_check()
        
        # 计算停止顺序（启动顺序的逆序）
        shutdown_order = self._calculate_startup_order()
        shutdown_order.reverse()
        
        success = True
        for service_name in shutdown_order:
            if not await self.stop_service(service_name):
                success = False
                # 继续停止其他服务，不中断
        
        if success:
            logger.info("✅ 所有服务停止完成")
        else:
            logger.warning("⚠️ 部分服务停止时出现问题")
        
        return success
    
    def _calculate_startup_order(self) -> list:
        """计算服务启动顺序（拓扑排序）"""
        # 简单的拓扑排序实现
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise Exception(f"检测到循环依赖: {service_name}")
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            if service_name in self.services:
                for dep in self.services[service_name].dependencies:
                    visit(dep)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        for service_name in self.services:
            if service_name not in visited:
                visit(service_name)
        
        return order
    
    async def _start_health_check(self):
        """启动健康检查任务"""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("✅ 健康检查任务已启动")
    
    async def _stop_health_check(self):
        """停止健康检查任务"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ 健康检查任务已停止")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 健康检查出错: {e}")
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        for name, service_info in self.services.items():
            if service_info.status == ServiceStatus.RUNNING and service_info.health_check:
                try:
                    is_healthy = await service_info.health_check(service_info.instance)
                    if not is_healthy:
                        logger.warning(f"⚠️ 服务 {name} 健康检查失败")
                        service_info.status = ServiceStatus.ERROR
                        service_info.error_message = "健康检查失败"
                except Exception as e:
                    logger.error(f"❌ 服务 {name} 健康检查异常: {e}")
                    service_info.status = ServiceStatus.ERROR
                    service_info.error_message = f"健康检查异常: {e}"

# 健康检查函数示例
async def default_health_check(service_instance) -> bool:
    """默认健康检查函数"""
    try:
        if hasattr(service_instance, 'health_check'):
            return await service_instance.health_check()
        elif hasattr(service_instance, 'is_healthy'):
            return service_instance.is_healthy()
        else:
            # 如果没有健康检查方法，认为服务健康
            return True
    except Exception:
        return False

# 全局服务管理器实例
service_manager = ServiceManager()
