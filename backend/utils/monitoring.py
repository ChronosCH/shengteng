"""
性能监控和指标收集模块
提供系统监控、性能分析、指标统计等功能
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
import json

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("警告: prometheus_client 未安装，将使用基础监控")

from utils.logger import setup_logger
from utils.config import settings

logger = setup_logger(__name__)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # 时间窗口配置
        self.window_size = 300  # 5分钟
        self.max_points = 1000
        
        # Prometheus 指标（如果可用）
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        logger.info("指标收集器初始化完成")
    
    def _setup_prometheus_metrics(self):
        """设置Prometheus指标"""
        self.prom_request_counter = Counter(
            'signavatar_requests_total',
            'Total requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.prom_request_duration = Histogram(
            'signavatar_request_duration_seconds',
            'Request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.prom_model_inference_duration = Histogram(
            'signavatar_model_inference_duration_seconds',
            'Model inference duration',
            ['model_name'],
            registry=self.registry
        )
        
        self.prom_active_connections = Gauge(
            'signavatar_active_connections',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        self.prom_cache_hits = Counter(
            'signavatar_cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.prom_cache_misses = Counter(
            'signavatar_cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """记录请求指标"""
        # 内部指标
        metric_key = f"request_{method}_{endpoint}"
        self.metrics[metric_key].append({
            'timestamp': time.time(),
            'duration': duration,
            'status_code': status_code
        })
        self._trim_metrics(metric_key)
        
        # Prometheus 指标
        if PROMETHEUS_AVAILABLE:
            self.prom_request_counter.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
            
            self.prom_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
    
    def record_model_inference(self, model_name: str, duration: float, success: bool = True):
        """记录模型推理指标"""
        metric_key = f"model_inference_{model_name}"
        self.metrics[metric_key].append({
            'timestamp': time.time(),
            'duration': duration,
            'success': success
        })
        self._trim_metrics(metric_key)
        
        if PROMETHEUS_AVAILABLE:
            self.prom_model_inference_duration.labels(
                model_name=model_name
            ).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """记录缓存命中"""
        self.counters[f"cache_hit_{cache_type}"] += 1
        
        if PROMETHEUS_AVAILABLE:
            self.prom_cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """记录缓存未命中"""
        self.counters[f"cache_miss_{cache_type}"] += 1
        
        if PROMETHEUS_AVAILABLE:
            self.prom_cache_misses.labels(cache_type=cache_type).inc()
    
    def set_active_connections(self, count: int):
        """设置活跃连接数"""
        self.gauges["active_connections"] = count
        
        if PROMETHEUS_AVAILABLE:
            self.prom_active_connections.set(count)
    
    def increment_counter(self, name: str, value: int = 1):
        """递增计数器"""
        self.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """设置仪表值"""
        self.gauges[name] = value
    
    def add_histogram_value(self, name: str, value: float):
        """添加直方图值"""
        self.histograms[name].append({
            'timestamp': time.time(),
            'value': value
        })
        self._trim_histogram(name)
    
    def _trim_metrics(self, metric_key: str):
        """修剪指标数据"""
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        self.metrics[metric_key] = [
            m for m in self.metrics[metric_key]
            if m['timestamp'] > cutoff_time
        ]
        
        # 限制最大点数
        if len(self.metrics[metric_key]) > self.max_points:
            self.metrics[metric_key] = self.metrics[metric_key][-self.max_points:]
    
    def _trim_histogram(self, name: str):
        """修剪直方图数据"""
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        self.histograms[name] = [
            h for h in self.histograms[name]
            if h['timestamp'] > cutoff_time
        ]
        
        if len(self.histograms[name]) > self.max_points:
            self.histograms[name] = self.histograms[name][-self.max_points:]
    
    def get_metrics_summary(self) -> Dict:
        """获取指标摘要"""
        summary = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timestamp': time.time()
        }
        
        # 计算时间序列指标的统计信息
        for metric_key, data_points in self.metrics.items():
            if data_points:
                durations = [dp['duration'] for dp in data_points if 'duration' in dp]
                if durations:
                    summary[f"{metric_key}_avg_duration"] = sum(durations) / len(durations)
                    summary[f"{metric_key}_max_duration"] = max(durations)
                    summary[f"{metric_key}_min_duration"] = min(durations)
                    summary[f"{metric_key}_p95_duration"] = self._calculate_percentile(durations, 95)
                    summary[f"{metric_key}_p99_duration"] = self._calculate_percentile(durations, 99)
                
                summary[f"{metric_key}_count"] = len(data_points)
                summary[f"{metric_key}_error_rate"] = self._calculate_error_rate(data_points)
        
        # 直方图统计
        for hist_name, values in self.histograms.items():
            if values:
                vals = [v['value'] for v in values]
                summary[f"{hist_name}_avg"] = sum(vals) / len(vals)
                summary[f"{hist_name}_max"] = max(vals)
                summary[f"{hist_name}_min"] = min(vals)
                summary[f"{hist_name}_p95"] = self._calculate_percentile(vals, 95)
                summary[f"{hist_name}_p99"] = self._calculate_percentile(vals, 99)
        
        return summary
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]
    
    def _calculate_error_rate(self, data_points: List[Dict]) -> float:
        """计算错误率"""
        if not data_points:
            return 0.0
        
        error_count = sum(1 for dp in data_points 
                         if dp.get('status_code', 200) >= 400 or not dp.get('success', True))
        return error_count / len(data_points)
    
    def record_custom_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录自定义指标"""
        metric_data = {
            'timestamp': time.time(),
            'value': value,
            'tags': tags or {}
        }
        
        self.metrics[f"custom_{name}"].append(metric_data)
        self._trim_metrics(f"custom_{name}")
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        current_time = time.time()
        
        # 基础统计
        total_requests = sum(len(points) for key, points in self.metrics.items() 
                           if key.startswith('request_'))
        
        # 计算吞吐量（每秒请求数）
        window_start = current_time - self.window_size
        recent_requests = sum(
            len([p for p in points if p['timestamp'] > window_start])
            for key, points in self.metrics.items() 
            if key.startswith('request_')
        )
        throughput = recent_requests / self.window_size if self.window_size > 0 else 0
        
        # 平均响应时间
        all_durations = []
        for key, points in self.metrics.items():
            if key.startswith('request_'):
                all_durations.extend([p['duration'] for p in points if 'duration' in p])
        
        avg_response_time = sum(all_durations) / len(all_durations) if all_durations else 0
        
        # 错误率
        all_errors = []
        for key, points in self.metrics.items():
            if key.startswith('request_'):
                all_errors.extend([self._calculate_error_rate([p]) for p in points])
        
        error_rate = sum(all_errors) / len(all_errors) if all_errors else 0
        
        return {
            'timestamp': current_time,
            'total_requests': total_requests,
            'throughput_rps': throughput,
            'avg_response_time': avg_response_time,
            'error_rate': error_rate,
            'active_connections': self.gauges.get('active_connections', 0),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'summary': self.get_metrics_summary()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        total_hits = sum(count for key, count in self.counters.items() 
                        if key.startswith('cache_hit_'))
        total_misses = sum(count for key, count in self.counters.items() 
                          if key.startswith('cache_miss_'))
        
        total_requests = total_hits + total_misses
        return total_hits / total_requests if total_requests > 0 else 0.0
        

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.is_running = False
        self.monitor_thread = None
        self.system_metrics = deque(maxlen=100)
        self.metrics_collector = MetricsCollector()
        
        logger.info("系统监控器初始化完成")
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # 启动 Prometheus 服务器（如果可用）
        if PROMETHEUS_AVAILABLE and settings.ENABLE_METRICS:
            try:
                start_http_server(settings.METRICS_PORT, registry=self.metrics_collector.registry)
                logger.info(f"Prometheus 指标服务器启动在端口 {settings.METRICS_PORT}")
            except Exception as e:
                logger.error(f"启动 Prometheus 服务器失败: {e}")
        
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self._collect_system_metrics()
                time.sleep(10)  # 每10秒收集一次
            except Exception as e:
                logger.error(f"系统监控错误: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # 网络统计
            network = psutil.net_io_counters()
            
            # 进程信息
            process = psutil.Process()
            process_memory = process.memory_info()
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_percent': memory.percent,
                'disk_total': disk.total,
                'disk_used': disk.used,
                'disk_percent': disk.percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_memory_rss': process_memory.rss,
                'process_memory_vms': process_memory.vms,
                'process_cpu_percent': process.cpu_percent()
            }
            
            self.system_metrics.append(metrics)
            
            # 更新指标收集器
            self.metrics_collector.set_gauge("cpu_percent", cpu_percent)
            self.metrics_collector.set_gauge("memory_percent", memory.percent)
            self.metrics_collector.set_gauge("disk_percent", disk.percent)
            self.metrics_collector.set_gauge("process_memory_mb", process_memory.rss / 1024 / 1024)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    def get_current_metrics(self) -> Dict:
        """获取当前系统指标"""
        if self.system_metrics:
            return self.system_metrics[-1]
        return {}
    
    def get_metrics_history(self, minutes: int = 10) -> List[Dict]:
        """获取指标历史"""
        cutoff_time = time.time() - (minutes * 60)
        return [
            metric for metric in self.system_metrics
            if metric['timestamp'] > cutoff_time
        ]
    
    def get_metrics_summary(self) -> Dict:
        """获取指标摘要"""
        system_summary = {}
        
        if self.system_metrics:
            latest = self.system_metrics[-1]
            system_summary = {
                'current_cpu_percent': latest.get('cpu_percent', 0),
                'current_memory_percent': latest.get('memory_percent', 0),
                'current_disk_percent': latest.get('disk_percent', 0),
                'process_memory_mb': latest.get('process_memory_rss', 0) / 1024 / 1024
            }
        
        # 合并应用指标
        app_summary = self.metrics_collector.get_metrics_summary()
        
        return {
            'system': system_summary,
            'application': app_summary,
            'monitoring_status': 'active' if self.is_running else 'inactive'
        }


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.request_times = defaultdict(list)
        self.slow_requests = []
        self.slow_threshold = 2.0  # 2秒
        
        logger.info("性能分析器初始化完成")
    
    def analyze_request_performance(self, endpoint: str, duration: float, 
                                  method: str = "GET", details: Dict = None):
        """分析请求性能"""
        key = f"{method}_{endpoint}"
        self.request_times[key].append({
            'duration': duration,
            'timestamp': time.time(),
            'details': details or {}
        })
        
        # 检测慢请求
        if duration > self.slow_threshold:
            self.slow_requests.append({
                'endpoint': endpoint,
                'method': method,
                'duration': duration,
                'timestamp': time.time(),
                'details': details or {}
            })
            
            # 保持慢请求列表大小
            if len(self.slow_requests) > 100:
                self.slow_requests = self.slow_requests[-100:]
        
        # 保持请求时间列表大小
        if len(self.request_times[key]) > 1000:
            self.request_times[key] = self.request_times[key][-1000:]
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'slow_requests_count': len(self.slow_requests),
            'recent_slow_requests': self.slow_requests[-10:],
            'endpoint_performance': {}
        }
        
        for endpoint, times in self.request_times.items():
            if times:
                durations = [t['duration'] for t in times]
                report['endpoint_performance'][endpoint] = {
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'request_count': len(durations),
                    'slow_requests': sum(1 for d in durations if d > self.slow_threshold)
                }
        
        return report
    
    def detect_performance_issues(self) -> List[Dict]:
        """检测性能问题"""
        issues = []
        
        for endpoint, times in self.request_times.items():
            if len(times) < 10:  # 需要足够的数据点
                continue
            
            recent_times = times[-50:]  # 最近50个请求
            durations = [t['duration'] for t in recent_times]
            avg_duration = sum(durations) / len(durations)
            
            # 检测平均响应时间过长
            if avg_duration > 1.0:
                issues.append({
                    'type': 'slow_average_response',
                    'endpoint': endpoint,
                    'avg_duration': avg_duration,
                    'severity': 'high' if avg_duration > 3.0 else 'medium'
                })
            
            # 检测高频慢请求
            slow_count = sum(1 for d in durations if d > self.slow_threshold)
            slow_ratio = slow_count / len(durations)
            
            if slow_ratio > 0.1:  # 超过10%的请求是慢请求
                issues.append({
                    'type': 'high_slow_request_ratio',
                    'endpoint': endpoint,
                    'slow_ratio': slow_ratio,
                    'severity': 'high' if slow_ratio > 0.3 else 'medium'
                })
        
        return issues


# 性能监控装饰器
def monitor_performance(endpoint_name: str = None):
    """性能监控装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = endpoint_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成功的请求
                performance_monitor.metrics_collector.record_request(
                    method="FUNC",
                    endpoint=endpoint,
                    status_code=200,
                    duration=duration
                )
                
                performance_monitor.performance_analyzer.analyze_request_performance(
                    endpoint=endpoint,
                    duration=duration,
                    method="FUNC"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录失败的请求
                performance_monitor.metrics_collector.record_request(
                    method="FUNC",
                    endpoint=endpoint,
                    status_code=500,
                    duration=duration
                )
                
                raise
        
        return wrapper
    return decorator


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_channels = []
        
        # 默认告警规则
        self._setup_default_rules()
        
        logger.info("告警管理器初始化完成")
    
    def _setup_default_rules(self):
        """设置默认告警规则"""
        self.alert_rules = [
            {
                'name': 'high_cpu_usage',
                'condition': lambda metrics: metrics.get('cpu_percent', 0) > 80,
                'severity': 'warning',
                'message': 'CPU使用率过高: {cpu_percent}%'
            },
            {
                'name': 'high_memory_usage',
                'condition': lambda metrics: metrics.get('memory_percent', 0) > 85,
                'severity': 'warning',
                'message': '内存使用率过高: {memory_percent}%'
            },
            {
                'name': 'high_disk_usage',
                'condition': lambda metrics: metrics.get('disk_percent', 0) > 90,
                'severity': 'critical',
                'message': '磁盘使用率过高: {disk_percent}%'
            },
            {
                'name': 'high_error_rate',
                'condition': lambda metrics: metrics.get('error_rate', 0) > 0.05,
                'severity': 'warning',
                'message': '错误率过高: {error_rate:.2%}'
            },
            {
                'name': 'slow_response_time',
                'condition': lambda metrics: metrics.get('avg_response_time', 0) > 2.0,
                'severity': 'warning',
                'message': '平均响应时间过长: {avg_response_time:.2f}s'
            }
        ]
    
    def add_alert_rule(self, name: str, condition, severity: str, message: str):
        """添加告警规则"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message
        })
    
    def check_alerts(self, metrics: Dict):
        """检查告警条件"""
        current_time = time.time()
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    alert_id = f"{rule['name']}_{current_time}"
                    
                    # 检查是否已存在相同类型的活跃告警
                    existing_alert = None
                    for alert in self.active_alerts.values():
                        if alert['rule_name'] == rule['name']:
                            existing_alert = alert
                            break
                    
                    if not existing_alert:
                        # 创建新告警
                        alert = {
                            'id': alert_id,
                            'rule_name': rule['name'],
                            'severity': rule['severity'],
                            'message': rule['message'].format(**metrics),
                            'start_time': current_time,
                            'status': 'firing',
                            'metrics_snapshot': metrics.copy()
                        }
                        
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert.copy())
                        new_alerts.append(alert)
                        
                        logger.warning(f"新告警触发: {alert['message']}")
                else:
                    # 检查是否需要解除告警
                    alerts_to_resolve = []
                    for alert_id, alert in self.active_alerts.items():
                        if alert['rule_name'] == rule['name']:
                            alerts_to_resolve.append(alert_id)
                    
                    for alert_id in alerts_to_resolve:
                        alert = self.active_alerts.pop(alert_id)
                        alert['status'] = 'resolved'
                        alert['end_time'] = current_time
                        self.alert_history.append(alert.copy())
                        
                        logger.info(f"告警已解除: {alert['rule_name']}")
                        
            except Exception as e:
                logger.error(f"检查告警规则 {rule['name']} 失败: {e}")
        
        return new_alerts
    
    def get_active_alerts(self) -> List[Dict]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """获取告警历史"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert for alert in self.alert_history
            if alert['start_time'] > cutoff_time
        ]
    
    def resolve_alert(self, alert_id: str):
        """手动解除告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert['status'] = 'resolved'
            alert['end_time'] = time.time()
            alert['resolved_manually'] = True
            self.alert_history.append(alert.copy())
            
            logger.info(f"告警已手动解除: {alert['rule_name']}")
            return True
        return False


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.health_checks = {}
        self.check_results = {}
        
        # 注册默认健康检查
        self._register_default_checks()
        
        logger.info("健康检查器初始化完成")
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_usage", self._check_memory_usage)
    
    def register_check(self, name: str, check_func):
        """注册健康检查"""
        self.health_checks[name] = check_func
    
    async def run_all_checks(self) -> Dict:
        """运行所有健康检查"""
        results = {}
        overall_status = "healthy"
        
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = result
                
                if result['status'] != 'healthy':
                    overall_status = 'unhealthy'
                    
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'message': f"健康检查失败: {str(e)}",
                    'timestamp': time.time()
                }
                overall_status = 'unhealthy'
                logger.error(f"健康检查 {name} 失败: {e}")
        
        self.check_results = results
        
        return {
            'overall_status': overall_status,
            'checks': results,
            'timestamp': time.time()
        }
    
    def _check_system_resources(self) -> Dict:
        """检查系统资源"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            messages = []
            
            if cpu_percent > 90:
                status = "unhealthy"
                messages.append(f"CPU使用率过高: {cpu_percent}%")
            elif cpu_percent > 75:
                status = "warning"
                messages.append(f"CPU使用率较高: {cpu_percent}%")
            
            if memory.percent > 90:
                status = "unhealthy"
                messages.append(f"内存使用率过高: {memory.percent}%")
            elif memory.percent > 80:
                if status == "healthy":
                    status = "warning"
                messages.append(f"内存使用率较高: {memory.percent}%")
            
            return {
                'status': status,
                'message': '; '.join(messages) if messages else "系统资源正常",
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / 1024 / 1024 / 1024
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"无法获取系统资源信息: {str(e)}",
                'timestamp': time.time()
            }
    
    def _check_disk_space(self) -> Dict:
        """检查磁盘空间"""
        try:
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            
            status = "healthy"
            if percent_used > 95:
                status = "unhealthy"
            elif percent_used > 85:
                status = "warning"
            
            return {
                'status': status,
                'message': f"磁盘使用率: {percent_used:.1f}%",
                'details': {
                    'disk_percent': percent_used,
                    'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                    'disk_total_gb': disk.total / 1024 / 1024 / 1024
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"无法获取磁盘信息: {str(e)}",
                'timestamp': time.time()
            }
    
    def _check_memory_usage(self) -> Dict:
        """检查内存使用情况"""
        try:
            process = psutil.Process()
            process_memory = process.memory_info()
            memory_mb = process_memory.rss / 1024 / 1024
            
            status = "healthy"
            if memory_mb > 2048:  # 2GB
                status = "warning"
            if memory_mb > 4096:  # 4GB
                status = "unhealthy"
            
            return {
                'status': status,
                'message': f"进程内存使用: {memory_mb:.1f}MB",
                'details': {
                    'process_memory_mb': memory_mb,
                    'process_memory_vms_mb': process_memory.vms / 1024 / 1024
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"无法获取进程内存信息: {str(e)}",
                'timestamp': time.time()
            }


class MetricsExporter:
    """指标导出器"""
    
    def __init__(self):
        self.export_formats = ['json', 'csv', 'prometheus']
        
        logger.info("指标导出器初始化完成")
    
    def export_metrics(self, metrics: Dict, format: str = 'json') -> str:
        """导出指标数据"""
        if format == 'json':
            return self._export_json(metrics)
        elif format == 'csv':
            return self._export_csv(metrics)
        elif format == 'prometheus':
            return self._export_prometheus(metrics)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _export_json(self, metrics: Dict) -> str:
        """导出为JSON格式"""
        return json.dumps(metrics, indent=2, default=str)
    
    def _export_csv(self, metrics: Dict) -> str:
        """导出为CSV格式"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        writer.writerow(['metric_name', 'value', 'timestamp'])
        
        # 写入数据
        timestamp = metrics.get('timestamp', time.time())
        for key, value in metrics.items():
            if key != 'timestamp' and isinstance(value, (int, float)):
                writer.writerow([key, value, timestamp])
        
        return output.getvalue()
    
    def _export_prometheus(self, metrics: Dict) -> str:
        """导出为Prometheus格式"""
        lines = []
        timestamp = int(metrics.get('timestamp', time.time()) * 1000)
        
        for key, value in metrics.items():
            if key != 'timestamp' and isinstance(value, (int, float)):
                metric_name = f"signavatar_{key.replace('.', '_').replace('-', '_')}"
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {value} {timestamp}")
        
        return '\n'.join(lines)


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self):
        self.baseline_data = defaultdict(list)
        self.anomaly_threshold = 2.0  # 标准差倍数
        self.min_data_points = 30
        
        logger.info("异常检测器初始化完成")
    
    def add_data_point(self, metric_name: str, value: float):
        """添加数据点"""
        self.baseline_data[metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
        
        # 保持数据点数量
        if len(self.baseline_data[metric_name]) > 1000:
            self.baseline_data[metric_name] = self.baseline_data[metric_name][-1000:]
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> Dict:
        """检测异常"""
        if metric_name not in self.baseline_data:
            return {'is_anomaly': False, 'reason': 'insufficient_data'}
        
        data_points = self.baseline_data[metric_name]
        
        if len(data_points) < self.min_data_points:
            return {'is_anomaly': False, 'reason': 'insufficient_data'}
        
        # 计算基线统计
        values = [dp['value'] for dp in data_points[-100:]]  # 使用最近100个点
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return {'is_anomaly': False, 'reason': 'no_variance'}
        
        # 计算Z分数
        z_score = abs(current_value - mean) / std_dev
        
        is_anomaly = z_score > self.anomaly_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'threshold': self.anomaly_threshold,
            'mean': mean,
            'std_dev': std_dev,
            'current_value': current_value,
            'confidence': min(z_score / self.anomaly_threshold, 1.0) if is_anomaly else 0.0
        }
    
    def get_anomaly_report(self) -> Dict:
        """获取异常检测报告"""
        report = {
            'total_metrics': len(self.baseline_data),
            'metrics_with_sufficient_data': 0,
            'recent_anomalies': []
        }
        
        for metric_name, data_points in self.baseline_data.items():
            if len(data_points) >= self.min_data_points:
                report['metrics_with_sufficient_data'] += 1
                
                # 检查最近的异常
                if data_points:
                    latest_value = data_points[-1]['value']
                    anomaly_result = self.detect_anomaly(metric_name, latest_value)
                    
                    if anomaly_result['is_anomaly']:
                        report['recent_anomalies'].append({
                            'metric_name': metric_name,
                            'timestamp': data_points[-1]['timestamp'],
                            **anomaly_result
                        })
        
        return report


# 更新PerformanceMonitor类
class PerformanceMonitor:
    """性能监控主类"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.metrics_collector = MetricsCollector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.metrics_exporter = MetricsExporter()
        self.anomaly_detector = AnomalyDetector()
        
        # 监控循环
        self.monitoring_task = None
        
        logger.info("性能监控系统初始化完成")
    
    async def initialize(self):
        """初始化监控系统"""
        self.system_monitor.start_monitoring()
        
        # 启动监控循环
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("性能监控系统已启动")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                # 获取当前指标
                metrics = self.get_comprehensive_report()
                
                # 检查告警
                self.alert_manager.check_alerts(metrics['system_metrics']['application'])
                
                # 异常检测
                for key, value in metrics['system_metrics']['application'].items():
                    if isinstance(value, (int, float)):
                        self.anomaly_detector.add_data_point(key, value)
                
                await asyncio.sleep(30)  # 每30秒执行一次
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(10)
    
    def get_comprehensive_report(self) -> Dict:
        """获取综合性能报告"""
        return {
            'system_metrics': self.system_monitor.get_metrics_summary(),
            'performance_analysis': self.performance_analyzer.get_performance_report(),
            'performance_issues': self.performance_analyzer.detect_performance_issues(),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'anomaly_report': self.anomaly_detector.get_anomaly_report(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_health_status(self) -> Dict:
        """获取健康状态"""
        return await self.health_checker.run_all_checks()
    
    def export_metrics(self, format: str = 'json') -> str:
        """导出指标"""
        metrics = self.get_comprehensive_report()
        return self.metrics_exporter.export_metrics(metrics, format)
    
    async def cleanup(self):
        """清理监控资源"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.system_monitor.stop_monitoring()
        logger.info("性能监控系统已清理")


# 全局性能监控实例
performance_monitor = PerformanceMonitor()