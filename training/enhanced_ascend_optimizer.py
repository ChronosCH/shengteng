"""
增强的华为昇腾AI处理器优化工具
整合了最新的优化策略和性能调优方法
支持昇腾910/310系列处理器和MindSpore框架
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mindspore as ms
from mindspore import context, nn, ops, Tensor, Parameter
from mindspore.train import Model
from mindspore.profiler import Profiler
from mindspore.communication.management import init, get_rank, get_group_size
import mindspore.dataset as ds
from mindspore.dataset import vision, transforms
from mindspore.common import set_seed
from mindspore import amp
from mindspore.nn import learning_rate_schedule as lr_schedule

logger = logging.getLogger(__name__)

@dataclass
class AscendOptimizationConfig:
    """昇腾优化配置"""
    # 设备配置
    device_target: str = "Ascend"
    device_id: int = 0
    device_num: int = 1
    
    # 内存配置
    max_device_memory: str = "30GB"  # 昇腾910规格
    variable_memory_max_size: str = "31GB"
    
    # 性能优化
    enable_amp: bool = True
    amp_level: str = "O1"  # O0, O1, O2, O3
    enable_graph_kernel: bool = True
    enable_reduce_precision: bool = True
    
    # 并行配置
    enable_parallel: bool = False
    parallel_mode: str = "data_parallel"  # data_parallel, model_parallel, hybrid_parallel
    gradients_mean: bool = True
    
    # 调试和分析
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_communication: bool = True
    save_graphs: bool = False
    
    # 数据处理优化
    dataset_sink_mode: bool = True
    prefetch_size: int = 16
    num_parallel_workers: int = 8

class EnhancedAscendOptimizer:
    """增强的昇腾处理器优化器"""
    
    def __init__(self, config: AscendOptimizationConfig):
        self.config = config
        self.profiler = None
        self.rank_id = 0
        self.rank_size = 1
        
        # 初始化环境
        self._setup_environment()
        self._setup_parallel_training()
    
    def _detect_available_devices(self):
        """检测可用的设备类型"""
        available_devices = ["CPU"]  # CPU总是可用的
        
        try:
            # 尝试检测GPU
            import mindspore
            context.set_context(device_target="GPU")
            available_devices.append("GPU")
            logger.info("检测到GPU设备")
        except:
            pass
        
        try:
            # 尝试检测Ascend
            context.set_context(device_target="Ascend") 
            available_devices.append("Ascend")
            logger.info("检测到Ascend设备")
        except:
            pass
        
        # 重置为CPU以避免影响后续设置
        context.set_context(device_target="CPU")
        logger.info(f"可用设备: {available_devices}")
        return available_devices
    
    def _setup_environment(self):
        """设置运行环境，自动检测可用设备"""
        try:
            # 自动检测可用设备
            available_devices = self._detect_available_devices()
            if self.config.device_target not in available_devices:
                logger.warning(f"设备 {self.config.device_target} 不可用，切换到 {available_devices[0]}")
                self.config.device_target = available_devices[0]
                if self.config.device_target == "CPU":
                    self.config.device_id = 0  # CPU模式下设备ID应为0
            
            logger.info(f"使用设备: {self.config.device_target}")
            
            # 设置基本上下文
            context_config = {
                'mode': context.GRAPH_MODE,
                'device_target': self.config.device_target,
                'save_graphs': self.config.save_graphs,
                'max_call_depth': 10000
            }
            
            # 只有非CPU设备才需要设置device_id
            if self.config.device_target != "CPU":
                context_config['device_id'] = self.config.device_id
            
            if self.config.save_graphs:
                context_config['save_graphs_path'] = "./graphs"
                
            context.set_context(**context_config)
            
            # CPU模式下的内存配置
            if self.config.device_target == "CPU":
                # CPU模式下的优化配置
                context.set_context(
                    max_call_depth=10000,
                    enable_compile_cache=True,
                    compile_cache_path="./cache"
                )
            else:
                # 其他设备的内存配置
                context.set_context(
                    max_device_memory=self.config.max_device_memory,
                    variable_memory_max_size=self.config.variable_memory_max_size
                )
            
            # 性能优化（CPU模式下某些优化可能不适用）
            if self.config.enable_graph_kernel and self.config.device_target != "CPU":
                context.set_context(enable_graph_kernel=True)
            
            if self.config.enable_reduce_precision and self.config.device_target != "CPU":
                context.set_context(enable_reduce_precision=True)
            
            # 设置随机种子
            set_seed(42)
            
            logger.info(f"昇腾环境配置完成 (设备ID: {self.config.device_id})")
            
        except Exception as e:
            logger.error(f"昇腾环境配置失败: {e}")
            raise
    
    def _setup_parallel_training(self):
        """设置并行训练"""
        if self.config.enable_parallel:
            try:
                # 初始化分布式训练
                init()
                self.rank_id = get_rank()
                self.rank_size = get_group_size()
                
                # 设置并行模式
                if self.config.parallel_mode == "data_parallel":
                    parallel_mode = context.ParallelMode.DATA_PARALLEL
                elif self.config.parallel_mode == "model_parallel":
                    parallel_mode = context.ParallelMode.HYBRID_PARALLEL
                else:
                    parallel_mode = context.ParallelMode.DATA_PARALLEL
                
                context.set_auto_parallel_context(
                    parallel_mode=parallel_mode,
                    gradients_mean=self.config.gradients_mean,
                    device_num=self.config.device_num
                )
                
                logger.info(f"并行训练配置完成 - Rank: {self.rank_id}/{self.rank_size}")
                
            except Exception as e:
                logger.warning(f"并行训练配置失败: {e}")
                self.config.enable_parallel = False
    
    def optimize_model_architecture(self, model: nn.Cell) -> nn.Cell:
        """优化模型架构"""
        try:
            # 1. 启用自动混合精度
            if self.config.enable_amp:
                model = self._enable_amp_training(model)
            
            # 2. 算子融合优化
            model = self._optimize_operators(model)
            
            # 3. 内存优化
            model = self._optimize_memory(model)
            
            logger.info("模型架构优化完成")
            return model
            
        except Exception as e:
            logger.error(f"模型架构优化失败: {e}")
            return model
    
    def _enable_amp_training(self, model: nn.Cell) -> nn.Cell:
        """启用自动混合精度训练"""
        try:
            level = self.config.amp_level
            
            if level == "O0":
                # 不使用混合精度
                return model
            elif level == "O1":
                # 部分算子使用fp16
                amp_model = amp.build_train_network(
                    model,
                    level=level,
                    keep_batchnorm_fp32=True,
                    cast_model_type=ms.float16
                )
            elif level == "O2":
                # 大部分算子使用fp16
                amp_model = amp.build_train_network(
                    model,
                    level=level,
                    keep_batchnorm_fp32=True,
                    cast_model_type=ms.float16
                )
            elif level == "O3":
                # 全部算子使用fp16
                amp_model = amp.build_train_network(
                    model,
                    level=level,
                    cast_model_type=ms.float16
                )
            else:
                logger.warning(f"不支持的AMP级别: {level}，使用O1")
                amp_model = amp.build_train_network(
                    model,
                    level="O1",
                    keep_batchnorm_fp32=True,
                    cast_model_type=ms.float16
                )
            
            logger.info(f"AMP训练已启用 (级别: {level})")
            return amp_model
            
        except Exception as e:
            logger.error(f"AMP训练启用失败: {e}")
            return model
    
    def _optimize_operators(self, model: nn.Cell) -> nn.Cell:
        """优化算子"""
        try:
            # 这里可以添加算子融合、算子替换等优化
            # 当前MindSpore会自动进行大部分算子优化
            
            # 示例：替换某些算子为昇腾优化版本
            for name, cell in model.cells_and_names():
                if isinstance(cell, nn.BatchNorm2d):
                    # 昇腾上BatchNorm可以进行特殊优化
                    cell.set_train(True)  # 确保训练模式下的优化
                elif isinstance(cell, nn.ReLU):
                    # 可以考虑使用其他激活函数
                    pass
            
            return model
            
        except Exception as e:
            logger.warning(f"算子优化失败: {e}")
            return model
    
    def _optimize_memory(self, model: nn.Cell) -> nn.Cell:
        """内存优化"""
        try:
            # 启用梯度检查点（如果模型支持）
            # 这可以减少内存使用，但会增加计算时间
            
            # 示例：为大型模型启用重计算
            for name, cell in model.cells_and_names():
                if hasattr(cell, 'recompute'):
                    cell.recompute()
            
            return model
            
        except Exception as e:
            logger.warning(f"内存优化失败: {e}")
            return model
    
    def create_optimized_optimizer(self, model: nn.Cell, 
                                  learning_rate: float = 1e-4,
                                  weight_decay: float = 1e-4,
                                  warmup_steps: int = 1000) -> nn.Optimizer:
        """创建优化的优化器"""
        try:
            # 学习率调度（昇腾优化版）
            lr_scheduler = lr_schedule.CosineDecayLR(
                min_lr=learning_rate * 0.01,
                max_lr=learning_rate,
                decay_steps=100000,  # 根据实际训练步数调整
                warmup_steps=warmup_steps
            )
            
            # 参数分组优化
            backbone_params = []
            head_params = []
            bn_params = []
            
            for name, param in model.parameters_and_names():
                if 'backbone' in name or 'encoder' in name:
                    backbone_params.append(param)
                elif 'bn' in name or 'norm' in name:
                    bn_params.append(param)
                else:
                    head_params.append(param)
            
            # 不同参数组使用不同配置
            group_params = []
            
            if backbone_params:
                group_params.append({
                    'params': backbone_params,
                    'lr': lr_scheduler * 0.1,  # 骨干网络使用较小学习率
                    'weight_decay': weight_decay
                })
            
            if head_params:
                group_params.append({
                    'params': head_params,
                    'lr': lr_scheduler,
                    'weight_decay': weight_decay
                })
            
            if bn_params:
                group_params.append({
                    'params': bn_params,
                    'lr': lr_scheduler,
                    'weight_decay': 0.0  # BN层不使用权重衰减
                })
            
            # 如果没有参数分组，使用全部参数
            if not group_params:
                group_params = model.trainable_params()
            
            # 创建优化器（昇腾推荐AdamWeightDecay）
            optimizer = nn.AdamWeightDecay(
                group_params,
                learning_rate=lr_scheduler,
                weight_decay=weight_decay,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8
            )
            
            logger.info("优化器创建完成")
            return optimizer
            
        except Exception as e:
            logger.error(f"优化器创建失败: {e}")
            # 回退到简单优化器
            return nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    
    def create_optimized_dataset(self, dataset_files: Union[str, List[str]], 
                                batch_size: int = 32,
                                shuffle: bool = True) -> ds.Dataset:
        """创建优化的数据集"""
        try:
            # 支持多种数据格式
            if isinstance(dataset_files, str):
                if dataset_files.endswith('.mindrecord'):
                    dataset = ds.MindDataset(
                        dataset_files=dataset_files,
                        shuffle=shuffle,
                        num_parallel_workers=self.config.num_parallel_workers
                    )
                else:
                    # 其他格式的数据集
                    dataset = ds.GeneratorDataset(
                        source=self._data_generator(dataset_files),
                        column_names=['data', 'label'],
                        shuffle=shuffle,
                        num_parallel_workers=self.config.num_parallel_workers
                    )
            else:
                # 多文件
                dataset = ds.MindDataset(
                    dataset_files=dataset_files,
                    shuffle=shuffle,
                    num_parallel_workers=self.config.num_parallel_workers
                )
            
            # 数据预处理优化
            dataset = self._apply_data_transformations(dataset)
            
            # 批处理
            dataset = dataset.batch(
                batch_size=batch_size,
                drop_remainder=True,
                num_parallel_workers=self.config.num_parallel_workers
            )
            
            # 预取优化
            dataset = dataset.prefetch(buffer_size=self.config.prefetch_size)
            
            # 分布式训练的数据分片
            if self.config.enable_parallel and self.rank_size > 1:
                dataset = dataset.shard(
                    num_shards=self.rank_size,
                    shard_id=self.rank_id
                )
            
            logger.info(f"优化数据集创建完成 (批次大小: {batch_size})")
            return dataset
            
        except Exception as e:
            logger.error(f"数据集创建失败: {e}")
            raise
    
    def _data_generator(self, data_path: str):
        """数据生成器"""
        # 这里应该根据具体数据格式实现
        # 示例实现
        def generator():
            # 加载数据逻辑
            for i in range(1000):  # 示例
                data = np.random.random((224, 224, 3)).astype(np.float32)
                label = np.random.randint(0, 10)
                yield data, label
        
        return generator
    
    def _apply_data_transformations(self, dataset):
        """应用数据变换"""
        try:
            # 昇腾优化的数据变换
            transforms_list = [
                vision.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]
            
            dataset = dataset.map(
                operations=transforms_list,
                input_columns=['data'],
                num_parallel_workers=self.config.num_parallel_workers
            )
            
            return dataset
            
        except Exception as e:
            logger.warning(f"数据变换应用失败: {e}")
            return dataset
    
    def enable_profiling(self, output_path: str = "./profiling_data"):
        """启用性能分析"""
        if not self.config.enable_profiling:
            return
        
        try:
            os.makedirs(output_path, exist_ok=True)
            
            self.profiler = Profiler(
                output_path=output_path,
                profile_memory=self.config.profile_memory,
                profile_communication=self.config.profile_communication,
                op_time=True,
                parallel_strategy=True
            )
            
            logger.info(f"性能分析已启用，输出路径: {output_path}")
            
        except Exception as e:
            logger.error(f"性能分析启用失败: {e}")
    
    def disable_profiling(self):
        """禁用性能分析"""
        if self.profiler:
            try:
                self.profiler.stop()
                self.profiler = None
                logger.info("性能分析已停止")
            except Exception as e:
                logger.error(f"性能分析停止失败: {e}")
    
    def benchmark_model(self, model: nn.Cell, input_data: Tensor, 
                       num_iterations: int = 100,
                       warmup_iterations: int = 10) -> Dict:
        """模型性能基准测试"""
        try:
            model.set_train(False)
            
            # 预热
            logger.info(f"开始预热 ({warmup_iterations} 次迭代)...")
            for i in range(warmup_iterations):
                _ = model(input_data)
            
            # 基准测试
            logger.info(f"开始基准测试 ({num_iterations} 次迭代)...")
            
            # 记录内存使用
            start_memory = self._get_memory_usage()
            
            start_time = time.time()
            
            for i in range(num_iterations):
                if i % 10 == 0:
                    logger.debug(f"基准测试进度: {i}/{num_iterations}")
                _ = model(input_data)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # 计算统计信息
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            throughput = 1.0 / avg_time
            batch_size = input_data.shape[0] if len(input_data.shape) > 0 else 1
            samples_per_second = throughput * batch_size
            
            results = {
                'total_time': total_time,
                'average_inference_time': avg_time,
                'throughput_fps': throughput,
                'samples_per_second': samples_per_second,
                'iterations': num_iterations,
                'batch_size': batch_size,
                'memory_usage_mb': end_memory - start_memory,
                'device_target': self.config.device_target,
                'device_id': self.config.device_id
            }
            
            logger.info("基准测试完成:")
            logger.info(f"  平均推理时间: {avg_time*1000:.2f} ms")
            logger.info(f"  吞吐量: {throughput:.2f} FPS")
            logger.info(f"  样本处理速度: {samples_per_second:.2f} samples/s")
            logger.info(f"  内存使用: {end_memory - start_memory:.2f} MB")
            
            return results
            
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """获取内存使用情况"""
        try:
            # 这里应该实现获取昇腾设备内存使用的逻辑
            # 当前返回模拟值
            return 0.0
        except:
            return 0.0
    
    def optimize_for_inference(self, model: nn.Cell, 
                              example_inputs: Tensor,
                              output_file: str = "optimized_model") -> str:
        """为推理优化模型"""
        try:
            # 设置为推理模式
            model.set_train(False)
            
            # 应用推理优化
            model = self._apply_inference_optimizations(model)
            
            # 导出为MindIR格式
            output_path = f"{output_file}.mindir"
            ms.export(
                model,
                example_inputs,
                file_name=output_file,
                file_format='MINDIR'
            )
            
            logger.info(f"推理优化模型已导出: {output_path}")
            
            # 生成优化报告
            self._generate_optimization_report(model, output_file)
            
            return output_path
            
        except Exception as e:
            logger.error(f"推理优化失败: {e}")
            raise
    
    def _apply_inference_optimizations(self, model: nn.Cell) -> nn.Cell:
        """应用推理优化"""
        try:
            # 1. 算子融合
            # 2. 常量折叠
            # 3. 死代码消除
            # 这些优化通常由MindSpore自动完成
            
            # 确保所有层都在推理模式
            for cell in model.cells():
                cell.set_train(False)
            
            return model
            
        except Exception as e:
            logger.warning(f"推理优化应用失败: {e}")
            return model
    
    def _generate_optimization_report(self, model: nn.Cell, output_file: str):
        """生成优化报告"""
        try:
            report = {
                'model_info': {
                    'total_parameters': self._count_parameters(model),
                    'model_size_mb': self._estimate_model_size(model),
                },
                'optimization_config': {
                    'amp_enabled': self.config.enable_amp,
                    'amp_level': self.config.amp_level,
                    'graph_kernel_enabled': self.config.enable_graph_kernel,
                    'parallel_enabled': self.config.enable_parallel,
                },
                'device_info': {
                    'device_target': self.config.device_target,
                    'device_id': self.config.device_id,
                }
            }
            
            report_path = f"{output_file}_optimization_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"优化报告已生成: {report_path}")
            
        except Exception as e:
            logger.warning(f"优化报告生成失败: {e}")
    
    def _count_parameters(self, model: nn.Cell) -> int:
        """计算模型参数量"""
        total_params = 0
        for param in model.get_parameters():
            total_params += np.prod(param.shape)
        return total_params
    
    def _estimate_model_size(self, model: nn.Cell) -> float:
        """估算模型大小（MB）"""
        total_params = self._count_parameters(model)
        # 假设float32，每个参数4字节
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)
        return size_mb

class AscendPerformanceMonitor:
    """昇腾性能监控器"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_times = []
        self.memory_usage = []
        self.current_step = 0
    
    def on_step_begin(self):
        """步骤开始"""
        self.step_start_time = time.time()
    
    def on_step_end(self):
        """步骤结束"""
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # 记录内存使用（如果可用）
        memory = self._get_current_memory()
        if memory is not None:
            self.memory_usage.append(memory)
        
        self.current_step += 1
        
        # 定期输出统计信息
        if self.current_step % self.log_interval == 0:
            self._log_performance_stats()
    
    def _get_current_memory(self) -> Optional[float]:
        """获取当前内存使用"""
        try:
            # 这里应该实现获取昇腾设备内存的逻辑
            return None
        except:
            return None
    
    def _log_performance_stats(self):
        """输出性能统计信息"""
        if self.step_times:
            recent_times = self.step_times[-self.log_interval:]
            avg_time = np.mean(recent_times)
            min_time = np.min(recent_times)
            max_time = np.max(recent_times)
            
            logger.info(f"性能统计 (步骤 {self.current_step}):")
            logger.info(f"  平均步骤时间: {avg_time*1000:.2f} ms")
            logger.info(f"  最小步骤时间: {min_time*1000:.2f} ms")
            logger.info(f"  最大步骤时间: {max_time*1000:.2f} ms")
            
            if self.memory_usage:
                recent_memory = self.memory_usage[-self.log_interval:]
                avg_memory = np.mean(recent_memory)
                logger.info(f"  平均内存使用: {avg_memory:.2f} MB")

def create_ascend_optimizer(device_id: int = 0, 
                           enable_amp: bool = True,
                           enable_parallel: bool = False) -> EnhancedAscendOptimizer:
    """创建昇腾优化器的便捷函数"""
    config = AscendOptimizationConfig(
        device_id=device_id,
        enable_amp=enable_amp,
        enable_parallel=enable_parallel
    )
    
    return EnhancedAscendOptimizer(config)

# 使用示例
if __name__ == "__main__":
    # 创建优化器
    optimizer = create_ascend_optimizer(device_id=0, enable_amp=True)
    
    # 示例：优化一个简单模型
    class SimpleModel(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(784, 10)
        
        def construct(self, x):
            return self.dense(x)
    
    model = SimpleModel()
    
    # 优化模型
    optimized_model = optimizer.optimize_model_architecture(model)
    
    # 创建优化器
    model_optimizer = optimizer.create_optimized_optimizer(optimized_model)
    
    # 基准测试
    dummy_input = Tensor(np.random.random((32, 784)).astype(np.float32))
    results = optimizer.benchmark_model(optimized_model, dummy_input)
    
    print("基准测试结果:", results)
