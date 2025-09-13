#!/usr/bin/env python3
"""
GPU优化的连续手语识别TFNet训练脚本
针对MindSpore框架的GPU执行进行优化
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, save_checkpoint, load_checkpoint, load_param_into_net, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import numpy as np  # 修复: 训练循环中使用 np.ndarray 判定但未导入

# 尝试导入新API，如果不可用则回退到旧版本
try:
    from mindspore import set_device
    MINDSPORE_NEW_API = True
except ImportError:
    MINDSPORE_NEW_API = False

# 将当前目录添加到导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from tfnet_model import TFNetModel, SeqKD
from data_processor import build_vocabulary, create_dataset
from decoder import CTCDecoder, calculate_wer_score
from utils import (
    normalize_path, ensure_directory_exists, safe_file_path,
    check_file_exists, check_directory_exists, print_error_details,
    validate_dataset_structure, print_dataset_validation
)

class GPUTFNetTrainer:
    """GPU优化的TFNet模型训练器"""
    
    def __init__(self, config_path=None):
        try:
            # 初始化配置
            print("初始化GPU优化的TFNet训练器...")
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config

            # 首先验证和创建目录（在日志设置之前）
            print("设置目录...")
            if not self._setup_directories():
                raise RuntimeError("创建所需目录失败")

            # 验证数据集结构
            print("验证数据集...")
            if not self._validate_dataset():
                raise RuntimeError("数据集验证失败")

            # 使用优化设置GPU上下文
            print("设置GPU上下文...")
            self._setup_gpu_context()

            # 初始化日志记录（在目录创建后）
            self._setup_logging()

            # Initialize components
            self.model = None
            self.train_dataset = None
            self.valid_dataset = None
            self.word2idx = None
            self.idx2word = None
            self.decoder = None

            # Training state
            self.current_epoch = 0
            self.best_wer = float('inf')
            self.best_epoch = 0

            self.logger.info("GPU-Optimized TFNet Trainer initialized successfully")

        except Exception as e:
            print_error_details(e, "GPU TFNet Trainer initialization")
            raise

    def _setup_gpu_context(self):
        """设置带有优化的GPU上下文"""
        try:
            device_target = self.config_manager.get("model.device_target", "GPU")
            device_id = self.config_manager.get("model.device_id", 0)
            
            # 检查是否有可用的GPU
            if not self._check_gpu_availability():
                raise RuntimeError("GPU not available or MindSpore GPU version not installed")

            # 使用配置中的优化设置设置上下文
            gpu_config = self.config_manager.get("gpu_optimization", {})
            
            if gpu_config.get("enable_graph_mode", True):
                mode = context.GRAPH_MODE
                print("✓ Using GRAPH_MODE for better GPU performance")
            else:
                mode = context.PYNATIVE_MODE
                print("✓ Using PYNATIVE_MODE for debugging")

            # 配置context
            context.set_context(
                mode=mode,
                device_target=device_target,
                device_id=device_id,
                save_graphs=False,  # Set to True for debugging
                save_graphs_path="./graphs",
            )

            # 启用内存优化（使用 max_device_memory）
            if gpu_config.get("enable_mem_reuse", True):
                try:
                    # 使用 max_device_memory 进行内存优化
                    max_memory = gpu_config.get("max_device_memory", "4GB")
                    context.set_context(max_device_memory=max_memory)
                    print(f"✓ Memory optimization enabled (max_device_memory: {max_memory})")
                    
                    # 额外的内存优化设置
                    mempool_size = gpu_config.get("mempool_block_size", "512MB")
                    context.set_context(
                        mempool_block_size=mempool_size,  # 更小的内存池块
                        enable_reduce_precision=True  # 启用降精度以节省内存
                    )
                    print(f"✓ Additional memory optimizations enabled (mempool: {mempool_size})")
                    
                    # 如果可用则启用内存卸载或稀疏支持
                    if gpu_config.get("enable_memory_offload", False):
                        try:
                            context.set_context(enable_sparse=True)  # 启用稀疏张量支持
                            print("✓ Sparse tensor support enabled for memory savings")
                        except Exception as sparse_e:
                            print(f"Warning: Sparse tensor not supported: {sparse_e}")
                            
                except Exception as e:
                    print(f"Warning: Memory optimization not supported: {e}")

            # 禁用图内核优化以避免编译问题
            if self.config_manager.get("model.enable_graph_kernel", False):
                try:
                    # Graph kernel 可能并非所有版本都支持
                    context.set_context(enable_graph_kernel=True)
                    print("✓ Graph kernel optimization enabled")
                except Exception as e:
                    print(f"Warning: Graph kernel optimization not supported: {e}")
            else:
                context.set_context(enable_graph_kernel=False)
                print("✓ Graph kernel optimization disabled for stability")

            # 启用自动混合精度（如支持）
            if self.config_manager.get("model.enable_auto_mixed_precision", True):
                try:
                    # 尝试不同的方法以启用自动混合精度
                    try:
                        context.set_auto_parallel_context(enable_auto_mixed_precision=True)
                        print("✓ Auto mixed precision enabled (auto_parallel_context)")
                    except:
                        # 备用方法
                        context.set_context(enable_auto_mixed_precision=True)
                        print("✓ Auto mixed precision enabled (context)")
                except Exception as e:
                    print(f"Warning: Auto mixed precision not supported: {e}")

            # 如果新API可用，则使用它设置设备
            if MINDSPORE_NEW_API:
                try:
                    set_device(device_target, device_id)
                    print(f"✓ MindSpore device set to: {device_target}:{device_id} (new API)")
                except Exception as e:
                    print(f"Warning: New API failed, using context setting: {e}")
            
            print(f"✓ GPU context configured successfully: {device_target}:{device_id}")
            
            # 打印内存管理相关信息
            max_memory = gpu_config.get("max_device_memory")
            if max_memory:
                print(f"✓ Max device memory limit: {max_memory}")

        except Exception as e:
            print_error_details(e, "GPU context setup")
            raise

    def _check_gpu_availability(self):
        """检查GPU是否可用"""
        try:
            # 尝试获取GPU设备信息
            import mindspore as ms
            
            # 简单的测试以检查GPU可用性
            test_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
            
            # 尝试在GPU上创建一个简单操作
            context.set_context(device_target="GPU")
            result = test_tensor + 1
            
            print("✓ GPU is available and accessible")
            return True
            
        except Exception as e:
            print(f"✗ GPU not available: {e}")
            print("Please ensure:")
            print("  1. NVIDIA GPU is properly installed")
            print("  2. CUDA drivers are installed")  
            print("  3. MindSpore GPU version is installed")
            print("  4. Environment is properly activated (conda activate mindspore-gpu)")
            return False

    def _setup_directories(self):
        """设置并验证所需目录"""
        try:
            # 使用配置管理器创建目录
            if not self.config_manager.create_directories():
                return False

            # 对关键目录进行额外验证
            critical_dirs = [
                self.config_manager.get("paths.checkpoint_dir"),
                self.config_manager.get("paths.log_dir"),
                self.config_manager.get("paths.output_dir")
            ]

            for dir_path in critical_dirs:
                if dir_path and not ensure_directory_exists(dir_path, create=True):
                    print(f"✗ Failed to setup critical directory: {dir_path}")
                    return False

            return True

        except Exception as e:
            print_error_details(e, "Directory setup")
            return False

    def _validate_dataset(self):
        """验证数据集结构和路径"""
        try:
            dataset_config = self.config_manager.get_dataset_config()

            # 检查是否配置了数据集路径
            required_paths = [
                ('train_data_path', 'Training data directory'),
                ('train_label_path', 'Training labels file'),
                ('valid_data_path', 'Validation data directory'),
                ('valid_label_path', 'Validation labels file')
            ]

            all_valid = True

            for path_key, description in required_paths:
                path = dataset_config.get(path_key)
                if not path:
                    print(f"✗ {description}: Path not configured")
                    all_valid = False
                    continue

                # 根据 path_key 判断是文件还是目录
                if 'label' in path_key:
                    # 标签路径应为文件
                    if not check_file_exists(path, description):
                        all_valid = False
                else:
                    # 数据路径应为目录
                    if not check_directory_exists(path, description):
                        all_valid = False

            # 如果是 CE-CSL 数据集，验证其目录结构
            if dataset_config.get('name') == 'CE-CSL':
                # 试图找到 CE-CSL 的基路径
                train_data_path = dataset_config.get('train_data_path', '')
                if 'CE-CSL' in train_data_path:
                    base_path = train_data_path.split('CE-CSL')[0] + 'CE-CSL'
                    validation_results = validate_dataset_structure(base_path)
                    print_dataset_validation(validation_results)
                    if not validation_results['valid']:
                        all_valid = False

            return all_valid

        except Exception as e:
            print_error_details(e, "Dataset validation")
            return False
    
    def _setup_logging(self):
        """设置日志配置并改进错误处理"""
        log_level = getattr(logging, self.config_manager.get("logging.level", "INFO"))

        # 创建 logger
        self.logger = logging.getLogger('GPUTFNetTrainer')
        self.logger.setLevel(log_level)

        # 清除现有处理器
        self.logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 文件处理器，确保目录存在
        if self.config_manager.get("logging.save_logs", True):
            try:
                log_dir = self.config_manager.get_safe_path("paths.log_dir", create_if_missing=True)
                if log_dir:
                    log_file = os.path.join(log_dir, f"gpu_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                    # 规范化路径以兼容不同平台
                    log_file = os.path.normpath(log_file)

                    file_handler = logging.FileHandler(log_file, encoding='utf-8')
                    file_handler.setLevel(log_level)
                    file_handler.setFormatter(console_format)
                    self.logger.addHandler(file_handler)
                    print(f"✓ Log file: {log_file}")
                else:
                    print("Warning: Could not create log directory, file logging disabled")
            except Exception as e:
                print(f"Warning: Failed to setup file logging: {e}")
                print("Continuing with console logging only")
    
    def prepare_data(self):
        """为GPU训练准备数据集和词表"""
        self.logger.info("Preparing data for GPU training...")
        
        dataset_config = self.config_manager.get_dataset_config()
        
        # 构建词表
        self.word2idx, vocab_size, self.idx2word = build_vocabulary(
            dataset_config["train_label_path"],
            dataset_config["valid_label_path"],
            dataset_config["test_label_path"],
            dataset_config["name"]
        )
        
        self.logger.info(f"Vocabulary size: {vocab_size}")
        
        # 保存词表
        vocab_path = os.path.join(self.config_manager.get("paths.output_dir"), "vocabulary.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': vocab_size
            }, f, indent=2, ensure_ascii=False)
        
        # 使用GPU优化创建数据集
        batch_size = self.config_manager.get("training.batch_size")
        num_workers = self.config_manager.get("training.num_workers")
        prefetch_size = self.config_manager.get("training.prefetch_size", 2)
        max_rowsize = self.config_manager.get("training.max_rowsize", 32)
        crop_size = self.config_manager.get("dataset.crop_size", 224)
        max_frames = self.config_manager.get("dataset.max_frames", 150)
        
        self.logger.info(f"Creating datasets with GPU optimizations:")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Num workers: {num_workers}")
        self.logger.info(f"  - Prefetch size: {prefetch_size}")
        self.logger.info(f"  - Max rowsize: {max_rowsize}")
        self.logger.info(f"  - Crop size: {crop_size}")
        self.logger.info(f"  - Max frames: {max_frames}")
        
        self.train_dataset = create_dataset(
            data_path=dataset_config["train_data_path"],
            label_path=dataset_config["train_label_path"],
            word2idx=self.word2idx,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=True,
            dataset_name=dataset_config["name"],
            prefetch_size=prefetch_size,
            max_rowsize=max_rowsize,
            crop_size=crop_size,
            max_frames=max_frames
        )
        
        self.valid_dataset = create_dataset(
            data_path=dataset_config["valid_data_path"],
            label_path=dataset_config["valid_label_path"],
            word2idx=self.word2idx,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=False,
            dataset_name=dataset_config["name"],
            prefetch_size=prefetch_size,
            max_rowsize=max_rowsize,
            crop_size=crop_size,
            max_frames=max_frames
        )
        
        self.logger.info("Data preparation completed")
        
        return len(self.word2idx)
        
    def build_model(self, vocab_size):
        """构建用于GPU的TFNet模型"""
        self.logger.info("Building TFNet model for GPU...")
        
        hidden_size = self.config_manager.get("model.hidden_size")
        device_target = self.config_manager.get("model.device_target")
        dataset_name = self.config_manager.get_dataset_config()["name"]
        
        # 创建模型实例
        self.model = TFNetModel(
            hidden_size=hidden_size,
            word_set_num=vocab_size,
            device_target=device_target,
            dataset_name=dataset_name
        )
        
        self.logger.info(f"Model created with hidden_size={hidden_size}, vocab_size={vocab_size}")
        self.logger.info(f"Model device target: {device_target}")
        
        # 初始化解码器
        self.decoder = CTCDecoder(
            gloss_dict=self.word2idx,
            num_classes=vocab_size,
            search_mode='beam',
            blank_id=self.config_manager.get("loss.ctc_blank_id", 0)
        )
        
        return self.model
    
    def setup_training(self, vocab_size):
        """使用GPU优化设置训练组件"""
        self.logger.info("Setting up training components...")
        
        # 构建模型
        model = self.build_model(vocab_size)
        
        # 设置优化器
        learning_rate = self.config_manager.get("training.learning_rate")
        weight_decay = self.config_manager.get("training.weight_decay")
        
        optimizer = nn.Adam(
            params=model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # 设置损失函数 - 使用 CTCLoss 替代 SeqKD
        from mindspore.nn import CTCLoss
        blank_id = self.config_manager.get("loss.ctc_blank_id", 0)
        reduction = self.config_manager.get("loss.ctc_reduction", "mean")
        loss_fn = CTCLoss(blank=blank_id, reduction=reduction)
        
        self.logger.info("Training components setup completed")
        
        return model, optimizer, loss_fn

    def train(self):
        """带有GPU优化的主训练循环"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING GPU-OPTIMIZED TRAINING")
            self.logger.info("=" * 60)
            
            # 准备数据
            vocab_size = self.prepare_data()
            
            # 设置训练组件
            model, optimizer, loss_fn = self.setup_training(vocab_size)
            
            # 训练参数
            num_epochs = self.config_manager.get("training.num_epochs")
            save_interval = self.config_manager.get("training.save_interval")
            eval_interval = self.config_manager.get("training.eval_interval")
            gradient_clip_norm = self.config_manager.get("training.gradient_clip_norm")
            
            self.logger.info(f"Training for {num_epochs} epochs")
            self.logger.info(f"Save interval: {save_interval}, Eval interval: {eval_interval}")
            
            # 设置回调
            callbacks = self._setup_callbacks()
            
            # 训练循环
            start_time = time.time()
            
            for epoch in range(num_epochs):
                self.current_epoch = epoch + 1
                epoch_start = time.time()
                
                self.logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
                
                # 训练步骤
                train_loss = self._train_epoch(model, optimizer, loss_fn, gradient_clip_norm)
                
                # 验证步骤
                if self.current_epoch % eval_interval == 0:
                    val_wer = self._validate_epoch(model)
                    
                    # 保存最佳模型
                    if val_wer < self.best_wer:
                        self.best_wer = val_wer
                        self.best_epoch = self.current_epoch
                        self._save_best_model(model)
                
                # 保存检查点
                if self.current_epoch % save_interval == 0:
                    self._save_checkpoint(model, optimizer)
                
                epoch_time = time.time() - epoch_start
                self.logger.info(f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s")
                
                # 提前停止检查
                if self._should_early_stop():
                    self.logger.info("Early stopping triggered")
                    break
            
            total_time = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info(f"TRAINING COMPLETED in {total_time:.2f}s")
            self.logger.info(f"Best WER: {self.best_wer:.4f} at epoch {self.best_epoch}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            print_error_details(e, "Training")
            raise

    def _train_epoch(self, model, optimizer, loss_fn, gradient_clip_norm):
        """执行一个训练周期并进行GPU优化"""
        model.set_train(True)
        total_loss = 0.0
        batch_count = 0
        
        # 如果配置中启用，使用 data sink 模式
        enable_data_sink = self.config_manager.get("training.enable_data_sink", True)
        
        if enable_data_sink:
            self.logger.info("Using data sink mode for better GPU utilization")
        
        for batch_idx, (data, target, data_len, target_len) in enumerate(self.train_dataset):
            try:
                # 统一dtype与形状处理 (B, T, C, H, W) -> 模型自己处理
                if isinstance(target, np.ndarray):
                    target = Tensor(target, ms.int32)
                
                # target 形状: (B, S_max) MindSpore CTCLoss 要求 2-D
                if len(target.shape) == 1:
                    target = ops.expand_dims(target, 0)
                
                # 处理输入长度 data_len 与 标签长度 target_len -> Tensor[int32]
                if not isinstance(data_len, Tensor):
                    if isinstance(data_len, (list, tuple)):
                        data_len = Tensor([int(x) for x in data_len], ms.int32)
                    else:
                        data_len = Tensor([int(data_len)], ms.int32)
                else:
                    data_len = ops.cast(data_len, ms.int32)
                    if len(data_len.shape) == 0:
                        data_len = ops.expand_dims(data_len, 0)
                
                if not isinstance(target_len, Tensor):
                    if isinstance(target_len, (list, tuple)):
                        target_len = Tensor([int(x) for x in target_len], ms.int32)
                    else:
                        target_len = Tensor([int(target_len)], ms.int32)
                else:
                    target_len = ops.cast(target_len, ms.int32)
                    if len(target_len.shape) == 0:
                        target_len = ops.expand_dims(target_len, 0)
                
                # CTC 约束: input_length >= target_length >= 1
                min_one = Tensor(1, ms.int32)
                target_len = ops.maximum(target_len, min_one)
                data_len = ops.maximum(data_len, target_len)  # 确保 input_len >= target_len
                
                # 前向计算
                def forward_fn(seq_data, seq_label, data_len_tensor, label_len_tensor):
                    model_output = model(seq_data, data_len_tensor, is_train=True)
                    logits = model_output[0]
                    
                    # logits 期望 shape (T, N, C)，若模型输出不同需转换
                    # 若 logits.shape == (N, T, C) 则转置
                    if logits.ndim == 3 and logits.shape[0] == data_len_tensor.shape[0]:
                        # 可能是 (N, T, C) -> (T, N, C)
                        logits = ops.transpose(logits, (1, 0, 2))
                    
                    actual_time_steps = logits.shape[0]
                    ts_tensor = Tensor(actual_time_steps, ms.int32)
                    one_tensor = Tensor(1, ms.int32)
                    data_len_tensor = ops.minimum(data_len_tensor, ts_tensor)
                    data_len_tensor = ops.maximum(data_len_tensor, one_tensor)
                    
                    # 再次保证 input_length >= target_length
                    data_len_tensor = ops.maximum(data_len_tensor, label_len_tensor)
                    
                    loss = loss_fn(logits, seq_label, data_len_tensor, label_len_tensor)
                    return loss
                
                grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
                loss, grads = grad_fn(data, target, data_len, target_len)
                
                if gradient_clip_norm > 0:
                    grads = ops.clip_by_global_norm(grads, gradient_clip_norm)
                optimizer(grads)
                
                total_loss += loss.asnumpy()
                batch_count += 1
                if batch_idx % self.config_manager.get("logging.print_interval") == 0:
                    self.logger.info(f"Batch {batch_idx}, Loss: {loss.asnumpy():.6f}")
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(batch_count, 1)
        self.logger.info(f"Training loss: {avg_loss:.6f}")
        return avg_loss

    def _validate_epoch(self, model):
        """执行一个验证周期"""
        model.set_train(False)
        total_wer = 0.0
        batch_count = 0
        
        for batch_idx, (data, target, data_len, target_len) in enumerate(self.valid_dataset):
            try:
                # 前向推断
                logits = model(data, data_len, is_train=False)
                
                # 解码预测结果
                predictions = self.decoder.decode(logits.asnumpy(), data_len.asnumpy())
                references = self.decoder.decode_labels(target.asnumpy(), target_len.asnumpy())
                
                # 计算 WER
                batch_wer = calculate_wer_score(predictions, references)
                total_wer += batch_wer
                batch_count += 1
                
            except Exception as e:
                self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue
        
        avg_wer = total_wer / max(batch_count, 1)
        self.logger.info(f"Validation WER: {avg_wer:.4f}")
        return avg_wer

    def _setup_callbacks(self):
        """设置训练回调"""
        callbacks = []
        
        # 损失监控器
        callbacks.append(LossMonitor(per_print_times=self.config_manager.get("logging.print_interval")))
        
        # 时间监控器
        callbacks.append(TimeMonitor())
        
        # 检查点回调
        checkpoint_config = CheckpointConfig(
            save_checkpoint_steps=self.config_manager.get("training.save_interval"),
            keep_checkpoint_max=10
        )
        checkpoint_dir = self.config_manager.get("paths.checkpoint_dir")
        checkpoint_callback = ModelCheckpoint(
            prefix="tfnet_gpu",
            directory=checkpoint_dir,
            config=checkpoint_config
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks

    def _save_best_model(self, model):
        """保存最佳模型"""
        try:
            best_model_path = self.config_manager.get("paths.best_model_path")
            save_checkpoint(model, best_model_path)
            self.logger.info(f"Best model saved: {best_model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save best model: {e}")

    def _save_checkpoint(self, model, optimizer):
        """保存训练检查点"""
        try:
            checkpoint_path = self.config_manager.get("paths.current_model_path")
            save_checkpoint(model, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _should_early_stop(self):
        """检查是否应该触发早停"""
        patience = self.config_manager.get("training.early_stopping_patience")
        return (self.current_epoch - self.best_epoch) >= patience

def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-Optimized TFNet Training")
    parser.add_argument("--config", type=str, default="configs/gpu_config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # 验证环境
    print("Checking environment...")
    print(f"Current conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    
    if 'mindspore-gpu' not in os.environ.get('CONDA_DEFAULT_ENV', ''):
        print("Warning: Not in mindspore-gpu environment")
        print("Please run: conda activate mindspore-gpu")
    
    try:
        # 创建训练器
        trainer = GPUTFNetTrainer(args.config)
        
        # 开始训练
        trainer.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print_error_details(e, "Main training")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
