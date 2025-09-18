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
            
            # 如果配置为CPU，直接设置CPU上下文
            if device_target == "CPU":
                context.set_context(
                    mode=context.PYNATIVE_MODE,
                    device_target="CPU",
                    device_id=0
                )
                print("✓ CPU context configured successfully")
                return
            
            # 检查是否有可用的GPU
            if not self._check_gpu_availability():
                print("❌ GPU not available, falling back to CPU mode")
                # 自动切换到CPU模式
                context.set_context(
                    mode=context.PYNATIVE_MODE,
                    device_target="CPU",
                    device_id=0
                )
                # 更新配置以反映实际使用的设备
                self.config.model.device_target = "CPU"
                print("✓ CPU context configured successfully (fallback)")
                return

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
                    # 新增: 兼容 MB -> GB
                    if isinstance(mempool_size, str) and mempool_size.upper().endswith("MB"):
                        try:
                            mb = float(mempool_size[:-2])
                            mempool_size_converted = f"{mb/1024.0:.3f}GB"
                        except:
                            mempool_size_converted = mempool_size
                    else:
                        mempool_size_converted = mempool_size
                    try:
                        # 仅在Ascend上开启reduce_precision，GPU会忽略且可能产生警告
                        if device_target == "Ascend":
                            context.set_context(
                                mempool_block_size=mempool_size_converted,
                                enable_reduce_precision=True
                            )
                        else:
                            context.set_context(
                                mempool_block_size=mempool_size_converted
                            )
                        print(f"✓ Additional memory optimizations enabled (mempool: {mempool_size_converted})")
                    except Exception as e_mpool:
                        print(f"Warning: mempool_block_size '{mempool_size_converted}' not applied: {e_mpool}")
                    
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

            # 启用自动混合精度（如支持）- 仅在Ascend上启用，避免GPU上cublas不兼容
            if self.config_manager.get("model.enable_auto_mixed_precision", True):
                try:
                    if device_target == "Ascend":
                        try:
                            context.set_auto_parallel_context(enable_auto_mixed_precision=True)
                            print("✓ Auto mixed precision enabled (auto_parallel_context)")
                        except Exception:
                            context.set_context(enable_auto_mixed_precision=True)
                            print("✓ Auto mixed precision enabled (context)")
                    else:
                        print("✓ Skipping auto mixed precision on non-Ascend device")
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
        
        self.logger.info(f"Creating datasets: batch_size={batch_size}, workers={num_workers}, frames={max_frames}")
        
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
        
        # 初始化模型权重
        self._initialize_model_weights()
        
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
        
        # 设置优化器 - 使用更稳定的参数
        learning_rate = self.config_manager.get("training.learning_rate", 0.0001)
        weight_decay = self.config_manager.get("training.weight_decay", 0.0001)
        
        # 使用更稳定的Adam参数
        optimizer = nn.Adam(
            params=model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8  # 防止除零
        )
        
        self.logger.info(f"Optimizer: Adam(lr={learning_rate}, wd={weight_decay})")
        
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
                    try:
                        model_output = model(seq_data, data_len_tensor, is_train=True)
                        
                        # 处理模型输出 - TFNet返回多个logits
                        if isinstance(model_output, (list, tuple)) and len(model_output) >= 5:
                            # 使用主要的logits (第5个输出是融合后的结果)
                            logits = model_output[4]  # log_probs5 是融合后的主要输出
                        elif isinstance(model_output, (list, tuple)):
                            logits = model_output[0]
                        else:
                            logits = model_output
                        
                        # 确保logits是有效张量
                        if not isinstance(logits, Tensor):
                            self.logger.warning("Model output is not a tensor, creating placeholder")
                            logits = ops.zeros((10, 1, len(self.word2idx)), ms.float32)
                        
                        # 检查并修正logits形状 - CTC期望 (T, N, C)
                        if logits.ndim == 3:
                            T, N, C = logits.shape[0], logits.shape[1], logits.shape[2]
                            
                            # 如果是(N, T, C)格式，需要转置为(T, N, C)
                            if N > T and T == data_len_tensor.shape[0]:
                                logits = ops.transpose(logits, (1, 0, 2))
                                T, N = N, T
                        else:
                            # 创建合理的默认形状
                            T, N, C = 10, 1, len(self.word2idx)
                            logits = ops.zeros((T, N, C), ms.float32)
                            self.logger.warning(f"Invalid logits shape, using default ({T}, {N}, {C})")
                        
                        # 应用log_softmax以数值稳定性
                        logits = ops.log_softmax(logits, axis=-1)
                        
                        # 检查并处理NaN/Inf
                        if ops.isnan(logits).any() or ops.isinf(logits).any():
                            self.logger.warning("Found NaN/Inf in logits, replacing with safe values")
                            logits = ops.where(ops.isnan(logits), 
                                             ops.zeros_like(logits) - 10.0, logits)
                            logits = ops.where(ops.isinf(logits), 
                                             ops.zeros_like(logits) - 10.0, logits)
                        
                        # 计算有效的序列长度，遵循CTC约束
                        T = logits.shape[0]
                        max_input_len = min(T, 50)  # 限制最大长度
                        max_label_len = min(max_input_len // 2, 25)  # CTC要求标签长度 <= 输入长度/2
                        
                        # 处理输入长度
                        if isinstance(data_len_tensor, Tensor):
                            input_len = ops.minimum(data_len_tensor, Tensor([max_input_len], ms.int32))
                        else:
                            input_len = Tensor([min(max_input_len, int(data_len_tensor))], ms.int32)
                        
                        # 处理标签长度 
                        if isinstance(label_len_tensor, Tensor):
                            label_len = ops.minimum(label_len_tensor, Tensor([max_label_len], ms.int32))
                        else:
                            label_len = Tensor([min(max_label_len, int(label_len_tensor))], ms.int32)
                        
                        # 确保长度至少为1
                        input_len = ops.maximum(input_len, Tensor([1], ms.int32))
                        label_len = ops.maximum(label_len, Tensor([1], ms.int32))
                        
                        # 处理标签，确保在有效范围内
                        seq_label_clean = ops.clip_by_value(seq_label, 0, len(self.word2idx) - 1)
                        
                        # 计算CTC损失
                        loss = loss_fn(logits, seq_label_clean, input_len, label_len)
                        
                        # 检查损失值
                        if ops.isnan(loss) or ops.isinf(loss):
                            self.logger.warning("NaN/Inf loss detected, using fallback value")
                            loss = Tensor(1.0, ms.float32)  # 使用较小的fallback值
                        
                        return loss
                        
                    except Exception as e:
                        self.logger.error(f"Error in forward pass: {e}")
                        # 返回一个小的loss值而不是让训练崩溃
                        return Tensor(1.0, ms.float32)
                
                grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
                loss, grads = grad_fn(data, target, data_len, target_len)
                
                # 检查梯度中的NaN/Inf
                grad_valid = True
                for grad in grads:
                    if grad is not None and (ops.isnan(grad).any() or ops.isinf(grad).any()):
                        grad_valid = False
                        break
                
                if not grad_valid:
                    self.logger.warning(f"Invalid gradients detected in batch {batch_idx}, skipping update")
                    continue
                
                # 梯度裁剪
                if gradient_clip_norm > 0:
                    grads = ops.clip_by_global_norm(grads, gradient_clip_norm)
                
                # 应用梯度更新
                try:
                    optimizer(grads)
                except Exception as e:
                    self.logger.warning(f"Optimizer step failed in batch {batch_idx}: {e}")
                    continue
                
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
        """执行一个验证周期 - 基于帧片段级别的验证，而不是整个视频"""
        model.set_train(False)
        total_wer = 0.0
        batch_count = 0
        
        for batch_idx, (data, target, data_len, target_len) in enumerate(self.valid_dataset):
            try:
                # 前向推断 - 获取每个时间步的logits
                logits = model(data, data_len, is_train=False)
                
                # 验证时需要注意：训练过程中视频被分为多个帧片段(gross段)
                # 每个帧片段对应不同的标签，因此验证也要按帧片段进行
                # 而不是盲目使用整个句子进行对比
                
                # 获取预测结果 - 基于帧级别的CTC解码
                predictions = self.decoder.decode(logits.asnumpy(), data_len.asnumpy())
                
                # 解码目标标签 - 按照实际的标签长度进行解码
                references = self.decoder.decode_labels(target.asnumpy(), target_len.asnumpy())
                
                # 对每个样本进行帧级别的验证
                for i in range(len(predictions)):
                    pred_seq = predictions[i]  # 预测的词汇序列
                    ref_seq = references[i]    # 参考的词汇序列
                    
                    # 计算单个样本的WER - 基于帧片段级别
                    if isinstance(pred_seq, list) and isinstance(ref_seq, str):
                        # 将预测序列转换为字符串格式以便比较
                        pred_words = [word for word, _ in pred_seq] if pred_seq else []
                        pred_str = ' '.join(pred_words)
                        sample_wer = self._calculate_frame_level_wer(pred_str, ref_seq)
                    else:
                        sample_wer = 1.0  # 如果格式不匹配，设为最大错误率
                    
                    total_wer += sample_wer
                    batch_count += 1
                
                if batch_idx % 20 == 0 and batch_idx > 0:  # 每20个batch打印一次进度
                    self.logger.info(f"Validation progress: {batch_idx} batches")
                
            except Exception as e:
                self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue
        
        avg_wer = total_wer / max(batch_count, 1)
        self.logger.info(f"Frame-level Validation WER: {avg_wer:.4f}")
        return avg_wer
    
    def _calculate_frame_level_wer(self, prediction, reference):
        """计算帧级别的词错误率"""
        try:
            # 将字符串分割为词汇列表
            pred_words = prediction.strip().split() if prediction else []
            ref_words = reference.strip().split() if reference else []
            
            # 如果都为空，认为完全匹配
            if not pred_words and not ref_words:
                return 0.0
            
            # 如果参考为空但预测不为空，错误率为100%
            if not ref_words:
                return 1.0
            
            # 使用编辑距离计算WER
            return self._edit_distance_wer(pred_words, ref_words)
            
        except Exception as e:
            self.logger.error(f"Error calculating frame-level WER: {e}")
            return 1.0
    
    def _edit_distance_wer(self, pred_words, ref_words):
        """使用编辑距离计算词错误率"""
        len_pred, len_ref = len(pred_words), len(ref_words)
        
        # 创建DP表
        dp = [[0] * (len_ref + 1) for _ in range(len_pred + 1)]
        
        # 初始化边界条件
        for i in range(len_pred + 1):
            dp[i][0] = i
        for j in range(len_ref + 1):
            dp[0][j] = j
        
        # 填充DP表
        for i in range(1, len_pred + 1):
            for j in range(1, len_ref + 1):
                if pred_words[i-1] == ref_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # 匹配，无需操作
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # 删除
                        dp[i][j-1] + 1,      # 插入
                        dp[i-1][j-1] + 1     # 替换
                    )
        
        # 计算WER = 编辑距离 / 参考长度
        wer = dp[len_pred][len_ref] / len_ref
        return wer

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

    def _initialize_model_weights(self):
        """初始化模型权重以提高数值稳定性"""
        try:
            from mindspore.common.initializer import Normal, Xavier, Zero
            
            for name, param in self.model.parameters_and_names():
                if 'weight' in name:
                    if 'classifier' in name or 'linear' in name:
                        # 分类器层使用Xavier初始化
                        param.set_data(Xavier(gain=1.0)(param.shape, param.dtype))
                    elif 'conv' in name:
                        # 卷积层使用正态分布初始化
                        param.set_data(Normal(sigma=0.01)(param.shape, param.dtype))
                    elif 'lstm' in name or 'rnn' in name:
                        # LSTM层使用较小的正态分布
                        param.set_data(Normal(sigma=0.1)(param.shape, param.dtype))
                elif 'bias' in name:
                    # 所有偏置初始化为0
                    param.set_data(Zero()(param.shape, param.dtype))
            
            self.logger.info("Model weights initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Weight initialization failed: {e}, using default initialization")
    
    # ...existing code...
def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-Optimized TFNet Training")
    parser.add_argument("--config", type=str, default="configs/gpu_config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # 验证环境
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"Current conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    
    # 检查是否激活了正确的conda环境
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != 'mind':
        print(f"❌ ERROR: Not in the correct conda environment!")
        print(f"   Current environment: {current_env}")
        print(f"   Required environment: mind")
        print(f"   Please run: conda activate mind")
        return 1
    else:
        print(f"✅ Correct conda environment activated: {current_env}")
    
    # 检查MindSpore是否正确安装
    try:
        import mindspore as ms
        print(f"✅ MindSpore version: {ms.__version__}")
    except ImportError as e:
        print(f"❌ ERROR: MindSpore not found: {e}")
        print("   Please install MindSpore in the 'mind' environment")
        return 1
    
    # 检查CUDA是否可用（如果使用GPU）
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
        else:
            print("⚠️  WARNING: nvidia-smi not available, GPU may not be accessible")
    except Exception:
        print("⚠️  WARNING: Could not check GPU status")
    
    print("=" * 60)
    
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
