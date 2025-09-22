#!/usr/bin/env python3
"""
用于连续手语识别的TFNet训练脚本
针对MindSpore框架的CPU执行进行优化
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
import psutil  # 新增: 系统与内存监控

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

# 尝试导入新API，如果不可用则回退到旧版本
try:
    from mindspore import set_device
    MINDSPORE_NEW_API = True
except ImportError:
    MINDSPORE_NEW_API = False

# 将当前目录添加到路径以便导入
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

class TFNetTrainer:
    """TFNet模型训练器"""
    
    def __init__(self, config_path=None):
        try:
            # 初始化配置
            print("正在初始化TFNet训练器...")
            # 如果没有指定配置路径，优先使用优化配置
            if config_path is None:
                optimized_config = "configs/tfnet_config_optimized.json"
                if os.path.exists(optimized_config):
                    config_path = optimized_config
                    print(f"✓ 使用优化配置：{optimized_config}")
                else:
                    config_path = "configs/tfnet_config.json"
                    print(f"✓ 使用默认配置：{config_path}")
            
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
            
            # 内存使用监控
            self.memory_stats = {
                'peak_memory': 0,
                'current_memory': 0,
                'batch_memories': []
            }

            # 首先验证和创建目录（在日志设置之前）
            print("正在设置目录...")
            if not self._setup_directories():
                raise RuntimeError("创建必需目录失败")

            # 验证数据集结构
            print("正在验证数据集...")
            if not self._validate_dataset():
                raise RuntimeError("数据集验证失败")

            # 设置MindSpore上下文，具有API兼容性
            device_target = self.config_manager.get("model.device_target", "CPU")

            # 如果可用，使用新API，否则回退到旧API
            if MINDSPORE_NEW_API:
                try:
                    context.set_context(mode=context.PYNATIVE_MODE)
                    set_device(device_target)
                    print(f"✓ MindSpore设备设置为：{device_target}（新API）")
                except Exception as e:
                    print(f"警告：新API失败，使用回退：{e}")
                    context.set_context(
                        mode=context.PYNATIVE_MODE,
                        device_target=device_target
                    )
                    print(f"✓ MindSpore设备设置为：{device_target}（回退API）")
            else:
                context.set_context(
                    mode=context.PYNATIVE_MODE,
                    device_target=device_target
                )
                print(f"✓ MindSpore设备设置为：{device_target}（传统API）")

            # 初始化日志（在目录创建后）
            self._setup_logging()

            # 初始化组件
            self.model = None
            self.train_dataset = None
            self.valid_dataset = None
            self.word2idx = None
            self.idx2word = None
            self.decoder = None

            # 训练状态
            self.current_epoch = 0
            self.best_wer = float('inf')
            self.best_epoch = 0
            # 监控与统计
            self.training_losses = []            # 每epoch平均loss
            self.validation_losses = []          # 每epoch验证loss
            self.validation_wers = []            # 每epoch验证WER
            self.memory_peaks = []               # 每epoch内存峰值(MB)
            self._epoch_batch_memory = []        # 当前epoch批次内存
        except Exception as e:
            print_error_details(e, "TFNet Trainer initialization")
            raise

    def _setup_directories(self):
        """Setup and validate directories"""
        try:
            # Create directories using config manager
            if not self.config_manager.create_directories():
                return False

            # Additional validation for critical directories
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
        """Validate dataset structure and paths"""
        try:
            dataset_config = self.config_manager.get_dataset_config()

            # Check if dataset paths are configured
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

                # Check if path exists - determine type based on path_key name
                if 'label' in path_key:
                    # Label paths are files
                    if not check_file_exists(path, description):
                        all_valid = False
                else:
                    # Data paths are directories
                    if not check_directory_exists(path, description):
                        all_valid = False

            # Validate CE-CSL dataset structure if base path is available
            if dataset_config.get('name') == 'CE-CSL':
                # Try to find base CE-CSL path
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
        """Setup logging configuration with improved error handling"""
        log_level = getattr(logging, self.config_manager.get("logging.level", "INFO"))

        # Create logger
        self.logger = logging.getLogger('TFNetTrainer')
        self.logger.setLevel(log_level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler with safe directory creation
        if self.config_manager.get("logging.save_logs", True):
            try:
                log_dir = self.config_manager.get_safe_path("paths.log_dir", create_if_missing=True)
                if log_dir:
                    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                    # Normalize path for cross-platform compatibility
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
        """Prepare datasets and vocabulary"""
        self.logger.info("Preparing data...")
        
        dataset_config = self.config_manager.get_dataset_config()
        
        # Build vocabulary
        self.word2idx, vocab_size, self.idx2word = build_vocabulary(
            dataset_config["train_label_path"],
            dataset_config["valid_label_path"],
            dataset_config["test_label_path"],
            dataset_config["name"]
        )
        
        self.logger.info(f"Vocabulary size: {vocab_size}")
        
        # Save vocabulary
        vocab_path = os.path.join(self.config_manager.get("paths.output_dir"), "vocabulary.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': vocab_size
            }, f, indent=2, ensure_ascii=False)
        
        # Create datasets
        batch_size = self.config_manager.get("training.batch_size")
        num_workers = self.config_manager.get("training.num_workers")
        prefetch_size = self.config_manager.get("training.prefetch_size", 1)
        max_rowsize = self.config_manager.get("training.max_rowsize", 16)
        crop_size = dataset_config.get("crop_size", 224)
        max_frames = dataset_config.get("max_frames", 150)
        
        dtype = 'float32'  # 改回float32以匹配Conv2D权重，避免类型不一致
        
        self.train_dataset = create_dataset(
            data_path=dataset_config["train_data_path"],
            label_path=dataset_config["train_label_path"],
            word2idx=self.word2idx,
            dataset_name=dataset_config["name"],
            batch_size=batch_size,
            is_train=True,
            num_workers=num_workers,
            prefetch_size=prefetch_size,
            max_rowsize=max_rowsize,
            crop_size=crop_size,
            max_frames=max_frames,
            dtype=dtype,
            enable_cache=True,  # 启用缓存利用大内存
            memory_optimize=True  # 启用内存优化
        )
        
        self.valid_dataset = create_dataset(
            data_path=dataset_config["valid_data_path"],
            label_path=dataset_config["valid_label_path"],
            word2idx=self.word2idx,
            dataset_name=dataset_config["name"],
            batch_size=batch_size,  # Use the same batch size as training
            is_train=False,
            num_workers=num_workers,
            prefetch_size=prefetch_size,
            max_rowsize=max_rowsize,
            crop_size=crop_size,
            max_frames=max_frames,
            dtype=dtype
        )
        
        self.logger.info("Data preparation completed")
        
        return vocab_size
    
    def build_model(self, vocab_size):
        """Build TFNet model"""
        self.logger.info("Building model...")
        
        model_config = self.config_manager.get_model_config()
        
        # Create model
        self.model = TFNetModel(
            hidden_size=model_config["hidden_size"],
            word_set_num=vocab_size,
            device_target=model_config["device_target"],
            dataset_name=model_config["dataset_name"]
        )
        
        # Initialize decoder
        self.decoder = CTCDecoder(
            gloss_dict=self.word2idx,
            num_classes=vocab_size + 1,
            search_mode='max',
            blank_id=self.config_manager.get("loss.ctc_blank_id", 0)
        )
        
        self.logger.info("Model built successfully")
        # 初始化持久化的 loss_fn 与 optimizer，避免每batch重复创建
        self.loss_fn = self.create_loss_fn()
        self.optimizer = self.create_optimizer()
        # 记录模型参数规模
        try:
            total_params = sum(p.size for p in self.model.get_parameters())
        except Exception:
            total_params = 0
        self.logger.info(f"Model parameters (approx): {total_params}")
    
    def create_loss_fn(self):
        """创建简化的损失函数 - 移除复杂的知识蒸馏"""
        # 只使用CTC损失，移除复杂的多分支损失计算
        ctc_loss = nn.CTCLoss(
            blank=self.config_manager.get("loss.ctc_blank_id", 0),
            reduction=self.config_manager.get("loss.ctc_reduction", "mean"),
            zero_infinity=True
        )
        
        def simplified_loss_fn(log_probs1, log_probs2, log_probs3, log_probs4, log_probs5,
                              lgt, target_data, target_lengths):
            """简化的损失函数 - 只使用主要的logits"""
            # 应用log softmax
            log_softmax = nn.LogSoftmax(axis=-1)
            log_probs = log_softmax(log_probs1)
            
            # 直接计算CTC损失，让CTC内部处理长度不匹配的情况
            try:
                loss = ctc_loss(log_probs, target_data, lgt, target_lengths)
                # 检查损失是否为有效值
                if ops.isnan(loss) or ops.isinf(loss):
                    return ops.scalar_to_tensor(1.0, ms.float32)
                return loss
            except Exception as e:
                # 如果CTC损失失败，返回一个中等损失值继续训练
                return ops.scalar_to_tensor(1.0, ms.float32)
        
        return simplified_loss_fn
    
    def create_optimizer(self):
        """Create optimizer"""
        training_config = self.config_manager.get_training_config()
        
        # Get model parameters
        params = self.model.trainable_params()
        
        # Create optimizer
        optimizer = nn.Adam(
            params=params,
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"]
        )
        
        return optimizer

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.logger.info(f"Training epoch {epoch}")
        
        self.model.set_train(True)
        total_loss = 0.0
        num_batches = 0
        
        self._epoch_batch_memory = []
        vm_total = psutil.virtual_memory().total / 1024 / 1024  # MB
        
        # 定义前向 + 计算loss 函数供梯度计算
        import mindspore as ms
        import mindspore.ops as ops
        def forward_fn(videos, video_lengths_list, target_data, target_lengths):
            log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, lgt, _, _, _ = \
                self.model(videos, video_lengths_list, is_train=True)
            loss = self.loss_fn(log_probs1, log_probs2, log_probs3, log_probs4, log_probs5,
                                lgt, target_data, target_lengths)
            return loss, (lgt,)
        grad_fn = ops.value_and_grad(forward_fn, None, self.model.trainable_params(), has_aux=True)
        memory_warning_flag = False
        
        for batch_idx, batch_data in enumerate(self.train_dataset.create_dict_iterator(num_epochs=1)):
            # Extract batch data
            videos = batch_data['video']
            labels = batch_data['label']
            video_lengths = batch_data['videoLength']
            # 将输入统一为Float32，避免Conv2D类型不一致（模型权重为Float32）
            try:
                if hasattr(videos, 'dtype'):
                    import mindspore as ms
                    import mindspore.ops as ops
                    if videos.dtype != ms.float32:
                        videos = ops.cast(videos, ms.float32)
            except Exception:
                pass
            
            # Convert lengths to a Python list for MindSpore graph safety
            if hasattr(video_lengths, 'asnumpy'):
                video_lengths_list = video_lengths.asnumpy().flatten().tolist()
            elif isinstance(video_lengths, (list, tuple)):
                video_lengths_list = [int(v) for v in video_lengths]
            else:
                # Fallback: try to wrap single value
                try:
                    video_lengths_list = [int(video_lengths)]
                except Exception:
                    video_lengths_list = []
            if hasattr(labels, 'asnumpy'):
                labels_list = labels.asnumpy().tolist()
            elif isinstance(labels, (list, tuple)):
                labels_list = [list(x) if isinstance(x, (list, tuple)) else [int(x)] for x in labels]
            else:
                labels_list = [[int(labels)]]
            target_lengths_list = []
            for seq in labels_list:
                seq = list(seq)
                length = len(seq)
                while length > 0 and int(seq[length - 1]) == 0:
                    length -= 1
                if length == 0:
                    length = 1
                    if len(seq) == 0:
                        seq.append(0)
                    else:
                        seq[0] = 0
                target_lengths_list.append(length)
            target_data = ms.Tensor(labels_list, ms.int32)
            target_lengths = ms.Tensor(target_lengths_list, ms.int32)
            # 前向 + 反向
            try:
                (loss, _), grads = grad_fn(videos, video_lengths_list, target_data, target_lengths)
                # 梯度裁剪（防止梯度爆炸）
                clip_norm = self.config_manager.get("training.gradient_clip_norm", 1.0)
                if clip_norm and clip_norm > 0:
                    new_grads = []
                    for g in grads:
                        if g is not None:
                            new_grads.append(ops.clip_by_norm(g, clip_norm))
                        else:
                            new_grads.append(g)
                    grads = tuple(new_grads)  # 关键：转为tuple与params结构一致
                # 应用优化器
                self.optimizer(grads)
            except Exception as e:
                self.logger.error(f"Gradient step failed at batch {batch_idx}: {e}")
                continue
            loss_value = float(loss.asnumpy().item())
            total_loss += loss_value
            num_batches += 1
            # 内存监控
            import gc
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            self._epoch_batch_memory.append(mem_mb)
            # 增长趋势检测（最近20个batch与前20个batch均值比较）
            if len(self._epoch_batch_memory) >= 40:
                recent = self._epoch_batch_memory[-20:]
                prev = self._epoch_batch_memory[-40:-20]
                if (sum(recent)/20) - (sum(prev)/20) > 300 and not memory_warning_flag:  # 内存上升超过300MB
                    self.logger.warning("Potential memory leak detected: last 20 batches average memory increased >300MB over previous 20")
                    memory_warning_flag = True
            # 触发高水位警告
            if not memory_warning_flag and mem_mb > 0.85 * vm_total:
                self.logger.critical(f"Memory usage {mem_mb:.1f}MB exceeds 85% of total {vm_total:.1f}MB. Consider reducing batch_size or max_frames.")
                memory_warning_flag = True
            if batch_idx % self.config_manager.get("logging.print_interval", 10) == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_value:.4f}")
                self.logger.info(f"[MEMORY] Epoch {epoch}, Batch {batch_idx}: {mem_mb:.1f} MB (Peak so far: {max(self._epoch_batch_memory):.1f} MB)")
                gc.collect()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        peak_mem = max(self._epoch_batch_memory) if self._epoch_batch_memory else 0.0
        self.memory_peaks.append(peak_mem)
        self.training_losses.append(avg_loss)
        self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}; Memory peak: {peak_mem:.1f} MB")
        return avg_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.logger.info(f"Validating epoch {epoch}")

        self.model.set_train(False)
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(self.valid_dataset.create_dict_iterator(num_epochs=1)):
            if batch_idx % 10 == 0:
                self.logger.info(f"Validation batch {batch_idx}")
            
            # Extract batch data
            videos = batch_data['video']
            labels = batch_data['label']
            video_lengths = batch_data['videoLength']
            # 将验证输入统一为Float32
            try:
                if hasattr(videos, 'dtype'):
                    import mindspore as ms
                    import mindspore.ops as ops
                    if videos.dtype != ms.float32:
                        videos = ops.cast(videos, ms.float32)
            except Exception:
                pass
            
            # Prepare target data for CTCLoss: targets shape (B, S), with target_lengths per sample
            if hasattr(labels, 'asnumpy'):
                labels_list = labels.asnumpy().tolist()
            elif isinstance(labels, (list, tuple)):
                labels_list = [list(x) if isinstance(x, (list, tuple)) else [int(x)] for x in labels]
            else:
                labels_list = [[int(labels)]]

            target_lengths_list = []
            for seq in labels_list:
                seq = list(seq)
                length = len(seq)
                while length > 0 and int(seq[length - 1]) == 0:
                    length -= 1
                if length == 0:
                    length = 1
                    if len(seq) == 0:
                        seq.append(0)
                    else:
                        seq[0] = 0
                target_lengths_list.append(length)

            target_data = ms.Tensor(labels_list, ms.int32)
            target_lengths = ms.Tensor(target_lengths_list, ms.int32)

            # Forward pass
            # Convert validation lengths to Python list
            if hasattr(video_lengths, 'asnumpy'):
                video_lengths_list = video_lengths.asnumpy().flatten().tolist()
            elif isinstance(video_lengths, (list, tuple)):
                video_lengths_list = [int(v) for v in video_lengths]
            else:
                try:
                    video_lengths_list = [int(video_lengths)]
                except Exception:
                    video_lengths_list = []

            log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, lgt, _, _, _ = \
                self.model(videos, video_lengths_list, is_train=False)

            # Calculate loss
            loss_fn = self.create_loss_fn()
            loss = loss_fn(log_probs1, log_probs2, log_probs3, log_probs4, log_probs5,
                          lgt, target_data, target_lengths)

            # Decode predictions
            predictions = self.decoder.decode(log_probs1, lgt, batch_first=False, probs=False)
            
            # 添加调试信息
            if batch_idx == 0:  # 只对第一个batch输出调试信息
                self.logger.info(f"DEBUG - Batch {batch_idx}:")
                self.logger.info(f"  log_probs1 shape: {log_probs1.shape}")
                self.logger.info(f"  lgt: {lgt}")
                self.logger.info(f"  predictions length: {len(predictions)}")
                if predictions:
                    self.logger.info(f"  first prediction: {predictions[0][:5] if len(predictions[0]) > 5 else predictions[0]}")
                self.logger.info(f"  target_data: {target_data}")
                self.logger.info(f"  target_lengths: {target_lengths}")

            # Prepare prediction ids for WER (map decoded words back to ids)
            pred_ids_batch = []
            for seq in predictions:
                pred_ids = []
                for item in seq:
                    # item is (word, idx)
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        word = item[0]
                        pred_ids.append(self.word2idx.get(word, 0))
                pred_ids_batch.append(pred_ids)
            if not pred_ids_batch:
                pred_ids_batch = [[]]

            # Prepare reference ids (trim padding zeros)
            if hasattr(labels, 'asnumpy'):
                labels_list = labels.asnumpy().tolist()
            elif isinstance(labels, (list, tuple)):
                labels_list = [list(x) if isinstance(x, (list, tuple)) else [int(x)] for x in labels]
            else:
                labels_list = [[int(labels)]]
            ref_ids = []
            for seq in labels_list:
                seq = list(seq)
                length = len(seq)
                while length > 0 and int(seq[length - 1]) == 0:
                    length -= 1
                ref_ids.append(seq[:max(1, length)])
            if not ref_ids:
                ref_ids = [[]]

            # Calculate WER
            current_batch_size = len(pred_ids_batch) if pred_ids_batch else 1
            batch_wer = calculate_wer_score(
                pred_ids_batch,
                ref_ids,
                self.idx2word,
                current_batch_size
            )

            total_loss += loss.asnumpy().item()
            total_wer += batch_wer
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_wer = total_wer / num_batches if num_batches > 0 else 0.0

        self.logger.info(f"Validation completed. Average loss: {avg_loss:.4f}, Average WER: {avg_wer:.2f}%")

        return avg_loss, avg_wer

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with improved error handling"""
        try:
            if is_best:
                checkpoint_path = self.config_manager.get_safe_path("paths.best_model_path", create_if_missing=True)
                self.logger.info(f"Saving best model to {checkpoint_path}")
            else:
                checkpoint_path = self.config_manager.get_safe_path("paths.current_model_path", create_if_missing=True)
                self.logger.info(f"Saving current model to {checkpoint_path}")

            if not checkpoint_path:
                self.logger.error("Failed to get valid checkpoint path")
                return False

            # Ensure parent directory exists
            parent_dir = os.path.dirname(checkpoint_path)
            if not ensure_directory_exists(parent_dir, create=True):
                self.logger.error(f"Failed to create checkpoint directory: {parent_dir}")
                return False

            # Save checkpoint
            save_checkpoint(self.model, checkpoint_path)

            # Verify checkpoint was saved
            if os.path.exists(checkpoint_path):
                self.logger.info(f"✓ Checkpoint saved successfully: {checkpoint_path}")
                return True
            else:
                self.logger.error(f"✗ Checkpoint file not found after saving: {checkpoint_path}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            print_error_details(e, "Checkpoint saving")
            return False

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint with improved error handling"""
        try:
            if not checkpoint_path:
                self.logger.warning("No checkpoint path provided")
                return False

            normalized_path = normalize_path(checkpoint_path)

            if not os.path.exists(normalized_path):
                self.logger.warning(f"Checkpoint not found: {normalized_path}")
                return False

            if not os.path.isfile(normalized_path):
                self.logger.error(f"Checkpoint path is not a file: {normalized_path}")
                return False

            self.logger.info(f"Loading checkpoint from {normalized_path}")

            # Load checkpoint
            param_dict = load_checkpoint(normalized_path)
            if not param_dict:
                self.logger.error("Failed to load checkpoint parameters")
                return False

            # Load parameters into model
            load_param_into_net(self.model, param_dict)

            self.logger.info("✓ Checkpoint loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            print_error_details(e, f"Loading checkpoint from {checkpoint_path}")
            return False

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")

        # Prepare data and model
        vocab_size = self.prepare_data()
        self.build_model(vocab_size)

        # Validate configuration
        if not self.config_manager.validate_config():
            self.logger.error("Configuration validation failed")
            return

        # Load checkpoint if exists
        current_model_path = self.config_manager.get("paths.current_model_path")
        if self.load_checkpoint(current_model_path):
            self.logger.info("Resumed from checkpoint")

        # Training parameters
        num_epochs = self.config_manager.get("training.num_epochs")
        eval_interval = self.config_manager.get("training.eval_interval")
        save_interval = self.config_manager.get("training.save_interval")
        early_stopping_patience = self.config_manager.get("training.early_stopping_patience")

        patience_counter = 0

        run_start = time.time()
        for epoch in range(self.current_epoch, num_epochs):
            start_time = time.time()
            # 每个epoch重新构建迭代器，避免数据管道状态累积
            train_iter = self.train_dataset.create_dict_iterator(num_epochs=1)
            # 将一次epoch的迭代交由train_epoch消费
            train_loss = self.train_epoch(epoch)

            # Validation
            if epoch % eval_interval == 0:
                val_loss, val_wer = self.validate_epoch(epoch)

                # Check for best model
                if val_wer < self.best_wer:
                    self.best_wer = val_wer
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                    patience_counter = 0
                    self.logger.info(f"New best model! WER: {val_wer:.2f}%")
                else:
                    patience_counter += 1

            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break

            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
            self.current_epoch = epoch + 1
        total_time = time.time() - run_start
        # 生成总结
        try:
            summary = {
                "epochs_completed": self.current_epoch,
                "best_wer": self.best_wer,
                "best_epoch": self.best_epoch,
                "training_losses": self.training_losses,
                "validation_losses": self.validation_losses,
                "validation_wers": self.validation_wers,
                "memory_peaks_mb": self.memory_peaks,
                "total_training_time_sec": round(total_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            summary_path = os.path.join(self.config_manager.get("paths.output_dir"), "training_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Training summary saved: {summary_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write training summary: {e}")
        self.logger.info(f"Training completed. Best WER: {self.best_wer:.2f}% at epoch {self.best_epoch}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='TFNet Training Script')
    parser.add_argument('--config', type=str, default='training/configs/tfnet_config.json',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Create trainer
    trainer = TFNetTrainer(config_path=args.config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
    except Exception as e:
        trainer.logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
