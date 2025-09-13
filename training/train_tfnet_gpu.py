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
        """Setup GPU context with optimizations"""
        try:
            device_target = self.config_manager.get("model.device_target", "GPU")
            device_id = self.config_manager.get("model.device_id", 0)
            
            # Check if GPU is available
            if not self._check_gpu_availability():
                raise RuntimeError("GPU not available or MindSpore GPU version not installed")

            # Set context for GPU with optimizations
            gpu_config = self.config_manager.get("gpu_optimization", {})
            
            if gpu_config.get("enable_graph_mode", True):
                mode = context.GRAPH_MODE
                print("✓ Using GRAPH_MODE for better GPU performance")
            else:
                mode = context.PYNATIVE_MODE
                print("✓ Using PYNATIVE_MODE for debugging")

            # Configure context
            context.set_context(
                mode=mode,
                device_target=device_target,
                device_id=device_id,
                save_graphs=False,  # Set to True for debugging
                save_graphs_path="./graphs",
            )

            # Enable memory optimization (using max_device_memory instead)
            if gpu_config.get("enable_mem_reuse", True):
                try:
                    # Use max_device_memory for memory optimization
                    max_memory = gpu_config.get("max_device_memory", "4GB")
                    context.set_context(max_device_memory=max_memory)
                    print(f"✓ Memory optimization enabled (max_device_memory: {max_memory})")
                    
                    # Additional memory optimization settings
                    mempool_size = gpu_config.get("mempool_block_size", "512MB")
                    context.set_context(
                        mempool_block_size=mempool_size,  # Smaller memory pool blocks
                        enable_reduce_precision=True  # Enable reduce precision to save memory
                    )
                    print(f"✓ Additional memory optimizations enabled (mempool: {mempool_size})")
                    
                    # Enable memory offload if available
                    if gpu_config.get("enable_memory_offload", False):
                        try:
                            context.set_context(enable_sparse=True)  # Enable sparse tensor support
                            print("✓ Sparse tensor support enabled for memory savings")
                        except Exception as sparse_e:
                            print(f"Warning: Sparse tensor not supported: {sparse_e}")
                            
                except Exception as e:
                    print(f"Warning: Memory optimization not supported: {e}")

            # Disable graph kernel optimization to avoid compilation issues
            if self.config_manager.get("model.enable_graph_kernel", False):
                try:
                    # Graph kernel might not be available in all versions
                    context.set_context(enable_graph_kernel=True)
                    print("✓ Graph kernel optimization enabled")
                except Exception as e:
                    print(f"Warning: Graph kernel optimization not supported: {e}")
            else:
                context.set_context(enable_graph_kernel=False)
                print("✓ Graph kernel optimization disabled for stability")

            # Enable auto mixed precision if supported
            if self.config_manager.get("model.enable_auto_mixed_precision", True):
                try:
                    # Try different methods for auto mixed precision
                    try:
                        context.set_auto_parallel_context(enable_auto_mixed_precision=True)
                        print("✓ Auto mixed precision enabled (auto_parallel_context)")
                    except:
                        # Alternative approach
                        context.set_context(enable_auto_mixed_precision=True)
                        print("✓ Auto mixed precision enabled (context)")
                except Exception as e:
                    print(f"Warning: Auto mixed precision not supported: {e}")

            # Set device using new API if available
            if MINDSPORE_NEW_API:
                try:
                    set_device(device_target, device_id)
                    print(f"✓ MindSpore device set to: {device_target}:{device_id} (new API)")
                except Exception as e:
                    print(f"Warning: New API failed, using context setting: {e}")
            
            print(f"✓ GPU context configured successfully: {device_target}:{device_id}")
            
            # Set memory management
            max_memory = gpu_config.get("max_device_memory")
            if max_memory:
                print(f"✓ Max device memory limit: {max_memory}")

        except Exception as e:
            print_error_details(e, "GPU context setup")
            raise

    def _check_gpu_availability(self):
        """Check if GPU is available"""
        try:
            # Try to get GPU device count
            import mindspore as ms
            
            # Simple test to check GPU availability
            test_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
            
            # Try to create a simple operation on GPU
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
        self.logger = logging.getLogger('GPUTFNetTrainer')
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
                    log_file = os.path.join(log_dir, f"gpu_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        """Prepare datasets and vocabulary with GPU optimizations"""
        self.logger.info("Preparing data for GPU training...")
        
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
        
        # Create datasets with GPU optimizations
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
        """Build TFNet model with GPU optimizations"""
        self.logger.info("Building TFNet model for GPU...")
        
        hidden_size = self.config_manager.get("model.hidden_size")
        device_target = self.config_manager.get("model.device_target")
        dataset_name = self.config_manager.get_dataset_config()["name"]
        
        # Create model
        self.model = TFNetModel(
            hidden_size=hidden_size,
            word_set_num=vocab_size,
            device_target=device_target,
            dataset_name=dataset_name
        )
        
        self.logger.info(f"Model created with hidden_size={hidden_size}, vocab_size={vocab_size}")
        self.logger.info(f"Model device target: {device_target}")
        
        # Initialize decoder
        self.decoder = CTCDecoder(
            gloss_dict=self.word2idx,
            num_classes=vocab_size,
            search_mode='beam',
            blank_id=self.config_manager.get("loss.ctc_blank_id", 0)
        )
        
        return self.model
    
    def setup_training(self, vocab_size):
        """Setup training components with GPU optimizations"""
        self.logger.info("Setting up training components...")
        
        # Build model
        model = self.build_model(vocab_size)
        
        # Setup optimizer
        learning_rate = self.config_manager.get("training.learning_rate")
        weight_decay = self.config_manager.get("training.weight_decay")
        
        optimizer = nn.Adam(
            params=model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function - use CTCLoss instead of SeqKD
        from mindspore.nn import CTCLoss
        blank_id = self.config_manager.get("loss.ctc_blank_id", 0)
        reduction = self.config_manager.get("loss.ctc_reduction", "mean")
        loss_fn = CTCLoss(blank=blank_id, reduction=reduction)
        
        self.logger.info("Training components setup completed")
        
        return model, optimizer, loss_fn

    def train(self):
        """Main training loop with GPU optimizations"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING GPU-OPTIMIZED TRAINING")
            self.logger.info("=" * 60)
            
            # Prepare data
            vocab_size = self.prepare_data()
            
            # Setup training
            model, optimizer, loss_fn = self.setup_training(vocab_size)
            
            # Training parameters
            num_epochs = self.config_manager.get("training.num_epochs")
            save_interval = self.config_manager.get("training.save_interval")
            eval_interval = self.config_manager.get("training.eval_interval")
            gradient_clip_norm = self.config_manager.get("training.gradient_clip_norm")
            
            self.logger.info(f"Training for {num_epochs} epochs")
            self.logger.info(f"Save interval: {save_interval}, Eval interval: {eval_interval}")
            
            # Setup callbacks
            callbacks = self._setup_callbacks()
            
            # Training loop
            start_time = time.time()
            
            for epoch in range(num_epochs):
                self.current_epoch = epoch + 1
                epoch_start = time.time()
                
                self.logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
                
                # Training step
                train_loss = self._train_epoch(model, optimizer, loss_fn, gradient_clip_norm)
                
                # Validation step
                if self.current_epoch % eval_interval == 0:
                    val_wer = self._validate_epoch(model)
                    
                    # Save best model
                    if val_wer < self.best_wer:
                        self.best_wer = val_wer
                        self.best_epoch = self.current_epoch
                        self._save_best_model(model)
                
                # Save checkpoint
                if self.current_epoch % save_interval == 0:
                    self._save_checkpoint(model, optimizer)
                
                epoch_time = time.time() - epoch_start
                self.logger.info(f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s")
                
                # Early stopping check
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
        """Train for one epoch with GPU optimizations"""
        model.set_train(True)
        total_loss = 0.0
        batch_count = 0
        
        # Enable data sink mode if configured
        enable_data_sink = self.config_manager.get("training.enable_data_sink", True)
        
        if enable_data_sink:
            self.logger.info("Using data sink mode for better GPU utilization")
        
        for batch_idx, (data, target, data_len, target_len) in enumerate(self.train_dataset):
            try:
                # Forward pass
                def forward_fn(seq_data, seq_label, data_len, label_len):
                    model_output = model(seq_data, data_len, is_train=True)
                    # Model returns: (log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, lgt_tensor, None, None, None)
                    logits = model_output[0]
                    
                    # 统一将data_len转换为Tensor[int32]
                    if not isinstance(data_len, Tensor):
                        if isinstance(data_len, (list, tuple)):
                            data_len = Tensor([int(l.item() if hasattr(l, 'item') else int(l)) for l in data_len], ms.int32)
                        else:
                            data_len = Tensor([int(data_len)], ms.int32)
                    else:
                        # 若是标量，包装成一维
                        if len(data_len.shape) == 0:
                            data_len = ops.expand_dims(data_len, 0)
                        data_len = ops.cast(data_len, ms.int32)
                    
                    # 获取模型输出时间步（logits形状: T x N x C 或类似），取第0维
                    actual_time_steps = logits.shape[0]
                    ts_tensor = Tensor(actual_time_steps, ms.int32)
                    one_tensor = Tensor(1, ms.int32)
                    
                    # 保证长度在[1, actual_time_steps]
                    data_len = ops.minimum(data_len, ts_tensor)
                    data_len = ops.maximum(data_len, one_tensor)
                    
                    loss = loss_fn(logits, seq_label, data_len, label_len)
                    return loss
                
                # Compute gradients
                grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
                loss, grads = grad_fn(data, target, data_len, target_len)
                
                # Gradient clipping
                if gradient_clip_norm > 0:
                    grads = ops.clip_by_global_norm(grads, gradient_clip_norm)
                
                # Update parameters
                optimizer(grads)
                
                total_loss += loss.asnumpy()
                batch_count += 1
                
                # Print progress
                if batch_idx % self.config_manager.get("logging.print_interval") == 0:
                    self.logger.info(f"Batch {batch_idx}, Loss: {loss.asnumpy():.6f}")
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(batch_count, 1)
        self.logger.info(f"Training loss: {avg_loss:.6f}")
        return avg_loss

    def _validate_epoch(self, model):
        """Validate for one epoch"""
        model.set_train(False)
        total_wer = 0.0
        batch_count = 0
        
        for batch_idx, (data, target, data_len, target_len) in enumerate(self.valid_dataset):
            try:
                # Forward pass
                logits = model(data, data_len, is_train=False)
                
                # Decode predictions
                predictions = self.decoder.decode(logits.asnumpy(), data_len.asnumpy())
                references = self.decoder.decode_labels(target.asnumpy(), target_len.asnumpy())
                
                # Calculate WER
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
        """Setup training callbacks"""
        callbacks = []
        
        # Loss monitor
        callbacks.append(LossMonitor(per_print_times=self.config_manager.get("logging.print_interval")))
        
        # Time monitor
        callbacks.append(TimeMonitor())
        
        # Checkpoint callback
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
        """Save best model"""
        try:
            best_model_path = self.config_manager.get("paths.best_model_path")
            save_checkpoint(model, best_model_path)
            self.logger.info(f"Best model saved: {best_model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save best model: {e}")

    def _save_checkpoint(self, model, optimizer):
        """Save training checkpoint"""
        try:
            checkpoint_path = self.config_manager.get("paths.current_model_path")
            save_checkpoint(model, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _should_early_stop(self):
        """Check if early stopping should be triggered"""
        patience = self.config_manager.get("training.early_stopping_patience")
        return (self.current_epoch - self.best_epoch) >= patience

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-Optimized TFNet Training")
    parser.add_argument("--config", type=str, default="configs/gpu_config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Verify environment
    print("Checking environment...")
    print(f"Current conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    
    if 'mindspore-gpu' not in os.environ.get('CONDA_DEFAULT_ENV', ''):
        print("Warning: Not in mindspore-gpu environment")
        print("Please run: conda activate mindspore-gpu")
    
    try:
        # Create trainer
        trainer = GPUTFNetTrainer(args.config)
        
        # Start training
        trainer.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print_error_details(e, "Main training")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
