#!/usr/bin/env python3
"""
TFNet Training Script for Continuous Sign Language Recognition
Optimized for CPU execution with MindSpore framework
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
from mindspore import context, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

# Try to import new API, fallback to old if not available
try:
    from mindspore import set_device
    MINDSPORE_NEW_API = True
except ImportError:
    MINDSPORE_NEW_API = False

# Add current directory to path for imports
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
    """TFNet model trainer"""
    
    def __init__(self, config_path=None):
        try:
            # Initialize configuration
            print("Initializing TFNet Trainer...")
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config

            # Validate and create directories FIRST (before logging setup)
            print("Setting up directories...")
            if not self._setup_directories():
                raise RuntimeError("Failed to create required directories")

            # Validate dataset structure
            print("Validating dataset...")
            if not self._validate_dataset():
                raise RuntimeError("Dataset validation failed")

            # Set MindSpore context with API compatibility
            device_target = self.config_manager.get("model.device_target", "CPU")

            # Use new API if available, otherwise fallback to old API
            if MINDSPORE_NEW_API:
                try:
                    context.set_context(mode=context.PYNATIVE_MODE)
                    set_device(device_target)
                    print(f"✓ MindSpore device set to: {device_target} (new API)")
                except Exception as e:
                    print(f"Warning: New API failed, using fallback: {e}")
                    context.set_context(
                        mode=context.PYNATIVE_MODE,
                        device_target=device_target
                    )
                    print(f"✓ MindSpore device set to: {device_target} (fallback API)")
            else:
                context.set_context(
                    mode=context.PYNATIVE_MODE,
                    device_target=device_target
                )
                print(f"✓ MindSpore device set to: {device_target} (legacy API)")

            # Initialize logging (after directories are created)
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

            self.logger.info("TFNet Trainer initialized successfully")

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
        
        self.train_dataset = create_dataset(
            data_path=dataset_config["train_data_path"],
            label_path=dataset_config["train_label_path"],
            word2idx=self.word2idx,
            dataset_name=dataset_config["name"],
            batch_size=batch_size,
            is_train=True,
            num_workers=num_workers
        )
        
        self.valid_dataset = create_dataset(
            data_path=dataset_config["valid_data_path"],
            label_path=dataset_config["valid_label_path"],
            word2idx=self.word2idx,
            dataset_name=dataset_config["name"],
            batch_size=1,  # Use batch size 1 for validation
            is_train=False,
            num_workers=num_workers
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
    
    def create_loss_fn(self):
        """Create loss function"""
        # CTC Loss
        ctc_loss = nn.CTCLoss(
            blank=self.config_manager.get("loss.ctc_blank_id", 0),
            reduction=self.config_manager.get("loss.ctc_reduction", "mean"),
            zero_infinity=True
        )
        
        # Knowledge Distillation Loss
        kd_loss = SeqKD(T=self.config_manager.get("loss.kd_temperature", 8))
        
        def loss_fn(log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, 
                   lgt, target_data, target_lengths):
            """Combined loss function for TFNet"""
            
            # Apply log softmax
            log_softmax = nn.LogSoftmax(axis=-1)
            log_probs1 = log_softmax(log_probs1)
            log_probs2 = log_softmax(log_probs2)
            log_probs3 = log_softmax(log_probs3)
            log_probs4 = log_softmax(log_probs4)
            log_probs5 = log_softmax(log_probs5)
            
            # CTC losses
            loss1 = ctc_loss(log_probs1, target_data, lgt, target_lengths)
            loss2 = ctc_loss(log_probs2, target_data, lgt, target_lengths)
            loss4 = ctc_loss(log_probs3, target_data, lgt, target_lengths)
            loss5 = ctc_loss(log_probs4, target_data, lgt, target_lengths)
            loss7 = ctc_loss(log_probs5, target_data, lgt, target_lengths)
            
            # Knowledge distillation losses
            kd_weight = self.config_manager.get("loss.kd_weight", 25.0)
            loss3 = kd_weight * kd_loss(log_probs2, log_probs1, use_blank=False)
            loss6 = kd_weight * kd_loss(log_probs4, log_probs3, use_blank=False)
            
            # Total loss
            total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
            
            return total_loss
        
        return loss_fn
    
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

        for batch_idx, batch_data in enumerate(self.train_dataset.create_dict_iterator()):
            # Extract batch data
            videos = batch_data['video']
            labels = batch_data['label']
            video_lengths = batch_data['videoLength']

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

            # Prepare target data for CTC loss (MindSpore CTCLoss expects 2D targets: [B, S])
            # Convert labels to list of lists
            if hasattr(labels, 'asnumpy'):
                labels_list = labels.asnumpy().tolist()
            elif isinstance(labels, (list, tuple)):
                labels_list = [list(x) if isinstance(x, (list, tuple)) else [int(x)] for x in labels]
            else:
                # Fallback for unexpected types
                labels_list = [[int(labels)]]

            # Compute target lengths by trimming trailing zeros (padding token is 0)
            target_lengths_list = []
            for seq in labels_list:
                # Ensure it's a list of ints
                seq = list(seq)
                length = len(seq)
                while length > 0 and int(seq[length - 1]) == 0:
                    length -= 1
                # Avoid empty target: keep one blank if necessary
                if length == 0:
                    length = 1
                    if len(seq) == 0:
                        seq.append(0)
                    else:
                        seq[0] = 0
                target_lengths_list.append(length)

            # targets must be 2D: (batch_size, max_target_length)
            target_data = ms.Tensor(labels_list, ms.int32)
            target_lengths = ms.Tensor(target_lengths_list, ms.int32)

            # Forward pass
            log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, lgt, _, _, _ = \
                self.model(videos, video_lengths_list, is_train=True)

            # Calculate loss
            loss_fn = self.create_loss_fn()
            loss = loss_fn(log_probs1, log_probs2, log_probs3, log_probs4, log_probs5,
                          lgt, target_data, target_lengths)

            # Backward pass
            optimizer = self.create_optimizer()

            # Update metrics
            total_loss += loss.asnumpy().item()
            num_batches += 1

            # Print progress
            if batch_idx % self.config_manager.get("logging.print_interval", 10) == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.asnumpy().item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        return avg_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.logger.info(f"Validating epoch {epoch}")

        self.model.set_train(False)
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(self.valid_dataset.create_dict_iterator()):
            # Extract batch data
            videos = batch_data['video']
            labels = batch_data['label']
            video_lengths = batch_data['videoLength']

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
            batch_wer = calculate_wer_score(
                pred_ids_batch,
                ref_ids,
                self.idx2word,
                1  # batch size is 1 for validation
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

        for epoch in range(self.current_epoch, num_epochs):
            start_time = time.time()

            # Training
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
