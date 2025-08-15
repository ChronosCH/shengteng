#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实用优化版CE-CSL手语识别训练器
解决准确率过低和训练时间过短的问题，使用更稳定的实现
"""

import os
import sys
import json
import logging
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, save_checkpoint
from mindspore.dataset import GeneratorDataset
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizedCECSLConfig:
    """优化版CE-CSL训练配置"""
    # 模型配置
    d_model: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    
    # 训练配置
    batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    
    # 数据配置
    data_root: str = "../data/CE-CSL"
    max_sequence_length: int = 100
    image_size: Tuple[int, int] = (224, 224)
    
    # 数据增强配置
    augment_factor: int = 8
    noise_std: float = 0.02
    
    # 设备配置
    device_target: str = "CPU"
    
    # 训练策略
    patience: int = 15
    min_delta: float = 0.001

class SimpleDataAugmentor:
    """简单数据增强器"""
    
    def __init__(self, config: OptimizedCECSLConfig):
        self.config = config
        
    def add_noise(self, frames: np.ndarray) -> np.ndarray:
        """添加噪声"""
        noise = np.random.normal(0, self.config.noise_std, frames.shape).astype(np.float32)
        return np.clip(frames + noise, 0, 1)
    
    def scale_intensity(self, frames: np.ndarray) -> np.ndarray:
        """强度缩放"""
        scale = random.uniform(0.8, 1.2)
        return np.clip(frames * scale, 0, 1)
    
    def augment_sample(self, frames: np.ndarray) -> List[np.ndarray]:
        """对单个样本进行增强"""
        augmented_samples = [frames]  # 原始样本
        
        for _ in range(self.config.augment_factor - 1):
            aug_frames = frames.copy()
            
            # 随机应用增强
            if random.random() < 0.5:
                aug_frames = self.add_noise(aug_frames)
            
            if random.random() < 0.3:
                aug_frames = self.scale_intensity(aug_frames)
            
            augmented_samples.append(aug_frames)
        
        return augmented_samples

class OptimizedCECSLDataset:
    """优化版CE-CSL数据集"""
    
    def __init__(self, config: OptimizedCECSLConfig, split: str = 'train', use_augmentation: bool = True):
        self.config = config
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.data_root = Path(config.data_root)
        
        # 数据增强器
        self.augmentor = SimpleDataAugmentor(config) if self.use_augmentation else None
        
        # 加载词汇表
        self.word2idx = {}
        self.idx2word = []
        self._build_vocabulary()
        
        # 加载数据
        self.samples = []
        self._load_data()
        
        # 应用数据增强
        if self.use_augmentation:
            self._apply_augmentation()
        
        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")
        logger.info(f"词汇表大小: {len(self.word2idx)}")
    
    def _build_vocabulary(self):
        """构建词汇表"""
        all_labels = set()
        
        for split in ['train', 'dev', 'test']:
            csv_file = self.data_root / f"{split}.corpus.csv"
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)
                all_labels.update(df['label'].unique())
        
        # 添加特殊标记
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        self.idx2word = ['<PAD>', '<UNK>']
        
        # 添加所有标签
        for label in sorted(all_labels):
            if label not in self.word2idx:
                self.word2idx[label] = len(self.idx2word)
                self.idx2word.append(label)
        
        logger.info(f"词汇表构建完成: {sorted(all_labels)}")
    
    def _load_data(self):
        """加载预处理数据"""
        metadata_file = self.data_root / "processed" / self.split / f"{self.split}_metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"元数据文件不存在: {metadata_file}")
            return
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for item in metadata:
            video_id = item['video_id']
            label = item['text']
            
            frames_file = self.data_root / "processed" / self.split / f"{video_id}_frames.npy"
            
            if frames_file.exists():
                try:
                    frames = self._load_frames(str(frames_file))
                    label_idx = self.word2idx.get(label, self.word2idx['<UNK>'])
                    
                    self.samples.append({
                        'frames': frames,
                        'label': label,
                        'label_idx': label_idx,
                        'video_id': video_id,
                        'is_augmented': False
                    })
                except Exception as e:
                    logger.error(f"加载数据失败 {frames_file}: {e}")
    
    def _load_frames(self, frames_file: str) -> np.ndarray:
        """加载视频帧数据"""
        frames = np.load(frames_file)
        
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32)
        
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        # 调整序列长度
        if len(frames) > self.config.max_sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.config.max_sequence_length, dtype=int)
            frames = frames[indices]
        elif len(frames) < self.config.max_sequence_length:
            pad_length = self.config.max_sequence_length - len(frames)
            pad_frames = np.zeros((pad_length,) + frames.shape[1:], dtype=frames.dtype)
            frames = np.concatenate([frames, pad_frames], axis=0)
        
        # 转换格式
        if len(frames.shape) == 4 and frames.shape[-1] in [1, 3]:
            frames = np.transpose(frames, (0, 3, 1, 2))
        
        return frames
    
    def _apply_augmentation(self):
        """应用数据增强"""
        if not self.augmentor:
            return
        
        original_samples = self.samples.copy()
        augmented_samples = []
        
        for sample in original_samples:
            augmented_frames_list = self.augmentor.augment_sample(sample['frames'])
            
            # 添加增强样本（跳过第一个，因为是原始样本）
            for i, aug_frames in enumerate(augmented_frames_list[1:], 1):
                augmented_samples.append({
                    'frames': aug_frames,
                    'label': sample['label'],
                    'label_idx': sample['label_idx'],
                    'video_id': f"{sample['video_id']}_aug_{i}",
                    'is_augmented': True
                })
        
        self.samples.extend(augmented_samples)
        logger.info(f"数据增强完成: 原始 {len(original_samples)} -> 总计 {len(self.samples)}")
    
    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.samples[index]
        frames = sample['frames']
        
        # 展平帧数据
        seq_len = frames.shape[0]
        features = np.prod(frames.shape[1:])
        frames_flat = frames.reshape(seq_len, features)
        
        return frames_flat.astype(np.float32), np.array(sample['label_idx'], dtype=np.int32)
    
    def __len__(self):
        return len(self.samples)

class OptimizedCECSLModel(nn.Cell):
    """优化版CE-CSL手语识别模型"""
    
    def __init__(self, config: OptimizedCECSLConfig, vocab_size: int):
        super().__init__()
        self.config = config
        
        input_size = 3 * config.image_size[0] * config.image_size[1]
        
        # 特征提取网络
        self.feature_extractor = nn.SequentialCell([
            nn.Dense(input_size, config.d_model),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Dense(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(p=config.dropout)
        ])
        
        # LSTM网络
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0
        )
        
        # 分类器
        self.classifier = nn.SequentialCell([
            nn.Dense(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Dense(config.d_model // 2, vocab_size)
        ])
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def construct(self, x, labels=None):
        batch_size, seq_len, input_size = x.shape
        
        # 特征提取
        x_reshaped = x.view(batch_size * seq_len, input_size)
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, self.config.d_model)
        
        # LSTM处理
        output, _ = self.lstm(features)
        
        # 取最后一个时间步的输出
        final_output = output[:, -1, :]  # (batch_size, d_model)
        
        # 分类
        logits = self.classifier(final_output)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits

class OptimizedCECSLTrainer:
    """优化版CE-CSL训练器"""
    
    def __init__(self, config: OptimizedCECSLConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.best_val_acc = 0
        self.patience_counter = 0
        
        # 设置设备
        ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
        
        # 创建输出目录
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"优化版CE-CSL训练器初始化完成 - 设备: {config.device_target}")
    
    def load_data(self):
        """加载数据"""
        logger.info("加载优化版CE-CSL数据集...")
        
        # 创建数据集（训练集使用增强）
        train_data = OptimizedCECSLDataset(self.config, 'train', use_augmentation=True)
        val_data = OptimizedCECSLDataset(self.config, 'dev', use_augmentation=False)
        
        if len(train_data) == 0:
            raise ValueError("训练数据集为空，请检查数据路径和预处理数据")
        
        # 创建MindSpore数据集
        self.train_dataset = GeneratorDataset(
            train_data, 
            column_names=["sequence", "label"],
            shuffle=True
        ).batch(self.config.batch_size)
        
        self.val_dataset = GeneratorDataset(
            val_data, 
            column_names=["sequence", "label"],
            shuffle=False
        ).batch(self.config.batch_size)
        
        # 保存词汇表信息
        self.vocab_size = len(train_data.word2idx)
        self.word2idx = train_data.word2idx
        self.idx2word = train_data.idx2word
        
        logger.info(f"训练集: {len(train_data)} 样本（包含增强数据）")
        logger.info(f"验证集: {len(val_data)} 样本")
        logger.info(f"词汇表大小: {self.vocab_size}")
        logger.info(f"标签类别: {list(self.word2idx.keys())}")
    
    def build_model(self):
        """构建模型"""
        logger.info("构建优化版CE-CSL模型...")
        
        if not hasattr(self, 'vocab_size'):
            raise ValueError("请先调用load_data()加载数据")
        
        # 创建模型
        self.model = OptimizedCECSLModel(self.config, self.vocab_size)
        
        # 计算参数量
        param_count = sum(p.size for p in self.model.get_parameters())
        logger.info(f"模型构建完成 - 参数量: {param_count}")
        
        # 创建优化器
        self.optimizer = nn.Adam(
            params=self.model.trainable_params(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logger.info("优化器创建完成")
    
    def train_step(self, data, label):
        """训练步骤"""
        def forward_fn(data, label):
            loss, logits = self.model(data, label)
            return loss, logits
        
        grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
        (loss, logits), grads = grad_fn(data, label)
        self.optimizer(grads)
        
        return loss, logits
    
    def evaluate(self):
        """评估模型"""
        self.model.set_train(False)
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        class_correct = {}
        class_total = {}
        
        # 初始化类别统计
        for word in self.word2idx.keys():
            if word not in ['<PAD>', '<UNK>']:
                class_correct[word] = 0
                class_total[word] = 0
        
        for data, labels in self.val_dataset:
            data = Tensor(data, ms.float32)
            labels = Tensor(labels, ms.int32)
            
            loss, logits = self.model(data, labels)
            predictions = ops.Argmax(axis=1)(logits)
            
            total_loss += loss.asnumpy()
            correct = (predictions == labels).sum()
            total_correct += correct.asnumpy()
            batch_size = labels.shape[0]
            total_samples += batch_size
            
            # 统计各类别准确率
            for i in range(batch_size):
                true_label = labels[i].asnumpy()
                pred_label = predictions[i].asnumpy()
                
                if true_label < len(self.idx2word):
                    true_word = self.idx2word[true_label]
                    if true_word in class_total:
                        class_total[true_word] += 1
                        if true_label == pred_label:
                            class_correct[true_word] += 1
        
        avg_loss = total_loss / len(self.val_dataset)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # 计算各类别准确率
        class_accuracies = {}
        for word in class_total:
            if class_total[word] > 0:
                class_accuracies[word] = class_correct[word] / class_total[word]
            else:
                class_accuracies[word] = 0.0
        
        return avg_loss, accuracy, class_accuracies
    
    def train(self):
        """开始训练"""
        logger.info("开始优化版CE-CSL真实数据训练...")
        
        training_history = []
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # 训练
            self.model.set_train(True)
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            logger.info(f"开始第 {epoch+1}/{self.config.epochs} 轮训练...")
            
            for batch_idx, (data, labels) in enumerate(self.train_dataset):
                data = Tensor(data, ms.float32)
                labels = Tensor(labels, ms.int32)
                
                loss, logits = self.train_step(data, labels)
                predictions = ops.Argmax(axis=1)(logits)
                
                epoch_loss += loss.asnumpy()
                correct = (predictions == labels).sum()
                epoch_correct += correct.asnumpy()
                epoch_total += labels.shape[0]
                
                if batch_idx % 5 == 0:  # 每5个batch打印一次
                    logger.info(f"Batch {batch_idx}: Loss = {loss.asnumpy():.4f}")
            
            avg_train_loss = epoch_loss / len(self.train_dataset)
            train_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            
            logger.info(f"Epoch {epoch+1} 训练完成:")
            logger.info(f"  平均损失: {avg_train_loss:.4f}")
            logger.info(f"  训练准确率: {train_accuracy:.4f}")
            
            # 验证
            logger.info("开始模型评估...")
            val_loss, val_accuracy, class_accuracies = self.evaluate()
            
            logger.info("评估完成:")
            logger.info(f"  验证损失: {val_loss:.4f}")
            logger.info(f"  验证准确率: {val_accuracy:.4f}")
            
            logger.info("各类别准确率:")
            for word, acc in class_accuracies.items():
                logger.info(f"  {word}: {acc:.4f}")
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} 总结:")
            logger.info(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}")
            logger.info(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
            logger.info(f"  耗时: {epoch_time:.2f}秒")
            
            # 保存最佳模型
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.patience_counter = 0
                logger.info(f"新的最佳验证准确率: {self.best_val_acc:.4f}")
                
                best_model_path = self.output_dir / "optimized_cecsl_best_model.ckpt"
                save_checkpoint(self.model, str(best_model_path))
                logger.info(f"最佳模型已保存: {best_model_path}")
            else:
                self.patience_counter += 1
                logger.info(f"验证准确率未提升，耐心计数: {self.patience_counter}/{self.config.patience}")
            
            # 记录训练历史
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_accuracy,
                'val_loss': val_loss,
                'val_acc': val_accuracy,
                'class_accuracies': class_accuracies
            })
            
            # 早停检查
            if self.patience_counter >= self.config.patience:
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        logger.info(f"训练完成! 最佳验证准确率: {self.best_val_acc:.4f}")
        
        # 保存训练历史
        history_file = self.output_dir / "optimized_training_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        
        return self.model
    
    def save_model(self, save_path: str):
        """保存模型"""
        save_checkpoint(self.model, save_path)
        
        # 保存词汇表
        vocab_path = Path(save_path).parent / "optimized_vocab.json"
        vocab_info = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ 优化版模型和词汇表保存完成")
