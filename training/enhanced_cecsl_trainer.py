#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版CE-CSL手语识别训练器
针对小数据集优化，包含数据增强、改进模型架构和训练策略
"""

import os
import sys
import json
import logging
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, save_checkpoint, load_checkpoint
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import random
import csv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedCECSLConfig:
    """增强版CE-CSL训练配置"""
    # 模型配置
    vocab_size: int = 1000
    d_model: int = 192         # 降低维度，减少显存/内存
    n_layers: int = 2          # 降低层数
    dropout: float = 0.3
    
    # 训练配置
    batch_size: int = 1        # 更小的 batch
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    epochs: int = 100
    warmup_epochs: int = 10
    
    # 数据配置
    data_root: str = "../data/CS-CSL"
    max_sequence_length: int = 64  # 先将序列长度降一点
    image_size: Tuple[int, int] = (112, 112)  # 配合 2× 降采样
    
    # 数据增强配置
    augment_factor: int = 1  # 关闭离线增广，改为在线
    noise_std: float = 0.01
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)  # 时间拉伸范围
    
    # 设备配置
    device_target: str = "CPU"
    
    # 早停配置
    patience: int = 20  # 早停耐心值
    min_delta: float = 0.001  # 最小改善阈值

class DataAugmentor:
    """数据增强器"""
    
    def __init__(self, config: EnhancedCECSLConfig):
        self.config = config
        
    def add_noise(self, frames: np.ndarray) -> np.ndarray:
        """添加随机噪声"""
        noise = np.random.normal(0, self.config.noise_std, frames.shape).astype(np.float32)
        return np.clip(frames + noise, 0, 1)
    
    def time_stretch(self, frames: np.ndarray) -> np.ndarray:
        """时间拉伸"""
        stretch_factor = random.uniform(*self.config.time_stretch_range)
        original_length = len(frames)
        new_length = int(original_length * stretch_factor)
        
        if new_length <= 0:
            return frames
            
        # 重采样
        indices = np.linspace(0, original_length - 1, new_length)
        indices = np.clip(indices, 0, original_length - 1).astype(int)
        stretched = frames[indices]
        
        # 调整到目标长度
        if len(stretched) > self.config.max_sequence_length:
            indices = np.linspace(0, len(stretched) - 1, self.config.max_sequence_length, dtype=int)
            stretched = stretched[indices]
        elif len(stretched) < self.config.max_sequence_length:
            pad_length = self.config.max_sequence_length - len(stretched)
            pad_frames = np.zeros((pad_length,) + stretched.shape[1:], dtype=stretched.dtype)
            stretched = np.concatenate([stretched, pad_frames], axis=0)
            
        return stretched
    
    def spatial_jitter(self, frames: np.ndarray) -> np.ndarray:
        """空间抖动"""
        # 添加小幅度的随机偏移
        jitter_std = 0.02
        spatial_noise = np.random.normal(0, jitter_std, frames.shape).astype(np.float32)
        return np.clip(frames + spatial_noise, 0, 1)
    
    def augment_sample(self, frames: np.ndarray) -> List[np.ndarray]:
        """对单个样本进行多种增强"""
        augmented_samples = [frames]  # 原始样本
        
        for _ in range(self.config.augment_factor - 1):
            aug_frames = frames.copy()
            
            # 随机应用不同的增强
            if random.random() < 0.7:
                aug_frames = self.add_noise(aug_frames)
            
            if random.random() < 0.5:
                aug_frames = self.time_stretch(aug_frames)
            
            if random.random() < 0.6:
                aug_frames = self.spatial_jitter(aug_frames)
            
            augmented_samples.append(aug_frames)
        
        return augmented_samples

class EnhancedCECSLDataset:
    """增强版CE-CSL数据集"""
    
    def __init__(self, config: EnhancedCECSLConfig, split: str = 'train', use_augmentation: bool = True):
        self.config = config
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.data_root = Path(config.data_root)
        # 新增：在线增广器（即使不使用也要定义为 None 防止属性不存在）
        self.augmentor = DataAugmentor(config) if self.use_augmentation else None
        
        # 若首选 CS-CSL 不存在则回退到 CE-CSL
        if not self.data_root.exists():
            alt = Path(str(self.data_root).replace("CS-CSL", "CE-CSL"))
            if alt.exists():
                logger.warning(f"未找到数据目录 {self.data_root} ，自动回退到 {alt}")
                self.data_root = alt
            else:
                logger.warning(f"未找到数据目录 {self.data_root} ，请确认数据是否已放置")
        
        # 加载词汇表
        self.word2idx = {}
        self.idx2word = []
        self._build_vocabulary()
        
        # 加载数据
        self.samples = []
        self._load_data()

        # 重要：不再在初始化中做离线增广，转为在线增广
        # if self.use_augmentation:
        #     self._apply_augmentation()

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
        """加载预处理数据（仅保存路径，懒加载）"""
        metadata_file = self.data_root / "processed" / self.split / f"{self.split}_metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"元数据文件不存在: {metadata_file}")
            self._load_from_corpus()
            return
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for item in metadata:
                video_id = item['video_id']
                # 从corpus获取标签
                label = self._get_label_from_corpus(video_id)
                if label is None:
                    continue
                
                frames_file = self.data_root / "processed" / self.split / f"{video_id}_frames.npy"
                
                if frames_file.exists():
                    try:
                        label_idx = self.word2idx.get(label, self.word2idx.get('<UNK>', 1))
                        
                        self.samples.append({
                            'frames_path': str(frames_file),  # 仅保存路径
                            'label': label,
                            'label_idx': label_idx,
                            'video_id': video_id,
                            'is_augmented': False
                        })
                    except Exception as e:
                        logger.error(f"记录数据失败 {frames_file}: {e}")
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            self._load_from_corpus()
    
    def _load_from_corpus(self):
        """直接从corpus文件加载数据（仅保存路径，懒加载）"""
        corpus_file = self.data_root / f"{self.split}.corpus.csv"
        if not corpus_file.exists():
            logger.error(f"Corpus文件不存在: {corpus_file}")
            return
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id']
                label = row['label']
                
                frames_file = self.data_root / "processed" / self.split / f"{video_id}_frames.npy"
                
                if frames_file.exists():
                    try:
                        label_idx = self.word2idx.get(label, self.word2idx.get('<UNK>', 1))
                        
                        self.samples.append({
                            'frames_path': str(frames_file),  # 仅保存路径
                            'label': label,
                            'label_idx': label_idx,
                            'video_id': video_id,
                            'is_augmented': False
                        })
                    except Exception as e:
                        logger.error(f"记录数据失败 {frames_file}: {e}")
        
        logger.info(f"从corpus加载 {len(self.samples)} 个样本")
    
    def _get_label_from_corpus(self, video_id: str) -> Optional[str]:
        """从corpus文件获取视频的标签"""
        corpus_file = self.data_root / f"{self.split}.corpus.csv"
        if not corpus_file.exists():
            return None
        
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['video_id'] == video_id:
                        return row['label']
        except Exception as e:
            logger.error(f"读取corpus文件失败: {e}")
        
        return None
    
    def __getitem__(self, index):
        """懒加载 + 在线增广 + 可选 2× 降采样 + 展平"""
        sample = self.samples[index]
        path = sample.get('frames_path')

        # 1) 懒加载 + 内存映射
        frames = np.load(path, mmap_mode='r')  # 可能是 (T,H,W,C) 或 (T,C,H,W)

        # 2) 统一为 (T,H,W,C) 以便处理，再转回 (T,C,H,W)
        if frames.ndim != 4:
            raise ValueError(f"不支持的帧维度: {frames.shape}")
        if frames.shape[-1] in (1, 3):
            layout = "THWC"
            thwc = frames
        elif frames.shape[1] in (1, 3):
            layout = "TCHW"
            # 转为 THWC
            thwc = np.transpose(frames, (0, 2, 3, 1))
        else:
            # 尝试猜测通道在最后
            layout = "THWC"
            thwc = frames

        # 3) 可选 2× 降采样（224->112），与 config.image_size 配套
        if thwc.shape[1] == 224 and thwc.shape[2] == 224 and self.config.image_size == (112, 112):
            thwc = thwc[:, ::2, ::2, :]

        # 4) 归一化到 [0,1] + float32
        if thwc.dtype != np.float32:
            thwc = thwc.astype(np.float32, copy=False)
        if thwc.max() > 1.0:
            thwc = thwc / 255.0

        # 5) 在线增广（仅训练集）
        if self.use_augmentation and self.augmentor is not None:
            # 简化的在线增广组合
            if random.random() < 0.7:
                thwc = self.augmentor.add_noise(thwc)
            if random.random() < 0.5:
                thwc = self.augmentor.time_stretch(thwc)  # 会改变时间长度
            if random.random() < 0.6:
                thwc = self.augmentor.spatial_jitter(thwc)

        # 6) 转回 (T,C,H,W)
        if thwc.ndim != 4:
            raise ValueError(f"增广后帧维度异常: {thwc.shape}")
        tchw = np.transpose(thwc, (0, 3, 1, 2))

        # 7) 调整序列长度
        seq_len = tchw.shape[0]
        if seq_len > self.config.max_sequence_length:
            idx = np.linspace(0, seq_len - 1, self.config.max_sequence_length, dtype=int)
            tchw = tchw[idx]
        elif seq_len < self.config.max_sequence_length:
            pad = np.zeros(
                (self.config.max_sequence_length - seq_len,) + tchw.shape[1:],
                dtype=tchw.dtype
            )
            tchw = np.concatenate([tchw, pad], axis=0)

        # 8) 展平为 (T, F)
        T = tchw.shape[0]
        F = int(np.prod(tchw.shape[1:], dtype=np.int64))
        frames_flat = tchw.reshape(T, F).astype(np.float32, copy=False)

        return frames_flat, np.array(sample['label_idx'], dtype=np.int32)
    
    def __len__(self):
        return len(self.samples)

class ImprovedCECSLModel(nn.Cell):
    """改进的CE-CSL手语识别模型"""
    
    def __init__(self, config: EnhancedCECSLConfig, vocab_size: int):
        super().__init__()
        self.config = config
        
        input_size = 3 * config.image_size[0] * config.image_size[1]
        
        # 改进的特征提取网络
        self.feature_extractor = nn.SequentialCell([
            nn.Dense(input_size, config.d_model * 2),
            nn.LayerNorm([config.d_model * 2]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            
            nn.Dense(config.d_model * 2, config.d_model),
            nn.LayerNorm([config.d_model]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            
            nn.Dense(config.d_model, config.d_model),
            nn.LayerNorm([config.d_model]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout / 2)
        ])
        
        # 双向LSTM
        self.temporal_model = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0
        )
        
        # 注意力机制
        self.attention = nn.SequentialCell([
            nn.Dense(config.d_model * 2, config.d_model),
            nn.Tanh(),
            nn.Dense(config.d_model, 1)
        ])
        
        # 改进的分类器
        self.classifier = nn.SequentialCell([
            nn.Dense(config.d_model * 2, config.d_model),
            nn.LayerNorm([config.d_model]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            
            nn.Dense(config.d_model, config.d_model // 2),
            nn.LayerNorm([config.d_model // 2]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout / 2),
            
            nn.Dense(config.d_model // 2, vocab_size)
        ])
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def construct(self, x, labels=None):
        batch_size, seq_len, input_size = x.shape
        
        # 特征提取
        x_reshaped = x.view(batch_size * seq_len, input_size)
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, self.config.d_model)
        
        # 双向LSTM
        lstm_output, _ = self.temporal_model(features)  # (batch, seq, hidden*2)
        
        # 注意力权重
        attention_weights = self.attention(lstm_output)  # (batch, seq, 1)
        attention_weights = ops.Softmax(axis=1)(attention_weights)
        
        # 加权平均
        attended_output = ops.ReduceSum()(lstm_output * attention_weights, axis=1)  # (batch, hidden*2)
        
        # 分类
        logits = self.classifier(attended_output)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits

class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, base_lr: float, warmup_epochs: int, total_epochs: int):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # 线性预热
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.base_lr * (1 + np.cos(np.pi * progress)) / 2

class EarlyStoppingCallback:
    """早停回调"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop

class EnhancedCECSLTrainer:
    """增强版CE-CSL训练器"""
    
    def __init__(self, config: EnhancedCECSLConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.lr_scheduler = None
        self.early_stopping = None
        
        # 设置设备（新接口优先，兼容旧接口）
        try:
            if hasattr(ms, "set_device"):
                ms.set_device(config.device_target)
            else:
                ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
        except Exception:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
        
        # 创建输出目录
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"增强版CE-CSL训练器初始化完成 - 设备: {config.device_target}")
    
    def load_data(self):
        """加载数据"""
        logger.info("加载增强版CE-CSL数据集...")
        
        # 创建数据集（训练集使用增强）
        train_data = EnhancedCECSLDataset(self.config, 'train', use_augmentation=True)
        val_data = EnhancedCECSLDataset(self.config, 'dev', use_augmentation=False)
        
        if len(train_data) == 0:
            raise ValueError("训练数据集为空，请检查数据路径和预处理数据")
        
        # 创建MindSpore数据集，控制并行度避免内存峰值
        self.train_dataset = GeneratorDataset(
            train_data,
            column_names=["sequence", "label"],
            shuffle=True,
            num_parallel_workers=1,
            python_multiprocessing=False
        ).batch(self.config.batch_size)

        self.val_dataset = GeneratorDataset(
            val_data,
            column_names=["sequence", "label"],
            shuffle=False,
            num_parallel_workers=1,
            python_multiprocessing=False
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
        logger.info("构建增强版CE-CSL模型...")
        
        if not hasattr(self, 'vocab_size'):
            raise ValueError("请先调用load_data()加载数据")
        
        # 创建模型
        self.model = ImprovedCECSLModel(self.config, self.vocab_size)
        
        # 计算参数量
        param_count = sum(p.size for p in self.model.get_parameters())
        logger.info(f"模型构建完成 - 参数量: {param_count}")
        
        # 创建优化器
        self.optimizer = nn.AdamWeightDecay(
            params=self.model.trainable_params(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 创建学习率调度器
        self.lr_scheduler = LearningRateScheduler(
            self.config.learning_rate,
            self.config.warmup_epochs,
            self.config.epochs
        )
        
        # 创建早停回调
        self.early_stopping = EarlyStoppingCallback(
            self.config.patience,
            self.config.min_delta
        )
        
        logger.info("优化器和调度器创建完成")
    
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
        logger.info("开始增强版CE-CSL真实数据训练...")
        
        best_val_acc = 0
        training_history = []
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # 更新学习率
            current_lr = self.lr_scheduler.get_lr(epoch)
            # MindSpore优化器不支持param_groups，需要重新创建优化器
            self.optimizer = nn.AdamWeightDecay(
                params=self.model.trainable_params(),
                learning_rate=current_lr,
                weight_decay=self.config.weight_decay
            )
            
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
                
                if batch_idx % 10 == 0:  # 每10个batch打印一次
                    logger.info(f"Batch {batch_idx}: Loss = {loss.asnumpy():.4f}")
            
            avg_train_loss = epoch_loss / len(self.train_dataset)
            train_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            
            logger.info(f"Epoch {epoch+1} 训练完成:")
            logger.info(f"  平均损失: {avg_train_loss:.4f}")
            logger.info(f"  训练准确率: {train_accuracy:.4f}")
            logger.info(f"  学习率: {current_lr:.6f}")
            
            # 验证
            logger.info("开始模型评估...")
            val_loss, val_accuracy, class_accuracies = self.evaluate()
            
            logger.info("评估完成:")
            logger.info(f"  验证损失: {val_loss:.4f}")
            logger.info(f"  验证准确率: {val_accuracy:.4f}")
            
            logger.info("各类别准确率:")
            for word, acc in class_accuracies.items():
                total = sum(1 for w in class_accuracies if w == word)
                logger.info(f"  {word}: {acc:.4f}")
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} 总结:")
            logger.info(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}")
            logger.info(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
            logger.info(f"  耗时: {epoch_time:.2f}秒")
            
            # 保存最佳模型
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                logger.info(f"新的最佳验证准确率: {best_val_acc:.4f}")
                
                best_model_path = self.output_dir / "enhanced_cecsl_best_model.ckpt"
                save_checkpoint(self.model, str(best_model_path))
                logger.info(f"最佳模型已保存: {best_model_path}")
            
            # 记录训练历史
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_accuracy,
                'val_loss': val_loss,
                'val_acc': val_accuracy,
                'learning_rate': current_lr,
                'class_accuracies': class_accuracies
            })
            
            # 早停检查
            if self.early_stopping(val_accuracy):
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        
        # 保存训练历史
        history_file = self.output_dir / "enhanced_training_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        
        return self.model
    
    def save_model(self, save_path: str):
        """保存模型"""
        save_checkpoint(self.model, save_path)
        
        # 保存词汇表
        vocab_path = Path(save_path).parent / "enhanced_vocab.json"
        vocab_info = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ 增强版模型和词汇表保存完成")
def main():
    """主函数 - 运行增强版CE-CSL训练"""
    try:
        # 创建配置（已更新默认值以更省内存）
        config = EnhancedCECSLConfig()
        
        # 打印配置信息
        logger.info("=" * 60)
        logger.info("增强版CE-CSL手语识别训练器启动")
        logger.info("=" * 60)
        logger.info(f"数据路径: {config.data_root}")
        logger.info(f"批次大小: {config.batch_size}")
        logger.info(f"学习率: {config.learning_rate}")
        logger.info(f"训练轮数: {config.epochs}")
        logger.info(f"设备: {config.device_target}")
        logger.info(f"数据增强倍数: {config.augment_factor}")
        logger.info("=" * 60)
        
        # 创建训练器
        trainer = EnhancedCECSLTrainer(config)
        
        # 加载数据
        logger.info("步骤 1: 加载数据...")
        trainer.load_data()
        
        # 构建模型
        logger.info("步骤 2: 构建模型...")
        trainer.build_model()
        
        # 开始训练
        logger.info("步骤 3: 开始训练...")
        model = trainer.train()
        
        # 保存模型
        logger.info("步骤 4: 保存模型...")
        final_model_path = trainer.output_dir / "enhanced_cecsl_final_model.ckpt"
        trainer.save_model(str(final_model_path))
        
        logger.info("=" * 60)
        logger.info("✅ 增强版CE-CSL训练完成!")
        logger.info(f"最终模型保存至: {final_model_path}")
        logger.info(f"训练历史保存至: {trainer.output_dir / 'enhanced_training_history.json'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
