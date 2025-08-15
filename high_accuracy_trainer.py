#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高准确率CE-CSL手语识别训练器 - 专门针对小数据集优化
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import numpy as np

import mindspore as ms
from mindspore import nn, ops, context, Model, load_checkpoint, save_checkpoint
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import Callback
from mindspore.communication.management import init, get_rank, get_group_size

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HighAccuracyConfig:
    """高准确率训练配置"""
    # 数据配置
    data_dir: str = "data/CE-CSL"
    vocab_file: str = "backend/models/vocab.json"
    
    # 模型配置 - 针对小数据集优化
    input_size: int = 150528  # 224*224*3
    hidden_size: int = 128  # 减小以防止过拟合
    num_layers: int = 1  # 简化模型
    num_classes: int = 10
    dropout_rate: float = 0.5  # 增强正则化
    
    # 训练配置 - 保守设置
    batch_size: int = 2  # 小批次
    learning_rate: float = 0.001  # 稍高学习率
    epochs: int = 50
    weight_decay: float = 0.01  # 强正则化
    
    # 数据增强
    augment_factor: int = 20  # 大幅增强数据
    
    # 早停和保存
    patience: int = 25
    min_epochs: int = 10
    
    # 设备配置
    device_target: str = "CPU"

class AggressiveDataAugmentor:
    """激进的数据增强器 - 专门针对小数据集"""
    
    def __init__(self, config: HighAccuracyConfig):
        self.config = config
        
    def augment_sample(self, frames: np.ndarray, label: int) -> List[Tuple[np.ndarray, int]]:
        """对单个样本进行激进的数据增强"""
        augmented_samples = [(frames, label)]  # 原始样本
        
        for i in range(self.config.augment_factor - 1):
            # 随机选择增强方法
            aug_type = random.choice(['time_warp', 'noise', 'brightness', 'contrast', 'flip', 'crop'])
            
            if aug_type == 'time_warp':
                aug_frames = self._time_warp(frames, strength=random.uniform(0.1, 0.3))
            elif aug_type == 'noise':
                aug_frames = self._add_noise(frames, noise_factor=random.uniform(0.05, 0.15))
            elif aug_type == 'brightness':
                aug_frames = self._adjust_brightness(frames, factor=random.uniform(0.7, 1.3))
            elif aug_type == 'contrast':
                aug_frames = self._adjust_contrast(frames, factor=random.uniform(0.8, 1.2))
            elif aug_type == 'flip':
                aug_frames = self._horizontal_flip(frames)
            else:  # crop
                aug_frames = self._random_crop_resize(frames, crop_ratio=random.uniform(0.85, 0.95))
            
            augmented_samples.append((aug_frames, label))
        
        return augmented_samples
    
    def _time_warp(self, frames: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """时间扭曲 - 改变序列长度"""
        seq_len = frames.shape[0]
        if seq_len <= 10:
            return frames
        
        # 随机选择新的序列长度
        new_len = int(seq_len * (1 + random.uniform(-strength, strength)))
        new_len = max(10, min(new_len, seq_len * 2))
        
        # 重采样
        indices = np.linspace(0, seq_len - 1, new_len).astype(int)
        return frames[indices]
    
    def _add_noise(self, frames: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_factor * 255, frames.shape).astype(np.float32)
        noisy_frames = frames.astype(np.float32) + noise
        return np.clip(noisy_frames, 0, 255).astype(np.uint8)
    
    def _adjust_brightness(self, frames: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """调整亮度"""
        bright_frames = frames.astype(np.float32) * factor
        return np.clip(bright_frames, 0, 255).astype(np.uint8)
    
    def _adjust_contrast(self, frames: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """调整对比度"""
        mean = np.mean(frames)
        contrast_frames = mean + factor * (frames.astype(np.float32) - mean)
        return np.clip(contrast_frames, 0, 255).astype(np.uint8)
    
    def _horizontal_flip(self, frames: np.ndarray) -> np.ndarray:
        """水平翻转"""
        return np.flip(frames, axis=2)  # 翻转宽度维度
    
    def _random_crop_resize(self, frames: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
        """随机裁剪并调整大小"""
        seq_len, h, w, c = frames.shape
        
        # 计算裁剪尺寸
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        # 随机裁剪位置
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        # 裁剪
        cropped = frames[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]
        
        # 这里简化处理，直接填充到原尺寸
        resized = np.zeros((seq_len, h, w, c), dtype=frames.dtype)
        resized[:, :crop_h, :crop_w, :] = cropped
        
        return resized

class HighAccuracyModel(nn.Cell):
    """高准确率手语识别模型 - 针对小数据集优化"""
    
    def __init__(self, config: HighAccuracyConfig):
        super().__init__()
        self.config = config
        
        # 特征降维 - 关键优化
        self.feature_reducer = nn.SequentialCell([
            nn.Dense(config.input_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Dense(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Dense(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        ])
        
        # 时序建模 - 简化的LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=float(config.dropout_rate) if config.num_layers > 1 else 0.0
        )
        
        # 分类头 - 增加正则化
        self.classifier = nn.SequentialCell([
            nn.Dense(config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.7),  # 高dropout
            nn.Dense(64, config.num_classes)
        ])
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(ms.common.initializer.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LSTM):
                for name, param in cell.parameters_and_names():
                    if 'weight' in name:
                        param.set_data(ms.common.initializer.initializer(
                            ms.common.initializer.XavierUniform(), param.shape, param.dtype))
    
    def construct(self, x):
        # x shape: (batch, seq_len, height, width, channels)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 展平空间维度
        x = x.view(batch_size * seq_len, -1)  # (batch*seq, features)
        
        # 特征降维
        x = self.feature_reducer(x)  # (batch*seq, 128)
        
        # 重塑为序列
        x = x.view(batch_size, seq_len, -1)  # (batch, seq, 128)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch, hidden)
        
        # 分类
        logits = self.classifier(last_output)  # (batch, num_classes)
        
        return logits

class HighAccuracyDataset:
    """高准确率数据集"""
    
    def __init__(self, data_dir: str, split: str, config: HighAccuracyConfig, 
                 vocab: Dict[str, int], augmentor: Optional[AggressiveDataAugmentor] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config
        self.vocab = vocab
        self.augmentor = augmentor
        
        # 加载数据
        self.samples = self._load_samples()
        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")
    
    def _load_samples(self) -> List[Tuple[np.ndarray, int]]:
        """加载数据样本"""
        samples = []
        
        # 加载元数据
        metadata_file = self.data_dir / "processed" / self.split / f"{self.split}_metadata.json"
        if not metadata_file.exists():
            logger.error(f"元数据文件不存在: {metadata_file}")
            return samples
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for item in metadata:
            try:
                # 加载帧数据
                frames_path = self.data_dir / "processed" / self.split / f"{item['video_id']}_frames.npy"
                if not frames_path.exists():
                    logger.warning(f"帧文件不存在: {frames_path}")
                    continue
                
                frames = np.load(frames_path)
                
                # 获取标签
                gloss = item['gloss_sequence'][0]  # 假设只有一个手语词
                if gloss not in self.vocab:
                    logger.warning(f"未知词汇: {gloss}")
                    continue
                
                label = self.vocab[gloss]
                
                # 数据预处理
                frames = self._preprocess_frames(frames)
                
                # 如果是训练集且有增强器，进行数据增强
                if self.split == "train" and self.augmentor:
                    augmented = self.augmentor.augment_sample(frames, label)
                    samples.extend(augmented)
                else:
                    samples.append((frames, label))
                
            except Exception as e:
                logger.error(f"加载样本失败 {item.get('video_id', 'unknown')}: {e}")
        
        return samples
    
    def _preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """预处理帧数据"""
        # 归一化到[0,1]
        frames = frames.astype(np.float32) / 255.0
        
        # 固定序列长度
        target_len = 100
        seq_len = frames.shape[0]
        
        if seq_len > target_len:
            # 均匀采样
            indices = np.linspace(0, seq_len - 1, target_len).astype(int)
            frames = frames[indices]
        elif seq_len < target_len:
            # 重复最后一帧进行填充
            padding = np.repeat(frames[-1:], target_len - seq_len, axis=0)
            frames = np.concatenate([frames, padding], axis=0)
        
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        return frames, label

class HighAccuracyTrainer:
    """高准确率训练器"""
    
    def __init__(self, config: HighAccuracyConfig):
        self.config = config
        self.setup_environment()
        self.load_vocab()
        self.setup_data()
        self.setup_model()
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def setup_environment(self):
        """设置环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target=self.config.device_target)
        logger.info(f"高准确率CE-CSL训练器初始化完成 - 设备: {self.config.device_target}")
    
    def load_vocab(self):
        """加载词汇表"""
        vocab_path = Path(self.config.vocab_file)
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.vocab = vocab_data.get('vocab', {})
        else:
            # 基于数据构建词汇表
            self.vocab = self._build_vocab()
        
        # 确保包含特殊token
        if '<PAD>' not in self.vocab:
            self.vocab['<PAD>'] = 0
        if '<UNK>' not in self.vocab:
            self.vocab['<UNK>'] = 1
        
        self.config.num_classes = len(self.vocab)
        logger.info(f"词汇表构建完成: {list(self.vocab.keys())}")
        logger.info(f"词汇表大小: {self.config.num_classes}")
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建词汇表"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # 扫描训练数据
        train_metadata = self.config.data_dir + "/processed/train/train_metadata.json"
        if Path(train_metadata).exists():
            with open(train_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for item in metadata:
                for gloss in item.get('gloss_sequence', []):
                    if gloss not in vocab:
                        vocab[gloss] = len(vocab)
        
        return vocab
    
    def setup_data(self):
        """设置数据"""
        logger.info("📊 加载数据（包含激进数据增强）...")
        
        # 创建数据增强器
        augmentor = AggressiveDataAugmentor(self.config)
        
        # 创建数据集
        self.train_dataset = HighAccuracyDataset(
            self.config.data_dir, "train", self.config, self.vocab, augmentor
        )
        self.val_dataset = HighAccuracyDataset(
            self.config.data_dir, "dev", self.config, self.vocab
        )
        
        logger.info(f"训练集: {len(self.train_dataset)} 样本（包含增强数据）")
        logger.info(f"验证集: {len(self.val_dataset)} 样本")
    
    def setup_model(self):
        """设置模型"""
        logger.info("🧠 构建高准确率模型...")
        
        self.model = HighAccuracyModel(self.config)
        
        # 计算参数量
        total_params = sum(p.size for p in self.model.trainable_params())
        logger.info(f"模型构建完成 - 参数量: {total_params}")
        
        # 设置优化器和损失函数
        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        logger.info("优化器和损失函数创建完成")
    
    def create_dataloader(self, dataset, shuffle=True):
        """创建数据加载器"""
        def generator():
            indices = list(range(len(dataset)))
            if shuffle:
                random.shuffle(indices)
            
            batch_frames = []
            batch_labels = []
            
            for idx in indices:
                frames, label = dataset[idx]
                batch_frames.append(frames)
                batch_labels.append(label)
                
                if len(batch_frames) == self.config.batch_size:
                    yield (np.stack(batch_frames), np.array(batch_labels))
                    batch_frames = []
                    batch_labels = []
            
            # 处理最后一个不完整的批次
            if batch_frames:
                yield (np.stack(batch_frames), np.array(batch_labels))
        
        return generator
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.set_train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        
        dataloader = self.create_dataloader(self.train_dataset, shuffle=True)
        
        for batch_frames, batch_labels in dataloader():
            batch_count += 1
            
            # 转换为Tensor
            frames_tensor = ms.Tensor(batch_frames, ms.float32)
            labels_tensor = ms.Tensor(batch_labels, ms.int32)
            
            # 前向传播
            def forward_fn():
                logits = self.model(frames_tensor)
                loss = self.loss_fn(logits, labels_tensor)
                return loss, logits
            
            grad_fn = ms.ops.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
            (loss, logits), grads = grad_fn()
            
            # 反向传播
            self.optimizer(grads)
            
            # 统计
            predictions = ops.argmax(logits, axis=1)
            correct = ops.equal(predictions, labels_tensor).sum()
            
            epoch_loss += loss.asnumpy()
            epoch_correct += correct.asnumpy()
            epoch_total += len(batch_labels)
            
            if batch_count % 5 == 0:
                logger.info(f"Batch {batch_count}: Loss = {loss.asnumpy():.4f}")
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self):
        """评估模型"""
        self.model.set_train(False)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        # 类别统计
        class_correct = {i: 0 for i in range(self.config.num_classes)}
        class_total = {i: 0 for i in range(self.config.num_classes)}
        
        dataloader = self.create_dataloader(self.val_dataset, shuffle=False)
        
        for batch_frames, batch_labels in dataloader():
            batch_count += 1
            
            frames_tensor = ms.Tensor(batch_frames, ms.float32)
            labels_tensor = ms.Tensor(batch_labels, ms.int32)
            
            # 前向传播
            logits = self.model(frames_tensor)
            loss = self.loss_fn(logits, labels_tensor)
            
            # 预测
            predictions = ops.argmax(logits, axis=1)
            correct = ops.equal(predictions, labels_tensor)
            
            total_loss += loss.asnumpy()
            total_correct += correct.sum().asnumpy()
            total_samples += len(batch_labels)
            
            # 统计各类别准确率
            for i in range(len(batch_labels)):
                true_label = batch_labels[i]
                pred_label = predictions[i].asnumpy()
                
                class_total[true_label] += 1
                if pred_label == true_label:
                    class_correct[true_label] += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # 打印各类别准确率
        vocab_items = list(self.vocab.items())
        logger.info("各类别准确率:")
        for class_id, count in class_total.items():
            if count > 0:
                class_name = next((name for name, id in vocab_items if id == class_id), f"Class_{class_id}")
                class_acc = class_correct[class_id] / count
                logger.info(f"  {class_name}: {class_acc:.4f}")
        
        return avg_loss, accuracy
    
    def train(self):
        """开始训练"""
        logger.info("🎯 开始高准确率训练...")
        logger.info(f"⏱️  预期每轮训练时间将大幅增加（数据增强{self.config.augment_factor}倍）")
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            
            logger.info(f"开始第 {epoch}/{self.config.epochs} 轮训练...")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            logger.info(f"Epoch {epoch} 训练完成:")
            logger.info(f"  平均损失: {train_loss:.4f}")
            logger.info(f"  训练准确率: {train_acc:.4f}")
            
            # 评估
            logger.info("开始模型评估...")
            val_loss, val_acc = self.evaluate()
            
            logger.info(f"评估完成:")
            logger.info(f"  验证损失: {val_loss:.4f}")
            logger.info(f"  验证准确率: {val_acc:.4f}")
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['epoch_times'].append(epoch_time)
            
            logger.info(f"Epoch {epoch} 总结:")
            logger.info(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            logger.info(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            logger.info(f"  耗时: {epoch_time:.2f}秒")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                save_checkpoint(self.model, "output/high_accuracy_best_model.ckpt")
                logger.info(f"新的最佳验证准确率: {val_acc:.4f}")
                logger.info("最佳模型已保存: output/high_accuracy_best_model.ckpt")
            else:
                self.patience_counter += 1
                logger.info(f"验证准确率未提升，耐心计数: {self.patience_counter}/{self.config.patience}")
            
            # 早停检查
            if epoch >= self.config.min_epochs and self.patience_counter >= self.config.patience:
                logger.info(f"早停触发，在第 {epoch} 轮停止训练")
                break
        
        logger.info(f"训练完成! 最佳验证准确率: {self.best_val_acc:.4f}")
        return self.best_val_acc
    
    def save_final_model(self):
        """保存最终模型"""
        # 保存模型
        save_checkpoint(self.model, "output/high_accuracy_final_model.ckpt")
        
        # 保存词汇表
        vocab_data = {
            'vocab': self.vocab,
            'num_classes': self.config.num_classes,
            'label_names': list(self.vocab.keys())
        }
        
        with open("output/high_accuracy_vocab.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # 保存训练历史
        with open("output/high_accuracy_training_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info("✅ 高准确率模型和词汇表保存完成")

def main():
    """主函数"""
    print("🚀 高准确率CE-CSL手语识别训练启动")
    print("🔧 主要优化:")
    print("  ✓ 激进数据增强: 每个样本增强20倍，解决数据严重不足")
    print("  ✓ 模型简化: 防止过拟合，专为小数据集设计")
    print("  ✓ 强正则化: 高dropout + 权重衰减")
    print("  ✓ 保守训练: 小批次 + 适中学习率")
    
    # 创建配置
    config = HighAccuracyConfig()
    
    print("📊 详细配置:")
    print(f"  - 训练轮数: {config.epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 权重衰减: {config.weight_decay}")
    print(f"  - 设备: {config.device_target}")
    print(f"  - 隐藏维度: {config.hidden_size}")
    print(f"  - Dropout率: {config.dropout_rate}")
    print(f"  - 数据目录: {config.data_dir}")
    print(f"  - 数据增强倍数: {config.augment_factor}")
    print(f"  - 早停耐心值: {config.patience}")
    
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    
    # 创建训练器
    trainer = HighAccuracyTrainer(config)
    
    # 开始训练
    best_acc = trainer.train()
    
    # 保存模型
    print("💾 保存最终模型...")
    trainer.save_final_model()
    
    print("🎉 高准确率训练完成！")
    print(f"📁 模型已保存到: ./output/high_accuracy_final_model.ckpt")
    print(f"📊 训练历史已保存到: ./output/high_accuracy_training_history.json")
    print(f"🏆 最佳验证准确率: {best_acc:.4f}")
    print("✨ 主要改进效果:")
    print("  ✓ 数据量大幅增加（20倍增强）")
    print("  ✓ 模型专为小数据集优化")
    print("  ✓ 强正则化防止过拟合")
    print("  ✓ 保守训练策略确保稳定收敛")

if __name__ == "__main__":
    main()
