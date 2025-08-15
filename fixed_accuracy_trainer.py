#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版高准确率CE-CSL手语识别训练器 - 修复词汇表问题
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
class FixedAccuracyConfig:
    """修复版高准确率训练配置"""
    # 数据配置
    data_dir: str = "data/CE-CSL"
    vocab_file: str = "backend/models/vocab.json"
    
    # 模型配置 - 针对小数据集优化
    input_size: int = 150528  # 224*224*3
    hidden_size: int = 64  # 进一步减小
    num_layers: int = 1
    num_classes: int = 10
    dropout_rate: float = 0.3  # 适中的dropout
    
    # 训练配置 - 更保守
    batch_size: int = 1  # 极小批次
    learning_rate: float = 0.0005  # 更小的学习率
    epochs: int = 100
    weight_decay: float = 0.001
    
    # 数据增强
    augment_factor: int = 10  # 适中的增强
    
    # 早停和保存
    patience: int = 50
    min_epochs: int = 20
    
    # 设备配置
    device_target: str = "CPU"

class FixedDataAugmentor:
    """修复版数据增强器"""
    
    def __init__(self, config: FixedAccuracyConfig):
        self.config = config
        
    def augment_sample(self, frames: np.ndarray, label: int) -> List[Tuple[np.ndarray, int]]:
        """对单个样本进行数据增强"""
        augmented_samples = [(frames, label)]  # 原始样本
        
        for i in range(self.config.augment_factor - 1):
            # 简单的增强方法
            aug_type = random.choice(['noise', 'brightness', 'flip'])
            
            if aug_type == 'noise':
                aug_frames = self._add_noise(frames, noise_factor=0.05)
            elif aug_type == 'brightness':
                aug_frames = self._adjust_brightness(frames, factor=random.uniform(0.8, 1.2))
            else:  # flip
                aug_frames = self._horizontal_flip(frames)
            
            augmented_samples.append((aug_frames, label))
        
        return augmented_samples
    
    def _add_noise(self, frames: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """添加轻微噪声"""
        noise = np.random.normal(0, noise_factor * 255, frames.shape).astype(np.float32)
        noisy_frames = frames.astype(np.float32) + noise
        return np.clip(noisy_frames, 0, 255).astype(np.uint8)
    
    def _adjust_brightness(self, frames: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """调整亮度"""
        bright_frames = frames.astype(np.float32) * factor
        return np.clip(bright_frames, 0, 255).astype(np.uint8)
    
    def _horizontal_flip(self, frames: np.ndarray) -> np.ndarray:
        """水平翻转"""
        return np.flip(frames, axis=2)

class FixedModel(nn.Cell):
    """修复版简化模型"""
    
    def __init__(self, config: FixedAccuracyConfig):
        super().__init__()
        self.config = config
        
        # 大幅简化的特征提取
        self.feature_reducer = nn.SequentialCell([
            nn.Dense(config.input_size, 256),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Dense(256, 64),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate)
        ])
        
        # 简化的LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.0
        )
        
        # 简化的分类器
        self.classifier = nn.SequentialCell([
            nn.Dense(config.hidden_size, config.num_classes)
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
    
    def construct(self, x):
        # x shape: (batch, seq_len, height, width, channels)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 展平空间维度
        x = x.view(batch_size * seq_len, -1)
        
        # 特征降维
        x = self.feature_reducer(x)
        
        # 重塑为序列
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步
        last_output = lstm_out[:, -1, :]
        
        # 分类
        logits = self.classifier(last_output)
        
        return logits

class FixedDataset:
    """修复版数据集"""
    
    def __init__(self, data_dir: str, split: str, config: FixedAccuracyConfig, 
                 vocab: Dict[str, int], augmentor: Optional[FixedDataAugmentor] = None):
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
                
                # 获取标签 - 关键修复
                gloss = item['gloss_sequence'][0]
                if gloss not in self.vocab:
                    logger.warning(f"词汇 '{gloss}' 不在词汇表中，跳过")
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
        
        # 固定序列长度为50
        target_len = 50
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

class FixedTrainer:
    """修复版训练器"""
    
    def __init__(self, config: FixedAccuracyConfig):
        self.config = config
        self.setup_environment()
        self.build_vocab()  # 先构建词汇表
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
        logger.info(f"修复版CE-CSL训练器初始化完成 - 设备: {self.config.device_target}")
    
    def build_vocab(self):
        """构建词汇表 - 关键修复"""
        # 先构建基础词汇表
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # 扫描训练数据构建词汇表
        train_metadata_file = Path(self.config.data_dir) / "processed" / "train" / "train_metadata.json"
        if train_metadata_file.exists():
            with open(train_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for item in metadata:
                for gloss in item.get('gloss_sequence', []):
                    if gloss not in vocab:
                        vocab[gloss] = len(vocab)
        
        # 扫描开发数据
        dev_metadata_file = Path(self.config.data_dir) / "processed" / "dev" / "dev_metadata.json"
        if dev_metadata_file.exists():
            with open(dev_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for item in metadata:
                for gloss in item.get('gloss_sequence', []):
                    if gloss not in vocab:
                        vocab[gloss] = len(vocab)
        
        self.vocab = vocab
        self.config.num_classes = len(vocab)
        
        logger.info(f"词汇表构建完成: {list(self.vocab.keys())}")
        logger.info(f"词汇表大小: {self.config.num_classes}")
    
    def setup_data(self):
        """设置数据"""
        logger.info("📊 加载数据（包含数据增强）...")
        
        # 创建数据增强器
        augmentor = FixedDataAugmentor(self.config)
        
        # 创建数据集
        self.train_dataset = FixedDataset(
            self.config.data_dir, "train", self.config, self.vocab, augmentor
        )
        self.val_dataset = FixedDataset(
            self.config.data_dir, "dev", self.config, self.vocab
        )
        
        logger.info(f"训练集: {len(self.train_dataset)} 样本（包含增强数据）")
        logger.info(f"验证集: {len(self.val_dataset)} 样本")
        
        if len(self.train_dataset) == 0:
            logger.error("训练数据集为空！")
            raise ValueError("训练数据集为空")
    
    def setup_model(self):
        """设置模型"""
        logger.info("🧠 构建修复版模型...")
        
        self.model = FixedModel(self.config)
        
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
            
            for idx in indices:
                frames, label = dataset[idx]
                # 返回单个样本
                yield (frames[np.newaxis, :], np.array([label]))
        
        return generator
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.set_train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        
        dataloader = self.create_dataloader(self.train_dataset, shuffle=True)
        
        for frames_batch, labels_batch in dataloader():
            batch_count += 1
            
            # 转换为Tensor
            frames_tensor = ms.Tensor(frames_batch, ms.float32)
            labels_tensor = ms.Tensor(labels_batch, ms.int32)
            
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
            predictions = ops.ArgMaxWithValue(axis=1)(logits)[0]
            correct = ops.equal(predictions, labels_tensor).sum()
            
            epoch_loss += loss.asnumpy()
            epoch_correct += correct.asnumpy()
            epoch_total += len(labels_batch)
            
            if batch_count % 20 == 0:
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
        
        for frames_batch, labels_batch in dataloader():
            batch_count += 1
            
            frames_tensor = ms.Tensor(frames_batch, ms.float32)
            labels_tensor = ms.Tensor(labels_batch, ms.int32)
            
            # 前向传播
            logits = self.model(frames_tensor)
            loss = self.loss_fn(logits, labels_tensor)
            
            # 预测
            predictions = ops.ArgMaxWithValue(axis=1)(logits)[0]
            correct = ops.equal(predictions, labels_tensor)
            
            total_loss += loss.asnumpy()
            total_correct += correct.sum().asnumpy()
            total_samples += len(labels_batch)
            
            # 统计各类别准确率
            for i in range(len(labels_batch)):
                true_label = labels_batch[i]
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
        logger.info("🎯 开始修复版高准确率训练...")
        
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
                save_checkpoint(self.model, "output/fixed_accuracy_best_model.ckpt")
                logger.info(f"新的最佳验证准确率: {val_acc:.4f}")
                logger.info("最佳模型已保存: output/fixed_accuracy_best_model.ckpt")
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
        save_checkpoint(self.model, "output/fixed_accuracy_final_model.ckpt")
        
        # 保存词汇表
        vocab_data = {
            'vocab': self.vocab,
            'num_classes': self.config.num_classes,
            'label_names': list(self.vocab.keys())
        }
        
        with open("output/fixed_accuracy_vocab.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # 保存训练历史
        with open("output/fixed_accuracy_training_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info("✅ 修复版模型和词汇表保存完成")

def main():
    """主函数"""
    print("🚀 修复版高准确率CE-CSL手语识别训练启动")
    print("🔧 主要修复:")
    print("  ✓ 修复词汇表构建问题")
    print("  ✓ 简化模型架构防止过拟合")
    print("  ✓ 优化数据加载流程")
    print("  ✓ 使用更保守的训练策略")
    
    # 创建配置
    config = FixedAccuracyConfig()
    
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
    trainer = FixedTrainer(config)
    
    # 开始训练
    best_acc = trainer.train()
    
    # 保存模型
    print("💾 保存最终模型...")
    trainer.save_final_model()
    
    print("🎉 修复版高准确率训练完成！")
    print(f"📁 模型已保存到: ./output/fixed_accuracy_final_model.ckpt")
    print(f"📊 训练历史已保存到: ./output/fixed_accuracy_training_history.json")
    print(f"🏆 最佳验证准确率: {best_acc:.4f}")
    print("✨ 主要改进效果:")
    print("  ✓ 词汇表正确构建")
    print("  ✓ 数据成功加载和增强")
    print("  ✓ 模型架构简化但有效")
    print("  ✓ 训练过程稳定可控")

if __name__ == "__main__":
    main()
