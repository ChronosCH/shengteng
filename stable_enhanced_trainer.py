#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定增强CE-CSL手语识别训练器
基于enhanced_ultra_simple_trainer.py的稳定版本，添加渐进式优化技术
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, Parameter
from mindspore.nn import Cell
from mindspore.train import Model
from mindspore.nn.loss import CrossEntropyLoss
from mindspore.nn.optim import Adam
from mindspore.nn.metrics import Accuracy

# 配置环境
os.environ['GLOG_v'] = '2'
os.environ['GLOG_logtostderr'] = '1'
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 0.001
    min_learning_rate: float = 0.0001
    hidden_dim: int = 20  # 稍微增加隐藏层
    dropout_rate: float = 0.3
    warmup_epochs: int = 5
    patience: int = 20
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    data_augmentation_factor: int = 12  # 增强倍数
    vocab_size: int = 10
    sequence_length: int = 100
    num_frames: int = 543  # 每个视频的关键点数量
    checkpoint_dir: str = "checkpoints/stable_enhanced"

class StableEnhancedModel(Cell):
    """稳定的增强模型"""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # 特征提取层
        self.feature_extractor = nn.SequentialCell([
            nn.Dense(config.num_frames, config.hidden_dim * 2),
            nn.BatchNorm1d(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate)
        ])
        
        # 时序建模层 - 简化的注意力机制
        self.temporal_layer = nn.SequentialCell([
            nn.Dense(config.hidden_dim * 2, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate)
        ])
        
        # 分类层
        self.classifier = nn.SequentialCell([
            nn.Dense(config.hidden_dim, config.vocab_size),
        ])
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for cell in self.cells():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(ms.common.initializer.initializer(
                    "xavier_uniform", cell.weight.shape, cell.weight.dtype
                ))
                if cell.bias is not None:
                    cell.bias.set_data(ms.common.initializer.initializer(
                        "zeros", cell.bias.shape, cell.bias.dtype
                    ))
    
    def construct(self, x):
        """前向传播"""
        # x shape: (batch_size, sequence_length, num_frames)
        batch_size, seq_len, _ = x.shape
        
        # 处理每个时间步
        outputs = []
        for i in range(seq_len):
            frame = x[:, i, :]  # (batch_size, num_frames)
            
            # 特征提取
            features = self.feature_extractor(frame)  # (batch_size, hidden_dim*2)
            
            # 时序建模
            temporal_features = self.temporal_layer(features)  # (batch_size, hidden_dim)
            outputs.append(temporal_features)
        
        # 平均池化融合时序信息
        sequence_features = ops.Stack(axis=1)(outputs)  # (batch_size, seq_len, hidden_dim)
        pooled_features = ops.ReduceMean(keep_dims=False)(sequence_features, 1)  # (batch_size, hidden_dim)
        
        # 分类
        logits = self.classifier(pooled_features)  # (batch_size, vocab_size)
        
        return logits

class LabelSmoothingCrossEntropy(Cell):
    """标签平滑交叉熵损失"""
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.log_softmax = nn.LogSoftmax(axis=-1)
        
    def construct(self, logits, target):
        """计算标签平滑交叉熵"""
        log_probs = self.log_softmax(logits)
        
        # 使用简化的标签平滑
        smooth_factor = self.smoothing / (self.num_classes - 1)
        one_hot = ops.OneHot()(target, self.num_classes, 
                               Tensor(self.confidence, ms.float32), 
                               Tensor(smooth_factor, ms.float32))
        
        # 计算损失
        loss = -ops.ReduceSum(keep_dims=False)(one_hot * log_probs, -1)
        return ops.ReduceMean()(loss)

class CosineAnnealingLR:
    """余弦退火学习率调度器"""
    def __init__(self, base_lr: float, min_lr: float, total_epochs: int, warmup_epochs: int = 0):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
    
    def get_lr(self, epoch: int) -> float:
        """获取当前epoch的学习率"""
        if epoch < self.warmup_epochs:
            # Warmup阶段
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火阶段
            cosine_epoch = epoch - self.warmup_epochs
            cosine_total = self.total_epochs - self.warmup_epochs
            cosine_factor = 0.5 * (1 + np.cos(np.pi * cosine_epoch / cosine_total))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

class StableEnhancedTrainer:
    """稳定增强训练器"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.vocab = self._build_vocab()
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        
        # 创建检查点目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info("稳定增强CE-CSL训练器初始化完成")
        logger.info(f"词汇表: {self.vocab}")
        logger.info(f"词汇表大小: {len(self.vocab)}")
    
    def _build_vocab(self) -> List[str]:
        """构建词汇表"""
        return ['<PAD>', '<UNK>', '请', '谢谢', '你好', '再见', '好的', '是的', '我', '不是']
    
    def _create_enhanced_mock_data(self, split: str) -> Tuple[List[np.ndarray], List[int]]:
        """创建增强的模拟数据"""
        logger.info(f"创建增强模拟数据...")
        
        np.random.seed(42 if split == 'train' else 123)
        
        if split == 'train':
            base_samples = 250  # 基础样本数
            total_samples = base_samples * self.config.data_augmentation_factor
        else:
            total_samples = 32
        
        data, labels = [], []
        
        for i in range(total_samples):
            # 选择类别 (跳过PAD和UNK)
            label = np.random.choice(range(2, len(self.vocab)))
            
            # 生成基础模式
            base_pattern = self._generate_class_pattern(label)
            
            # 数据增强
            if split == 'train':
                augmented_pattern = self._apply_data_augmentation(base_pattern)
            else:
                augmented_pattern = base_pattern
            
            data.append(augmented_pattern)
            labels.append(label)
        
        logger.info(f"增强模拟数据创建完成 - {split}集: {len(data)} 样本")
        return data, labels
    
    def _generate_class_pattern(self, class_id: int) -> np.ndarray:
        """为特定类别生成特征模式"""
        pattern = np.random.randn(self.config.sequence_length, self.config.num_frames) * 0.1
        
        # 为每个类别添加独特的信号模式
        class_patterns = {
            2: (0.8, 0.3),   # 请
            3: (0.6, 0.7),   # 谢谢  
            4: (0.9, 0.2),   # 你好
            5: (0.4, 0.8),   # 再见
            6: (0.7, 0.5),   # 好的
            7: (0.5, 0.9),   # 是的
            8: (0.3, 0.6),   # 我
            9: (0.2, 0.4),   # 不是
        }
        
        if class_id in class_patterns:
            amp1, amp2 = class_patterns[class_id]
            
            # 添加时间模式
            for t in range(self.config.sequence_length):
                time_factor = t / self.config.sequence_length
                
                # 第一个特征组 (前1/3)
                feature_range1 = slice(0, self.config.num_frames // 3)
                pattern[t, feature_range1] += amp1 * np.sin(2 * np.pi * time_factor * (class_id + 1))
                
                # 第二个特征组 (中1/3)  
                feature_range2 = slice(self.config.num_frames // 3, 2 * self.config.num_frames // 3)
                pattern[t, feature_range2] += amp2 * np.cos(2 * np.pi * time_factor * (class_id + 2))
                
                # 第三个特征组 (后1/3) - 组合模式
                feature_range3 = slice(2 * self.config.num_frames // 3, self.config.num_frames)
                pattern[t, feature_range3] += (amp1 + amp2) / 2 * np.sin(4 * np.pi * time_factor * class_id)
        
        return pattern.astype(np.float32)
    
    def _apply_data_augmentation(self, pattern: np.ndarray) -> np.ndarray:
        """应用数据增强技术"""
        augmented = pattern.copy()
        
        # 1. 高斯噪声
        noise_level = np.random.uniform(0.01, 0.05)
        augmented += np.random.normal(0, noise_level, augmented.shape)
        
        # 2. 时间平移
        if np.random.random() < 0.3:
            shift = np.random.randint(-5, 6)
            if shift > 0:
                augmented = np.concatenate([augmented[shift:], augmented[:shift]], axis=0)
            elif shift < 0:
                augmented = np.concatenate([augmented[shift:], augmented[:shift]], axis=0)
        
        # 3. 特征缩放
        if np.random.random() < 0.3:
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented *= scale_factor
        
        # 4. 随机遮挡
        if np.random.random() < 0.2:
            mask_length = np.random.randint(5, 15)
            mask_start = np.random.randint(0, self.config.sequence_length - mask_length)
            augmented[mask_start:mask_start+mask_length] *= 0.1
        
        # 5. 特征维度扰动
        if np.random.random() < 0.3:
            feature_mask = np.random.random(self.config.num_frames) > 0.1
            augmented[:, ~feature_mask] *= np.random.uniform(0.5, 1.5)
        
        return augmented.astype(np.float32)
    
    def create_dataset(self, split: str):
        """创建数据集"""
        data, labels = self._create_enhanced_mock_data(split)
        
        # 转换为numpy数组
        data_array = np.array(data, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)
        
        # 创建数据集
        dataset = ms.dataset.NumpySlicesDataset((data_array, labels_array), column_names=["data", "label"])
        dataset = dataset.batch(self.config.batch_size, drop_remainder=False)
        
        return dataset
    
    def train(self):
        """执行训练"""
        logger.info("🎯 开始稳定增强训练...")
        
        # 创建数据集
        train_dataset = self.create_dataset('train')
        val_dataset = self.create_dataset('dev')
        
        # 创建模型
        model = StableEnhancedModel(self.config)
        
        # 创建损失函数
        loss_fn = LabelSmoothingCrossEntropy(
            num_classes=self.config.vocab_size,
            smoothing=self.config.label_smoothing
        )
        
        # 创建优化器
        optimizer = Adam(
            model.trainable_params(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        lr_scheduler = CosineAnnealingLR(
            base_lr=self.config.learning_rate,
            min_lr=self.config.min_learning_rate,
            total_epochs=self.config.num_epochs,
            warmup_epochs=self.config.warmup_epochs
        )
        
        # 训练状态
        best_val_acc = 0.0
        patience_counter = 0
        train_step = 0
        
        # 开始训练
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # 更新学习率 - MindSpore中需要重新创建优化器
            current_lr = lr_scheduler.get_lr(epoch)
            if epoch > 0:  # 第一个epoch使用初始学习率
                optimizer = Adam(
                    model.trainable_params(),
                    learning_rate=current_lr,
                    weight_decay=self.config.weight_decay
                )
            
            logger.info(f"开始第 {epoch+1}/{self.config.num_epochs} 轮训练... LR: {current_lr:.6f}")
            
            # 训练阶段
            model.set_train(True)
            train_losses = []
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch_data in enumerate(train_dataset.create_tuple_iterator()):
                train_step += 1
                
                # 解包数据 - MindSpore返回的是tuple
                data, labels = batch_data
                if not isinstance(data, Tensor):
                    data = Tensor(data, ms.float32)
                if not isinstance(labels, Tensor):
                    labels = Tensor(labels, ms.int32)
                
                # 前向传播
                def forward_fn(data, labels):
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    return loss, logits
                
                # 计算梯度
                grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                (loss, logits), grads = grad_fn(data, labels)
                
                # 梯度裁剪
                grads = ops.clip_by_global_norm(grads, self.config.gradient_clip_norm)
                
                # 更新参数
                optimizer(grads)
                
                # 统计
                train_losses.append(loss.asnumpy())
                preds = ops.Argmax(axis=1)(logits)
                train_correct += ops.ReduceSum()(ops.Equal()(preds, labels)).asnumpy()
                train_total += labels.shape[0]
                
                # 打印进度
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = np.mean(train_losses[-50:])
                    acc = train_correct / train_total
                    logger.info(f"批次 {batch_idx + 1}: Loss = {avg_loss:.4f}, 准确率 = {acc:.4f}")
            
            # 训练epoch统计
            epoch_train_loss = np.mean(train_losses)
            epoch_train_acc = train_correct / train_total
            
            logger.info(f"Epoch {epoch+1} 训练完成:")
            logger.info(f"  平均损失: {epoch_train_loss:.4f}")
            logger.info(f"  训练准确率: {epoch_train_acc:.4f}")
            
            # 验证阶段
            model.set_train(False)
            val_losses = []
            val_correct = 0
            val_total = 0
            class_correct = {i: 0 for i in range(2, self.config.vocab_size)}
            class_total = {i: 0 for i in range(2, self.config.vocab_size)}
            
            logger.info("开始验证...")
            for batch_data in val_dataset.create_tuple_iterator():
                # 解包数据 - MindSpore返回的是tuple
                data, labels = batch_data
                if not isinstance(data, Tensor):
                    data = Tensor(data, ms.float32)
                if not isinstance(labels, Tensor):
                    labels = Tensor(labels, ms.int32)
                
                logits = model(data)
                loss = loss_fn(logits, labels)
                
                val_losses.append(loss.asnumpy())
                preds = ops.Argmax(axis=1)(logits)
                
                # 总体统计
                val_correct += ops.ReduceSum()(ops.Equal()(preds, labels)).asnumpy()
                val_total += labels.shape[0]
                
                # 各类别统计
                for i in range(labels.shape[0]):
                    true_label = int(labels[i].asnumpy())
                    pred_label = int(preds[i].asnumpy())
                    
                    if true_label in class_total:
                        class_total[true_label] += 1
                        if true_label == pred_label:
                            class_correct[true_label] += 1
            
            # 验证epoch统计
            epoch_val_loss = np.mean(val_losses)
            epoch_val_acc = val_correct / val_total
            
            # 打印各类别准确率
            logger.info("各类别准确率:")
            for class_id in range(2, self.config.vocab_size):
                class_name = self.id_to_word[class_id]
                if class_total[class_id] > 0:
                    class_acc = class_correct[class_id] / class_total[class_id]
                    logger.info(f"  {class_name}: {class_acc:.4f} ({class_correct[class_id]}/{class_total[class_id]})")
                else:
                    logger.info(f"  {class_name}: 无样本")
            
            logger.info(f"验证完成:")
            logger.info(f"  验证损失: {epoch_val_loss:.4f}")
            logger.info(f"  验证准确率: {epoch_val_acc:.4f}")
            
            # 检查是否为最佳模型
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0
                
                # 保存最佳模型
                checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.ckpt")
                ms.save_checkpoint(model, checkpoint_path)
                logger.info(f"🎉 新的最佳验证准确率: {best_val_acc:.4f}")
                logger.info(f"最佳模型已保存到: {checkpoint_path}")
            else:
                patience_counter += 1
                logger.info(f"验证准确率未提升，耐心计数: {patience_counter}/{self.config.patience}")
            
            # 计算耗时
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} 总结:")
            logger.info(f"  训练: Loss={epoch_train_loss:.4f}, Acc={epoch_train_acc:.4f}")
            logger.info(f"  验证: Loss={epoch_val_loss:.4f}, Acc={epoch_val_acc:.4f}")
            logger.info(f"  学习率: {current_lr:.6f}, 耗时: {epoch_time:.2f}秒")
            
            # 早停检查
            if patience_counter >= self.config.patience:
                logger.info(f"早停触发！最佳验证准确率: {best_val_acc:.4f}")
                break
        
        logger.info("🎉 训练完成！")
        logger.info(f"📊 最终统计:")
        logger.info(f"  最佳验证准确率: {best_val_acc:.4f}")
        logger.info(f"  训练轮数: {epoch+1}")
        logger.info(f"  模型保存路径: {self.config.checkpoint_dir}")

def main():
    """主函数"""
    print("🚀 稳定增强CE-CSL手语识别训练启动")
    print("🎯 渐进式优化方案:")
    print("  ✓ 增强数据增强 - 12倍数据")
    print("  ✓ 标签平滑 - 提升泛化")
    print("  ✓ 余弦退火学习率 - 稳定收敛")
    print("  ✓ 梯度裁剪 - 防止爆炸")
    print("  ✓ 权重衰减 - 正则化")
    print("  ✓ Warmup策略 - 稳定启动")
    print("  ✓ 时序特征提取 - 改进架构")
    
    config = TrainingConfig()
    print(f"📊 训练配置:")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 隐藏维度: {config.hidden_dim}")
    print(f"  - 数据增强倍数: {config.data_augmentation_factor}")
    print(f"  - 标签平滑: {config.label_smoothing}")
    print(f"  - 权重衰减: {config.weight_decay}")
    
    trainer = StableEnhancedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
