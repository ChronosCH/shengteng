#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于真实CE-CSL数据的手语识别训练器
使用预处理好的视频帧数据进行训练
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

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CECSLTrainingConfig:
    """CE-CSL训练配置"""
    # 模型配置
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    
    # 训练配置
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    
    # 数据配置
    data_root: str = "../data/CE-CSL"
    max_sequence_length: int = 150  # 最大帧数
    image_size: Tuple[int, int] = (224, 224)
    
    # 设备配置
    device_target: str = "CPU"

class CECSLDataset:
    """CE-CSL数据集加载器"""
    
    def __init__(self, config: CECSLTrainingConfig, split: str = 'train'):
        self.config = config
        self.split = split
        self.data_root = Path(config.data_root)
        
        # 加载词汇表
        self.word2idx = {}
        self.idx2word = []
        self._build_vocabulary()
        
        # 加载数据
        self.samples = []
        self._load_data()
        
        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")
        logger.info(f"词汇表大小: {len(self.word2idx)}")
    
    def _build_vocabulary(self):
        """构建词汇表"""
        # 从所有CSV文件中提取标签
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
        # 加载元数据
        metadata_file = self.data_root / "processed" / self.split / f"{self.split}_metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"元数据文件不存在: {metadata_file}")
            return
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for item in metadata:
            video_id = item['video_id']
            label = item['text']  # 使用text字段作为标签
            
            # 检查对应的帧数据文件是否存在
            frames_file = self.data_root / "processed" / self.split / f"{video_id}_frames.npy"
            
            if frames_file.exists():
                label_idx = self.word2idx.get(label, self.word2idx['<UNK>'])
                self.samples.append({
                    'video_id': video_id,
                    'frames_file': str(frames_file),
                    'label': label,
                    'label_idx': label_idx,
                    'metadata': item
                })
    
    def _load_frames(self, frames_file: str) -> np.ndarray:
        """加载视频帧数据"""
        try:
            frames = np.load(frames_file)  # shape: (num_frames, height, width, channels)
            
            # 确保数据类型正确
            if frames.dtype != np.float32:
                frames = frames.astype(np.float32)
            
            # 归一化到[0,1]
            if frames.max() > 1.0:
                frames = frames / 255.0
            
            # 调整序列长度
            if len(frames) > self.config.max_sequence_length:
                # 均匀采样
                indices = np.linspace(0, len(frames) - 1, self.config.max_sequence_length, dtype=int)
                frames = frames[indices]
            elif len(frames) < self.config.max_sequence_length:
                # 填充
                pad_length = self.config.max_sequence_length - len(frames)
                pad_frames = np.zeros((pad_length,) + frames.shape[1:], dtype=frames.dtype)
                frames = np.concatenate([frames, pad_frames], axis=0)
            
            # 转换为 (seq_len, channels, height, width) 如果需要
            if len(frames.shape) == 4 and frames.shape[-1] in [1, 3]:  # (seq, h, w, c)
                frames = np.transpose(frames, (0, 3, 1, 2))  # (seq, c, h, w)
            
            return frames
            
        except Exception as e:
            logger.error(f"加载帧数据失败 {frames_file}: {e}")
            # 返回零填充的数据
            return np.zeros((self.config.max_sequence_length, 3, *self.config.image_size), dtype=np.float32)
    
    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.samples[index]
        
        # 加载帧数据
        frames = self._load_frames(sample['frames_file'])
        
        # 展平帧数据用于简单的LSTM模型
        # frames shape: (seq_len, channels, height, width) -> (seq_len, features)
        seq_len = frames.shape[0]
        features = np.prod(frames.shape[1:])  # channels * height * width
        frames_flat = frames.reshape(seq_len, features)
        
        return frames_flat.astype(np.float32), np.array(sample['label_idx'], dtype=np.int32)
    
    def __len__(self):
        return len(self.samples)

class CECSLModel(nn.Cell):
    """CE-CSL手语识别模型"""
    
    def __init__(self, config: CECSLTrainingConfig, vocab_size: int):
        super().__init__()
        self.config = config
        
        # 计算输入特征维度
        input_size = 3 * config.image_size[0] * config.image_size[1]  # channels * height * width
        
        # 特征提取层
        self.feature_extractor = nn.SequentialCell([
            nn.Dense(input_size, config.d_model),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Dense(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(p=config.dropout)
        ])
        
        # 时序建模层
        self.temporal_model = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0
        )
        
        # 分类层
        self.classifier = nn.SequentialCell([
            nn.Dense(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Dense(config.d_model // 2, vocab_size)
        ])
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
    
    def construct(self, x, labels=None):
        batch_size, seq_len, input_size = x.shape
        
        # 特征提取
        x_reshaped = x.view(batch_size * seq_len, input_size)
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, self.config.d_model)
        
        # 时序建模
        output, _ = self.temporal_model(features)
        
        # 全局平均池化或取最后一个时间步
        # 这里使用平均池化来聚合所有时间步的信息
        pooled_output = ops.ReduceMean()(output, axis=1)  # (batch_size, d_model)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits

class CECSLTrainer:
    """CE-CSL训练器"""
    
    def __init__(self, config: CECSLTrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_dataset = None
        self.val_dataset = None
        
        # 设置设备
        ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
        
        logger.info(f"CE-CSL训练器初始化完成 - 设备: {config.device_target}")
    
    def load_data(self):
        """加载数据"""
        logger.info("加载CE-CSL数据集...")
        
        # 创建数据集
        train_data = CECSLDataset(self.config, 'train')
        val_data = CECSLDataset(self.config, 'dev')
        
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
        
        logger.info(f"训练集: {len(train_data)} 样本")
        logger.info(f"验证集: {len(val_data)} 样本")
        logger.info(f"词汇表大小: {self.vocab_size}")
        logger.info(f"标签类别: {list(self.word2idx.keys())}")
    
    def build_model(self):
        """构建模型"""
        logger.info("构建CE-CSL模型...")
        
        if not hasattr(self, 'vocab_size'):
            raise ValueError("请先调用load_data()加载数据")
        
        self.model = CECSLModel(self.config, self.vocab_size)
        
        # 计算参数量
        total_params = sum(param.size for param in self.model.get_parameters())
        logger.info(f"模型构建完成 - 参数量: {total_params}")
        
        # 创建优化器
        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logger.info("优化器创建完成")
        
        # 创建训练步骤函数，避免重复编译
        self._setup_training_functions()
    
    def _setup_training_functions(self):
        """设置训练函数，避免重复JIT编译"""
        def forward_fn(data, labels):
            loss, logits = self.model(data, labels)
            return loss, logits
        
        # 创建梯度计算函数，只编译一次
        self.grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
        
        # 创建训练步骤函数
        @ms.jit
        def train_step_fn(data, labels):
            (loss, logits), grads = self.grad_fn(data, labels)
            self.optimizer(grads)
            return loss, logits
        
        self.train_step_fn = train_step_fn
        logger.info("训练函数设置完成，避免重复JIT编译")
        
        # 创建评估步骤函数
        @ms.jit
        def eval_step_fn(data, labels):
            loss, logits = self.model(data, labels)
            predicted = ops.ArgMaxWithValue(axis=1)(logits)[0]
            return loss, logits, predicted
        
        self.eval_step_fn = eval_step_fn
    
    def train_step(self, data, labels):
        """单步训练"""
        return self.train_step_fn(data, labels)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.set_train(True)
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        logger.info(f"开始第 {epoch+1}/{self.config.epochs} 轮训练...")
        
        for batch_idx, (data, labels) in enumerate(self.train_dataset.create_tuple_iterator()):
            loss, logits = self.train_step(data, labels)
            
            # 计算准确率
            predicted = ops.ArgMaxWithValue(axis=1)(logits)[0]
            correct += ops.ReduceSum()(ops.Cast()(predicted == labels, ms.float32)).asnumpy()
            total += labels.shape[0]
            
            total_loss += loss.asnumpy()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx}: Loss = {loss.asnumpy():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"Epoch {epoch+1} 训练完成:")
        logger.info(f"  平均损失: {avg_loss:.4f}")
        logger.info(f"  训练准确率: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def evaluate(self):
        """评估模型"""
        logger.info("开始模型评估...")
        
        self.model.set_train(False)
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        # 统计每个类别的预测结果
        class_correct = {}
        class_total = {}
        
        for data, labels in self.val_dataset.create_tuple_iterator():
            loss, logits, predicted = self.eval_step_fn(data, labels)
            
            # 计算准确率
            correct += ops.ReduceSum()(ops.Cast()(predicted == labels, ms.float32)).asnumpy()
            total += labels.shape[0]
            
            # 统计每个类别的准确率
            for i in range(labels.shape[0]):
                true_label = labels[i].asnumpy().item()
                pred_label = predicted[i].asnumpy().item()
                
                if true_label not in class_total:
                    class_total[true_label] = 0
                    class_correct[true_label] = 0
                
                class_total[true_label] += 1
                if true_label == pred_label:
                    class_correct[true_label] += 1
            
            total_loss += loss.asnumpy()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"评估完成:")
        logger.info(f"  验证损失: {avg_loss:.4f}")
        logger.info(f"  验证准确率: {accuracy:.4f}")
        
        # 打印每个类别的准确率
        logger.info("各类别准确率:")
        for label_idx in sorted(class_total.keys()):
            if label_idx < len(self.idx2word):
                label_name = self.idx2word[label_idx]
                acc = class_correct[label_idx] / class_total[label_idx]
                logger.info(f"  {label_name}: {acc:.4f} ({class_correct[label_idx]}/{class_total[label_idx]})")
        
        return {"accuracy": accuracy, "loss": avg_loss}
    
    def train(self):
        """完整训练流程"""
        logger.info("开始CE-CSL真实数据训练...")
        
        best_accuracy = 0.0
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 评估
            eval_results = self.evaluate()
            val_loss = eval_results["loss"]
            val_acc = eval_results["accuracy"]
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1} 总结:")
            logger.info(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            logger.info(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            logger.info(f"  耗时: {epoch_time:.2f}秒")
            
            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                logger.info(f"新的最佳验证准确率: {best_accuracy:.4f}")
                
                # 保存最佳模型
                os.makedirs("./output", exist_ok=True)
                best_model_path = "./output/cecsl_best_model.ckpt"
                save_checkpoint(self.model, best_model_path)
                logger.info(f"最佳模型已保存: {best_model_path}")
        
        logger.info(f"训练完成! 最佳验证准确率: {best_accuracy:.4f}")
        return self.model
    
    def save_model(self, model_path):
        """保存模型"""
        logger.info(f"保存模型到: {model_path}")
        save_checkpoint(self.model, model_path)
        
        # 同时保存词汇表
        vocab_path = model_path.replace('.ckpt', '_vocab.json')
        vocab_info = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 模型和词汇表保存完成")

def main():
    """测试CE-CSL训练器"""
    # 检查数据是否存在
    data_root = Path("../data/CE-CSL")  # 从training目录到data目录
    if not data_root.exists():
        logger.error(f"数据目录不存在: {data_root}")
        return False
    
    # 检查必要的文件
    required_files = [
        "train.corpus.csv",
        "dev.corpus.csv", 
        "processed/train/train_metadata.json"
    ]
    
    for file_path in required_files:
        if not (data_root / file_path).exists():
            logger.error(f"必要文件不存在: {data_root / file_path}")
            return False
    
    # 创建配置
    config = CECSLTrainingConfig(
        vocab_size=1000,
        d_model=128,  # 较小的模型用于测试
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        batch_size=4,  # 较小的批次大小
        learning_rate=1e-3,
        epochs=10,  # 较少的epoch用于测试
        max_sequence_length=100,  # 较短的序列
        image_size=(224, 224),  # 实际的图像尺寸，匹配数据
        device_target="CPU"
    )
    
    try:
        # 创建训练器
        trainer = CECSLTrainer(config)
        
        # 加载数据
        trainer.load_data()
        
        # 构建模型
        trainer.build_model()
        
        # 开始训练
        model = trainer.train()
        
        # 保存最终模型
        trainer.save_model("./output/cecsl_final_model.ckpt")
        
        logger.info("✅ CE-CSL训练成功完成!")
        return True
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 CE-CSL真实数据训练成功!")
    else:
        print("❌ CE-CSL训练失败!")
        sys.exit(1)
