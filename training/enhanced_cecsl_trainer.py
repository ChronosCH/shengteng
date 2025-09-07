#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版CE-CSL手语识别训练器（修正版）
- 兼容 corpus 的 frames_path，支持 .npz/.npy，路径自动拼接
- 词表仅用 train 构建，过滤空/NaN
- 数据加载严格过滤空标签
- 优化器 warmup+cosine schedule，一次性创建，保留动量
- 梯度裁剪、类别权重、随机种子
- 评估日志显示每类样本数
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

# 随机种子
def set_seed(seed=20250831):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
set_seed()

@dataclass
class EnhancedCECSLConfig:
    """增强版CE-CSL训练配置"""
    vocab_size: int = 1000
    d_model: int = 192
    n_layers: int = 2
    dropout: float = 0.1  # 降低丢弃率提升早期收敛稳定性
    batch_size: int = 1
    learning_rate: float = 3e-4  # 提高初始学习率，加快摆脱全空白预测
    weight_decay: float = 1e-3
    epochs: int = 100
    warmup_epochs: int = 10
    data_root: str = "../data/CE-CSL"
    max_sequence_length: int = 64
    image_size: Tuple[int, int] = (112, 112)
    augment_factor: int = 1
    noise_std: float = 0.01
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)
    device_target: str = "CPU"
    patience: int = 20
    min_delta: float = 0.001
    max_target_length: int = 48  # 或更大，视数据分布
    # 新增调试参数
    debug_overfit: bool = False          # 设为 True 时只在第一批上过拟合，验证梯度是否有效
    debug_overfit_steps: int = 120       # 过拟合迭代步数
    log_first_batch_logits: bool = True  # 记录首批次logits分布

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
    
    def __init__(self, config: EnhancedCECSLConfig, split: str = 'train', use_augmentation: bool = True, token2idx=None):
        self.config = config
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.data_root = Path(config.data_root)
        self.token2idx = token2idx
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
        self._load_from_corpus()

        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")
        logger.info(f"词汇表大小: {len(self.word2idx)}")
    
    def _build_vocabulary(self):
        """构建词汇表"""
        import pandas as pd
        csv_file = self.data_root / "train.corpus.csv"
        all_labels = []
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            labels = [str(x).strip() for x in df["label"].tolist() if isinstance(x, str) and str(x).strip() != ""]
            all_labels.extend(labels)
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = ['<PAD>', '<UNK>']
        for label in sorted(set(all_labels)):
            if label not in self.word2idx:
                self.word2idx[label] = len(self.idx2word)
                self.idx2word.append(label)
        logger.info(f"词汇表构建完成: 类别数={len(self.idx2word)}（不含 <PAD>/<UNK>）")
    
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
                label = (row['label'] or "").strip()
                if self.split in ('train', 'dev') and label == "":
                    continue
                rel = row.get('frames_path', '')
                frames_file = (self.data_root / rel).resolve() if rel and not os.path.isabs(rel) else Path(rel)
                if frames_file.exists():
                    try:
                        label_idx = self.word2idx.get(label, self.word2idx['<UNK>'])
                        
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
    
    def __getitem__(self, index):
        """懒加载 + 在线增广 + 可选 2× 降采样 + 展平"""
        sample = self.samples[index]
        path = sample['frames_path']

        # 1) 懒加载 + 内存映射
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            with np.load(path) as z:
                thwc = z['frames']
        else:
            arr = np.load(path, mmap_mode='r' if ext == ".npy" else None)
            if arr.ndim != 4:
                raise ValueError(f"不支持的帧维度: {arr.shape}")
            thwc = arr if arr.shape[-1] in (1, 3) else np.transpose(arr, (0, 2, 3, 1))

        # 2) 可选 2× 降采样（224->112），与 config.image_size 配套
        if thwc.shape[1] == 224 and thwc.shape[2] == 224 and self.config.image_size == (112, 112):
            thwc = thwc[:, ::2, ::2, :]

        # 3) 归一化到 [0,1] + float32
        if thwc.dtype != np.float32:
            thwc = thwc.astype(np.float32, copy=False)
        if thwc.max() > 1.0:
            thwc = thwc / 255.0

        # 4) 在线增广（仅训练集）
        if self.use_augmentation and self.augmentor is not None:
            # 简化的在线增广组合
            if random.random() < 0.7:
                thwc = self.augmentor.add_noise(thwc)
            if random.random() < 0.5:
                thwc = self.augmentor.time_stretch(thwc)  # 会改变时间长度
            if random.random() < 0.6:
                thwc = self.augmentor.spatial_jitter(thwc)

        # 5) 转回 (T,C,H,W)
        tchw = np.transpose(thwc, (0, 3, 1, 2))

        # 6) 调整序列长度
        Tcur = tchw.shape[0]  # 有效帧数（增广后，padding前）
        if Tcur > self.config.max_sequence_length:
            idx = np.linspace(0, Tcur-1, self.config.max_sequence_length, dtype=int)
            tchw = tchw[idx]
        elif Tcur < self.config.max_sequence_length:
            pad = np.zeros((self.config.max_sequence_length - Tcur, *tchw.shape[1:]), dtype=tchw.dtype)
            tchw = np.concatenate([tchw, pad], axis=0)

        # 7) 展平为 (T, F)
        T = tchw.shape[0]
        F = int(np.prod(tchw.shape[1:], dtype=np.int64))
        frames_flat = tchw.reshape(T, F).astype(np.float32, copy=False)

        # 取标签序列
        gloss = sample['label']
        tokens = [t.strip() for t in gloss.split('/') if t.strip()]
        # 将 OOV token 映射为 <UNK>，不要映射为 <BLK>（CTC 的 blank 不能作为目标标签）
        targets = [self.token2idx.get(t, self.token2idx['<UNK>']) for t in tokens]
        target_len = len(targets)
        max_target_len = 32  # 可根据数据实际调整
        if target_len < max_target_len:
            targets += [self.token2idx['<PAD>']] * (max_target_len - target_len)
        else:
            targets = targets[:max_target_len]
            target_len = max_target_len
        input_len = min(Tcur, self.config.max_sequence_length)
        return frames_flat, np.array(targets, np.int32), np.array(input_len, np.int32), np.array(target_len, np.int32)
    
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
        
        # CTC头
        self.ctc_head = nn.Dense(config.d_model * 2, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0)  # blank id=0

        # 减少初始时刻对 <BLK> 的偏置，避免一开始全是空白预测
        try:
            b = self.ctc_head.bias.asnumpy()
            if b is not None and b.shape[0] >= 1:
                b[0] = -3.0  # <BLK>=0 的 bias 设为负
                self.ctc_head.bias.set_data(Tensor(b, ms.float32))
        except Exception:
            pass

    def construct(self, x, targets, input_len, target_len):
        batch_size, seq_len, input_size = x.shape
        
        # 特征提取
        x_reshaped = ops.reshape(x, (batch_size * seq_len, input_size))
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, self.config.d_model)
        
        # 双向LSTM
        lstm_output, _ = self.temporal_model(features)  # (batch, seq, hidden*2)
        
        # CTC解码
        logits = self.ctc_head(lstm_output)  # (batch, seq, vocab_size)
        log_probs = ops.log_softmax(logits, axis=2)
        log_probs = ops.transpose(log_probs, (1, 0, 2))   # (seq, batch, vocab)
        loss = self.ctc_loss(log_probs, targets, input_len, target_len)
        
        return loss, log_probs

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
        corpus_csv = Path(self.config.data_root) / "train.corpus.csv"
        token2idx, idx2token = build_token_vocab(corpus_csv)
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.vocab_size = len(token2idx)
        # 为稳定起见，先关闭在线增强（如需开启可将 use_augmentation=True）
        train_data = EnhancedCECSLDataset(self.config, 'train', use_augmentation=False, token2idx=token2idx)
        val_data = EnhancedCECSLDataset(self.config, 'dev', use_augmentation=False, token2idx=token2idx)
        
        if len(train_data) == 0:
            raise ValueError("训练数据集为空，请检查数据路径和预处理数据")
        
        # 创建MindSpore数据集，控制并行度避免内存峰值
        self.train_dataset = GeneratorDataset(
            train_data,
            column_names=["sequence", "targets", "input_len", "target_len"],
            shuffle=True, num_parallel_workers=1, python_multiprocessing=False
        ).batch(self.config.batch_size)

        self.val_dataset = GeneratorDataset(
            val_data,
            column_names=["sequence", "targets", "input_len", "target_len"],
            shuffle=False, num_parallel_workers=1, python_multiprocessing=False
        ).batch(self.config.batch_size)
        
        # 保持与 token 级词表一致（用于CTC）
        # 注意：不要用 EnhancedCECSLDataset 内部的整词 word2idx 覆盖，否则会与CTC的 token 级标签不一致
        logger.info(f"训练集: {len(train_data)} 样本（包含增强数据）")
        logger.info(f"验证集: {len(val_data)} 样本")
        logger.info(f"词汇表大小(CTC token级): {self.vocab_size}")
        logger.info(f"标签类别示例(前50个，去除特殊符号): {[t for t in self.idx2token if t not in ('<BLK>', '<PAD>')][:50]}")
    
    def _build_warmup_cosine_lr(self, base_lr, warmup_epochs, total_epochs, steps_per_epoch):
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        lr = np.zeros(total_steps, dtype=np.float32)
        for s in range(warmup_steps):
            lr[s] = base_lr * (s + 1) / warmup_steps
        for s in range(warmup_steps, total_steps):
            progress = (s - warmup_steps) / max(1, (total_steps - warmup_steps))
            lr[s] = base_lr * (1 + np.cos(np.pi * progress)) / 2
        return lr

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
        
        steps_per_epoch = max(1, len(self.train_dataset))
        lr_array = self._build_warmup_cosine_lr(
            self.config.learning_rate, self.config.warmup_epochs, self.config.epochs, steps_per_epoch
        )
        self.optimizer = nn.AdamWeightDecay(
            params=self.model.trainable_params(),
            learning_rate=Tensor(lr_array, ms.float32),
            weight_decay=self.config.weight_decay
        )
        
        # 类别权重
        from collections import Counter
        cnt = Counter()
        for batch in self.train_dataset.create_dict_iterator(output_numpy=True):
            for target_seq in batch["targets"]:
                for y in target_seq:
                    if int(y) > 1:  # 跳过 <BLK> 和 <PAD>
                        cnt[int(y)] += 1
        weights = np.ones(self.vocab_size, dtype=np.float32)
        for k, v in cnt.items():
            if v > 0:
                weights[k] = (sum(cnt.values()) / (len(cnt) * v))
        weights[0] = 0.0
        self.model.loss_fn = nn.CrossEntropyLoss(weight=Tensor(weights, ms.float32))
        
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
    
    def train_step(self, data, targets, input_len, target_len):
        def forward_fn(d, t, il, tl):
            loss, log_probs = self.model(d, t, il, tl)
            return loss, log_probs
        grad_fn = ms.value_and_grad(forward_fn, None, self.model.trainable_params(), has_aux=True)
        (loss, log_probs), grads = grad_fn(data, targets, input_len, target_len)
        # 修正：部分 MindSpore 版本 clip_by_global_norm 只返回裁剪梯度
        clipped_grads = ops.clip_by_global_norm(grads, 1.0)
        self.optimizer(clipped_grads)
        return loss, log_probs, None
    
    def evaluate(self):
        """评估模型"""
        self.model.set_train(False)
        total_loss, n_batches = 0.0, 0
        exact_match, total_samples = 0, 0
        total_edits, total_ref_tokens = 0, 0

        # 新增：类别统计（与CTC token词表一致）
        class_total = {w: 0 for w in self.idx2token}
        class_correct = {w: 0 for w in self.idx2token}

        for data, targets, input_len, target_len in self.val_dataset:
            data       = Tensor(data,       ms.float32)
            targets    = Tensor(targets,    ms.int32)
            input_len  = Tensor(input_len,  ms.int32)
            target_len = Tensor(target_len, ms.int32)

            loss, log_probs = self.model(data, targets, input_len, target_len)
            total_loss += float(loss.asnumpy()); n_batches += 1

            preds = ctc_greedy_decode(log_probs, self.idx2token, input_len)  # List[List[token]]

            targets_np    = targets.asnumpy()
            target_len_np = target_len.asnumpy()
            for b in range(targets_np.shape[0]):
                L = int(target_len_np[b])
                ref_ids = targets_np[b, :L].tolist()
                ref_tokens = [ self.idx2token[i] for i in ref_ids if i not in (0,1) ]
                hyp_tokens = preds[b]

                # 统计类别
                if len(ref_tokens) > 0:
                    main_label = ref_tokens[0]
                    class_total[main_label] += 1
                    if hyp_tokens == ref_tokens:
                        class_correct[main_label] += 1

                if hyp_tokens == ref_tokens:
                    exact_match += 1
                total_samples += 1

                d = edit_distance(hyp_tokens, ref_tokens)
                total_edits      += d
                total_ref_tokens += max(1, len(ref_tokens))

        avg_loss = total_loss / max(1, n_batches)
        em  = exact_match / max(1, total_samples)
        cer = total_edits / max(1, total_ref_tokens)
        # 新增：返回类别准确率统计
        class_accuracies = {w: (class_correct[w] / class_total[w] if class_total[w] > 0 else 0.0) for w in class_total}
        return avg_loss, em, cer, class_accuracies, class_total
    
    def train(self):
        """开始训练"""
        logger.info("开始增强版CE-CSL真实数据训练...")
        best_val_acc = 0
        training_history = []

        # 可选：取第一批用于过拟合调试
        first_batch_cache = None
        if self.config.debug_overfit:
            logger.warning("DEBUG: 启用单批次过拟合模式 (debug_overfit=True)，不会进行正常全数据训练！")
            for batch in self.train_dataset.create_tuple_iterator():
                first_batch_cache = batch
                break
            if first_batch_cache is None:
                raise RuntimeError("无法取得首批数据用于过拟合调试")
            logger.warning(f"DEBUG: 首批数据 shapes: {[x.shape for x in first_batch_cache]}")

            for step in range(self.config.debug_overfit_steps):
                data, targets, input_len, target_len = first_batch_cache
                data = Tensor(data, ms.float32)
                targets = Tensor(targets, ms.int32)
                input_len = Tensor(input_len, ms.int32)
                target_len = Tensor(target_len, ms.int32)
                self.model.set_train(True)
                loss, log_probs, gnorm = self.train_step(data, targets, input_len, target_len)
                if step % 10 == 0:
                    # 观察预测（按有效时长裁剪）
                    preds = ctc_greedy_decode(log_probs, self.idx2token, input_len)
                    tgt_np = targets.asnumpy(); tgt_len_np = target_len.asnumpy()
                    ref_ids = tgt_np[0, :int(tgt_len_np[0])].tolist()
                    ref_tokens = [self.idx2token[i] for i in ref_ids if i not in (0,1)]
                    gnorm_str = f"{gnorm:.3f}" if isinstance(gnorm, (int, float)) else str(gnorm)
                    logger.info(f"[Overfit Step {step}] loss={float(loss.asnumpy()):.4f} gnorm={gnorm_str} ref={ref_tokens} pred={preds[0]}")
                # 若已精确匹配可提前结束
                preds = ctc_greedy_decode(log_probs, self.idx2token, input_len)
                tgt_np = targets.asnumpy(); tgt_len_np = target_len.asnumpy()
                ref_ids = tgt_np[0, :int(tgt_len_np[0])].tolist()
                ref_tokens = [self.idx2token[i] for i in ref_ids if i not in (0,1)]
                if preds[0] == ref_tokens and step > 5:
                    logger.info(f"DEBUG: 成功在 {step} 步过拟合单批次，梯度更新有效。")
                    break
            logger.warning("DEBUG: 过拟合调试结束，退出训练流程。请关闭 debug_overfit 进行正式训练。")
            return self.model

        for epoch in range(self.config.epochs):
            start_time = time.time()
            self.model.set_train(True)
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            sum_gn = 0.0; gn_steps = 0
            logger.info(f"开始第 {epoch+1}/{self.config.epochs} 轮训练...")

            for batch_idx, (data, targets, input_len, target_len) in enumerate(self.train_dataset):
                data = Tensor(data, ms.float32)
                targets = Tensor(targets, ms.int32)
                input_len = Tensor(input_len, ms.int32)
                target_len = Tensor(target_len, ms.int32)

                loss, log_probs, gnorm = self.train_step(data, targets, input_len, target_len)
                if gnorm is not None:
                    sum_gn += gnorm
                    gn_steps += 1
                epoch_loss += float(loss.asnumpy())

                # 首批次可选记录 logits 分布（检查是否严重偏向 blank）
                if self.config.log_first_batch_logits and batch_idx == 0 and epoch == 0:
                    lp_np = log_probs.asnumpy()  # (T,B,V)
                    vocab_mean = lp_np.mean(axis=(0,1))  # (V,)
                    top5 = np.argsort(-vocab_mean)[:5]
                    logger.info("首批平均log概率Top5: " + ", ".join([f"{self.idx2token[i]}:{vocab_mean[i]:.2f}" for i in top5]))
                    logger.info(f"首批 blank(0) 平均logp: {vocab_mean[0]:.2f} | pad(1): {vocab_mean[1]:.2f}")

                # 计算训练EM（按有效时长裁剪）
                preds = ctc_greedy_decode(log_probs, self.idx2token, input_len)
                targets_np = targets.asnumpy(); target_len_np = target_len.asnumpy()
                for b in range(targets_np.shape[0]):
                    L = int(target_len_np[b])
                    ref_ids = targets_np[b, :L].tolist()
                    ref_tokens = [self.idx2token[i] for i in ref_ids if i not in (0,1)]
                    if preds[b] == ref_tokens:
                        epoch_correct += 1
                    epoch_total += 1

                if batch_idx % 10 == 0:
                    gnorm_str = f"{gnorm:.3f}" if gnorm is not None else "None"
                    logger.info(f"Batch {batch_idx}: Loss={float(loss.asnumpy()):.4f} TrainEM={epoch_correct/max(1,epoch_total):.4f} gNorm={gnorm_str}")

            avg_train_loss = epoch_loss / max(1, len(self.train_dataset))
            train_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            avg_gn = sum_gn / max(1, gn_steps)
            logger.info(f"Epoch {epoch+1} 训练完成: 平均损失={avg_train_loss:.4f} 训练EM={train_accuracy:.4f} 平均梯度范数={avg_gn:.3f}")
            logger.info(f"学习率(基准): {self.config.learning_rate:.6f}")

            # 验证
            logger.info("开始模型评估...")
            val_loss, val_em, val_cer, class_accuracies, class_total = self.evaluate()
            val_accuracy = 1.0 - val_cer
            logger.info(f"  验证损失: {val_loss:.4f} | EM: {val_em:.4f} | CER: {val_cer:.4f} | TokenAcc: {val_accuracy:.4f}")
            
            logger.info("各类别准确率（样本数）:")
            for word in sorted(class_total.keys()):
                if class_total[word] > 0:
                    logger.info(f"  {word}: {class_accuracies[word]:.4f}  (n={class_total[word]})")
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} 总结:")
            logger.info(f"  训练损失: {avg_train_loss:.4f}, 训练EM: {train_accuracy:.4f}")
            logger.info(f"  验证损失: {val_loss:.4f}, 验证EM: {val_em:.4f}, 验证TokenAcc: {val_accuracy:.4f}")
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
                'learning_rate': self.config.learning_rate,
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
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'vocab_size': self.vocab_size,
            'blank_id': 0,
            'pad_id': 1
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ 增强版模型和词汇表保存完成")

def build_token_vocab(corpus_csv):
    tokens = set()
    with open(corpus_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row['label']
            for token in gloss.split('/'):
                token = token.strip()
                if token:
                    tokens.add(token)
    # 加入 <UNK>，避免 OOV 错映射到 <BLK>
    idx2token = ['<BLK>', '<PAD>', '<UNK>'] + sorted(tokens)
    token2idx = {t: i for i, t in enumerate(idx2token)}
    return token2idx, idx2token

def ctc_greedy_decode(log_probs, idx2token, input_lengths=None):
    """
    基于贪心的CTC解码，支持按各样本有效时长裁剪，避免padding区域产生伪预测导致EM恒为0。
    Args:
        log_probs: (seq_len, batch_size, vocab_size) 的对数概率
        idx2token: 词表索引到token的映射
        input_lengths: (batch_size,) 每个样本的有效时长T，若为None则使用完整seq_len
    Returns:
        List[List[str]]: 每个样本的token序列
    """
    lp_np = log_probs.asnumpy() if hasattr(log_probs, 'asnumpy') else np.asarray(log_probs)
    seq_len, batch_size, _ = lp_np.shape
    if input_lengths is not None:
        il_np = input_lengths.asnumpy() if hasattr(input_lengths, 'asnumpy') else np.asarray(input_lengths)
        il_np = il_np.astype(np.int32)
    else:
        il_np = None

    pred_ids = np.argmax(lp_np, axis=2)  # (seq_len, batch_size)
    results = []
    for b in range(pred_ids.shape[1]):
        valid_len = int(il_np[b]) if il_np is not None else seq_len
        valid_len = max(0, min(valid_len, seq_len))
        seq = pred_ids[:valid_len, b]
        tokens = []
        prev = -1
        for t in seq:
            # 忽略 <BLK>(0) 和 <PAD>(1)，并进行相邻去重
            if t not in (0, 1) and t != prev:
                tokens.append(idx2token[t])
            prev = t
        results.append(tokens)
    return results

def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[la][lb]

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
        trainer.train()

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
