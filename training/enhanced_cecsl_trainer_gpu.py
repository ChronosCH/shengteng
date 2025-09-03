#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版CE-CSL手语识别训练器（GPU加速版，修正版）
- 兼容 corpus 的 frames_path，支持 .npz/.npy，路径自动拼接（懒加载/内存映射）
- 词表仅用 train 构建（CTC token级），含 <BLK>=0 / <PAD>=1
- 数据加载严格过滤空标签，支持在线增强 + 时间拉伸 + 抖动
- 优化器 warmup+cosine schedule（一次性创建全局 lr 序列，动量不丢失）
- GPU 加速：GRAPH_MODE、可选 AMP(O2)、训练步 ms.jit 编译
- 梯度裁剪、类别权重（可选）、随机种子
- 评估日志显示每类样本数（按 token 主标签统计）
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
import csv

# =========================
# 日志 & 随机种子
# =========================
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("CE-CSL-GPU")

def set_seed(seed=20250831):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)

set_seed()

# =========================
# 配置
# =========================
@dataclass
class EnhancedCECSLConfig:
    """增强版CE-CSL训练配置"""
    vocab_size: int = 1000
    d_model: int = 192
    n_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    epochs: int = 100
    warmup_epochs: int = 8
    data_root: str = "../data/CE-CSL"
    max_sequence_length: int = 64
    image_size: Tuple[int, int] = (112, 112)
    augment_factor: int = 1
    noise_std: float = 0.01
    time_stretch_range: Tuple[float, float] = (0.85, 1.15)
    device_target: str = "GPU"     # <<< 默认使用 GPU
    patience: int = 20
    min_delta: float = 1e-3
    max_target_length: int = 48
    # GPU/AMP 相关
    enable_amp: bool = False       # True 可开启 O2 混合精度（若遇到数值不稳请关闭）
    amp_level: str = "O2"          # O0/O1/O2
    num_workers: int = max(2, min(8, os.cpu_count() or 4))
    prefetch_size: int = 8

# =========================
# 数据增强
# =========================
class DataAugmentor:
    def __init__(self, config: EnhancedCECSLConfig):
        self.config = config

    def add_noise(self, frames: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.config.noise_std, frames.shape).astype(np.float32)
        return np.clip(frames + noise, 0, 1)

    def time_stretch(self, frames: np.ndarray) -> np.ndarray:
        stretch_factor = random.uniform(*self.config.time_stretch_range)
        original_length = len(frames)
        new_length = int(max(1, round(original_length * stretch_factor)))

        indices = np.linspace(0, original_length - 1, new_length)
        indices = np.clip(indices, 0, original_length - 1).astype(int)
        stretched = frames[indices]

        # pad/trim 至目标长度
        target_len = self.config.max_sequence_length
        if len(stretched) > target_len:
            idx = np.linspace(0, len(stretched) - 1, target_len, dtype=int)
            stretched = stretched[idx]
        elif len(stretched) < target_len:
            pad = np.zeros((target_len - len(stretched),) + stretched.shape[1:], dtype=stretched.dtype)
            stretched = np.concatenate([stretched, pad], axis=0)
        return stretched

    def spatial_jitter(self, frames: np.ndarray) -> np.ndarray:
        jitter_std = 0.02
        spatial_noise = np.random.normal(0, jitter_std, frames.shape).astype(np.float32)
        return np.clip(frames + spatial_noise, 0, 1)

# =========================
# 数据集
# =========================
class EnhancedCECSLDataset:
    """增强版CE-CSL数据集（懒加载 + 在线增强）"""
    def __init__(self, config: EnhancedCECSLConfig, split: str = 'train', use_augmentation: bool = True, token2idx=None):
        self.config = config
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.data_root = Path(config.data_root)
        self.token2idx = token2idx
        self.augmentor = DataAugmentor(config) if self.use_augmentation else None

        if not self.data_root.exists():
            alt = Path(str(self.data_root).replace("CS-CSL", "CE-CSL"))
            if alt.exists():
                logger.warning(f"未找到数据目录 {self.data_root} ，自动回退到 {alt}")
                self.data_root = alt
            else:
                logger.warning(f"未找到数据目录 {self.data_root} ，请确认数据是否已放置")

        self.samples = []
        self._load_from_corpus()
        logger.info(f"[{split}] 样本数: {len(self.samples)}")

    def _load_from_corpus(self):
        corpus_file = self.data_root / f"{self.split}.corpus.csv"
        if not corpus_file.exists():
            logger.error(f"Corpus文件不存在: {corpus_file}")
            return

        with open(corpus_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get('video_id', '')
                label = (row.get('label') or "").strip()
                if self.split in ('train', 'dev') and label == "":
                    continue
                rel = row.get('frames_path', '')
                frames_file = (self.data_root / rel).resolve() if rel and not os.path.isabs(rel) else Path(rel)
                if frames_file.exists():
                    try:
                        self.samples.append({
                            'frames_path': str(frames_file),
                            'label': label,
                            'video_id': video_id
                        })
                    except Exception as e:
                        logger.error(f"记录数据失败 {frames_file}: {e}")

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample['frames_path']

        # 懒加载 + 内存映射
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            with np.load(path) as z:
                thwc = z['frames']
        else:
            arr = np.load(path, mmap_mode='r' if ext == ".npy" else None)
            if arr.ndim != 4:
                raise ValueError(f"不支持的帧维度: {arr.shape}")
            thwc = arr if arr.shape[-1] in (1, 3) else np.transpose(arr, (0, 2, 3, 1))

        # 可选 2× 降采样（224->112）
        if thwc.shape[1] == 224 and thwc.shape[2] == 224 and self.config.image_size == (112, 112):
            thwc = thwc[:, ::2, ::2, :]

        # 归一化
        thwc = thwc.astype(np.float32, copy=False)
        if thwc.max() > 1.0:
            thwc = thwc / 255.0

        # 在线增强（仅训练）
        if self.augmentor is not None:
            if random.random() < 0.7:
                thwc = self.augmentor.add_noise(thwc)
            if random.random() < 0.5:
                thwc = self.augmentor.time_stretch(thwc)
            if random.random() < 0.6:
                thwc = self.augmentor.spatial_jitter(thwc)

        # THWC -> TCHW
        tchw = np.transpose(thwc, (0, 3, 1, 2))

        # 调整序列长度
        Tcur = tchw.shape[0]
        target_T = self.config.max_sequence_length
        if Tcur > target_T:
            idx = np.linspace(0, Tcur - 1, target_T, dtype=int)
            tchw = tchw[idx]
            Tcur = target_T
        elif Tcur < target_T:
            pad = np.zeros((target_T - Tcur, *tchw.shape[1:]), dtype=tchw.dtype)
            tchw = np.concatenate([tchw, pad], axis=0)

        # 展平 (T, F)
        T = tchw.shape[0]
        F = int(np.prod(tchw.shape[1:], dtype=np.int64))
        frames_flat = tchw.reshape(T, F).astype(np.float32, copy=False)

        # 组装 CTC 标签序列（按 token）
        gloss = sample['label']
        tokens = [t.strip() for t in gloss.split('/') if t.strip()]
        targets = [self.token2idx.get(t, 0) for t in tokens]  # 0 是 <BLK>
        target_len = len(targets)
        max_target_len = self.config.max_target_length
        if target_len < max_target_len:
            targets += [1] * (max_target_len - target_len)     # 1 是 <PAD>
        else:
            targets = targets[:max_target_len]
            target_len = max_target_len

        input_len = min(Tcur, target_T)
        return frames_flat, np.array(targets, np.int32), np.array(input_len, np.int32), np.array(target_len, np.int32)

    def __len__(self):
        return len(self.samples)

# =========================
# 模型
# =========================
class ImprovedCECSLModel(nn.Cell):
    """改进的CE-CSL手语识别模型（MLP + BiLSTM + CTC）"""
    def __init__(self, config: EnhancedCECSLConfig, vocab_size: int):
        super().__init__()
        self.config = config
        input_size = 3 * config.image_size[0] * config.image_size[1]

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
            nn.Dropout(p=config.dropout / 2),
        ])

        self.temporal_model = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0
        )

        self.ctc_head = nn.Dense(config.d_model * 2, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0)  # <BLK>=0

    def construct(self, x, targets, input_len, target_len):
        # x: (B, T, F)
        B, T, F = x.shape
        x_reshaped = ops.reshape(x, (B * T, F))
        feats = self.feature_extractor(x_reshaped)
        feats = ops.reshape(feats, (B, T, self.config.d_model))

        y, _ = self.temporal_model(feats)           # (B, T, 2*d)
        logits = self.ctc_head(y)                   # (B, T, V)
        log_probs = ops.log_softmax(logits, axis=2)
        log_probs = ops.transpose(log_probs, (1, 0, 2))  # (T, B, V)
        loss = self.ctc_loss(log_probs, targets, input_len, target_len)
        return loss, log_probs

# =========================
# 学习率调度/早停
# =========================
class LearningRateScheduler:
    def __init__(self, base_lr: float, warmup_epochs: int, total_epochs: int):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        return self.base_lr * (1 + np.cos(np.pi * progress)) / 2

class EarlyStoppingCallback:
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return False
        if val_score < self.best_score + self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        self.best_score = val_score
        self.counter = 0
        return False

# =========================
# 训练器
# =========================
class EnhancedCECSLTrainer:
    def __init__(self, config: EnhancedCECSLConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.lr_scheduler = None
        self.early_stopping = None

        # ---- 设备设置（GPU优先）----
        try:
            device_target = config.device_target.upper()
        except:
            device_target = "GPU"

        # 推断 device_id
        def _infer_device_id():
            env_id = os.getenv("LOCAL_RANK") or os.getenv("RANK_ID")
            if env_id is not None and env_id.isdigit():
                return int(env_id)
            cuda = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            return int(cuda) if cuda.isdigit() else 0

        if device_target == "GPU":
            ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=_infer_device_id())
            # GraphKernel 对 LSTM 支持有限，谨慎开启。若报错可注释掉。
            # ms.set_context(enable_graph_kernel=False)
        else:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target)

        # 输出目录
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"训练器初始化完成 - 设备: {device_target}")

    # ---------- 数据 ----------
    def load_data(self):
        logger.info("加载数据与构建 CTC 词表...")
        corpus_csv = Path(self.config.data_root) / "train.corpus.csv"
        token2idx, idx2token = build_token_vocab(corpus_csv)
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.vocab_size = len(token2idx)

        train_data = EnhancedCECSLDataset(self.config, 'train', use_augmentation=True, token2idx=token2idx)
        val_split = 'dev' if (Path(self.config.data_root) / "dev.corpus.csv").exists() else 'test'
        val_data = EnhancedCECSLDataset(self.config, val_split, use_augmentation=False, token2idx=token2idx)

        if len(train_data) == 0:
            raise ValueError("训练数据集为空，请检查数据路径和预处理数据")

        # 更高并发与预取，提升 GPU 吞吐
        self.train_dataset = GeneratorDataset(
            train_data,
            column_names=["sequence", "targets", "input_len", "target_len"],
            shuffle=True,
            num_parallel_workers=self.config.num_workers,
            python_multiprocessing=True
        ).batch(self.config.batch_size, drop_remainder=False).repeat(1).prefetch(self.config.prefetch_size)

        self.val_dataset = GeneratorDataset(
            val_data,
            column_names=["sequence", "targets", "input_len", "target_len"],
            shuffle=False,
            num_parallel_workers=max(2, self.config.num_workers // 2),
            python_multiprocessing=True
        ).batch(self.config.batch_size, drop_remainder=False).prefetch(self.config.prefetch_size)

        logger.info(f"训练集: {len(train_data)} 样本 | 验证集: {len(val_data)} 样本 | 词表大小: {self.vocab_size}")
        logger.info(f"示例标签(前30): {[t for t in self.idx2token if t not in ('<BLK>','<PAD>')][:30]}")

    # ---------- 优化器/模型 ----------
    def _build_warmup_cosine_lr(self, base_lr, warmup_epochs, total_epochs, steps_per_epoch):
        total_steps = max(1, total_epochs * max(1, steps_per_epoch))
        warmup_steps = max(1, warmup_epochs * max(1, steps_per_epoch))
        lr = np.zeros(total_steps, dtype=np.float32)
        for s in range(warmup_steps):
            lr[s] = base_lr * (s + 1) / warmup_steps
        for s in range(warmup_steps, total_steps):
            progress = (s - warmup_steps) / max(1, (total_steps - warmup_steps))
            lr[s] = base_lr * (1 + np.cos(np.pi * progress)) / 2
        return lr

    def build_model(self):
        logger.info("构建模型与优化器...")
        if not hasattr(self, 'vocab_size'):
            raise ValueError("请先调用 load_data()")

        self.model = ImprovedCECSLModel(self.config, self.vocab_size)

        # 可选 AMP（建议先用 O0 稳定训练；确认稳定后再开启 O2 提升速度）
        if self.config.enable_amp:
            try:
                from mindspore import amp
                amp.auto_mixed_precision(self.model, amp_level=self.config.amp_level)
                logger.info(f"AMP 已启用：{self.config.amp_level}")
            except Exception as e:
                logger.warning(f"AMP 初始化失败，改用 O0：{e}")

        # 学习率序列
        steps_per_epoch = max(1, len(self.train_dataset))
        lr_array = self._build_warmup_cosine_lr(
            self.config.learning_rate, self.config.warmup_epochs, self.config.epochs, steps_per_epoch
        )

        self.optimizer = nn.AdamWeightDecay(
            params=self.model.trainable_params(),
            learning_rate=Tensor(lr_array, ms.float32),
            weight_decay=self.config.weight_decay
        )

        # 早停
        self.early_stopping = EarlyStoppingCallback(self.config.patience, self.config.min_delta)

        # JIT 编译训练步（显著降低 Python 开销）
        self.train_step = ms.jit(self.train_step)

        # 统计参数量
        param_count = sum(p.size for p in self.model.get_parameters())
        logger.info(f"模型参数量: {param_count}")

    # ---------- 一个训练步 ----------
    def train_step(self, data, targets, input_len, target_len):
        def forward_fn(d, t, il, tl):
            loss, _ = self.model(d, t, il, tl)
            return loss
        grad_fn = ms.value_and_grad(forward_fn, None, self.model.trainable_params())
        loss, grads = grad_fn(data, targets, input_len, target_len)
        grads = ops.clip_by_global_norm(grads, 1.0)
        self.optimizer(grads)
        return loss

    # ---------- 解码/评估 ----------
    def evaluate(self):
        self.model.set_train(False)
        total_loss, n_batches = 0.0, 0
        exact_match, total_samples = 0, 0
        total_edits, total_ref_tokens = 0, 0

        class_total = {w: 0 for w in self.idx2token}
        class_correct = {w: 0 for w in self.idx2token}

        for data, targets, input_len, target_len in self.val_dataset:
            data       = Tensor(data,       ms.float32)
            targets    = Tensor(targets,    ms.int32)
            input_len  = Tensor(input_len,  ms.int32)
            target_len = Tensor(target_len, ms.int32)

            loss, log_probs = self.model(data, targets, input_len, target_len)
            total_loss += float(loss.asnumpy()); n_batches += 1

            preds = ctc_greedy_decode(log_probs, self.idx2token)
            targets_np    = targets.asnumpy()
            target_len_np = target_len.asnumpy()
            for b in range(targets_np.shape[0]):
                L = int(target_len_np[b])
                ref_ids = targets_np[b, :L].tolist()
                ref_tokens = [self.idx2token[i] for i in ref_ids if i not in (0, 1)]
                hyp_tokens = preds[b]

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
        class_accuracies = {w: (class_correct[w] / class_total[w] if class_total[w] > 0 else 0.0) for w in class_total}
        return avg_loss, em, cer, class_accuracies, class_total

    # ---------- 训练主循环 ----------
    def train(self):
        logger.info("开始训练...")
        best_val_acc = 0.0
        training_history = []

        for epoch in range(self.config.epochs):
            t0 = time.time()
            self.model.set_train(True)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")

            for step, (data, targets, input_len, target_len) in enumerate(self.train_dataset):
                data       = Tensor(data,       ms.float32)
                targets    = Tensor(targets,    ms.int32)
                input_len  = Tensor(input_len,  ms.int32)
                target_len = Tensor(target_len, ms.int32)

                loss = self.train_step(data, targets, input_len, target_len)
                epoch_loss += float(loss.asnumpy())

                # 计算训练 EM
                _, log_probs = self.model(data, targets, input_len, target_len)
                preds = ctc_greedy_decode(log_probs, self.idx2token)
                targets_np    = targets.asnumpy()
                target_len_np = target_len.asnumpy()
                for b in range(targets_np.shape[0]):
                    L = int(target_len_np[b])
                    ref_ids = targets_np[b, :L].tolist()
                    ref_tokens = [self.idx2token[i] for i in ref_ids if i not in (0, 1)]
                    hyp_tokens = preds[b]
                    if hyp_tokens == ref_tokens:
                        epoch_correct += 1
                    epoch_total += 1

                if step % 10 == 0:
                    logger.info(f"  step {step:04d} | loss={float(loss.asnumpy()):.4f} | TrainEM={epoch_correct/max(1,epoch_total):.4f}")

            avg_train_loss = epoch_loss / max(1, len(self.train_dataset))
            train_accuracy = epoch_correct / max(1, epoch_total)

            # 验证
            val_loss, val_em, val_cer, class_accuracies, class_total = self.evaluate()
            val_accuracy = val_em

            # 日志
            t1 = time.time()
            logger.info(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} | train_EM={train_accuracy:.4f} | "
                        f"val_loss={val_loss:.4f} | val_EM={val_em:.4f} | val_CER={val_cer:.4f} | time={t1-t0:.2f}s")

            logger.info("类别准确率（样本数）:")
            for word in sorted(class_total.keys()):
                if class_total[word] > 0:
                    logger.info(f"  {word}: {class_accuracies[word]:.4f}  (n={class_total[word]})")

            # 保存最佳
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_path = self.output_dir / "enhanced_cecsl_best_model.ckpt"
                save_checkpoint(self.model, str(best_model_path))
                logger.info(f"✨ 新的最佳验证EM={best_val_acc:.4f}，已保存到: {best_model_path}")

            # 记录历史
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'train_acc': float(train_accuracy),
                'val_loss': float(val_loss),
                'val_acc': float(val_accuracy),
            })

            # 早停
            if self.early_stopping(val_accuracy):
                logger.info(f"⏹ 早停触发，停止于第 {epoch+1} 轮")
                break

        logger.info(f"训练结束！最佳验证EM: {best_val_acc:.4f}")

        # 保存训练历史
        history_file = self.output_dir / "enhanced_training_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)

        return self.model

    def save_model(self, save_path: str):
        save_checkpoint(self.model, save_path)
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
        logger.info(f"✅ 模型与词表已保存：{save_path} | {vocab_path}")

# =========================
# 工具函数
# =========================
def build_token_vocab(corpus_csv: Path):
    tokens = set()
    if not corpus_csv.exists():
        raise FileNotFoundError(f"未找到 {corpus_csv}")
    with open(corpus_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = (row.get('label') or "")
            for token in gloss.split('/'):
                token = token.strip()
                if token:
                    tokens.add(token)
    idx2token = ['<BLK>', '<PAD>'] + sorted(tokens)
    token2idx = {t: i for i, t in enumerate(idx2token)}
    return token2idx, idx2token

def ctc_greedy_decode(log_probs: Tensor, idx2token: List[str]):
    # log_probs: (T, B, V)
    pred_ids = np.argmax(log_probs.asnumpy(), axis=2)  # (T, B)
    results = []
    for b in range(pred_ids.shape[1]):
        seq = pred_ids[:, b]
        tokens = []
        prev = -1
        for t in seq:
            t = int(t)
            if t not in (0, 1) and t != prev:  # 去重并忽略 <BLK>/<PAD>
                tokens.append(idx2token[t])
            prev = t
        results.append(tokens)
    return results

def edit_distance(a: List[str], b: List[str]) -> int:
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[la][lb]

# =========================
# Main
# =========================
def main():
    try:
        config = EnhancedCECSLConfig()

        logger.info("=" * 60)
        logger.info("增强版CE-CSL手语识别训练器（GPU加速版）启动")
        logger.info("=" * 60)
        logger.info(f"数据路径: {config.data_root}")
        logger.info(f"批次大小: {config.batch_size}")
        logger.info(f"学习率: {config.learning_rate}")
        logger.info(f"训练轮数: {config.epochs}")
        logger.info(f"设备: {config.device_target}")
        logger.info(f"AMP: {config.enable_amp} ({config.amp_level})")
        logger.info(f"数据并行 workers: {config.num_workers}, 预取: {config.prefetch_size}")
        logger.info("=" * 60)

        trainer = EnhancedCECSLTrainer(config)
        logger.info("步骤 1: 加载数据...")
        trainer.load_data()

        logger.info("步骤 2: 构建模型...")
        trainer.build_model()

        logger.info("步骤 3: 开始训练...")
        trainer.train()

        logger.info("步骤 4: 保存最终模型...")
        final_model_path = trainer.output_dir / "enhanced_cecsl_final_model.ckpt"
        trainer.save_model(str(final_model_path))

        logger.info("=" * 60)
        logger.info("✅ 训练完成!")
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
