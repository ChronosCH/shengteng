"""
连续手语识别(CSLR)模型训练脚本
基于MindSpore实现ST-Transformer-CTC模型
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
import mindspore.dataset as ds
from mindspore.dataset import vision, transforms
from mindspore.common import set_seed

# 自定义模块
from mindspore import context

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置MindSpore上下文
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")  # 或 "GPU", "CPU"

class PositionalEncoding(nn.Cell):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = Parameter(Tensor(pe, ms.float32), requires_grad=False)
    
    def construct(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

class MultiHeadAttention(nn.Cell):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Dense(d_model, d_model)
        self.w_k = nn.Dense(d_model, d_model)
        self.w_v = nn.Dense(d_model, d_model)
        self.w_o = nn.Dense(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(axis=-1)
        
    def construct(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # 线性变换并重塑为多头形式
        Q = self.w_q(query).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = self.w_k(key).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = self.w_v(value).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力计算
        attention = self._attention(Q, K, V, mask)
        
        # 重塑并输出
        attention = attention.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attention)
    
    def _attention(self, Q, K, V, mask=None):
        """计算注意力"""
        d_k = Q.shape[-1]
        
        # 计算注意力分数
        scores = ops.matmul(Q, K.transpose(0, 1, 3, 2)) / ops.sqrt(Tensor(d_k, ms.float32))
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax和dropout
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        return ops.matmul(attention_weights, V)

class TransformerBlock(nn.Cell):
    """Transformer块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        
        self.feed_forward = nn.SequentialCell([
            nn.Dense(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(d_ff, d_model),
            nn.Dropout(p=dropout)
        ])
        
        self.dropout = nn.Dropout(p=dropout)
    
    def construct(self, x, mask=None):
        # 多头注意力 + 残差连接
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class SpatialTemporalTransformer(nn.Cell):
    """时空Transformer编码器"""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 300,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Dense(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.transformer_layers = nn.CellList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm((d_model,))
    
    def construct(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len)
        """
        # 输入投影和位置编码
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        return x

class CTCHead(nn.Cell):
    """CTC输出头"""
    
    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Dense(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(axis=-1)
    
    def construct(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, vocab_size)
        """
        x = self.dropout(x)
        x = self.classifier(x)
        return self.log_softmax(x)

class CSLRModel(nn.Cell):
    """连续手语识别模型"""
    
    def __init__(self,
                 input_dim: int = 1629,  # 543 keypoints * 3 coordinates
                 vocab_size: int = 1000,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 300,
                 dropout: float = 0.1):
        super().__init__()
        
        self.backbone = SpatialTemporalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.ctc_head = CTCHead(d_model, vocab_size, dropout)
    
    def construct(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) 序列实际长度
        """
        # 特征提取
        features = self.backbone(x)
        
        # CTC预测
        logits = self.ctc_head(features)
        
        return logits

class CTCLoss(nn.Cell):
    """CTC损失函数"""
    
    def __init__(self, blank_id: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=reduction, zero_infinity=True)
    
    def construct(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (seq_len, batch_size, vocab_size)
            targets: (batch_size, target_seq_len)
            input_lengths: (batch_size,)
            target_lengths: (batch_size,)
        """
        # 转置维度以适配CTC损失
        log_probs = log_probs.transpose(1, 0, 2)  # (seq_len, batch_size, vocab_size)
        
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss

class CSLRDataset:
    """手语识别数据集"""
    
    def __init__(self, data_dir: str, vocab_file: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.word_to_id = {word: idx for word, idx in self.vocab.items()}
        self.id_to_word = {idx: word for word, idx in self.vocab.items()}
        
        # 加载数据文件列表
        self.data_files = list(self.data_dir.glob(f"{split}*.npz"))
        logger.info(f"加载 {split} 数据集: {len(self.data_files)} 个文件")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        data_file = self.data_files[idx]
        
        # 加载数据
        data = np.load(data_file)
        keypoints = data['keypoints']  # (T, 543, 3)
        gloss_sequence = data['gloss_sequence']
        
        # 展平关键点数据
        keypoints_flat = keypoints.reshape(keypoints.shape[0], -1)  # (T, 1629)
        
        # 转换标签为ID序列
        target_ids = []
        for gloss in gloss_sequence:
            if gloss in self.word_to_id:
                target_ids.append(self.word_to_id[gloss])
            else:
                target_ids.append(self.word_to_id['<unk>'])
        
        return {
            'keypoints': keypoints_flat.astype(np.float32),
            'targets': np.array(target_ids, dtype=np.int32),
            'input_length': len(keypoints_flat),
            'target_length': len(target_ids)
        }
    
    def create_dataset(self, batch_size: int = 8, shuffle: bool = True, num_workers: int = 4):
        """创建MindSpore数据集"""
        
        def generator():
            for i in range(len(self)):
                yield self[i]
        
        dataset = ds.GeneratorDataset(
            generator,
            column_names=['keypoints', 'targets', 'input_length', 'target_length'],
            shuffle=shuffle,
            num_parallel_workers=num_workers
        )
        
        # 数据预处理
        dataset = dataset.batch(batch_size, pad_info={
            'keypoints': ([None, 1629], 0.0),
            'targets': ([None], 0),
            'input_length': ([], 0),
            'target_length': ([], 0)
        })
        
        return dataset

class CSLRTrainer:
    """CSLR模型训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device_target = config.get('device_target', 'Ascend')
        
        # 设置设备
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target=self.device_target
        )
        
        # 分布式训练
        if config.get('distributed', False):
            init()
            self.rank_id = get_rank()
            self.group_size = get_group_size()
        else:
            self.rank_id = 0
            self.group_size = 1
        
        # 设置随机种子
        set_seed(config.get('seed', 42))
        
        # 创建模型
        self.model = CSLRModel(
            input_dim=config['input_dim'],
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )
        
        # 损失函数
        self.loss_fn = CTCLoss(blank_id=config.get('blank_id', 0))
        
        # 优化器
        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        logger.info(f"模型初始化完成，参数量: {self._count_parameters()}")
    
    def _count_parameters(self):
        """计算模型参数量"""
        total_params = sum(p.size for p in self.model.trainable_params())
        return total_params
    
    def train(self, train_dataset, val_dataset=None, epochs: int = 100):
        """训练模型"""
        
        # 定义网络
        net_with_loss = nn.WithLossCell(self.model, self.loss_fn)
        train_net = nn.TrainOneStepCell(net_with_loss, self.optimizer)
        train_net.set_train()
        
        # 回调函数
        callbacks = []
        
        # 保存检查点
        if self.rank_id == 0:
            ckpt_config = CheckpointConfig(
                save_checkpoint_steps=self.config.get('save_steps', 1000),
                keep_checkpoint_max=self.config.get('keep_ckpt_max', 5)
            )
            ckpt_callback = ModelCheckpoint(
                prefix="cslr_model",
                directory=self.config['output_dir'],
                config=ckpt_config
            )
            callbacks.append(ckpt_callback)
        
        # 损失监控
        callbacks.extend([
            LossMonitor(per_print_times=self.config.get('print_steps', 100)),
            TimeMonitor()
        ])
        
        # 训练循环
        model = Model(train_net, eval_network=self.model, metrics={'accuracy'})
        
        logger.info(f"开始训练，共 {epochs} 个epoch")
        model.train(
            epoch=epochs,
            train_dataset=train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=True
        )
        
        # 保存最终模型
        if self.rank_id == 0:
            save_checkpoint(self.model, os.path.join(self.config['output_dir'], 'final_model.ckpt'))
            
            # 导出MindIR模型
            self._export_mindir()
    
    def _export_mindir(self):
        """导出MindIR格式模型"""
        try:
            # 创建示例输入
            batch_size = 1
            seq_len = 100
            input_dim = self.config['input_dim']
            
            dummy_input = Tensor(np.random.randn(batch_size, seq_len, input_dim), ms.float32)
            
            # 导出
            ms.export(
                self.model,
                dummy_input,
                file_name=os.path.join(self.config['output_dir'], 'cslr_model'),
                file_format='MINDIR'
            )
            
            logger.info("MindIR模型导出成功")
            
        except Exception as e:
            logger.error(f"MindIR模型导出失败: {e}")
    
    def evaluate(self, eval_dataset):
        """评估模型"""
        self.model.set_train(False)
        
        total_loss = 0
        total_samples = 0
        
        for data in eval_dataset:
            keypoints = data['keypoints']
            targets = data['targets']
            input_lengths = data['input_length']
            target_lengths = data['target_length']
            
            # 前向传播
            logits = self.model(keypoints)
            
            # 计算损失
            loss = self.loss_fn(logits, targets, input_lengths, target_lengths)
            
            batch_size = keypoints.shape[0]
            total_loss += loss.asnumpy() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        logger.info(f"验证集平均损失: {avg_loss:.4f}")
        
        return avg_loss

def main():
    parser = argparse.ArgumentParser(description='CSLR模型训练')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--data_dir', required=True, help='数据目录')
    parser.add_argument('--vocab_file', required=True, help='词汇表文件')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--distributed', action='store_true', help='分布式训练')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 更新配置
    config.update({
        'data_dir': args.data_dir,
        'vocab_file': args.vocab_file,
        'output_dir': args.output_dir,
        'distributed': args.distributed
    })
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据集
    train_dataset_obj = CSLRDataset(args.data_dir, args.vocab_file, 'train')
    val_dataset_obj = CSLRDataset(args.data_dir, args.vocab_file, 'val')
    
    train_dataset = train_dataset_obj.create_dataset(
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_dataset = val_dataset_obj.create_dataset(
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # 更新配置中的词汇表大小
    config['vocab_size'] = train_dataset_obj.vocab_size
    
    # 创建训练器
    trainer = CSLRTrainer(config)
    
    # 开始训练
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=config['epochs']
    )

if __name__ == "__main__":
    main()
