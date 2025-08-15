"""
TFNet模型的MindSpore实现
基于华为昇腾AI处理器优化的连续手语识别模型
从PyTorch TFNet迁移到MindSpore框架
"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import Normal, XavierUniform, HeUniform
from typing import Tuple, Optional

class Identity(nn.Cell):
    """恒等映射层"""
    def construct(self, x):
        return x

class NormLinear(nn.Cell):
    """标准化线性层"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Dense(in_features, out_features, has_bias=False)
        self.norm = ops.L2Normalize(axis=1)
        
    def construct(self, x):
        return self.norm(self.linear(x))

class TemporalConv(nn.Cell):
    """时序卷积模块"""
    def __init__(self, input_size: int, hidden_size: int, conv_type: int = 2):
        super().__init__()
        self.conv_type = conv_type
        
        if conv_type == 1:
            # 简单1D卷积
            self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=3, 
                                   stride=1, padding=1, pad_mode='pad')
        elif conv_type == 2:
            # 双层1D卷积
            self.conv1d_1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, 
                                     stride=1, padding=1, pad_mode='pad')
            self.conv1d_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, 
                                     stride=1, padding=1, pad_mode='pad')
            self.relu = nn.ReLU()
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
    
    def construct(self, x):
        # x shape: (batch_size, seq_len, features) -> (batch_size, features, seq_len)
        x = ops.transpose(x, (0, 2, 1))
        
        if self.conv_type == 1:
            out = self.conv1d(x)
        else:
            out = self.bn1(self.relu(self.conv1d_1(x)))
            out = self.bn2(self.relu(self.conv1d_2(out)))
        
        # 转回 (batch_size, seq_len, features)
        return ops.transpose(out, (0, 2, 1))

class PositionalEncoding(nn.Cell):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = Parameter(Tensor(pe, ms.float32), requires_grad=False)
    
    def construct(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

class MultiHeadAttention(nn.Cell):
    """多头注意力机制"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rpe_k: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rpe_k = rpe_k
        
        self.w_q = nn.Dense(d_model, d_model)
        self.w_k = nn.Dense(d_model, d_model)
        self.w_v = nn.Dense(d_model, d_model)
        self.w_o = nn.Dense(d_model, d_model)
        
        if rpe_k > 0:
            self.rpe_w = nn.Embedding(rpe_k * 2 + 1, 2 * d_model // n_heads)
        
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.sqrt_d_k = ops.Sqrt()
        
    def construct(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[0], query.shape[1]
        
        # 线性变换
        Q = self.w_q(query).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.w_k(key).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.w_v(value).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # 转置为 (batch_size, n_heads, seq_len, d_k)
        Q = ops.transpose(Q, (0, 2, 1, 3))
        K = ops.transpose(K, (0, 2, 1, 3))
        V = ops.transpose(V, (0, 2, 1, 3))
        
        # 计算注意力
        attention = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑并输出
        attention = ops.transpose(attention, (0, 2, 1, 3))
        attention = attention.reshape(batch_size, seq_len, self.d_model)
        
        return self.w_o(attention)
    
    def _scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Tensor(self.d_k, ms.float32)
        scores = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2))) / self.sqrt_d_k(d_k)
        
        if mask is not None:
            scores = ops.masked_fill(scores, mask, -1e9)
        
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        
        return ops.matmul(attn_weights, V)

class TransformerEncoder(nn.Cell):
    """Transformer编码器"""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, 
                 dropout: float = 0.1, rpe_k: int = 8):
        super().__init__()
        
        self.layers = nn.CellList([
            TransformerEncoderLayer(d_model, n_heads, dropout, rpe_k)
            for _ in range(n_layers)
        ])
        
    def construct(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerEncoderLayer(nn.Cell):
    """Transformer编码器层"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rpe_k: int = 8):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rpe_k)
        self.feed_forward = nn.SequentialCell([
            nn.Dense(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(d_model * 4, d_model),
            nn.Dropout(p=dropout)
        ])
        
        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        self.dropout = nn.Dropout(p=dropout)
        
    def construct(self, x, mask=None):
        # 自注意力
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class BiLSTMLayer(nn.Cell):
    """双向LSTM层"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                 bidirectional: bool = True, dropout: float = 0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           has_bias=True, batch_first=True, 
                           dropout=dropout, bidirectional=bidirectional)
        
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        
    def construct(self, x):
        # x shape: (batch_size, seq_len, input_size)
        output, _ = self.lstm(x)
        return output

class TFNetMindSpore(nn.Cell):
    """TFNet模型的MindSpore实现"""
    
    def __init__(self, hidden_size: int = 512, vocab_size: int = 1000, 
                 module_choice: str = "TFNet", dataset_name: str = "CE-CSL"):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.module_choice = module_choice
        self.dataset_name = dataset_name
        
        # 2D CNN特征提取器 (ResNet34)
        self.conv2d = self._build_resnet34_backbone()
        
        if module_choice == "TFNet":
            self._build_tfnet_components()
        elif module_choice == "MSTNet":
            self._build_mstnet_components()
        elif module_choice == "VAC":
            self._build_vac_components()
        
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.softmax = nn.Softmax(axis=-1)
        
    def _build_resnet34_backbone(self):
        """构建ResNet34骨干网络"""
        # 简化的ResNet34实现
        layers = nn.SequentialCell([
            # Conv1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode='pad'),
            
            # 简化的残差块
            self._make_layer(64, 64, 3),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ])
        
        return layers
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int = 1):
        """构建残差层"""
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(self._basic_block(in_channels, out_channels, stride))
        
        # 其余块
        for _ in range(1, num_blocks):
            layers.append(self._basic_block(out_channels, out_channels))
            
        return nn.SequentialCell(layers)
    
    def _basic_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """基础残差块"""
        layers = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                     padding=1, pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                     padding=1, pad_mode='pad'),
            nn.BatchNorm2d(out_channels)
        ])
        
        # 跳跃连接
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            ])
        else:
            self.downsample = None
            
        return layers
    
    def _build_tfnet_components(self):
        """构建TFNet特定组件"""
        # 1D时序卷积
        self.conv1d = TemporalConv(512, self.hidden_size, conv_type=2)
        
        # Transformer编码器
        self.temporal_model = TransformerEncoder(
            d_model=self.hidden_size,
            n_heads=8,
            n_layers=2,
            dropout=0.1,
            rpe_k=8
        )
        
        # 分类器
        self.classifier1 = NormLinear(self.hidden_size, self.vocab_size)
        self.classifier2 = NormLinear(self.hidden_size, self.vocab_size)
        
        if self.dataset_name in ['CE-CSL', 'RWTH']:
            self.classifier3 = NormLinear(self.hidden_size, self.vocab_size)
            self.classifier4 = NormLinear(512, self.vocab_size)
    
    def _build_mstnet_components(self):
        """构建MSTNet特定组件"""
        # 多尺度时序网络组件
        self.conv1D1_1 = nn.Conv1d(512, self.hidden_size, kernel_size=3, 
                                   stride=1, padding=1, pad_mode='pad')
        self.conv1D1_2 = nn.Conv1d(512, self.hidden_size, kernel_size=5, 
                                   stride=1, padding=2, pad_mode='pad')
        self.conv1D1_3 = nn.Conv1d(512, self.hidden_size, kernel_size=7, 
                                   stride=1, padding=3, pad_mode='pad')
        self.conv1D1_4 = nn.Conv1d(512, self.hidden_size, kernel_size=9, 
                                   stride=1, padding=4, pad_mode='pad')
        
        # 其他组件...
        self.temporal_model = TransformerEncoder(
            d_model=self.hidden_size,
            n_heads=8,
            n_layers=2,
            dropout=0.1,
            rpe_k=8
        )
        
        self.classifier1 = NormLinear(self.hidden_size, self.vocab_size)
        self.classifier2 = NormLinear(self.hidden_size, self.vocab_size)
    
    def _build_vac_components(self):
        """构建VAC特定组件"""
        self.conv1d = TemporalConv(512, self.hidden_size, conv_type=2)
        
        self.temporal_model = BiLSTMLayer(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=True
        )
        
        self.classifier = NormLinear(self.hidden_size, self.vocab_size)
        self.classifier1 = self.classifier
    
    def construct(self, x, lengths, is_train=True):
        """
        前向传播
        Args:
            x: 输入视频帧 (batch_size, seq_len, height, width, channels)
            lengths: 序列长度
            is_train: 是否训练模式
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 2D CNN特征提取
        # 重塑为 (batch_size * seq_len, channels, height, width)
        x_reshaped = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])
        features_2d = self.conv2d(x_reshaped)  # (batch_size * seq_len, 512)
        
        # 重塑回序列形式
        features_2d = features_2d.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, 512)
        
        if self.module_choice == "TFNet":
            return self._forward_tfnet(features_2d, lengths, is_train)
        elif self.module_choice == "MSTNet":
            return self._forward_mstnet(features_2d, lengths, is_train)
        elif self.module_choice == "VAC":
            return self._forward_vac(features_2d, lengths, is_train)
    
    def _forward_tfnet(self, features, lengths, is_train):
        """TFNet前向传播"""
        # 1D时序卷积
        conv_features = self.conv1d(features)  # (batch_size, seq_len, hidden_size)
        
        # Transformer编码
        temporal_features = self.temporal_model(conv_features)
        
        # 分类
        logits1 = self.classifier1(temporal_features)
        logits2 = self.classifier2(conv_features)
        
        if self.dataset_name in ['CE-CSL', 'RWTH']:
            logits3 = self.classifier3(temporal_features)
            logits4 = self.classifier4(features)
            return logits1, logits2, logits3, logits4, lengths, conv_features, temporal_features, features
        
        return logits1, logits2, None, None, lengths, conv_features, temporal_features, features
    
    def _forward_mstnet(self, features, lengths, is_train):
        """MSTNet前向传播"""
        # 多尺度时序卷积
        x = ops.transpose(features, (0, 2, 1))  # (batch_size, 512, seq_len)
        
        conv1 = self.conv1D1_1(x)
        conv2 = self.conv1D1_2(x)
        conv3 = self.conv1D1_3(x)
        conv4 = self.conv1D1_4(x)
        
        # 多尺度特征融合
        multi_scale = ops.concat([conv1, conv2, conv3, conv4], axis=1)
        multi_scale = ops.transpose(multi_scale, (0, 2, 1))
        
        # Transformer编码
        temporal_features = self.temporal_model(multi_scale)
        
        # 分类
        logits1 = self.classifier1(temporal_features)
        logits2 = self.classifier2(multi_scale)
        
        return logits1, logits2, None, None, lengths, multi_scale, temporal_features, features
    
    def _forward_vac(self, features, lengths, is_train):
        """VAC前向传播"""
        # 1D卷积
        conv_features = self.conv1d(features)
        
        # BiLSTM
        lstm_features = self.temporal_model(conv_features)
        
        # 分类
        logits = self.classifier(lstm_features)
        
        return logits, None, None, None, lengths, conv_features, lstm_features, features

class SeqKD(nn.Cell):
    """序列知识蒸馏损失"""
    def __init__(self, T: float = 8.0):
        super().__init__()
        self.T = T
        self.softmax = nn.Softmax(axis=-1)
        
    def construct(self, teacher_logits, student_logits, use_blank=False):
        """
        计算知识蒸馏损失
        Args:
            teacher_logits: 教师网络输出
            student_logits: 学生网络输出
            use_blank: 是否使用空白标签
        """
        # 温度缩放
        teacher_soft = self.softmax(teacher_logits / self.T)
        student_soft = self.softmax(student_logits / self.T)
        
        # KL散度
        kl_loss = ops.reduce_sum(
            teacher_soft * ops.log(teacher_soft / (student_soft + 1e-8)),
            axis=-1
        )
        
        return ops.reduce_mean(kl_loss) * (self.T ** 2)
