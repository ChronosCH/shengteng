"""
MindSpore implementation of SLR Model for inference
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x


class NormLinear(nn.Cell):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = ms.Parameter(
            ms.Tensor(np.random.randn(in_dim, out_dim).astype(np.float32))
        )
        self.l2_normalize = ops.L2Normalize(axis=0)

    def construct(self, x):
        normalized_weight = self.l2_normalize(self.weight)
        outputs = ops.matmul(x, normalized_weight)
        return outputs


class TemporalConv(nn.Cell):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), stride=int(ks[1])))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), 
                             stride=1, padding=0, pad_mode='valid', has_bias=True)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU())
        self.temporal_conv = nn.SequentialCell(modules)

        if self.num_classes != -1:
            self.fc = nn.Dense(self.hidden_size, self.num_classes)
        
        self.transpose = ops.Transpose()

    def update_lgt(self, lgt):
        """计算经过temporal conv后的序列长度"""
        if isinstance(lgt, ms.Tensor):
            lengths = lgt.asnumpy().astype(np.int32).tolist()
        elif isinstance(lgt, np.ndarray):
            lengths = lgt.astype(np.int32).tolist()
        elif isinstance(lgt, (list, tuple)):
            lengths = [int(x) for x in lgt]
        else:
            lengths = [int(lgt)]

        for ks in self.kernel_size:
            if ks[0] == 'P':
                lengths = [max(length // 2, 0) for length in lengths]
            else:
                reduction = int(ks[1]) - 1
                lengths = [max(length - reduction, 0) for length in lengths]

        return lengths

    def construct(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt_updated = self.update_lgt(lgt)
        
        logits = None
        if self.num_classes != -1:
            # visual_feat: (B, C, T) -> (B, T, C)
            transposed = self.transpose(visual_feat, (0, 2, 1))
            logits = self.fc(transposed)
            # logits: (B, T, C) -> (B, C, T)
            logits = self.transpose(logits, (0, 2, 1))
        
        # visual_feat: (B, C, T) -> (T, B, C)
        visual_feat = self.transpose(visual_feat, (2, 0, 1))
        if logits is not None:
            logits = self.transpose(logits, (2, 0, 1))

        return visual_feat, logits, lgt_updated


class BiLSTMLayer(nn.Cell):
    def __init__(self, input_size, hidden_size=512, num_layers=1, 
                 bidirectional=True, dropout=0.3):
        super(BiLSTMLayer, self).__init__()
        
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            has_bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def construct(self, src_feats, src_lens):
        """
        Args:
            src_feats: (T, B, D)
            src_lens: (B,)
        Returns:
            predictions: (T, B, hidden_size * num_directions)
        """
        # MindSpore LSTM doesn't support packed sequence well in inference
        # So we just use the full sequence
        outputs, _ = self.rnn(src_feats)
        return outputs


class BasicBlock(nn.Cell):
    """ResNet18的BasicBlock"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.add = ops.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet18Backbone(nn.Cell):
    """标准ResNet18骨干网络,匹配PyTorch torchvision实现"""
    def __init__(self):
        super(ResNet18Backbone, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, 
                              pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode='pad')
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                         has_bias=False, pad_mode='valid'),
                nn.BatchNorm2d(out_channels)
            ])

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class SLRModel(nn.Cell):
    def __init__(self, num_classes, hidden_size=1024, conv_type=2, use_bn=False,
                 weight_norm=True, share_classifier=True):
        super(SLRModel, self).__init__()
        
        self.num_classes = num_classes
        
        # 2D CNN backbone (ResNet18)
        self.conv2d = ResNet18Backbone()
        
        # 1D Temporal Convolution
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes
        )
        
        # BiLSTM
        self.temporal_model = BiLSTMLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.0  # inference时设为0
        )
        
        # Classifier
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Dense(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Dense(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def masked_bn(self, inputs, len_x):
        """处理变长序列的batch normalization"""
        # 如果len_x是单个值,转换为列表
        if isinstance(len_x, (int, np.integer)):
            len_x = [len_x]
        
        batch_size = len(len_x)
        max_len = len_x[0]
        
        # 分别处理每个样本
        features = []
        for idx in range(batch_size):
            lgt = len_x[idx]
            start = max_len * idx
            end = start + lgt
            feat = inputs[start:end]
            feat = self.conv2d(feat)
            
            # 填充到max_len
            if lgt < max_len:
                pad_size = max_len - lgt
                pad_shape = (pad_size,) + feat.shape[1:]
                padding = ops.zeros(pad_shape, feat.dtype)
                feat = ops.concat([feat, padding], axis=0)
            
            features.append(feat)
        
        # Stack所有样本
        x = ops.stack(features, axis=0)
        return x

    def construct(self, x, len_x):
        """
        Args:
            x: (B, T, C, H, W) 或 (B, C, T) 
            len_x: list of sequence lengths
        Returns:
            dict with recognized results
        """
        if len(x.shape) == 5:
            # 视频输入: (B, T, C, H, W)
            batch, temp, channel, height, width = x.shape
            inputs = self.reshape(x, (batch * temp, channel, height, width))
            framewise = self.masked_bn(inputs, len_x)
            # framewise: (B, T, 512) -> (B, 512, T)
            framewise = self.transpose(framewise, (0, 2, 1))
        else:
            # 帧级特征
            framewise = x

    # Temporal Convolution
        visual_feat, conv_logits, lgt = self.conv1d(framewise, len_x)
        lgt_array = np.array(lgt, dtype=np.int32)
        lgt_tensor = Tensor(lgt_array, dtype=ms.int32)
        
        # BiLSTM
        # visual_feat: (T, B, C)
        tm_outputs = self.temporal_model(visual_feat, lgt_tensor)
        
        # Classifier
        # tm_outputs: (T, B, C)
        sequence_logits = self.classifier(tm_outputs)
        
        return {
            "framewise_features": framewise,
            "visual_features": visual_feat,
            "feat_len": lgt_array,
            "conv_logits": conv_logits,
            "sequence_logits": sequence_logits,
        }
