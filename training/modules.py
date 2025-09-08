import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import copy

class Identity(nn.Cell):
    """Identity layer that returns input unchanged"""
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x

class TemporalConv(nn.Cell):
    """Temporal convolution module for processing sequential features"""
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K3', 'K3']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1])))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), 
                             stride=1, padding=0, has_bias=True, pad_mode='valid')
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU())

        self.temporal_conv = nn.SequentialCell(*modules)

    def update_lgt(self, lgt):
        """Update sequence lengths after convolution operations"""
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                # Pooling operation: divide by stride
                feat_len = [max(1, int(i // 2)) for i in feat_len]
            else:
                # Convolution operation: length = input_length - kernel_size + 1
                kernel_size = int(ks[1])
                feat_len = [max(1, i - kernel_size + 1) for i in feat_len]
        return feat_len

    def construct(self, frame_feat, lgt):
        # Only compute visual features inside graph; avoid Python ops on lengths here
        visual_feat = self.temporal_conv(frame_feat)
        return {
            "visual_feat": visual_feat,
            "feat_len": lgt,
        }

class NormLinear(nn.Cell):
    """Normalized linear layer"""
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        # Xavier uniform initialization
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weight_init = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        self.weight = ms.Parameter(
            Tensor(weight_init, ms.float32),
            name='weight'
        )
        self.l2_normalize = ops.L2Normalize(axis=0)
        self.matmul = ops.MatMul()

    def construct(self, x):
        # Support inputs with rank >= 2 by flattening the last feature dim
        x_shape = x.shape  # e.g., (T, B, C) or (N, C)
        in_feat = x_shape[-1]
        reshape = ops.Reshape()
        x2d = reshape(x, (-1, in_feat))

        normalized_weight = self.l2_normalize(self.weight)  # (in_dim, out_dim)
        out2d = self.matmul(x2d, normalized_weight)  # (-1, out_dim)

        out_dim = int(self.weight.shape[1])
        new_shape = x_shape[:-1] + (out_dim,)
        outputs = reshape(out2d, new_shape)
        return outputs

class BiLSTMLayer(nn.Cell):
    """Bidirectional LSTM layer"""
    def __init__(self, rnn_type='LSTM', input_size=512, hidden_size=512, 
                 num_layers=2, bidirectional=True):
        super(BiLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=False
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

    def construct(self, x, lgt=None):
        # x shape: (T, B, C)
        output, _ = self.rnn(x)
        
        if self.bidirectional:
            # Split bidirectional output and sum
            forward_out = output[:, :, :self.hidden_size]
            backward_out = output[:, :, self.hidden_size:]
            output = forward_out + backward_out
            
        return {
            "predictions": output,
            "feat_len": lgt
        }

class ResNet34Backbone(nn.Cell):
    """ResNet34 backbone for feature extraction"""
    def __init__(self):
        super(ResNet34Backbone, self).__init__()
        # Simplified ResNet34 implementation for CPU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode='pad')
        
        # Simplified residual blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # First block with potential stride
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, 
                               padding=1, has_bias=False, pad_mode='pad'))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, stride=1, 
                                   padding=1, has_bias=False, pad_mode='pad'))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            
        return nn.SequentialCell(*layers)

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
