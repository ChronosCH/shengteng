"""
I3D模型的MindSpore实现
基于Inception-v1 I3D架构
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class Unit3D(nn.Cell):
    """3D卷积单元"""
    
    def __init__(self, in_channels, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 use_batch_norm=True,
                 use_bias=False,
                 use_activation=True):
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._use_activation = use_activation
        self.padding = padding
        
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            pad_mode='pad',
            padding=0,
            has_bias=use_bias
        )
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.99)
        
        self.relu = nn.ReLU()
        self.pad_op = ops.Pad(((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
        
    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)
    
    def construct(self, x):
        # 计算same padding
        batch, channel, t, h, w = x.shape
        
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        
        # MindSpore padding格式: ((0,0), (0,0), (front,back), (top,bottom), (left,right))
        paddings = ((0, 0), (0, 0), (pad_t_f, pad_t_b), (pad_h_f, pad_h_b), (pad_w_f, pad_w_b))
        pad_op = ops.Pad(paddings)
        x = pad_op(x)
        
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._use_activation:
            x = self.relu(x)
        return x


class InceptionModule(nn.Cell):
    """Inception模块"""
    
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        
        self.b0 = Unit3D(in_channels, out_channels[0], kernel_shape=(1, 1, 1))
        
        self.b1a = Unit3D(in_channels, out_channels[1], kernel_shape=(1, 1, 1))
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=(3, 3, 3))
        
        self.b2a = Unit3D(in_channels, out_channels[3], kernel_shape=(1, 1, 1))
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=(3, 3, 3))
        
        self.b3a = nn.MaxPool3d(kernel_size=3, stride=1, pad_mode='same')
        self.b3b = Unit3D(in_channels, out_channels[5], kernel_shape=(1, 1, 1))
        
        self.concat = ops.Concat(axis=1)
    
    def construct(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return self.concat((b0, b1, b2, b3))


class InceptionI3d(nn.Cell):
    """Inception I3D模型"""
    
    def __init__(self, num_classes=400, in_channels=3, dropout_keep_prob=0.5):
        super(InceptionI3d, self).__init__()
        
        self._num_classes = num_classes
        self._spatial_squeeze = True
        
        # 构建网络层
        self.Conv3d_1a_7x7 = Unit3D(in_channels, 64, kernel_shape=(7, 7, 7), 
                                     stride=(2, 2, 2), padding=(3, 3, 3))
        self.MaxPool3d_2a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), 
                                             stride=(1, 2, 2), pad_mode='same')
        
        self.Conv3d_2b_1x1 = Unit3D(64, 64, kernel_shape=(1, 1, 1))
        self.Conv3d_2c_3x3 = Unit3D(64, 192, kernel_shape=(3, 3, 3), padding=(1, 1, 1))
        
        self.MaxPool3d_3a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), 
                                             stride=(1, 2, 2), pad_mode='same')
        
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        
        self.MaxPool3d_4a_3x3 = nn.MaxPool3d(kernel_size=3, stride=2, pad_mode='same')
        
        self.Mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        
        self.MaxPool3d_5a_2x2 = nn.MaxPool3d(kernel_size=2, stride=2, pad_mode='same')
        
        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        
        # Head
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.dropout = nn.Dropout(keep_prob=dropout_keep_prob)
        self.logits = Unit3D(1024, self._num_classes, kernel_shape=(1, 1, 1),
                            use_batch_norm=False, use_bias=True, use_activation=False)
        
        self.squeeze = ops.Squeeze((3, 4))
    
    def replace_logits(self, num_classes):
        """替换分类层"""
        self._num_classes = num_classes
        self.logits = Unit3D(1024, num_classes, kernel_shape=(1, 1, 1),
                            use_batch_norm=False, use_bias=True, use_activation=False)
    
    def construct(self, x):
        # Stem
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        
        # Inception blocks
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        
        # Head
        x = self.avg_pool(x)
        x = self.dropout(x)
        logits = self.logits(x)
        
        if self._spatial_squeeze:
            logits = self.squeeze(logits)
        
        return logits
