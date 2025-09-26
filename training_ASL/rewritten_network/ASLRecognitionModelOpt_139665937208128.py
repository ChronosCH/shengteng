import mindspore
from mindspore import nn
import sys
sys.path.insert(0, '/data/src')
import os
import numpy as np
try:
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor, Parameter
    from mindspore.common.initializer import Normal, Constant
    import mindspore as ms
    MINDSPORE_AVAILABLE = True
except ImportError:
    print('警告: MindSpore未安装。请运行: pip install mindspore-gpu')
    print('如果您使用的是CPU环境，请运行: pip install mindspore')
    MINDSPORE_AVAILABLE = False

    class MockClass:

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def construct(self, *args, **kwargs):
            return self
    nn = type('nn', (), {'Cell': MockClass, 'Conv3d': MockClass, 'BatchNorm3d': MockClass, 'ReLU': MockClass, 'Dropout': MockClass, 'MaxPool3d': MockClass, 'AdaptiveAvgPool3d': MockClass, 'SequentialCell': MockClass, 'Flatten': MockClass, 'Dense': MockClass, 'CrossEntropyLoss': MockClass})()
    ops = type('ops', (), {'one_hot': lambda *args, **kwargs: None, 'log_softmax': lambda *args, **kwargs: None, 'reduce_mean': lambda *args, **kwargs: None, 'reduce_sum': lambda *args, **kwargs: None})()
    ms = type('ms', (), {'float32': None, 'common': type('common', (), {'initializer': type('initializer', (), {'initializer': lambda *args: None})()})()})()
try:
    from mindspore.common.initializer import HeNormal, XavierUniform
    _use_adv_init = True
except Exception:
    _use_adv_init = False

class Conv3DBlockOpt_3(nn.Cell):
    """3D卷积块"""

    def __init__(self, obj):
        super().__init__()
        for (key, value) in obj.__dict__.items():
            setattr(self, key, value)
        self.outcast_conv3d = obj.outcast_conv3d

    def construct(self, x):
        x = self.conv3d(x)
        x = self.outcast_conv3d(x, mindspore.float32)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SpatioTemporalBlockOpt_1(nn.Cell):
    """时空注意力块"""

    def __init__(self, obj):
        super().__init__()
        for (key, value) in obj.__dict__.items():
            setattr(self, key, value)
        self.outcast_spatial_conv = obj.outcast_spatial_conv
        self.outcast_temporal_conv = obj.outcast_temporal_conv

    def construct(self, x):
        temporal = self.temporal_conv(x)
        temporal = self.outcast_temporal_conv(temporal, mindspore.float32)
        spatial = self.spatial_conv(x)
        spatial = self.outcast_spatial_conv(spatial, mindspore.float32)
        add_var = x + temporal
        out = add_var + spatial
        out = self.batch_norm(out)
        out = self.relu(out)
        return out

class Conv3DBlockOpt_2(nn.Cell):
    """3D卷积块"""

    def __init__(self, obj):
        super().__init__()
        for (key, value) in obj.__dict__.items():
            setattr(self, key, value)
        self.outcast_conv3d = obj.outcast_conv3d

    def construct(self, x):
        x = self.conv3d(x)
        x = self.outcast_conv3d(x, mindspore.float32)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SpatioTemporalBlockOpt(nn.Cell):
    """时空注意力块"""

    def __init__(self, obj):
        super().__init__()
        for (key, value) in obj.__dict__.items():
            setattr(self, key, value)
        self.outcast_spatial_conv = obj.outcast_spatial_conv
        self.outcast_temporal_conv = obj.outcast_temporal_conv

    def construct(self, x):
        temporal = self.temporal_conv(x)
        temporal = self.outcast_temporal_conv(temporal, mindspore.float32)
        spatial = self.spatial_conv(x)
        spatial = self.outcast_spatial_conv(spatial, mindspore.float32)
        add_var = x + temporal
        out = add_var + spatial
        out = self.batch_norm(out)
        out = self.relu(out)
        return out

class Conv3DBlockOpt_1(nn.Cell):
    """3D卷积块"""

    def __init__(self, obj):
        super().__init__()
        for (key, value) in obj.__dict__.items():
            setattr(self, key, value)
        self.outcast_conv3d = obj.outcast_conv3d

    def construct(self, x):
        x = self.conv3d(x)
        x = self.outcast_conv3d(x, mindspore.float32)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Conv3DBlockOpt(nn.Cell):
    """3D卷积块"""

    def __init__(self, obj):
        super().__init__()
        for (key, value) in obj.__dict__.items():
            setattr(self, key, value)
        self.outcast_conv3d = obj.outcast_conv3d

    def construct(self, x):
        x = self.conv3d(x)
        x = self.outcast_conv3d(x, mindspore.float32)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ASLRecognitionModelOpt(nn.Cell):
    """ASL手语识别模型"""

    def __init__(self, obj):
        super().__init__()
        for (key, value) in obj.__dict__.items():
            setattr(self, key, value)
        self.feature_extractor[0] = Conv3DBlockOpt(self.feature_extractor[0])
        self.feature_extractor[2] = Conv3DBlockOpt_1(self.feature_extractor[2])
        self.feature_extractor[4] = SpatioTemporalBlockOpt(self.feature_extractor[4])
        self.feature_extractor[5] = Conv3DBlockOpt_2(self.feature_extractor[5])
        self.feature_extractor[7] = SpatioTemporalBlockOpt_1(self.feature_extractor[7])
        self.feature_extractor[8] = Conv3DBlockOpt_3(self.feature_extractor[8])

    def init_weights(self):
        """初始化网络权重（ReLU配合He，Dense配合Xavier）。"""
        try:
            from mindspore.common.initializer import HeNormal, XavierUniform
            _use_adv_init = True
        except Exception:
            _use_adv_init = False
        for (_, cell) in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                if _use_adv_init:
                    cell.weight.set_data(ms.common.initializer.initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))
                else:
                    cell.weight.set_data(ms.common.initializer.initializer(Normal(0.02), cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(ms.common.initializer.initializer(Constant(0), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, (nn.Conv3d,)):
                if _use_adv_init:
                    cell.weight.set_data(ms.common.initializer.initializer(HeNormal(), cell.weight.shape, cell.weight.dtype))
                else:
                    cell.weight.set_data(ms.common.initializer.initializer(Normal(0.02), cell.weight.shape, cell.weight.dtype))

    def construct(self, x):
        features = self.feature_extractor(x)
        pooled = self.global_avg_pool(features)
        output = self.classifier(pooled)
        return output