import os
import sys

# 优雅的依赖导入处理
try:
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor, Parameter
    from mindspore.common.initializer import Normal, Constant
    import mindspore as ms
    MINDSPORE_AVAILABLE = True
except ImportError:
    print("警告: MindSpore未安装。请运行: pip install mindspore-gpu")
    print("如果您使用的是CPU环境，请运行: pip install mindspore")
    MINDSPORE_AVAILABLE = False
    
    # 创建模拟类以避免导入错误
    class MockClass:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def construct(self, *args, **kwargs):
            return self
            
    nn = type('nn', (), {
        'Cell': MockClass,
        'Conv3d': MockClass,
        'BatchNorm3d': MockClass,
        'ReLU': MockClass,
        'Dropout': MockClass,
        'MaxPool3d': MockClass,
        'AdaptiveAvgPool3d': MockClass,
        'SequentialCell': MockClass,
        'Flatten': MockClass,
        'Dense': MockClass,
        'CrossEntropyLoss': MockClass
    })()
    
    ops = type('ops', (), {
        'one_hot': lambda *args, **kwargs: None,
        'log_softmax': lambda *args, **kwargs: None,
        'reduce_mean': lambda *args, **kwargs: None,
        'reduce_sum': lambda *args, **kwargs: None
    })()
    
    ms = type('ms', (), {
        'float32': None,
        'common': type('common', (), {
            'initializer': type('initializer', (), {
                'initializer': lambda *args: None
            })()
        })()
    })()


def _to_6tuple_3dpad(pad):
    """将 padding 规范化为 MindSpore Conv3d 需要的 6 元组。
    规则：
      - int k -> (k,k, k,k, k,k)
      - (d,h,w) -> (d,d, h,h, w,w)
      - (f,b,t,bt,l,r) -> 原样
    """
    if isinstance(pad, int):
        return (pad, pad, pad, pad, pad, pad)
    if isinstance(pad, (tuple, list)):
        if len(pad) == 6:
            return tuple(pad)
        if len(pad) == 3:
            d, h, w = pad
            return (d, d, h, h, w, w)
    raise ValueError(f"Invalid padding for Conv3d: {pad}")


def _make_dropout(drop_prob: float):
    """兼容不同 MindSpore 版本的 Dropout 构造。
    期望传入丢弃概率 p（如 0.3）。
    新版接口：nn.Dropout(p=drop_prob)
    旧版接口：nn.Dropout(keep_prob=1 - drop_prob)
    """
    try:
        # 新版优先
        return nn.Dropout(p=drop_prob)
    except TypeError:
        # 旧版回退
        keep = 1.0 - float(drop_prob)
        return nn.Dropout(keep)


class Conv3DBlock(nn.Cell):
    """3D卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        pad6 = _to_6tuple_3dpad(padding)
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, pad_mode='pad', padding=pad6
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        # 降低早期Dropout强度，提升特征学习能力
        self.dropout = _make_dropout(0.1)
    
    def construct(self, x):
        x = self.conv3d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SpatioTemporalBlock(nn.Cell):
    """时空注意力块"""
    
    def __init__(self, channels):
        super(SpatioTemporalBlock, self).__init__()
        # 时间卷积：kernel=(3,1,1)，只在时间维前后各 pad 1
        self.temporal_conv = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1), 
            stride=1, pad_mode='pad', padding=(1, 1, 0, 0, 0, 0)
        )
        # 空间卷积：kernel=(1,3,3)，在高宽维各 pad 1
        self.spatial_conv = nn.Conv3d(
            channels, channels, kernel_size=(1, 3, 3), 
            stride=1, pad_mode='pad', padding=(0, 0, 1, 1, 1, 1)
        )
        self.batch_norm = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        
    def construct(self, x):
        # 时间卷积
        temporal = self.temporal_conv(x)
        # 空间卷积
        spatial = self.spatial_conv(x)
        # 残差连接
        out = x + temporal + spatial
        out = self.batch_norm(out)
        out = self.relu(out)
        return out


class ASLRecognitionModel(nn.Cell):
    """ASL手语识别模型"""
    
    def __init__(self, num_classes, sequence_length=16, input_size=(224, 224), base_channels=64):
        super(ASLRecognitionModel, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.base_channels = base_channels
        
        # 3D卷积特征提取器
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.feature_extractor = nn.SequentialCell([
            # 第一层: 输入 (B, 3, T, H, W)
            Conv3DBlock(3, c1, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # (B, 64, T, H/2, W/2)
            
            # 第二层
            Conv3DBlock(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # (B, 128, T/2, H/4, W/4)
            
            # 时空注意力块
            SpatioTemporalBlock(c2),
            
            # 第三层
            Conv3DBlock(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # (B, 256, T/4, H/8, W/8)
            
            # 时空注意力块
            SpatioTemporalBlock(c3),
            
            # 第四层
            Conv3DBlock(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # (B, 512, T/8, H/16, W/16)
        ])
        
        # 新增：全局平均池化，输出 (B, c4, 1, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 分类头中的 Dropout 使用兼容封装
        self.classifier = nn.SequentialCell([
            nn.Flatten(),  # (B, c4)
            nn.Dense(c4, 1024),
            nn.ReLU(),
            # 降低全连接层Dropout，避免梯度过弱
            _make_dropout(0.3),
            nn.Dense(1024, 512),
            nn.ReLU(),
            _make_dropout(0.3),
            nn.Dense(512, num_classes)
        ])
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化网络权重（ReLU配合He，Dense配合Xavier）。"""
        try:
            from mindspore.common.initializer import HeNormal, XavierUniform
            _use_adv_init = True
        except Exception:
            _use_adv_init = False
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                if _use_adv_init:
                    cell.weight.set_data(
                        ms.common.initializer.initializer(
                            XavierUniform(), cell.weight.shape, cell.weight.dtype
                        )
                    )
                else:
                    cell.weight.set_data(
                        ms.common.initializer.initializer(
                            Normal(0.02), cell.weight.shape, cell.weight.dtype
                        )
                    )
                cell.bias.set_data(
                    ms.common.initializer.initializer(
                        Constant(0), cell.bias.shape, cell.bias.dtype
                    )
                )
            elif isinstance(cell, (nn.Conv3d,)):
                if _use_adv_init:
                    cell.weight.set_data(
                        ms.common.initializer.initializer(
                            HeNormal(), cell.weight.shape, cell.weight.dtype
                        )
                    )
                else:
                    cell.weight.set_data(
                        ms.common.initializer.initializer(
                            Normal(0.02), cell.weight.shape, cell.weight.dtype
                        )
                    )
    
    def construct(self, x):
        # x shape: (B, C, T, H, W)
        features = self.feature_extractor(x)
        
        # 全局平均池化
        pooled = self.global_avg_pool(features)
        
        # 分类
        output = self.classifier(pooled)
        
        return output


class ASLLoss(nn.Cell):
    """ASL损失函数（带标签平滑的交叉熵）"""
    
    def __init__(self, num_classes, label_smoothing=0.1):
        super(ASLLoss, self).__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss()
        # 可选：使用 LogSoftmax 层，兼容不同版本
        try:
            self.log_softmax = nn.LogSoftmax(axis=1)
        except Exception:
            self.log_softmax = None
        # 预创建用于裁剪的常量（图内使用，不要在 construct 里 try/except）
        self._lbl_min = Tensor(0, ms.int32)
        self._lbl_max = Tensor(int(num_classes - 1), ms.int32)
        
    def construct(self, logits, labels):
        # 统一标签到 int32，并裁剪到合法范围（使用图算子，不使用 try/except）
        labels = ops.cast(labels, ms.int32)
        labels = ops.minimum(ops.maximum(labels, self._lbl_min), self._lbl_max)
        
        # 标签平滑
        if self.label_smoothing > 0:
            on_val = Tensor(1.0 - self.label_smoothing, ms.float32)
            off_val = Tensor(self.label_smoothing / (self.num_classes - 1), ms.float32)
            smooth_labels = ops.one_hot(labels, self.num_classes, on_val, off_val)
            if self.log_softmax is not None:
                log_probs = self.log_softmax(logits)
            else:
                log_probs = ops.log_softmax(logits, axis=1)
            loss_vec = -ops.reduce_sum(smooth_labels * log_probs, 1)
            loss = ops.reduce_mean(loss_vec)
        else:
            loss = self.cross_entropy(logits, labels)
        
        return loss


def create_model(num_classes, sequence_length=16, input_size=(224, 224), base_channels=64):
    """创建ASL识别模型"""
    model = ASLRecognitionModel(
        num_classes=num_classes,
        sequence_length=sequence_length,
        input_size=input_size,
        base_channels=base_channels
    )
    return model


if __name__ == "__main__":
    # 测试模型
    import numpy as np
    
    # 假设有1000个类别
    num_classes = 1000
    batch_size = 4
    sequence_length = 16
    
    model = create_model(num_classes, sequence_length)
    
    # 创建测试输入
    test_input = Tensor(np.random.randn(batch_size, 3, sequence_length, 224, 224).astype(np.float32))
    test_labels = Tensor(np.random.randint(0, num_classes, (batch_size,)).astype(np.int32))
    
    # 前向传播
    output = model(test_input)
    print(f"输出形状: {output.shape}")  # 应该是 (batch_size, num_classes)
    
    # 测试损失函数
    loss_fn = ASLLoss(num_classes)
    loss = loss_fn(output, test_labels)
    print(f"损失值: {loss}")
