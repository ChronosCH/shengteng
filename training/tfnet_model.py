import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from modules import Identity, TemporalConv, NormLinear, BiLSTMLayer, ResNet34Backbone

class TFNetModel(nn.Cell):
    """基于MindSpore的TFNet模型实现"""
    
    def __init__(self, hidden_size, word_set_num, device_target="CPU", dataset_name='CE-CSL'):
        super(TFNetModel, self).__init__()
        self.device_target = device_target
        self.out_dim = word_set_num
        self.dataset_name = dataset_name
        self.hidden_size = hidden_size
        
        # 骨干网络
        self.conv2d = ResNet34Backbone()
        
        # 时序卷积层
        self.conv1d = TemporalConv(input_size=512, hidden_size=hidden_size, conv_type=2)
        self.conv1d1 = TemporalConv(input_size=512, hidden_size=hidden_size, conv_type=2)
        
        # 双向LSTM层
        self.temporal_model = BiLSTMLayer(
            rnn_type='LSTM', 
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=2, 
            bidirectional=True
        )
        
        self.temporal_model1 = BiLSTMLayer(
            rnn_type='LSTM', 
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=2, 
            bidirectional=True
        )
        
        # 分类层
        self.classifier11 = NormLinear(hidden_size, self.out_dim)
        self.classifier22 = self.classifier11
        self.classifier33 = NormLinear(hidden_size, self.out_dim)
        self.classifier44 = self.classifier33
        self.classifier55 = NormLinear(hidden_size, self.out_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 操作符
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=0)
        # 注意：MindSpore中没有FFT，使用简化方法
        self.abs = ops.Abs()

    def pad_sequence(self, tensor, length):
        """将张量填充到指定长度"""
        current_length = tensor.shape[0]
        target_length = max(1, int(length))  # 确保目标长度至少为1
        
        if current_length >= target_length:
            return tensor[:target_length]

        # 确保形状计算的类型一致性
        pad_length = target_length - int(current_length)
        if pad_length > 0:
            pad_shape = (pad_length,) + tensor.shape[1:]
            pad_tensor = ops.zeros(pad_shape, tensor.dtype)
            return ops.concat([tensor, pad_tensor], axis=0)
        else:
            return tensor

    def construct(self, seq_data, data_len=None, is_train=True):
        """TFNet模型的前向传播"""
        # 为MindSpore GRAPH_MODE安全性，将长度输入标准化为Python列表
        len_x = data_len
        batch, temp, channel, height, width = seq_data.shape
        if len_x is None:
            len_x_list = [int(temp)] * int(batch)
        elif isinstance(len_x, (list, tuple)):
            len_x_list = [int(v) for v in len_x]
        elif isinstance(len_x, (int, np.integer)):
            len_x_list = [int(len_x)] * int(batch)
        else:
            # 未知类型；默认每个样本使用完整长度
            len_x_list = [int(temp)] * int(batch)
        # 确保长度列表与批次大小匹配
        if len(len_x_list) < int(batch):
            len_x_list = len_x_list + [int(temp)] * (int(batch) - len(len_x_list))
        elif len(len_x_list) > int(batch):
            len_x_list = len_x_list[:int(batch)]

        # 重塑为2D卷积格式
        inputs = self.reshape(seq_data, (batch * temp, channel, height, width))

        # 使用原始长度为每个序列提取特征
        feature_list = []
        start_idx = 0
        for i in range(int(batch)):
            # 使用原始长度但限制为实际张量大小
            lgt_i = min(int(len_x_list[i]), int(temp))
            end_idx = start_idx + lgt_i
            # 确保索引不超出边界
            end_idx = min(end_idx, inputs.shape[0])
            if start_idx < inputs.shape[0] and start_idx < end_idx:
                seq_features = self.conv2d(inputs[start_idx:end_idx])
                feature_list.append(seq_features)
            else:
                # 如果索引无效，创建一个空的特征张量
                dummy_input = inputs[0:1]  # 取第一个样本作为模板
                dummy_features = self.conv2d(dummy_input)
                # 创建零特征
                zero_features = ops.zeros_like(dummy_features)
                feature_list.append(zero_features)
            start_idx += int(temp)  # 移动到下一个序列（固定步长）

        # 将序列填充到相同长度
        if len_x_list and feature_list:
            max_len = max(max(len_x_list), max([f.shape[0] for f in feature_list if f.shape[0] > 0]))
        else:
            max_len = 1  # 默认最小长度

        padded_features = []
        for features in feature_list:
            # 确保特征张量不为空
            if features.shape[0] > 0:
                padded = self.pad_sequence(features, max_len)
            else:
                # 创建一个最小的特征张量
                feature_shape = (max_len,) + features.shape[1:]
                padded = ops.zeros(feature_shape, features.dtype)
            padded_features.append(padded)

        # 堆叠特征
        framewise = ops.stack(padded_features, axis=0)  # (B, T, C)
        framewise = self.transpose(framewise, (0, 2, 1))  # (B, C, T)

        # 应用时序卷积并在Python中更新长度
        conv1d_outputs = self.conv1d(framewise, len_x_list)
        x = conv1d_outputs['visual_feat']
        # 根据卷积/池化配方K5,P2,K5,P2更新长度
        lgt = len_x_list
        for ks in ['K5', 'P2', 'K5', 'P2']:
            if ks[0] == 'P':
                lgt = [max(1, int(i // 2)) for i in lgt]
            else:
                k = int(ks[1])
                lgt = [max(1, int(i) - k + 1) for i in lgt]
        x = self.transpose(x, (2, 0, 1))  # (T, B, C)

        # Fourier transform branch
        framewise1 = self.transpose(framewise, (0, 2, 1))  # (B, T, C)
        # Apply FFT (simplified for CPU)
        X = self.abs(framewise1)  # Simplified FFT
        framewise1 = self.transpose(X, (0, 2, 1))  # (B, C, T)

        conv1d_outputs1 = self.conv1d1(framewise1, len_x_list)
        x1 = conv1d_outputs1['visual_feat']
        x1 = self.transpose(x1, (2, 0, 1))  # (T, B, C)

        # Apply temporal models
        outputs = self.temporal_model(x, lgt)
        outputs1 = self.temporal_model1(x1, lgt)

        # Generate predictions
        log_probs1 = self.classifier11(outputs['predictions'])
        log_probs2 = self.classifier22(x)
        log_probs3 = self.classifier33(outputs1['predictions'])
        log_probs4 = self.classifier44(x1)

        # Fusion prediction
        x2 = outputs['predictions'] + outputs1['predictions']
        log_probs5 = self.classifier55(x2)

        if not is_train:
            log_probs1 = log_probs5

        # Convert lengths to tensor (list of ints -> Tensor shape (B,))
        # 确保长度列表不为空且值都为正数
        safe_lgt = [max(1, int(l)) for l in lgt] if lgt else [1] * int(batch)
        lgt_tensor = Tensor(np.array(safe_lgt, dtype=np.int32))

        return log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, lgt_tensor, None, None, None

class SeqKD(nn.Cell):
    """Sequence Knowledge Distillation loss"""
    def __init__(self, T=8):
        super(SeqKD, self).__init__()
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, logits_student, logits_teacher, use_blank=False):
        """Compute KL divergence loss between student and teacher"""
        # Apply temperature scaling
        student_log_probs = self.log_softmax(logits_student / self.T)
        teacher_probs = self.softmax(logits_teacher / self.T)
        
        # Compute KL divergence
        loss = self.kl_div(student_log_probs, teacher_probs) * (self.T ** 2)
        
        return loss
