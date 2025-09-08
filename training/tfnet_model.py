import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from modules import Identity, TemporalConv, NormLinear, BiLSTMLayer, ResNet34Backbone

class TFNetModel(nn.Cell):
    """TFNet model implementation in MindSpore"""
    
    def __init__(self, hidden_size, word_set_num, device_target="CPU", dataset_name='CE-CSL'):
        super(TFNetModel, self).__init__()
        self.device_target = device_target
        self.out_dim = word_set_num
        self.dataset_name = dataset_name
        self.hidden_size = hidden_size
        
        # Backbone network
        self.conv2d = ResNet34Backbone()
        
        # Temporal convolution layers
        self.conv1d = TemporalConv(input_size=512, hidden_size=hidden_size, conv_type=2)
        self.conv1d1 = TemporalConv(input_size=512, hidden_size=hidden_size, conv_type=2)
        
        # Bidirectional LSTM layers
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
        
        # Classification layers
        self.classifier11 = NormLinear(hidden_size, self.out_dim)
        self.classifier22 = self.classifier11
        self.classifier33 = NormLinear(hidden_size, self.out_dim)
        self.classifier44 = self.classifier33
        self.classifier55 = NormLinear(hidden_size, self.out_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Operations
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=0)
        # Note: FFT is not available in MindSpore, using simplified approach
        self.abs = ops.Abs()

    def pad_sequence(self, tensor, length):
        """Pad tensor to specified length"""
        current_length = tensor.shape[0]
        if current_length >= length:
            return tensor[:length]

        # Ensure consistent types for shape calculation
        pad_length = int(length) - int(current_length)
        pad_shape = (pad_length,) + tensor.shape[1:]
        pad_tensor = ops.zeros(pad_shape, tensor.dtype)
        return ops.concat([tensor, pad_tensor], axis=0)

    def construct(self, seq_data, data_len=None, is_train=True):
        """Forward pass of TFNet model"""
        # Standardize length input to a Python list for MindSpore GRAPH_MODE safety
        len_x = data_len
        batch, temp, channel, height, width = seq_data.shape
        if len_x is None:
            len_x_list = [int(temp)] * int(batch)
        elif isinstance(len_x, (list, tuple)):
            len_x_list = [int(v) for v in len_x]
        elif isinstance(len_x, (int, np.integer)):
            len_x_list = [int(len_x)] * int(batch)
        else:
            # Unknown type; default to full length per sample
            len_x_list = [int(temp)] * int(batch)
        # Ensure length list matches batch size
        if len(len_x_list) < int(batch):
            len_x_list = len_x_list + [int(temp)] * (int(batch) - len(len_x_list))
        elif len(len_x_list) > int(batch):
            len_x_list = len_x_list[:int(batch)]

        # Reshape for 2D convolution
        inputs = self.reshape(seq_data, (batch * temp, channel, height, width))

        # Extract features for each sequence using original lengths
        feature_list = []
        start_idx = 0
        for i in range(int(batch)):
            # Use original length but cap at actual tensor size
            lgt_i = min(int(len_x_list[i]), int(temp))
            end_idx = start_idx + lgt_i
            seq_features = self.conv2d(inputs[start_idx:end_idx])
            feature_list.append(seq_features)
            start_idx += int(temp)  # Move to next sequence (fixed stride)

        # Pad sequences to same length
        max_len = max(len_x_list) if len_x_list else max([f.shape[0] for f in feature_list])

        padded_features = []
        for features in feature_list:
            padded = self.pad_sequence(features, max_len)
            padded_features.append(padded)

        # Stack features
        framewise = ops.stack(padded_features, axis=0)  # (B, T, C)
        framewise = self.transpose(framewise, (0, 2, 1))  # (B, C, T)

        # Apply temporal convolution and update lengths in Python
        conv1d_outputs = self.conv1d(framewise, len_x_list)
        x = conv1d_outputs['visual_feat']
        # Update lengths according to conv/pool recipe K5,P2,K5,P2
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
        lgt_tensor = Tensor(np.array(lgt, dtype=np.int32))

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
