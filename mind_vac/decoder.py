"""
MindSpore解码器 - CTC Beam Search Decoder
"""
import numpy as np
from itertools import groupby

try:
    import torch
    import ctcdecode
    HAS_CTCDECODE = True
except ImportError:
    HAS_CTCDECODE = False


class Decode:
    def __init__(self, gloss_dict, num_classes, search_mode='beam', blank_id=0):
        """
        初始化解码器
        
        Args:
            gloss_dict: 手语词汇字典
            num_classes: 类别数量
            search_mode: 搜索模式 'beam' 或 'max'
            blank_id: blank标签的ID
        """
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self._ctc_decoder = None

        if self.search_mode == 'beam' and HAS_CTCDECODE:
            vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
            self._ctc_decoder = ctcdecode.CTCBeamDecoder(
                vocab,
                beam_width=10,
                blank_id=blank_id,
                num_processes=10
            )
    
    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        """
        解码神经网络输出
        
        Args:
            nn_output: 网络输出 (T, B, C) 或 (B, T, C)
            vid_lgt: 序列长度
            batch_first: 是否batch维度在前
            probs: 输出是否已经是概率值
        
        Returns:
            识别结果列表
        """
        # 转换为numpy
        if hasattr(nn_output, 'asnumpy'):
            nn_output = nn_output.asnumpy()
        if hasattr(vid_lgt, 'asnumpy'):
            vid_lgt = vid_lgt.asnumpy()
        
        # 确保batch_first
        if not batch_first:
            nn_output = np.transpose(nn_output, (1, 0, 2))  # (T, B, C) -> (B, T, C)
        
        if self.search_mode == "max" or self._ctc_decoder is None:
            if self.search_mode == 'beam' and self._ctc_decoder is None:
                print("Warning: Using greedy decode because ctcdecode is not available")
            return self.max_decode(nn_output, vid_lgt)

        return self.beam_search(nn_output, vid_lgt, probs)
    
    def max_decode(self, nn_output, vid_lgt):
        """
        贪婪解码 - 选择每个时间步最大概率的类别
        
        Args:
            nn_output: (B, T, C)
            vid_lgt: (B,) 或单个int值
        Returns:
            解码结果列表
        """
        # 获取最大概率的索引
        index_list = np.argmax(nn_output, axis=2)  # (B, T)
        batch_size = index_list.shape[0]
        
        # 处理 vid_lgt 可能是单个值或数组的情况
        if isinstance(vid_lgt, (int, float, np.integer)):
            vid_lgt = [int(vid_lgt)] * batch_size
        elif isinstance(vid_lgt, np.ndarray):
            if vid_lgt.ndim == 0:  # 标量数组
                vid_lgt = [int(vid_lgt)] * batch_size
            else:
                vid_lgt = vid_lgt.tolist()
        
        ret_list = []
        for batch_idx in range(batch_size):
            seq_len = int(vid_lgt[batch_idx])
            # 获取有效长度的序列
            sequence = index_list[batch_idx, :seq_len]
            
            # 去除连续重复
            group_result = [x[0] for x in groupby(sequence)]
            
            # 过滤掉blank
            filtered = [x for x in group_result if x != self.blank_id]
            
            # 再次去除重复（可能在去除blank后产生）
            if len(filtered) > 0:
                max_result = [x[0] for x in groupby(filtered)]
            else:
                max_result = filtered
            
            # 转换为词汇
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx) 
                for idx, gloss_id in enumerate(max_result)
            ])
        
        return ret_list
    
    def beam_search(self, nn_output, vid_lgt, probs=False):
        if not HAS_CTCDECODE or self._ctc_decoder is None:
            return self.max_decode(nn_output, vid_lgt)

        if isinstance(nn_output, np.ndarray):
            torch_output = torch.from_numpy(nn_output).float()
        elif hasattr(nn_output, 'asnumpy'):
            torch_output = torch.from_numpy(nn_output.asnumpy()).float()
        else:
            torch_output = nn_output

        if not probs:
            torch_output = torch.softmax(torch_output, dim=-1)

        if isinstance(vid_lgt, np.ndarray):
            vid_tensor = torch.from_numpy(vid_lgt.astype(np.int32))
        elif hasattr(vid_lgt, 'asnumpy'):
            vid_tensor = torch.from_numpy(vid_lgt.asnumpy().astype(np.int32))
        elif isinstance(vid_lgt, (list, tuple)):
            vid_tensor = torch.tensor(vid_lgt, dtype=torch.int32)
        else:
            vid_tensor = torch.tensor([int(vid_lgt)], dtype=torch.int32)

        beam_result, beam_scores, timesteps, out_seq_len = self._ctc_decoder.decode(
            torch_output, vid_tensor
        )

        ret_list = []
        batch_size = beam_result.size(0)
        for batch_idx in range(batch_size):
            take_len = out_seq_len[batch_idx][0].item()
            first_result = beam_result[batch_idx][0][:take_len]
            if first_result.numel() != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(first_result)
            ])

        return ret_list


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """Log Softmax函数"""
    return np.log(softmax(x, axis=axis) + 1e-10)
