import mindspore as ms
import mindspore.ops as ops
import numpy as np
from itertools import groupby

class CTCDecoder:
    """用于序列预测的CTC解码器"""
    
    def __init__(self, gloss_dict, num_classes, search_mode='beam', blank_id=0):
        self.g2i_dict = {}
        for k, v in gloss_dict.items():
            if v == 0:
                continue
            self.g2i_dict[k] = v
        self.i2g_dict = {v: k for k, v in self.g2i_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        
        # 操作符
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.Argmax(axis=2)
        
    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        """将神经网络输出解码为序列"""
        if not batch_first:
            nn_output = ops.transpose(nn_output, (1, 0, 2))
        
        if self.search_mode == "max":
            return self.max_decode(nn_output, vid_lgt)
        else:
            return self.beam_search(nn_output, vid_lgt, probs)
    
    def max_decode(self, nn_output, vid_lgt):
        """最大值解码（贪心搜索）"""
        # 转换为numpy以便处理
        if isinstance(nn_output, ms.Tensor):
            nn_output = nn_output.asnumpy()
        if isinstance(vid_lgt, ms.Tensor):
            vid_lgt = vid_lgt.asnumpy()
        
        index_list = np.argmax(nn_output, axis=2)
        batch_size, lgt = index_list.shape
        ret_list = []
        
        for batch_idx in range(batch_size):
            # 获取到实际长度的序列
            sequence = index_list[batch_idx][:vid_lgt[batch_idx]]
            
            # 移除连续重复
            group_result = [x[0] for x in groupby(sequence)]
            
            # 移除空白标记
            filtered = [x for x in group_result if x != self.blank_id]
            
            # 转换为词汇
            decoded_sequence = []
            for idx, gloss_id in enumerate(filtered):
                if gloss_id in self.i2g_dict:
                    decoded_sequence.append((self.i2g_dict[gloss_id], idx))
                    
            ret_list.append(decoded_sequence)
        
        return ret_list
    
    def beam_search(self, nn_output, vid_lgt, probs=False):
        """简化的束搜索解码"""
        # 为简化起见，回退到最大值解码
        # 在完整实现中，您将使用适当的束搜索算法
        return self.max_decode(nn_output, vid_lgt)

class WERCalculator:
    """词错误率计算器"""
    
    @staticmethod
    def calculate_wer(references, hypotheses):
        """计算参考序列和假设序列之间的词错误率"""
        total_error = total_del = total_ins = total_sub = total_ref_len = 0
        
        for ref, hyp in zip(references, hypotheses):
            result = WERCalculator._wer_single(ref, hyp)
            total_error += result["num_err"]
            total_del += result["num_del"]
            total_ins += result["num_ins"]
            total_sub += result["num_sub"]
            total_ref_len += result["num_ref"]
        
        if total_ref_len == 0:
            return {"wer": 0.0, "del_rate": 0.0, "ins_rate": 0.0, "sub_rate": 0.0}
        
        wer = (total_error / total_ref_len) * 100
        del_rate = (total_del / total_ref_len) * 100
        ins_rate = (total_ins / total_ref_len) * 100
        sub_rate = (total_sub / total_ref_len) * 100
        
        return {
            "wer": wer,
            "del_rate": del_rate,
            "ins_rate": ins_rate,
            "sub_rate": sub_rate,
        }
    
    @staticmethod
    def _wer_single(ref, hyp):
        """Calculate WER for a single reference-hypothesis pair"""
        if isinstance(ref, str):
            ref = ref.strip().split()
        if isinstance(hyp, str):
            hyp = hyp.strip().split()
        
        edit_distance_matrix = WERCalculator._edit_distance(ref, hyp)
        alignment, _ = WERCalculator._get_alignment(ref, hyp, edit_distance_matrix)
        
        num_cor = sum([s == "C" for s in alignment])
        num_del = sum([s == "D" for s in alignment])
        num_ins = sum([s == "I" for s in alignment])
        num_sub = sum([s == "S" for s in alignment])
        num_err = num_del + num_ins + num_sub
        num_ref = len(ref)
        
        return {
            "alignment": alignment,
            "num_cor": num_cor,
            "num_del": num_del,
            "num_ins": num_ins,
            "num_sub": num_sub,
            "num_err": num_err,
            "num_ref": num_ref,
        }
    
    @staticmethod
    def _edit_distance(ref, hyp):
        """Calculate edit distance matrix"""
        WER_COST_DEL = 1
        WER_COST_INS = 1
        WER_COST_SUB = 1
        
        d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
        
        for i in range(len(ref) + 1):
            for j in range(len(hyp) + 1):
                if i == 0:
                    d[0][j] = j * WER_COST_DEL
                elif j == 0:
                    d[i][0] = i * WER_COST_INS
        
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitute = d[i - 1][j - 1] + WER_COST_SUB
                    insert = d[i][j - 1] + WER_COST_DEL
                    delete = d[i - 1][j] + WER_COST_INS
                    d[i][j] = min(substitute, insert, delete)
        
        return d
    
    @staticmethod
    def _get_alignment(ref, hyp, d):
        """Get alignment between reference and hypothesis"""
        WER_COST_DEL = 1
        WER_COST_INS = 1
        WER_COST_SUB = 1
        
        x = len(ref)
        y = len(hyp)
        max_len = x + y
        
        alignlist = []
        
        while True:
            if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
                break
            elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                alignlist.append("C")
                x = max(x - 1, 0)
                y = max(y - 1, 0)
            elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_DEL:
                alignlist.append("D")
                x = max(x, 0)
                y = max(y - 1, 0)
            elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
                alignlist.append("S")
                x = max(x - 1, 0)
                y = max(y - 1, 0)
            elif x >= 1 and d[x][y] == d[x - 1][y] + WER_COST_INS:
                alignlist.append("I")
                x = max(x - 1, 0)
                y = max(y, 0)
        
        return alignlist[::-1], None

def calculate_wer_score(prediction_result, target_data, idx2word, batch_size):
    """Calculate WER score for batch predictions"""
    hypotheses = []
    references = []
    
    for i in range(batch_size):
        # Convert indices to words
        pred_words = [idx2word[j] for j in prediction_result[i] if j < len(idx2word)]
        target_words = [idx2word[j] for j in target_data[i] if j < len(idx2word)]
        
        # Join words with spaces
        hypothesis = ' '.join(pred_words)
        reference = ' '.join(target_words)
        
        hypotheses.append(hypothesis)
        references.append(reference)
    
    # Calculate WER
    wer_result = WERCalculator.calculate_wer(references, hypotheses)
    return wer_result["wer"]
