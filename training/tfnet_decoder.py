"""
TFNet解码和评估模块
基于MindSpore实现CTC解码和WER计算
从TFNet项目迁移并优化
"""

import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from typing import List, Tuple, Dict, Optional
import json
import editdistance
from collections import defaultdict

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTCDecoder:
    """CTC解码器"""
    
    def __init__(self, vocab_file: str, blank_id: int = 0):
        self.blank_id = blank_id
        self.load_vocabulary(vocab_file)
        
    def load_vocabulary(self, vocab_file: str):
        """加载词汇表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.vocab_size = len(self.idx2word)
        
        logger.info(f"词汇表加载完成，词汇数: {self.vocab_size}")
    
    def greedy_decode(self, log_probs: np.ndarray, input_lengths: np.ndarray) -> List[List[int]]:
        """
        贪婪解码
        Args:
            log_probs: (batch_size, seq_len, vocab_size) 或 (seq_len, batch_size, vocab_size)
            input_lengths: (batch_size,) 实际序列长度
        Returns:
            解码后的序列列表
        """
        if log_probs.ndim == 3:
            if log_probs.shape[0] == log_probs.shape[1]:
                # 如果是 (batch_size, seq_len, vocab_size)，转换为 (seq_len, batch_size, vocab_size)
                if len(input_lengths) == log_probs.shape[0]:
                    log_probs = np.transpose(log_probs, (1, 0, 2))
        
        seq_len, batch_size, vocab_size = log_probs.shape
        decoded_sequences = []
        
        for b in range(batch_size):
            # 获取当前序列的有效长度
            valid_len = min(int(input_lengths[b]), seq_len)
            
            # 获取最大概率的索引
            max_indices = np.argmax(log_probs[:valid_len, b, :], axis=1)
            
            # CTC解码：移除重复和空白标记
            decoded = self._ctc_collapse(max_indices.tolist())
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def beam_search_decode(self, log_probs: np.ndarray, input_lengths: np.ndarray,
                          beam_size: int = 10) -> List[List[int]]:
        """
        束搜索解码
        Args:
            log_probs: (seq_len, batch_size, vocab_size)
            input_lengths: (batch_size,)
            beam_size: 束大小
        Returns:
            解码后的序列列表
        """
        if log_probs.ndim == 3:
            if log_probs.shape[0] == log_probs.shape[1]:
                if len(input_lengths) == log_probs.shape[0]:
                    log_probs = np.transpose(log_probs, (1, 0, 2))
        
        seq_len, batch_size, vocab_size = log_probs.shape
        decoded_sequences = []
        
        for b in range(batch_size):
            valid_len = min(int(input_lengths[b]), seq_len)
            sequence_probs = log_probs[:valid_len, b, :]
            
            # 束搜索
            decoded = self._beam_search_single(sequence_probs, beam_size)
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def _beam_search_single(self, log_probs: np.ndarray, beam_size: int) -> List[int]:
        """
        单个序列的束搜索
        """
        seq_len, vocab_size = log_probs.shape
        
        # 初始化束
        # 每个beam: (sequence, log_prob)
        beams = [([self.blank_id], 0.0)]
        
        for t in range(seq_len):
            new_beams = []
            
            for sequence, log_prob in beams:
                for c in range(vocab_size):
                    new_log_prob = log_prob + log_probs[t, c]
                    
                    if c == self.blank_id:
                        # 空白标记
                        new_sequence = sequence.copy()
                    else:
                        # 非空白标记
                        if len(sequence) > 0 and sequence[-1] == c:
                            # 重复字符，需要空白分隔
                            new_sequence = sequence.copy()
                        else:
                            new_sequence = sequence + [c]
                    
                    new_beams.append((new_sequence, new_log_prob))
            
            # 按概率排序并保留top-k
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
        
        # 返回最佳序列，移除空白标记
        best_sequence = beams[0][0]
        return [c for c in best_sequence if c != self.blank_id]
    
    def _ctc_collapse(self, sequence: List[int]) -> List[int]:
        """
        CTC折叠：移除重复和空白标记
        """
        collapsed = []
        prev = None
        
        for token in sequence:
            if token != self.blank_id and token != prev:
                collapsed.append(token)
            prev = token
        
        return collapsed
    
    def decode_to_words(self, sequences: List[List[int]]) -> List[List[str]]:
        """
        将索引序列转换为词汇序列
        """
        word_sequences = []
        
        for sequence in sequences:
            words = []
            for idx in sequence:
                if 0 <= idx < len(self.idx2word):
                    words.append(self.idx2word[idx])
                else:
                    logger.warning(f"无效索引: {idx}")
            word_sequences.append(words)
        
        return word_sequences

class WERCalculator:
    """词错误率(WER)计算器"""
    
    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case
    
    def compute_wer(self, references: List[List[str]], 
                   hypotheses: List[List[str]]) -> Dict[str, float]:
        """
        计算WER
        Args:
            references: 参考句子列表
            hypotheses: 假设句子列表
        Returns:
            WER统计信息
        """
        assert len(references) == len(hypotheses), \
            f"参考句子数({len(references)})与假设句子数({len(hypotheses)})不匹配"
        
        total_words = 0
        total_errors = 0
        sentence_accuracies = []
        
        for ref, hyp in zip(references, hypotheses):
            if self.ignore_case:
                ref = [w.lower() for w in ref]
                hyp = [w.lower() for w in hyp]
            
            # 计算编辑距离
            errors = editdistance.eval(ref, hyp)
            total_errors += errors
            total_words += len(ref)
            
            # 句子级准确率
            sentence_acc = 1.0 if errors == 0 else 0.0
            sentence_accuracies.append(sentence_acc)
        
        wer = total_errors / total_words if total_words > 0 else 0.0
        sentence_accuracy = np.mean(sentence_accuracies)
        
        return {
            'wer': wer * 100,  # 转换为百分比
            'total_errors': total_errors,
            'total_words': total_words,
            'sentence_accuracy': sentence_accuracy * 100,
            'num_sentences': len(references)
        }
    
    def compute_detailed_wer(self, references: List[List[str]], 
                           hypotheses: List[List[str]]) -> Dict:
        """
        计算详细的WER统计
        """
        assert len(references) == len(hypotheses)
        
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_words = 0
        
        detailed_results = []
        
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            if self.ignore_case:
                ref = [w.lower() for w in ref]
                hyp = [w.lower() for w in hyp]
            
            # 详细的编辑操作分析
            ops = self._get_edit_operations(ref, hyp)
            
            substitutions = sum(1 for op in ops if op == 'substitute')
            deletions = sum(1 for op in ops if op == 'delete')
            insertions = sum(1 for op in ops if op == 'insert')
            
            total_substitutions += substitutions
            total_deletions += deletions
            total_insertions += insertions
            total_words += len(ref)
            
            detailed_results.append({
                'index': i,
                'reference': ref,
                'hypothesis': hyp,
                'substitutions': substitutions,
                'deletions': deletions,
                'insertions': insertions,
                'ref_length': len(ref),
                'hyp_length': len(hyp),
                'wer': (substitutions + deletions + insertions) / len(ref) if len(ref) > 0 else 0.0
            })
        
        overall_wer = (total_substitutions + total_deletions + total_insertions) / total_words if total_words > 0 else 0.0
        
        return {
            'overall_wer': overall_wer * 100,
            'total_substitutions': total_substitutions,
            'total_deletions': total_deletions,
            'total_insertions': total_insertions,
            'total_words': total_words,
            'substitution_rate': total_substitutions / total_words * 100 if total_words > 0 else 0.0,
            'deletion_rate': total_deletions / total_words * 100 if total_words > 0 else 0.0,
            'insertion_rate': total_insertions / total_words * 100 if total_words > 0 else 0.0,
            'detailed_results': detailed_results
        }
    
    def _get_edit_operations(self, ref: List[str], hyp: List[str]) -> List[str]:
        """
        获取编辑操作序列
        """
        # 动态规划计算编辑距离并记录操作
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        ops = [[None] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
            ops[i][0] = 'delete'
        for j in range(n + 1):
            dp[0][j] = j
            ops[0][j] = 'insert'
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = 'match'
                else:
                    substitute_cost = dp[i-1][j-1] + 1
                    delete_cost = dp[i-1][j] + 1
                    insert_cost = dp[i][j-1] + 1
                    
                    min_cost = min(substitute_cost, delete_cost, insert_cost)
                    dp[i][j] = min_cost
                    
                    if min_cost == substitute_cost:
                        ops[i][j] = 'substitute'
                    elif min_cost == delete_cost:
                        ops[i][j] = 'delete'
                    else:
                        ops[i][j] = 'insert'
        
        # 回溯获取操作序列
        operations = []
        i, j = m, n
        while i > 0 or j > 0:
            if i == 0:
                operations.append('insert')
                j -= 1
            elif j == 0:
                operations.append('delete')
                i -= 1
            elif ops[i][j] == 'match':
                i -= 1
                j -= 1
            elif ops[i][j] == 'substitute':
                operations.append('substitute')
                i -= 1
                j -= 1
            elif ops[i][j] == 'delete':
                operations.append('delete')
                i -= 1
            else:  # insert
                operations.append('insert')
                j -= 1
        
        return operations[::-1]

class TFNetEvaluator:
    """TFNet模型评估器"""
    
    def __init__(self, vocab_file: str, blank_id: int = 0):
        self.decoder = CTCDecoder(vocab_file, blank_id)
        self.wer_calculator = WERCalculator()
    
    def evaluate_predictions(self, predictions: List[np.ndarray], 
                           ground_truths: List[List[int]],
                           input_lengths: List[int],
                           decode_method: str = 'greedy',
                           beam_size: int = 10) -> Dict:
        """
        评估预测结果
        Args:
            predictions: 模型预测的log_probs列表
            ground_truths: 真实标签列表
            input_lengths: 输入长度列表
            decode_method: 解码方法 ('greedy' 或 'beam_search')
            beam_size: 束搜索大小
        Returns:
            评估结果
        """
        all_decoded = []
        all_references = []
        
        for pred, gt, length in zip(predictions, ground_truths, input_lengths):
            # 解码预测
            if decode_method == 'greedy':
                decoded = self.decoder.greedy_decode(pred[np.newaxis, :], np.array([length]))
            else:
                decoded = self.decoder.beam_search_decode(pred[np.newaxis, :], np.array([length]), beam_size)
            
            all_decoded.extend(decoded)
            
            # 处理真实标签（移除填充）
            gt_cleaned = [token for token in gt if token != 0]  # 0是填充标记
            all_references.append(gt_cleaned)
        
        # 转换为词汇
        decoded_words = self.decoder.decode_to_words(all_decoded)
        reference_words = self.decoder.decode_to_words(all_references)
        
        # 计算WER
        wer_results = self.wer_calculator.compute_wer(reference_words, decoded_words)
        detailed_wer = self.wer_calculator.compute_detailed_wer(reference_words, decoded_words)
        
        return {
            'wer': wer_results,
            'detailed_wer': detailed_wer,
            'decoded_words': decoded_words,
            'reference_words': reference_words,
            'num_samples': len(predictions)
        }
    
    def save_predictions(self, results: Dict, output_file: str):
        """保存预测结果"""
        output_data = {
            'wer_summary': results['wer'],
            'detailed_wer': results['detailed_wer'],
            'predictions': []
        }
        
        for i, (decoded, reference) in enumerate(zip(results['decoded_words'], results['reference_words'])):
            output_data['predictions'].append({
                'index': i,
                'decoded': decoded,
                'reference': reference,
                'decoded_sentence': ' '.join(decoded),
                'reference_sentence': ' '.join(reference)
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"预测结果已保存到: {output_file}")

if __name__ == "__main__":
    # 测试解码器
    vocab_file = "./backend/models/vocab.json"
    
    # 创建示例词汇表用于测试
    example_vocab = {
        'word2idx': {' ': 0, '你好': 1, '世界': 2, '手语': 3, '识别': 4},
        'idx2word': [' ', '你好', '世界', '手语', '识别'],
        'vocab_size': 5
    }
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(example_vocab, f, indent=2, ensure_ascii=False)
    
    # 测试解码器
    decoder = CTCDecoder(vocab_file, blank_id=0)
    
    # 模拟预测概率
    log_probs = np.random.randn(10, 1, 5)  # (seq_len, batch_size, vocab_size)
    input_lengths = np.array([8])
    
    # 贪婪解码
    greedy_results = decoder.greedy_decode(log_probs, input_lengths)
    print("贪婪解码结果:", greedy_results)
    
    # 束搜索解码
    beam_results = decoder.beam_search_decode(log_probs, input_lengths, beam_size=3)
    print("束搜索解码结果:", beam_results)
    
    # 转换为词汇
    word_results = decoder.decode_to_words(greedy_results)
    print("词汇结果:", word_results)
    
    # WER计算测试
    wer_calc = WERCalculator()
    
    references = [['你好', '世界'], ['手语', '识别']]
    hypotheses = [['你好', '世界'], ['手语']]
    
    wer_results = wer_calc.compute_wer(references, hypotheses)
    print("WER结果:", wer_results)
    
    logger.info("解码和评估模块测试完成")
