#!/usr/bin/env python3
"""
TFNet的模型评估脚本
"""

import os
import sys
import json
import logging
from datetime import datetime

import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net

# 尝试导入新API，如果不可用则回退到旧版本
try:
    from mindspore import set_device
    MINDSPORE_NEW_API = True
except ImportError:
    MINDSPORE_NEW_API = False

# 将当前目录添加到路径以便导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from tfnet_model import TFNetModel
from data_processor import build_vocabulary, create_dataset
from decoder import CTCDecoder, calculate_wer_score, WERCalculator
from utils import (
    normalize_path, ensure_directory_exists, safe_file_path,
    check_file_exists, print_error_details
)

class TFNetEvaluator:
    """TFNet模型评估器"""
    
    def __init__(self, config_path=None, model_path=None):
        try:
            # 初始化配置
            print("正在初始化TFNet评估器...")
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
            self.model_path = model_path

            # 确保输出目录存在
            output_dir = self.config_manager.get("paths.output_dir")
            if output_dir:
                ensure_directory_exists(output_dir, create=True)
        
        # 设置MindSpore上下文，具有API兼容性
        device_target = self.config_manager.get("model.device_target", "CPU")

        # 使用新API（如果可用），否则回退到旧API
        if MINDSPORE_NEW_API:
            try:
                context.set_context(mode=context.GRAPH_MODE)
                set_device(device_target)
                print(f"✓ MindSpore device set to: {device_target} (new API)")
            except Exception as e:
                print(f"Warning: New API failed, using fallback: {e}")
                context.set_context(
                    mode=context.GRAPH_MODE,
                    device_target=device_target
                )
                print(f"✓ MindSpore device set to: {device_target} (fallback API)")
        else:
            context.set_context(
                mode=context.GRAPH_MODE,
                device_target=device_target
            )
            print(f"✓ MindSpore device set to: {device_target} (legacy API)")
        
        # 初始化日志
        self._setup_logging()
        
        # 初始化组件
        self.model = None
        self.test_dataset = None
        self.word2idx = None
        self.idx2word = None
        self.decoder = None
        
            self.logger.info("TFNet Evaluator initialized successfully")

        except Exception as e:
            print_error_details(e, "TFNet Evaluator initialization")
            raise
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.config_manager.get("logging.level", "INFO"))
        
        # 创建记录器
        self.logger = logging.getLogger('TFNetEvaluator')
        self.logger.setLevel(log_level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def load_vocabulary(self):
        """从文件加载词汇表或从数据构建词汇表"""
        vocab_path = os.path.join(self.config_manager.get("paths.output_dir"), "vocabulary.json")
        
        if os.path.exists(vocab_path):
            self.logger.info(f"Loading vocabulary from {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.word2idx = vocab_data['word2idx']
            self.idx2word = vocab_data['idx2word']
            vocab_size = vocab_data['vocab_size']
        else:
            self.logger.info("Building vocabulary from data")
            dataset_config = self.config_manager.get_dataset_config()
            
            self.word2idx, vocab_size, self.idx2word = build_vocabulary(
                dataset_config["train_label_path"],
                dataset_config["valid_label_path"],
                dataset_config["test_label_path"],
                dataset_config["name"]
            )
        
        self.logger.info(f"Vocabulary size: {vocab_size}")
        return vocab_size
    
    def prepare_test_data(self):
        """准备测试数据集"""
        self.logger.info("Preparing test data...")
        
        dataset_config = self.config_manager.get_dataset_config()
        
        # 创建测试数据集
        self.test_dataset = create_dataset(
            data_path=dataset_config["test_data_path"],
            label_path=dataset_config["test_label_path"],
            word2idx=self.word2idx,
            dataset_name=dataset_config["name"],
            batch_size=1,  # Use batch size 1 for evaluation
            is_train=False,
            num_workers=1
        )
        
        self.logger.info("Test data preparation completed")
    
    def load_model(self, vocab_size):
        """加载训练好的模型"""
        self.logger.info("Loading model...")
        
        model_config = self.config_manager.get_model_config()
        
        # 创建模型
        self.model = TFNetModel(
            hidden_size=model_config["hidden_size"],
            word_set_num=vocab_size,
            device_target=model_config["device_target"],
            dataset_name=model_config["dataset_name"]
        )
        
        # 加载检查点
        if self.model_path and os.path.exists(self.model_path):
            self.logger.info(f"Loading model from {self.model_path}")
            param_dict = load_checkpoint(self.model_path)
            load_param_into_net(self.model, param_dict)
        else:
            # 尝试加载最佳模型
            best_model_path = self.config_manager.get("paths.best_model_path")
            if os.path.exists(best_model_path):
                self.logger.info(f"Loading best model from {best_model_path}")
                param_dict = load_checkpoint(best_model_path)
                load_param_into_net(self.model, param_dict)
            else:
                self.logger.error("No trained model found!")
                return False
        
        # 初始化解码器
        self.decoder = CTCDecoder(
            gloss_dict=self.word2idx,
            num_classes=vocab_size + 1,
            search_mode='max',
            blank_id=self.config_manager.get("loss.ctc_blank_id", 0)
        )
        
        self.logger.info("Model loaded successfully")
        return True
    
    def evaluate(self):
        """在测试集上评估模型"""
        self.logger.info("Starting evaluation...")
        
        # 准备数据和模型
        vocab_size = self.load_vocabulary()
        self.prepare_test_data()
        
        if not self.load_model(vocab_size):
            return None
        
        # 将模型设置为评估模式
        self.model.set_train(False)
        
        # 评估指标
        total_samples = 0
        total_wer = 0.0
        all_predictions = []
        all_references = []
        
        # 处理测试数据
        for batch_idx, batch_data in enumerate(self.test_dataset.create_dict_iterator()):
            # 提取批次数据
            videos = batch_data['video']
            labels = batch_data['label']
            video_lengths = batch_data['videoLength']
            info = batch_data['info']
            
            # 前向传递
            log_probs1, _, _, _, _, lgt, _, _, _ = \
                self.model(videos, video_lengths, is_train=False)
            
            # 解码预测结果
            predictions, _ = self.decoder.decode(log_probs1, lgt, batch_first=False, probs=False)
            
            # 处理预测和参考文本
            for i, (pred, label_seq, sample_info) in enumerate(zip(predictions, labels, info)):
                # 将预测转换为单词
                pred_words = [word for word, _ in pred] if pred else []
                pred_sentence = ' '.join(pred_words)
                
                # 将参考序列转换为单词
                if isinstance(label_seq, (list, tuple)):
                    ref_words = [self.idx2word[idx] for idx in label_seq if idx < len(self.idx2word)]
                else:
                    ref_words = [self.idx2word[label_seq.asnumpy().item()]] if label_seq.asnumpy().item() < len(self.idx2word) else []
                ref_sentence = ' '.join(ref_words)
                
                # 计算该样本的WER
                sample_wer = WERCalculator.calculate_wer([ref_sentence], [pred_sentence])["wer"]
                
                all_predictions.append(pred_sentence)
                all_references.append(ref_sentence)
                total_wer += sample_wer
                total_samples += 1
                
                # 记录样本结果
                if batch_idx % 10 == 0:
                    self.logger.info(f"Sample {total_samples}: {sample_info}")
                    self.logger.info(f"  Reference: {ref_sentence}")
                    self.logger.info(f"  Prediction: {pred_sentence}")
                    self.logger.info(f"  WER: {sample_wer:.2f}%")
        
        # 计算总体指标
        avg_wer = total_wer / total_samples if total_samples > 0 else 0.0
        overall_wer = WERCalculator.calculate_wer(all_references, all_predictions)
        
        # 记录结果
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Average WER: {avg_wer:.2f}%")
        self.logger.info(f"Overall WER: {overall_wer['wer']:.2f}%")
        self.logger.info(f"Deletion rate: {overall_wer['del_rate']:.2f}%")
        self.logger.info(f"Insertion rate: {overall_wer['ins_rate']:.2f}%")
        self.logger.info(f"Substitution rate: {overall_wer['sub_rate']:.2f}%")
        
        # 保存结果
        results = {
            'total_samples': total_samples,
            'average_wer': avg_wer,
            'overall_wer': overall_wer,
            'predictions': all_predictions,
            'references': all_references,
            'evaluation_time': datetime.now().isoformat()
        }
        
        results_path = os.path.join(
            self.config_manager.get("paths.output_dir"), 
            f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {results_path}")
        
        return results

def main():
    """主入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TFNet Evaluation Script')
    parser.add_argument('--config', type=str, default='training/configs/tfnet_config.json',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = TFNetEvaluator(config_path=args.config, model_path=args.model)
    
    # Start evaluation
    try:
        results = evaluator.evaluate()
        if results:
            print(f"\nEvaluation completed successfully!")
            print(f"Overall WER: {results['overall_wer']['wer']:.2f}%")
        else:
            print("Evaluation failed!")
    except Exception as e:
        evaluator.logger.error(f"Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
