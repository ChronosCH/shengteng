#!/usr/bin/env python3
"""
完整的视频测试脚本
"""

import os
import sys
import json
import numpy as np
import cv2
import csv
from pathlib import Path

import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net, Tensor

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfnet_model import TFNetModel
from decoder import CTCDecoder
from data_processor import preprocess_words

class VideoTester:
    """视频测试器"""
    
    def __init__(self, model_path, vocab_path):
        """初始化测试器"""
        self.model_path = model_path
        self.vocab_path = vocab_path
        
        # 设置MindSpore环境
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target="CPU"
        )
        
        # 加载词汇表
        self.load_vocabulary()
        
        # 初始化模型
        self.init_model()
        
        # 初始化解码器
        self.init_decoder()
        
        print("✓ 视频测试器初始化完成")
    
    def load_vocabulary(self):
        """加载词汇表"""
        print(f"正在加载词汇表: {self.vocab_path}")
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"✓ 词汇表加载完成，包含 {self.vocab_size} 个词汇")
    
    def init_model(self):
        """初始化模型"""
        print("正在初始化模型...")
        
        # 创建模型 - 使用safe_gpu_config.json的参数
        self.model = TFNetModel(
            hidden_size=128,
            word_set_num=self.vocab_size,
            device_target="CPU",
            dataset_name="CE-CSL"
        )
        
        # 加载模型参数
        param_dict = load_checkpoint(self.model_path)
        load_param_into_net(self.model, param_dict)
        
        # 设置为评估模式
        self.model.set_train(False)
        
        print("✓ 模型初始化完成")
    
    def init_decoder(self):
        """初始化解码器"""
        self.decoder = CTCDecoder(
            gloss_dict=self.word2idx,
            num_classes=self.vocab_size,
            search_mode='max',
            blank_id=0
        )
        print("✓ 解码器初始化完成")
    
    def load_video(self, video_path, max_frames=50, crop_size=160):
        """加载并预处理视频"""
        print(f"正在加载视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        original_length = len(frames)
        
        if len(frames) == 0:
            raise ValueError(f"视频文件为空: {video_path}")
        
        print(f"✓ 视频加载完成，原始帧数: {original_length}")
        
        # 限制最大帧数
        if len(frames) > max_frames:
            # 均匀采样帧
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            processed_length = max_frames
        else:
            processed_length = len(frames)
        
        # 调整帧大小
        resized_frames = []
        for frame in frames:
            # 调整大小并保持长宽比
            h, w = frame.shape[:2]
            if h > w:
                new_h, new_w = crop_size, int(w * crop_size / h)
            else:
                new_h, new_w = int(h * crop_size / w), crop_size
            
            resized = cv2.resize(frame, (new_w, new_h))
            
            # 填充到正方形
            padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            y_offset = (crop_size - new_h) // 2
            x_offset = (crop_size - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            resized_frames.append(padded)
        
        print(f"✓ 视频预处理完成，处理后帧数: {processed_length}")
        
        # 转换为模型输入格式 (T, H, W, C) -> (1, T, C, H, W)
        frames_array = np.array(resized_frames)
        frames_array = frames_array.transpose(0, 3, 1, 2)  # (T, C, H, W)
        frames_array = frames_array.astype(np.float32) / 255.0  # 归一化到 [0, 1]
        frames_array = np.expand_dims(frames_array, axis=0)  # (1, T, C, H, W)
        
        return frames_array, processed_length
    
    def predict_video(self, video_path):
        """预测单个视频的手语序列"""
        try:
            # 加载视频
            frames, video_length = self.load_video(video_path)
            
            # 转换为MindSpore张量
            video_tensor = Tensor(frames, ms.float32)
            length_tensor = Tensor([video_length], ms.int32)
            
            print(f"输入张量形状: {video_tensor.shape}")
            print(f"视频长度: {video_length}")
            
            # 模型推理
            print("正在进行模型推理...")
            outputs = self.model(video_tensor, length_tensor, is_train=False)
            
            # 获取主要输出 - 使用输出1，因为输出0几乎全为零
            logits = outputs[1]  # (T, B, C) - 使用更有活力的输出
            pred_lengths = outputs[5]  # (B,)
            
            print(f"模型输出形状: {logits.shape}")
            print(f"预测长度: {pred_lengths}")
            
            # CTC解码
            print("正在进行CTC解码...")
            predictions = self.decoder.decode(
                nn_output=logits,
                vid_lgt=pred_lengths,
                batch_first=False
            )
            
            # 提取预测结果
            if predictions and len(predictions) > 0:
                pred_sequence = predictions[0]  # 第一个样本的预测
                
                # 解码器返回(word, position)的元组列表
                predicted_words = []
                for item in pred_sequence:
                    if isinstance(item, tuple) and len(item) >= 1:
                        word = item[0]  # 获取单词部分
                        if word and word.strip() and word != ' ':  # 跳过空白和PAD标记
                            predicted_words.append(word.strip())
                
                return predicted_words
            else:
                return []
                
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def load_ground_truth(self, label_file, video_name):
        """从标签文件加载真实标签"""
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 4 and row[0] == video_name:
                        # 解析gloss标签（第4列，索引3）
                        gloss_words = row[3].split("/")
                        # 预处理单词
                        gloss_words = preprocess_words(gloss_words)
                        return [word for word in gloss_words if word.strip()]
            return []
        except Exception as e:
            print(f"加载真实标签时出错: {str(e)}")
            return []
    
    def test_video_with_ground_truth(self, video_path, label_file=None):
        """测试视频并与真实标签对比"""
        video_name = Path(video_path).stem
        print(f"\n{'='*60}")
        print(f"测试视频: {video_name}")
        print(f"{'='*60}")
        
        # 预测
        predicted_words = self.predict_video(video_path)
        
        print(f"\n预测结果:")
        print(f"  预测序列: {' / '.join(predicted_words) if predicted_words else '(空)'}")
        print(f"  预测词数: {len(predicted_words)}")
        
        result = {
            'video_name': video_name,
            'predicted': predicted_words
        }
        
        # 如果有标签文件，加载真实标签进行对比
        if label_file and os.path.exists(label_file):
            ground_truth = self.load_ground_truth(label_file, video_name)
            
            print(f"\n真实标签:")
            print(f"  真实序列: {' / '.join(ground_truth) if ground_truth else '(空)'}")
            print(f"  真实词数: {len(ground_truth)}")
            
            # 计算简单的准确率指标
            if predicted_words and ground_truth:
                # 词级别精确匹配数量
                correct = sum(1 for p, g in zip(predicted_words, ground_truth) if p == g)
                precision = correct / len(predicted_words) if predicted_words else 0
                recall = correct / len(ground_truth) if ground_truth else 0
                
                print(f"\n评估指标:")
                print(f"  正确匹配: {correct}")
                print(f"  精确率: {precision:.4f}")
                print(f"  召回率: {recall:.4f}")
                
                result.update({
                    'ground_truth': ground_truth,
                    'correct_matches': correct,
                    'precision': precision,
                    'recall': recall
                })
        
        return result

def main():
    """主测试函数"""
    # 配置路径
    model_path = "/data/shengteng/training/models/best_model.ckpt"
    vocab_path = "/data/shengteng/training/output_gpu/vocabulary.json" 
    
    # 测试视频路径
    test_videos = [
        "/data/shengteng/training/data/CE-CSL/video/dev/A/dev-00005.mp4",
        "/data/shengteng/training/data/CE-CSL/video/dev/A/dev-00001.mp4",
        "/data/shengteng/training/data/CE-CSL/video/dev/A/dev-00010.mp4"
    ]
    
    # 标签文件
    label_file = "/data/shengteng/training/data/CE-CSL/label/dev.csv"
    
    print("开始测试TFNet模型...")
    print(f"模型路径: {model_path}")
    print(f"词汇表路径: {vocab_path}")
    
    try:
        # 初始化测试器
        tester = VideoTester(model_path, vocab_path)
        
        # 测试多个视频
        results = []
        for video_path in test_videos:
            if os.path.exists(video_path):
                result = tester.test_video_with_ground_truth(video_path, label_file)
                results.append(result)
            else:
                print(f"视频文件不存在: {video_path}")
        
        # 总结结果
        print(f"\n{'='*60}")
        print("测试总结")
        print(f"{'='*60}")
        
        total_videos = len(results)
        total_correct = sum(r.get('correct_matches', 0) for r in results)
        total_predicted = sum(len(r.get('predicted', [])) for r in results)
        total_ground_truth = sum(len(r.get('ground_truth', [])) for r in results)
        
        if total_predicted > 0:
            overall_precision = total_correct / total_predicted
        else:
            overall_precision = 0.0
            
        if total_ground_truth > 0:
            overall_recall = total_correct / total_ground_truth
        else:
            overall_recall = 0.0
        
        print(f"测试视频数量: {total_videos}")
        print(f"总预测词数: {total_predicted}")
        print(f"总真实词数: {total_ground_truth}")
        print(f"总正确匹配: {total_correct}")
        print(f"整体精确率: {overall_precision:.4f}")
        print(f"整体召回率: {overall_recall:.4f}")
        
        # 保存结果
        output_file = "/data/shengteng/training/video_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
