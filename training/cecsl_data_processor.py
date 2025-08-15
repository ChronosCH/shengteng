"""
CE-CSL数据集预处理模块
基于MindSpore和昇腾AI处理器优化的数据处理
从TFNet项目迁移并优化
"""

import os
import csv
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import cv2
import imageio
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAD = ' '

def seed_everything(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    ms.set_seed(seed)

class CECSLVideoProcessor:
    """CE-CSL视频数据预处理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
    
    def video_to_frames(self, video_path: str, output_dir: str, max_frames: int = None):
        """
        将视频转换为帧序列
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            max_frames: 最大帧数限制
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            vid = imageio.get_reader(video_path)
            nframes = vid.count_frames()
            fps = vid.get_meta_data()['fps']
            duration = vid.get_meta_data()['duration']
            resolution = vid.get_meta_data()['size']
            
            # 限制帧数
            if max_frames and nframes > max_frames:
                # 均匀采样
                indices = np.linspace(0, nframes-1, max_frames, dtype=int)
            else:
                indices = list(range(nframes))
            
            frames_info = {
                'fps': fps,
                'duration': duration,
                'original_resolution': resolution,
                'target_resolution': self.target_size,
                'total_frames': len(indices)
            }
            
            for i, frame_idx in enumerate(indices):
                try:
                    image = vid.get_data(frame_idx)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.target_size)
                    
                    # 保存帧
                    frame_name = f"{i:05d}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    cv2.imencode('.jpg', image)[1].tofile(frame_path)
                    
                except Exception as e:
                    logger.warning(f"处理第{frame_idx}帧失败: {e}")
                    continue
            
            vid.close()
            
            # 保存元数据
            meta_path = os.path.join(output_dir, 'metadata.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(frames_info, f, indent=2, ensure_ascii=False)
                
            return frames_info
            
        except Exception as e:
            logger.error(f"处理视频{video_path}失败: {e}")
            return None
    
    def batch_process_dataset(self, data_path: str, save_path: str, 
                            max_frames: int = 300):
        """
        批量处理CE-CSL数据集
        Args:
            data_path: 原始视频数据路径
            save_path: 处理后数据保存路径
            max_frames: 每个视频最大帧数
        """
        logger.info(f"开始批量处理数据集: {data_path} -> {save_path}")
        
        file_types = sorted(os.listdir(data_path))
        
        stats = {
            'total_videos': 0,
            'success_videos': 0,
            'failed_videos': 0,
            'frame_stats': [],
            'fps_stats': [],
            'duration_stats': []
        }
        
        for file_type in file_types:
            type_path = os.path.join(data_path, file_type)
            save_type_path = os.path.join(save_path, file_type)
            
            if not os.path.isdir(type_path):
                continue
                
            translators = sorted(os.listdir(type_path))
            
            for translator in translators:
                translator_path = os.path.join(type_path, translator)
                save_translator_path = os.path.join(save_type_path, translator)
                
                if not os.path.isdir(translator_path):
                    continue
                
                videos = sorted(os.listdir(translator_path))
                
                for video in tqdm(videos, desc=f"处理{file_type}/{translator}"):
                    video_path = os.path.join(translator_path, video)
                    
                    if not video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        continue
                    
                    # 输出目录
                    video_name = os.path.splitext(video)[0]
                    output_dir = os.path.join(save_translator_path, video_name)
                    
                    stats['total_videos'] += 1
                    
                    # 处理视频
                    frames_info = self.video_to_frames(
                        video_path, output_dir, max_frames
                    )
                    
                    if frames_info:
                        stats['success_videos'] += 1
                        stats['frame_stats'].append(frames_info['total_frames'])
                        stats['fps_stats'].append(frames_info['fps'])
                        stats['duration_stats'].append(frames_info['duration'])
                    else:
                        stats['failed_videos'] += 1
        
        # 统计信息
        if stats['frame_stats']:
            logger.info(f"处理完成统计:")
            logger.info(f"总视频数: {stats['total_videos']}")
            logger.info(f"成功处理: {stats['success_videos']}")
            logger.info(f"处理失败: {stats['failed_videos']}")
            logger.info(f"帧数统计 - 最大: {max(stats['frame_stats'])}, "
                       f"最小: {min(stats['frame_stats'])}, "
                       f"平均: {np.mean(stats['frame_stats']):.1f}")
            logger.info(f"FPS统计 - 最大: {max(stats['fps_stats'])}, "
                       f"最小: {min(stats['fps_stats'])}, "
                       f"平均: {np.mean(stats['fps_stats']):.1f}")
        
        # 保存统计信息
        stats_path = os.path.join(save_path, 'processing_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return stats

class CECSLLabelProcessor:
    """CE-CSL标签处理器"""
    
    def __init__(self):
        self.vocab = {}
        self.idx2word = [PAD]  # 0是填充符
        self.word2idx = {PAD: 0}
    
    def preprocess_words(self, words: List[str]) -> List[str]:
        """
        预处理词汇，处理特殊字符和格式化
        """
        processed_words = []
        
        for word in words:
            # 移除括号内容
            n = 0
            sub_flag = False
            word_list = list(word)
            
            for j in range(len(word)):
                if word[j] in "({[（":
                    sub_flag = True
                
                if sub_flag:
                    if j - n < len(word_list):
                        word_list.pop(j - n)
                        n += 1
                
                if word[j] in ")}]）":
                    sub_flag = False
            
            word = "".join(word_list)
            
            # 处理数字后缀
            if word and word[-1].isdigit() and not word[0].isdigit():
                word = word[:-1]
            
            # 标准化标点符号
            if word and word[0] in ",，":
                word = "，" + word[1:]
            
            if word and word[0] in "?？":
                word = "？" + word[1:]
            
            # 标准化数字
            if word.isdigit():
                word = str(int(word))
            
            if word:  # 只保留非空词
                processed_words.append(word)
        
        return processed_words
    
    def build_vocabulary(self, label_files: List[str], 
                        min_freq: int = 1) -> Dict[str, int]:
        """
        构建词汇表
        Args:
            label_files: 标签文件路径列表
            min_freq: 最小词频
        """
        word_counts = defaultdict(int)
        
        for label_file in label_files:
            with open(label_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n == 0:  # 跳过标题行
                        continue
                    
                    if len(row) >= 4:
                        words = row[3].split("/")
                        words = self.preprocess_words(words)
                        
                        for word in words:
                            word_counts[word] += 1
        
        # 按频次排序并构建词汇表
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words:
            if count >= min_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.idx2word)
                    self.idx2word.append(word)
        
        logger.info(f"构建词汇表完成，总词汇数: {len(self.idx2word)}")
        logger.info(f"前10个高频词: {self.idx2word[1:11]}")
        
        return self.word2idx
    
    def save_vocabulary(self, vocab_path: str):
        """保存词汇表"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': len(self.idx2word)
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"词汇表已保存到: {vocab_path}")
    
    def load_vocabulary(self, vocab_path: str):
        """加载词汇表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        
        logger.info(f"词汇表已加载，词汇数: {len(self.idx2word)}")
    
    def process_labels(self, label_file: str) -> List[Dict]:
        """
        处理标签文件
        Args:
            label_file: 标签文件路径
        Returns:
            处理后的标签数据
        """
        labels = []
        
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n == 0:  # 跳过标题行
                    continue
                
                if len(row) >= 4:
                    video_id = row[0]
                    translator = row[1] if len(row) > 1 else ""
                    sentence = row[2] if len(row) > 2 else ""
                    gloss_sequence = row[3]
                    
                    # 处理手语词汇序列
                    words = gloss_sequence.split("/")
                    words = self.preprocess_words(words)
                    
                    # 转换为索引
                    word_indices = []
                    for word in words:
                        if word in self.word2idx:
                            word_indices.append(self.word2idx[word])
                        else:
                            logger.warning(f"未知词汇: {word}")
                            # 可选择跳过或使用UNK标记
                    
                    if word_indices:  # 只保留有效标签
                        labels.append({
                            'video_id': video_id,
                            'translator': translator,
                            'sentence': sentence,
                            'gloss_sequence': gloss_sequence,
                            'words': words,
                            'word_indices': word_indices,
                            'length': len(word_indices)
                        })
        
        logger.info(f"处理标签文件 {label_file}，有效样本数: {len(labels)}")
        return labels

class CECSLDataset:
    """CE-CSL数据集类"""
    
    def __init__(self, data_dir: str, labels: List[Dict], 
                 transform=None, max_frames: int = 300):
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label_data = self.labels[idx]
        video_id = label_data['video_id']
        
        # 构建视频帧目录路径
        # 根据实际数据结构调整路径构建逻辑
        frame_dir = self._get_frame_directory(video_id)
        
        # 加载视频帧
        frames = self._load_video_frames(frame_dir)
        
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'video': frames,
            'label': label_data['word_indices'],
            'video_length': len(frames),
            'label_length': label_data['length'],
            'video_id': video_id
        }
    
    def _get_frame_directory(self, video_id: str) -> str:
        """根据视频ID获取帧目录路径"""
        # 这里需要根据实际的数据结构来实现
        # 例如: data_dir/train/translator1/video_001/
        parts = video_id.split('_')
        if len(parts) >= 2:
            split_type = parts[0]  # train/dev/test
            translator = parts[1]
            video_name = '_'.join(parts[2:])
            return os.path.join(self.data_dir, split_type, translator, video_name)
        else:
            return os.path.join(self.data_dir, video_id)
    
    def _load_video_frames(self, frame_dir: str) -> np.ndarray:
        """加载视频帧"""
        if not os.path.exists(frame_dir):
            logger.warning(f"帧目录不存在: {frame_dir}")
            return np.zeros((1, 224, 224, 3), dtype=np.uint8)
        
        frame_files = sorted([f for f in os.listdir(frame_dir) 
                            if f.endswith('.jpg')])
        
        if not frame_files:
            logger.warning(f"帧目录为空: {frame_dir}")
            return np.zeros((1, 224, 224, 3), dtype=np.uint8)
        
        frames = []
        for frame_file in frame_files[:self.max_frames]:
            frame_path = os.path.join(frame_dir, frame_file)
            try:
                image = cv2.imread(frame_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frames.append(image)
            except Exception as e:
                logger.warning(f"加载帧失败 {frame_path}: {e}")
        
        if frames:
            return np.array(frames)
        else:
            return np.zeros((1, 224, 224, 3), dtype=np.uint8)

def create_cecsl_dataloaders(data_config: Dict, vocab_file: str) -> Tuple:
    """
    创建CE-CSL数据加载器
    Args:
        data_config: 数据配置
        vocab_file: 词汇表文件路径
    Returns:
        (train_loader, val_loader, test_loader, vocab_size)
    """
    # 初始化标签处理器
    label_processor = CECSLLabelProcessor()
    
    # 构建或加载词汇表
    if os.path.exists(vocab_file):
        label_processor.load_vocabulary(vocab_file)
    else:
        # 构建词汇表
        label_files = [
            data_config['train_label_path'],
            data_config['val_label_path'],
            data_config['test_label_path']
        ]
        label_processor.build_vocabulary(label_files)
        label_processor.save_vocabulary(vocab_file)
    
    # 处理标签
    train_labels = label_processor.process_labels(data_config['train_label_path'])
    val_labels = label_processor.process_labels(data_config['val_label_path'])
    test_labels = label_processor.process_labels(data_config['test_label_path'])
    
    # 数据变换
    train_transform = vision.Compose([
        vision.Resize((224, 224)),
        vision.RandomHorizontalFlip(0.5),
        vision.ToTensor(),
        vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = vision.Compose([
        vision.Resize((224, 224)),
        vision.ToTensor(),
        vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CECSLDataset(
        data_config['train_data_path'],
        train_labels,
        transform=train_transform,
        max_frames=data_config.get('max_frames', 300)
    )
    
    val_dataset = CECSLDataset(
        data_config['val_data_path'],
        val_labels,
        transform=test_transform,
        max_frames=data_config.get('max_frames', 300)
    )
    
    test_dataset = CECSLDataset(
        data_config['test_data_path'],
        test_labels,
        transform=test_transform,
        max_frames=data_config.get('max_frames', 300)
    )
    
    # 创建数据加载器
    train_loader = ds.GeneratorDataset(
        train_dataset,
        ['video', 'label', 'video_length', 'label_length', 'video_id'],
        shuffle=True
    ).batch(data_config['batch_size'], drop_remainder=True)
    
    val_loader = ds.GeneratorDataset(
        val_dataset,
        ['video', 'label', 'video_length', 'label_length', 'video_id'],
        shuffle=False
    ).batch(1, drop_remainder=False)
    
    test_loader = ds.GeneratorDataset(
        test_dataset,
        ['video', 'label', 'video_length', 'label_length', 'video_id'],
        shuffle=False
    ).batch(1, drop_remainder=False)
    
    vocab_size = len(label_processor.idx2word)
    
    logger.info(f"数据加载器创建完成:")
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"测试集: {len(test_dataset)} 样本")
    logger.info(f"词汇表大小: {vocab_size}")
    
    return train_loader, val_loader, test_loader, vocab_size

def collate_fn(batch):
    """
    批处理函数，处理变长序列
    """
    videos = [item['video'] for item in batch]
    labels = [item['label'] for item in batch]
    video_lengths = [item['video_length'] for item in batch]
    label_lengths = [item['label_length'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    
    # 填充视频序列
    max_video_len = max(video_lengths)
    padded_videos = []
    
    for video in videos:
        if len(video) < max_video_len:
            pad_size = max_video_len - len(video)
            padding = np.zeros((pad_size,) + video.shape[1:], dtype=video.dtype)
            video = np.concatenate([video, padding], axis=0)
        padded_videos.append(video)
    
    # 填充标签序列
    max_label_len = max(label_lengths)
    padded_labels = []
    
    for label in labels:
        if len(label) < max_label_len:
            pad_size = max_label_len - len(label)
            label = label + [0] * pad_size  # 0是PAD标记
        padded_labels.append(label)
    
    return {
        'video': np.array(padded_videos),
        'label': np.array(padded_labels),
        'video_length': np.array(video_lengths),
        'label_length': np.array(label_lengths),
        'video_id': video_ids
    }

if __name__ == "__main__":
    # 测试数据预处理
    seed_everything(42)
    
    # 视频预处理示例
    video_processor = CECSLVideoProcessor()
    
    # 批量处理视频数据
    # video_processor.batch_process_dataset(
    #     data_path="/path/to/CE-CSL/video",
    #     save_path="/path/to/CE-CSL/processed",
    #     max_frames=300
    # )
    
    # 标签处理示例
    label_processor = CECSLLabelProcessor()
    
    # 构建词汇表
    # label_files = [
    #     "/path/to/train.corpus.csv",
    #     "/path/to/dev.corpus.csv", 
    #     "/path/to/test.corpus.csv"
    # ]
    # label_processor.build_vocabulary(label_files)
    # label_processor.save_vocabulary("./vocab.json")
    
    logger.info("数据预处理模块测试完成")
