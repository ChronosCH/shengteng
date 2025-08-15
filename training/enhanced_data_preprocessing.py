"""
增强的手语数据预处理模块
支持CE-CSL数据集和TFNet模型的数据处理需求
结合MindSpore框架和华为昇腾优化
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# MindSpore相关导入
try:
    import mindspore as ms
    from mindspore import Tensor, ops, nn
    from mindspore.dataset import vision
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("警告: MindSpore未安装，部分功能不可用")

# MediaPipe导入（可选）
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("警告: MediaPipe未安装，关键点提取功能不可用")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """数据预处理配置"""
    # 视频处理参数
    target_fps: int = 25
    max_sequence_length: int = 300
    min_sequence_length: int = 10
    image_size: Tuple[int, int] = (224, 224)
    
    # 数据增强参数
    enable_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    random_crop_prob: float = 0.3
    
    # 关键点提取参数
    enable_keypoints: bool = True
    mediapipe_confidence: float = 0.5
    
    # 处理参数
    num_workers: int = 4
    batch_size: int = 32
    cache_dir: str = "./cache"
    
    # 质量控制参数 - 调整为更宽松的阈值
    min_quality_score: float = 0.1  # 降低最小质量分数要求
    blur_threshold: float = 50.0    # 降低模糊度阈值，手语视频允许一定模糊
    brightness_min: float = 20      # 最小亮度值
    brightness_max: float = 235     # 最大亮度值
    contrast_threshold: float = 0.15 # 降低对比度要求
    quality_frame_ratio: float = 0.3 # 降低好帧比例要求到30%
    enable_quality_check: bool = True # 是否启用质量检查
    skip_bad_quality: bool = False   # 是否跳过质量不佳的视频（False表示仅警告）

@dataclass 
class VideoSample:
    """视频样本数据结构"""
    video_path: str
    gloss_sequence: List[str]
    text: str = ""
    video_id: str = ""
    translator: str = ""
    start_frame: int = 0
    end_frame: Optional[int] = None
    fps: float = 25.0

@dataclass
class ProcessedSample:
    """处理后的样本数据结构"""
    video_id: str
    gloss_sequence: List[str]
    text: str
    frames: np.ndarray
    keypoints: Optional[np.ndarray] = None
    face_landmarks: Optional[np.ndarray] = None
    pose_landmarks: Optional[np.ndarray] = None
    left_hand_landmarks: Optional[np.ndarray] = None
    right_hand_landmarks: Optional[np.ndarray] = None
    duration: float = 0.0
    fps: float = 25.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = asdict(self)
        # 将numpy数组转换为列表（用于JSON序列化）
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.shape  # 只保存形状信息
        return result

class EnhancedSignLanguagePreprocessor:
    """增强的手语数据预处理器"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'avg_duration': 0.0,
            'quality_scores': []
        }
        
        # 初始化MediaPipe（如果可用）
        if MEDIAPIPE_AVAILABLE and config.enable_keypoints:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=config.mediapipe_confidence,
                min_tracking_confidence=config.mediapipe_confidence
            )
        else:
            self.holistic = None
            
        # 创建缓存目录
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def check_video_quality(self, frame: np.ndarray) -> Dict:
        """检查视频帧质量 - 优化后的质量检查算法"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 计算模糊度（拉普拉斯方差）
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 计算亮度统计
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            
            # 计算对比度
            contrast = brightness_std / brightness_mean if brightness_mean > 0 else 0
            
            # 更宽松的质量评估
            # 1. 模糊度评估 - 使用更低的阈值
            blur_ok = blur_score > self.config.blur_threshold
            
            # 2. 亮度评估 - 使用更宽松的范围
            brightness_ok = (brightness_mean > self.config.brightness_min and 
                           brightness_mean < self.config.brightness_max)
            
            # 3. 对比度评估 - 使用更低的阈值
            contrast_ok = contrast > self.config.contrast_threshold
            
            # 综合质量评分 - 使用加权平均
            # 给模糊度更低的权重，因为手语视频允许一定程度的模糊
            quality_score = (
                0.3 * min(1.0, blur_score / self.config.blur_threshold) +
                0.3 * min(1.0, contrast / 0.5) +  # 对比度权重
                0.4 * min(1.0, max(0, 1.0 - abs(brightness_mean - 128) / 128))  # 亮度权重
            )
            
            # 更宽松的质量判断 - 满足任意两个条件即可
            conditions_met = sum([blur_ok, brightness_ok, contrast_ok])
            is_good_quality = (
                conditions_met >= 2 or  # 满足任意两个条件
                quality_score > self.config.min_quality_score  # 或者综合分数达标
            )
            
            return {
                'blur_score': blur_score,
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'contrast': contrast,
                'quality_score': quality_score,
                'is_good_quality': is_good_quality,
                'blur_ok': blur_ok,
                'brightness_ok': brightness_ok,
                'contrast_ok': contrast_ok,
                'conditions_met': conditions_met
            }
            
        except Exception as e:
            logger.warning(f"质量检查失败: {e}")
            return {
                'blur_score': 0.0,
                'brightness_mean': 0.0,
                'brightness_std': 0.0,
                'contrast': 0.0,
                'quality_score': 0.0,
                'is_good_quality': False,
                'blur_ok': False,
                'brightness_ok': False,
                'contrast_ok': False,
                'conditions_met': 0
            }
    
    def extract_frames_from_video(self, video_path: str, 
                                 start_frame: int = 0, 
                                 end_frame: Optional[int] = None) -> Optional[np.ndarray]:
        """从视频中提取帧序列"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return None
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            if end_frame is None:
                end_frame = total_frames
            
            # 计算采样间隔
            frame_interval = max(1, int(fps / self.config.target_fps))
            
            # 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            good_frames = 0
            total_quality_score = 0
            
            for frame_idx in range(start_frame, min(end_frame, total_frames), frame_interval):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检查帧质量
                quality_info = self.check_video_quality(frame_rgb)
                total_quality_score += quality_info['quality_score']
                
                # 调整大小
                frame_resized = cv2.resize(frame_rgb, self.config.image_size)
                
                frames.append(frame_resized)
                
                if quality_info['is_good_quality']:
                    good_frames += 1
                
                # 限制最大帧数
                if len(frames) >= self.config.max_sequence_length:
                    break
            
            cap.release()
            
            if not frames:
                logger.warning(f"视频无有效帧: {video_path}")
                return None
            
            # 质量检查
            avg_quality = total_quality_score / len(frames)
            quality_ratio = good_frames / len(frames)
            
            if len(frames) < self.config.min_sequence_length:
                logger.warning(f"视频帧数不足: {video_path} ({len(frames)} < {self.config.min_sequence_length})")
                return None
            
            # 更宽松的质量检查和详细的日志输出
            if self.config.enable_quality_check and quality_ratio < self.config.quality_frame_ratio:
                logger.info(f"视频质量统计: {video_path}")
                logger.info(f"  总帧数: {len(frames)}, 良好帧数: {good_frames}")
                logger.info(f"  质量比例: {quality_ratio:.2f}, 平均质量分: {avg_quality:.2f}")
                logger.info(f"  阈值设置: 质量比例>={self.config.quality_frame_ratio}, 模糊度>={self.config.blur_threshold}")
                
                if self.config.skip_bad_quality:
                    logger.warning(f"跳过质量不佳的视频: {video_path} (质量比例: {quality_ratio:.2f})")
                    return None
                else:
                    logger.warning(f"视频质量提醒: {video_path} (质量比例: {quality_ratio:.2f}) - 继续使用")
            else:
                logger.debug(f"视频质量良好: {video_path} (质量比例: {quality_ratio:.2f}, 平均分: {avg_quality:.2f})")
            
            frames_array = np.array(frames)
            logger.debug(f"成功提取帧: {video_path} - {frames_array.shape}")
            
            return frames_array
            
        except Exception as e:
            logger.error(f"视频帧提取失败 {video_path}: {e}")
            return None
    
    def extract_keypoints_from_frames(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """从帧序列中提取关键点"""
        if not self.config.enable_keypoints or self.holistic is None:
            return {}
        
        try:
            all_keypoints = []
            face_landmarks_list = []
            pose_landmarks_list = []
            left_hand_landmarks_list = []
            right_hand_landmarks_list = []
            
            for frame in frames:
                # MediaPipe处理
                results = self.holistic.process(frame)
                
                # 提取关键点
                keypoints = self._extract_landmarks(results)
                all_keypoints.append(keypoints)
                
                # 分别提取各部分关键点
                face_landmarks_list.append(self._extract_face_landmarks(results))
                pose_landmarks_list.append(self._extract_pose_landmarks(results))
                left_hand_landmarks_list.append(self._extract_hand_landmarks(results.left_hand_landmarks))
                right_hand_landmarks_list.append(self._extract_hand_landmarks(results.right_hand_landmarks))
            
            return {
                'keypoints': np.array(all_keypoints),
                'face_landmarks': np.array(face_landmarks_list),
                'pose_landmarks': np.array(pose_landmarks_list),
                'left_hand_landmarks': np.array(left_hand_landmarks_list),
                'right_hand_landmarks': np.array(right_hand_landmarks_list)
            }
            
        except Exception as e:
            logger.error(f"关键点提取失败: {e}")
            return {}
    
    def _extract_landmarks(self, results) -> np.ndarray:
        """提取所有关键点"""
        landmarks = np.zeros((543, 3))  # 468面部 + 33姿态 + 21左手 + 21右手
        
        offset = 0
        
        # 面部关键点
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                landmarks[offset + i] = [landmark.x, landmark.y, landmark.z]
        offset += 468
        
        # 姿态关键点
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[offset + i] = [landmark.x, landmark.y, landmark.z]
        offset += 33
        
        # 左手关键点
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                landmarks[offset + i] = [landmark.x, landmark.y, landmark.z]
        offset += 21
        
        # 右手关键点
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                landmarks[offset + i] = [landmark.x, landmark.y, landmark.z]
        
        return landmarks
    
    def _extract_face_landmarks(self, results) -> np.ndarray:
        """提取面部关键点"""
        landmarks = np.zeros((468, 3))
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y, landmark.z]
        return landmarks
    
    def _extract_pose_landmarks(self, results) -> np.ndarray:
        """提取姿态关键点"""
        landmarks = np.zeros((33, 3))
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y, landmark.z]
        return landmarks
    
    def _extract_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """提取手部关键点"""
        landmarks = np.zeros((21, 3))
        if hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y, landmark.z]
        return landmarks
    
    def apply_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        if not self.config.enable_augmentation:
            return frames
        
        try:
            # 随机水平翻转
            if np.random.random() < self.config.horizontal_flip_prob:
                frames = np.flip(frames, axis=2)  # 沿宽度轴翻转
            
            # 随机亮度调整
            if np.random.random() < 0.5:
                brightness_factor = 1.0 + np.random.uniform(
                    -self.config.brightness_factor, 
                    self.config.brightness_factor
                )
                frames = np.clip(frames * brightness_factor, 0, 255)
            
            # 随机对比度调整
            if np.random.random() < 0.5:
                contrast_factor = 1.0 + np.random.uniform(
                    -self.config.contrast_factor,
                    self.config.contrast_factor
                )
                frames = np.clip(127 + contrast_factor * (frames - 127), 0, 255)
            
            # 随机裁剪和调整大小
            if np.random.random() < self.config.random_crop_prob:
                frames = self._random_crop_and_resize(frames)
            
            return frames.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"数据增强失败: {e}")
            return frames
    
    def _random_crop_and_resize(self, frames: np.ndarray) -> np.ndarray:
        """随机裁剪和调整大小"""
        t, h, w, c = frames.shape
        
        # 随机裁剪比例 (0.8-1.0)
        crop_ratio = np.random.uniform(0.8, 1.0)
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        # 随机起始位置
        start_h = np.random.randint(0, h - new_h + 1)
        start_w = np.random.randint(0, w - new_w + 1)
        
        # 裁剪
        cropped_frames = frames[:, start_h:start_h+new_h, start_w:start_w+new_w, :]
        
        # 调整回原始大小
        resized_frames = []
        for frame in cropped_frames:
            resized_frame = cv2.resize(frame, (w, h))
            resized_frames.append(resized_frame)
        
        return np.array(resized_frames)
    
    def process_single_video(self, video_sample: VideoSample) -> Optional[ProcessedSample]:
        """处理单个视频"""
        try:
            # 提取帧
            frames = self.extract_frames_from_video(
                video_sample.video_path,
                video_sample.start_frame,
                video_sample.end_frame
            )
            
            if frames is None:
                return None
            
            # 应用数据增强（仅在训练时）
            if self.config.enable_augmentation:
                frames = self.apply_augmentation(frames)
            
            # 提取关键点
            keypoint_data = self.extract_keypoints_from_frames(frames)
            
            # 计算视频统计信息
            duration = len(frames) / self.config.target_fps
            
            # 创建处理后的样本
            processed_sample = ProcessedSample(
                video_id=video_sample.video_id,
                gloss_sequence=video_sample.gloss_sequence,
                text=video_sample.text,
                frames=frames,
                keypoints=keypoint_data.get('keypoints'),
                face_landmarks=keypoint_data.get('face_landmarks'),
                pose_landmarks=keypoint_data.get('pose_landmarks'),
                left_hand_landmarks=keypoint_data.get('left_hand_landmarks'),
                right_hand_landmarks=keypoint_data.get('right_hand_landmarks'),
                duration=duration,
                fps=self.config.target_fps
            )
            
            # 更新统计信息
            self.stats['processed_videos'] += 1
            self.stats['total_frames'] += len(frames)
            self.stats['avg_duration'] = (
                (self.stats['avg_duration'] * (self.stats['processed_videos'] - 1) + duration) /
                self.stats['processed_videos']
            )
            
            logger.debug(f"处理完成: {video_sample.video_id} - {frames.shape}")
            return processed_sample
            
        except Exception as e:
            logger.error(f"视频处理失败 {video_sample.video_path}: {e}")
            self.stats['failed_videos'] += 1
            return None
    
    def process_video_batch(self, video_samples: List[VideoSample], 
                           num_workers: Optional[int] = None) -> List[ProcessedSample]:
        """批量处理视频"""
        if num_workers is None:
            num_workers = self.config.num_workers
        
        processed_samples = []
        
        if num_workers <= 1:
            # 单线程处理
            for sample in video_samples:
                result = self.process_single_video(sample)
                if result:
                    processed_samples.append(result)
        else:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_sample = {
                    executor.submit(self.process_single_video, sample): sample
                    for sample in video_samples
                }
                
                for future in as_completed(future_to_sample):
                    try:
                        result = future.result()
                        if result:
                            processed_samples.append(result)
                    except Exception as e:
                        sample = future_to_sample[future]
                        logger.error(f"批量处理失败 {sample.video_id}: {e}")
                        self.stats['failed_videos'] += 1
        
        logger.info(f"批量处理完成: {len(processed_samples)}/{len(video_samples)} 个视频成功")
        return processed_samples
    
    def save_processed_data(self, samples: List[ProcessedSample], 
                           output_dir: str, split: str = "train"):
        """保存处理后的数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存元数据
        metadata = []
        for i, sample in enumerate(samples):
            # 保存视频帧
            frames_path = output_dir / f"{sample.video_id}_frames.npy"
            np.save(frames_path, sample.frames)
            
            # 保存关键点（如果有）
            keypoints_path = None
            if sample.keypoints is not None:
                keypoints_path = output_dir / f"{sample.video_id}_keypoints.npy"
                np.save(keypoints_path, sample.keypoints)
            
            # 添加到元数据
            sample_meta = sample.to_dict()
            sample_meta['frames_path'] = str(frames_path)
            sample_meta['keypoints_path'] = str(keypoints_path) if keypoints_path else None
            metadata.append(sample_meta)
        
        # 保存元数据文件
        metadata_path = output_dir / f"{split}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats_path = output_dir / f"{split}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"处理后数据已保存: {output_dir}")
        logger.info(f"元数据文件: {metadata_path}")
        logger.info(f"统计信息: {stats_path}")
    
    def get_statistics(self) -> Dict:
        """获取处理统计信息"""
        return self.stats.copy()

class CECSLDatasetProcessor:
    """CE-CSL数据集专用处理器"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessor = EnhancedSignLanguagePreprocessor(config)
        self.video_mapping = {}  # 存储video_id到实际文件名的映射
        self._load_video_mapping()
    
    def _load_video_mapping(self):
        """加载视频ID到实际文件名的映射"""
        try:
            data_root = getattr(self.config, 'data_root', "./data/CE-CSL")
            
            # 加载train, dev, test的映射
            for split in ['train', 'dev', 'test']:
                label_detail_file = os.path.join(data_root, 'label', f'{split}.csv')
                if os.path.exists(label_detail_file):
                    with open(label_detail_file, 'r', encoding='utf-8') as f:
                        import csv
                        reader = csv.reader(f)
                        next(reader)  # 跳过标题行
                        
                        for row in reader:
                            if len(row) >= 2:
                                actual_video_name = row[0]  # train-00001
                                translator = row[1]  # A, B, C等
                                
                                # 为每个video_id创建映射
                                # 从train-00001转换为train_video_000格式
                                parts = actual_video_name.split('-')
                                if len(parts) == 2:
                                    prefix = parts[0]  # train
                                    number = int(parts[1]) - 1  # 00001 -> 0
                                    video_id = f"{prefix}_video_{number:03d}"  # train_video_000
                                    
                                    self.video_mapping[video_id] = {
                                        'actual_name': actual_video_name,
                                        'translator': translator,
                                        'split': split
                                    }
                                    
                    logger.info(f"加载 {split} 视频映射: {len([k for k in self.video_mapping.keys() if k.startswith(split)])} 个")
                            
        except Exception as e:
            logger.warning(f"加载视频映射失败: {e}")
    
    def load_cecsl_labels(self, label_file: str) -> List[VideoSample]:
        """加载CE-CSL标签文件"""
        samples = []
        
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                import csv
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:  # 跳过标题行
                        continue
                    
                    if len(row) >= 4:
                        video_id = row[0]
                        start_frame = int(row[1]) if row[1].isdigit() else 0
                        end_frame = int(row[2]) if row[2].isdigit() else 30
                        label = row[3]
                        
                        # 获取视频映射信息
                        if video_id in self.video_mapping:
                            mapping_info = self.video_mapping[video_id]
                            actual_name = mapping_info['actual_name']
                            translator = mapping_info['translator']
                            split = mapping_info['split']
                            
                            # 构建视频路径
                            video_path = self._construct_video_path_with_mapping(
                                actual_name, translator, split
                            )
                            
                            text = label  # 使用label作为文本
                            gloss_sequence = [label]  # 将label作为gloss序列
                            
                            if os.path.exists(video_path):
                                sample = VideoSample(
                                    video_path=video_path,
                                    gloss_sequence=gloss_sequence,
                                    text=text,
                                    video_id=video_id,
                                    translator=translator
                                )
                                samples.append(sample)
                            else:
                                logger.warning(f"视频文件不存在: {video_path}")
                        else:
                            logger.warning(f"未找到视频映射: {video_id}")
                    elif len(row) == 2:  # 处理只有video_id和label的简化格式
                        video_id = row[0]
                        label = row[1]
                        
                        # 获取视频映射信息
                        if video_id in self.video_mapping:
                            mapping_info = self.video_mapping[video_id]
                            actual_name = mapping_info['actual_name']
                            translator = mapping_info['translator']
                            split = mapping_info['split']
                            
                            # 构建视频路径
                            video_path = self._construct_video_path_with_mapping(
                                actual_name, translator, split
                            )
                            
                            text = label
                            gloss_sequence = [label]
                            
                            if os.path.exists(video_path):
                                sample = VideoSample(
                                    video_path=video_path,
                                    gloss_sequence=gloss_sequence,
                                    text=text,
                                    video_id=video_id,
                                    translator=translator
                                )
                                samples.append(sample)
                            else:
                                logger.warning(f"视频文件不存在: {video_path}")
                        else:
                            logger.warning(f"未找到视频映射: {video_id}")
            
            logger.info(f"加载标签文件: {label_file} - {len(samples)} 个样本")
            return samples
            
        except Exception as e:
            logger.error(f"标签文件加载失败 {label_file}: {e}")
            return []
    
    def _construct_video_path_with_mapping(self, actual_name: str, translator: str, split: str) -> str:
        """使用映射信息构建视频文件路径"""
        data_root = getattr(self.config, 'data_root', "./data/CE-CSL")
        
        # 构建路径: data_root/video/split/translator/actual_name.mp4
        video_path = os.path.join(
            data_root,
            "video",
            split,
            translator,
            f"{actual_name}.mp4"
        )
        
        return video_path
    
    def _construct_video_path(self, video_id: str, translator: str) -> str:
        """构建视频文件路径（保留原方法用于兼容性）"""
        # 首先尝试使用映射
        if video_id in self.video_mapping:
            mapping_info = self.video_mapping[video_id]
            return self._construct_video_path_with_mapping(
                mapping_info['actual_name'],
                mapping_info['translator'],
                mapping_info['split']
            )
        
        # 原始逻辑（作为后备）
        # 根据CE-CSL数据集的实际结构调整
        # 实际结构为: data_root/video/split/video_file.mp4
        parts = video_id.split('_')
        if len(parts) >= 2:
            split = parts[0]  # train/dev/test
            video_file = f"{video_id}.mp4"  # 视频文件名
            
            # 首先尝试直接在split目录下查找
            direct_path = os.path.join(
                self.config.data_root if hasattr(self.config, 'data_root') else "./data/CE-CSL",
                "video", split, video_file
            )
            
            if os.path.exists(direct_path):
                return direct_path
            
            # 如果直接路径不存在，尝试在translator子目录中查找
            if translator:
                translator_path = os.path.join(
                    self.config.data_root if hasattr(self.config, 'data_root') else "./data/CE-CSL",
                    "video", split, translator, video_file
                )
                if os.path.exists(translator_path):
                    return translator_path
            
            # 如果以上都不存在，尝试在所有子目录中查找
            video_dir = os.path.join(
                self.config.data_root if hasattr(self.config, 'data_root') else "./data/CE-CSL",
                "video", split
            )
            
            if os.path.exists(video_dir):
                for root, dirs, files in os.walk(video_dir):
                    if video_file in files:
                        return os.path.join(root, video_file)
        
        return ""
    
    def process_cecsl_split(self, label_file: str, output_dir: str, split: str):
        """处理CE-CSL数据集的一个分割"""
        logger.info(f"开始处理CE-CSL {split} 数据集...")
        
        # 加载样本
        samples = self.load_cecsl_labels(label_file)
        if not samples:
            logger.error(f"未找到有效样本: {label_file}")
            return
        
        # 批量处理
        processed_samples = self.preprocessor.process_video_batch(samples)
        
        if processed_samples:
            # 保存结果
            self.preprocessor.save_processed_data(processed_samples, output_dir, split)
            
            # 打印统计信息
            stats = self.preprocessor.get_statistics()
            logger.info(f"{split} 数据集处理完成:")
            logger.info(f"  成功处理: {stats['processed_videos']} 个视频")
            logger.info(f"  失败: {stats['failed_videos']} 个视频")
            logger.info(f"  总帧数: {stats['total_frames']}")
            logger.info(f"  平均时长: {stats['avg_duration']:.2f} 秒")
        else:
            logger.error(f"{split} 数据集处理失败")

class MindSporeDatasetGenerator:
    """MindSpore数据集生成器"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def create_mindspore_dataset(self, metadata_file: str, 
                                batch_size: int = 32,
                                shuffle: bool = True,
                                num_workers: int = 4):
        """创建MindSpore数据集"""
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore未安装")
        
        import mindspore.dataset as ds
        
        # 加载元数据
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 创建数据生成器
        def data_generator():
            for sample_meta in metadata:
                # 加载帧数据
                frames = np.load(sample_meta['frames_path'])
                
                # 加载关键点（如果有）
                keypoints = None
                if sample_meta.get('keypoints_path'):
                    keypoints = np.load(sample_meta['keypoints_path'])
                
                # 创建标签
                gloss_sequence = sample_meta['gloss_sequence']
                
                yield frames, gloss_sequence, keypoints
        
        # 创建数据集
        dataset = ds.GeneratorDataset(
            data_generator,
            column_names=['frames', 'gloss_sequence', 'keypoints'],
            shuffle=shuffle,
            num_parallel_workers=num_workers
        )
        
        # 批处理
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        return dataset

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CE-CSL数据集预处理')
    parser.add_argument('--data_root', type=str, required=True,
                       help='CE-CSL数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='处理后数据输出目录')
    parser.add_argument('--target_fps', type=int, default=25,
                       help='目标帧率')
    parser.add_argument('--max_length', type=int, default=300,
                       help='最大序列长度')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='图像大小')
    parser.add_argument('--enable_keypoints', action='store_true',
                       help='是否提取关键点')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行处理线程数')
    parser.add_argument('--splits', type=str, nargs='+', 
                       default=['train', 'dev', 'test'],
                       help='要处理的数据分割')
    
    args = parser.parse_args()
    
    # 创建配置
    config = PreprocessingConfig(
        target_fps=args.target_fps,
        max_sequence_length=args.max_length,
        image_size=tuple(args.image_size),
        enable_keypoints=args.enable_keypoints,
        num_workers=args.num_workers
    )
    config.data_root = args.data_root
    
    # 创建处理器
    processor = CECSLDatasetProcessor(config)
    
    # 处理各个分割
    for split in args.splits:
        label_file = os.path.join(args.data_root, f"{split}.corpus.csv")
        split_output_dir = os.path.join(args.output_dir, split)
        
        if os.path.exists(label_file):
            processor.process_cecsl_split(label_file, split_output_dir, split)
        else:
            logger.warning(f"标签文件不存在: {label_file}")
    
    logger.info("CE-CSL数据集预处理完成!")

if __name__ == "__main__":
    main()
