"""
增强的手语数据预处理管道
专为CE-CSL数据集优化，支持多模态特征提取
基于MindSpore和华为昇腾AI处理器
"""

import os
import cv2
import json
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from tqdm import tqdm

# MindSpore相关导入
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision, transforms
from mindspore import Tensor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoSample:
    """视频样本数据类"""
    video_path: str
    gloss_sequence: List[str]
    text: str
    video_id: str
    translator: str = ""
    start_frame: int = 0
    end_frame: Optional[int] = None

@dataclass
class ProcessedSample:
    """处理后的样本数据类"""
    video_id: str
    gloss_sequence: List[str]
    text: str
    frames: np.ndarray  # (T, H, W, C) 视频帧
    keypoints: Optional[np.ndarray] = None  # (T, 543, 3) 所有关键点
    face_landmarks: Optional[np.ndarray] = None  # (T, 468, 3)
    pose_landmarks: Optional[np.ndarray] = None  # (T, 33, 3)
    left_hand_landmarks: Optional[np.ndarray] = None  # (T, 21, 3)
    right_hand_landmarks: Optional[np.ndarray] = None  # (T, 21, 3)
    duration: float = 0.0
    fps: int = 25
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'video_id': self.video_id,
            'gloss_sequence': self.gloss_sequence,
            'text': self.text,
            'frames_shape': self.frames.shape if self.frames is not None else None,
            'duration': self.duration,
            'fps': self.fps
        }

@dataclass
class PreprocessingConfig:
    """预处理配置"""
    target_fps: int = 25
    min_sequence_length: int = 10
    max_sequence_length: int = 300
    image_size: Tuple[int, int] = (224, 224)
    enable_keypoints: bool = False  # 是否提取关键点
    enable_augmentation: bool = True
    num_workers: int = 4
    batch_size: int = 32
    
    # 数据增强参数
    random_crop_prob: float = 0.5
    horizontal_flip_prob: float = 0.5
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    
    # 质量控制
    min_confidence: float = 0.5
    blur_threshold: float = 100.0  # Laplacian方差阈值

class EnhancedSignLanguagePreprocessor:
    """增强的手语数据预处理器"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # 初始化MediaPipe（如果需要关键点）
        if config.enable_keypoints:
            self._init_mediapipe()
        
        # 统计信息
        self.stats = {
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'avg_duration': 0.0,
            'avg_fps': 0.0
        }
        
        logger.info("增强手语数据预处理器初始化完成")
    
    def _init_mediapipe(self):
        """初始化MediaPipe"""
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=self.config.min_confidence,
            min_tracking_confidence=self.config.min_confidence
        )
        logger.info("MediaPipe Holistic 初始化完成")
    
    def check_video_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """检查视频帧质量"""
        # 计算图像清晰度（Laplacian方差）
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 计算亮度
        brightness = np.mean(gray)
        
        # 计算对比度
        contrast = np.std(gray)
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'is_good_quality': blur_score > self.config.blur_threshold
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
                total_quality_score += quality_info['blur_score']
                
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
            
            if quality_ratio < 0.5:  # 至少50%的帧质量良好
                logger.warning(f"视频质量不佳: {video_path} (质量比例: {quality_ratio:.2f})")
            
            frames_array = np.array(frames)
            logger.debug(f"成功提取帧: {video_path} - {frames_array.shape}")
            
            return frames_array
            
        except Exception as e:
            logger.error(f"视频帧提取失败 {video_path}: {e}")
            return None
    
    def extract_keypoints_from_frames(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """从帧序列中提取关键点"""
        if not self.config.enable_keypoints:
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
                        translator = row[1] if len(row) > 1 else ""
                        text = row[2] if len(row) > 2 else ""
                        gloss_sequence = row[3].split("/")
                        
                        # 构建视频路径（需要根据实际数据结构调整）
                        video_path = self._construct_video_path(video_id, translator)
                        
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
            
            logger.info(f"加载标签文件: {label_file} - {len(samples)} 个样本")
            return samples
            
        except Exception as e:
            logger.error(f"标签文件加载失败 {label_file}: {e}")
            return []
    
    def _construct_video_path(self, video_id: str, translator: str) -> str:
        """构建视频文件路径"""
        # 根据CE-CSL数据集的实际结构调整
        # 假设结构为: data_root/video/split/translator/video_file
        parts = video_id.split('_')
        if len(parts) >= 3:
            split = parts[0]  # train/dev/test
            video_file = f"{video_id}.mp4"  # 或其他视频格式
            base_root = self.config.data_root if hasattr(self.config, 'data_root') else "./data/CS-CSL"
            video_path = os.path.join(base_root, "video", split, translator, video_file)
            
            # 若首选 CS-CSL 不存在则回退到 CE-CSL
            if not os.path.exists(video_path):
                alt_root = base_root.replace("CS-CSL", "CE-CSL")
                alt_path = os.path.join(alt_root, "video", split, translator, video_file)
                if os.path.exists(alt_path):
                    return alt_path
            return video_path
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

    # 以上为 CE-CSL 数据集预处理入口

    # 以下为其他数据集处理示例（可选）
    """
   其他数据集处理示例代码
    """
