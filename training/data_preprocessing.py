"""
手语数据预处理管道
将原始视频数据转换为模型训练所需的格式
"""

import os
import cv2
import json
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoSample:
    """视频样本数据类"""
    video_path: str
    gloss_sequence: List[str]
    text: str
    start_frame: int = 0
    end_frame: Optional[int] = None

@dataclass
class ProcessedSample:
    """处理后的样本数据类"""
    video_id: str
    gloss_sequence: List[str]
    text: str
    keypoints: np.ndarray  # (T, 543, 3)
    face_landmarks: np.ndarray  # (T, 468, 3)
    pose_landmarks: np.ndarray  # (T, 33, 3)
    left_hand_landmarks: np.ndarray  # (T, 21, 3)
    right_hand_landmarks: np.ndarray  # (T, 21, 3)
    duration: float
    fps: int

class SignLanguagePreprocessor:
    """手语数据预处理器"""
    
    def __init__(self, 
                 target_fps: int = 25,
                 min_sequence_length: int = 10,
                 max_sequence_length: int = 300,
                 image_size: Tuple[int, int] = (224, 224)):
        
        self.target_fps = target_fps
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.image_size = image_size
        
        # 初始化MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("手语数据预处理器初始化完成")
    
    def extract_keypoints_from_video(self, video_path: str, 
                                   start_frame: int = 0, 
                                   end_frame: Optional[int] = None) -> Optional[Dict]:
        """从视频中提取关键点"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return None
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if end_frame is None:
                end_frame = total_frames
            
            # 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            all_keypoints = []
            face_landmarks_list = []
            pose_landmarks_list = []
            left_hand_landmarks_list = []
            right_hand_landmarks_list = []
            
            frame_count = 0
            target_frame_interval = max(1, int(fps / self.target_fps))
            
            for frame_idx in range(start_frame, min(end_frame, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按目标FPS采样
                if frame_count % target_frame_interval != 0:
                    frame_count += 1
                    continue
                
                # 调整图像大小
                frame = cv2.resize(frame, self.image_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # MediaPipe处理
                results = self.holistic.process(frame_rgb)
                
                # 提取各部分关键点
                frame_keypoints = self._extract_frame_keypoints(results)
                all_keypoints.append(frame_keypoints)
                
                # 分别存储各部分关键点
                face_landmarks_list.append(self._extract_face_landmarks(results))
                pose_landmarks_list.append(self._extract_pose_landmarks(results))
                left_hand_landmarks_list.append(self._extract_hand_landmarks(results.left_hand_landmarks))
                right_hand_landmarks_list.append(self._extract_hand_landmarks(results.right_hand_landmarks))
                
                frame_count += 1
            
            cap.release()
            
            if len(all_keypoints) < self.min_sequence_length:
                logger.warning(f"序列长度不足: {len(all_keypoints)} < {self.min_sequence_length}")
                return None
            
            # 转换为numpy数组
            keypoints_array = np.array(all_keypoints)  # (T, 543, 3)
            face_array = np.array(face_landmarks_list)  # (T, 468, 3)
            pose_array = np.array(pose_landmarks_list)  # (T, 33, 3)
            left_hand_array = np.array(left_hand_landmarks_list)  # (T, 21, 3)
            right_hand_array = np.array(right_hand_landmarks_list)  # (T, 21, 3)
            
            # 序列长度限制
            if len(keypoints_array) > self.max_sequence_length:
                keypoints_array = keypoints_array[:self.max_sequence_length]
                face_array = face_array[:self.max_sequence_length]
                pose_array = pose_array[:self.max_sequence_length]
                left_hand_array = left_hand_array[:self.max_sequence_length]
                right_hand_array = right_hand_array[:self.max_sequence_length]
            
            return {
                'keypoints': keypoints_array,
                'face_landmarks': face_array,
                'pose_landmarks': pose_array,
                'left_hand_landmarks': left_hand_array,
                'right_hand_landmarks': right_hand_array,
                'duration': len(keypoints_array) / self.target_fps,
                'fps': self.target_fps,
                'original_fps': fps
            }
            
        except Exception as e:
            logger.error(f"提取关键点失败 {video_path}: {e}")
            return None
    
    def _extract_frame_keypoints(self, results) -> np.ndarray:
        """提取单帧的所有关键点 (543个点)"""
        keypoints = np.zeros((543, 3))
        
        # 面部关键点 (468个)
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y, landmark.z]
        
        # 身体姿态关键点 (33个)
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[468 + i] = [landmark.x, landmark.y, landmark.z]
        
        # 左手关键点 (21个)
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                keypoints[501 + i] = [landmark.x, landmark.y, landmark.z]
        
        # 右手关键点 (21个)
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                keypoints[522 + i] = [landmark.x, landmark.y, landmark.z]
        
        return keypoints
    
    def _extract_face_landmarks(self, results) -> np.ndarray:
        """提取面部关键点"""
        landmarks = np.zeros((468, 3))
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y, landmark.z]
        return landmarks
    
    def _extract_pose_landmarks(self, results) -> np.ndarray:
        """提取身体姿态关键点"""
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
    
    def process_dataset(self, 
                       annotation_file: str, 
                       video_dir: str, 
                       output_dir: str,
                       num_workers: int = 4) -> None:
        """处理整个数据集"""
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载标注文件
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        logger.info(f"开始处理数据集，共 {len(annotations)} 个样本")
        
        # 准备任务
        tasks = []
        for item in annotations:
            video_path = os.path.join(video_dir, item['video_file'])
            if os.path.exists(video_path):
                sample = VideoSample(
                    video_path=video_path,
                    gloss_sequence=item['gloss_sequence'],
                    text=item['text'],
                    start_frame=item.get('start_frame', 0),
                    end_frame=item.get('end_frame', None)
                )
                tasks.append((item['video_id'], sample))
        
        # 并行处理
        processed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(self._process_single_sample, video_id, sample, output_dir): (video_id, sample)
                for video_id, sample in tasks
            }
            
            for future in as_completed(future_to_task):
                video_id, sample = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    if (processed_count + failed_count) % 100 == 0:
                        logger.info(f"已处理: {processed_count + failed_count}/{len(tasks)}, "
                                  f"成功: {processed_count}, 失败: {failed_count}")
                        
                except Exception as e:
                    logger.error(f"处理样本 {video_id} 时出错: {e}")
                    failed_count += 1
        
        logger.info(f"数据集处理完成！成功: {processed_count}, 失败: {failed_count}")
        
        # 生成数据集统计信息
        self._generate_dataset_stats(output_dir, processed_count)
    
    def _process_single_sample(self, video_id: str, sample: VideoSample, output_dir: str) -> bool:
        """处理单个样本"""
        try:
            # 提取关键点
            keypoint_data = self.extract_keypoints_from_video(
                sample.video_path, 
                sample.start_frame, 
                sample.end_frame
            )
            
            if keypoint_data is None:
                return False
            
            # 创建处理后的样本
            processed_sample = ProcessedSample(
                video_id=video_id,
                gloss_sequence=sample.gloss_sequence,
                text=sample.text,
                keypoints=keypoint_data['keypoints'],
                face_landmarks=keypoint_data['face_landmarks'],
                pose_landmarks=keypoint_data['pose_landmarks'],
                left_hand_landmarks=keypoint_data['left_hand_landmarks'],
                right_hand_landmarks=keypoint_data['right_hand_landmarks'],
                duration=keypoint_data['duration'],
                fps=keypoint_data['fps']
            )
            
            # 保存处理后的数据
            output_file = os.path.join(output_dir, f"{video_id}.npz")
            np.savez_compressed(
                output_file,
                video_id=video_id,
                gloss_sequence=sample.gloss_sequence,
                text=sample.text,
                keypoints=processed_sample.keypoints,
                face_landmarks=processed_sample.face_landmarks,
                pose_landmarks=processed_sample.pose_landmarks,
                left_hand_landmarks=processed_sample.left_hand_landmarks,
                right_hand_landmarks=processed_sample.right_hand_landmarks,
                duration=processed_sample.duration,
                fps=processed_sample.fps
            )
            
            return True
            
        except Exception as e:
            logger.error(f"处理样本 {video_id} 失败: {e}")
            return False
    
    def _generate_dataset_stats(self, output_dir: str, sample_count: int) -> None:
        """生成数据集统计信息"""
        stats = {
            'total_samples': sample_count,
            'target_fps': self.target_fps,
            'image_size': self.image_size,
            'min_sequence_length': self.min_sequence_length,
            'max_sequence_length': self.max_sequence_length,
            'keypoint_dimensions': 543,
            'coordinate_dimensions': 3
        }
        
        stats_file = os.path.join(output_dir, 'dataset_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集统计信息已保存到: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='手语数据预处理')
    parser.add_argument('--annotation_file', required=True, help='标注文件路径')
    parser.add_argument('--video_dir', required=True, help='视频文件目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--target_fps', type=int, default=25, help='目标帧率')
    parser.add_argument('--min_length', type=int, default=10, help='最小序列长度')
    parser.add_argument('--max_length', type=int, default=300, help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=4, help='并行工作数')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = SignLanguagePreprocessor(
        target_fps=args.target_fps,
        min_sequence_length=args.min_length,
        max_sequence_length=args.max_length
    )
    
    # 处理数据集
    preprocessor.process_dataset(
        annotation_file=args.annotation_file,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
