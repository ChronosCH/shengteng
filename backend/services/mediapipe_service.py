"""
MediaPipe Holistic 关键点提取服务
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import time

from utils.logger import setup_logger
from utils.config import settings

logger = setup_logger(__name__)


class MediaPipeService:
    """MediaPipe Holistic 关键点提取服务"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化Holistic模型
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            enable_segmentation=False,  # 不需要分割，提高性能
            smooth_landmarks=True,
        )
        
        # 性能统计
        self.stats = {
            "total_frames_processed": 0,
            "average_processing_time": 0.0,
            "last_processing_time": 0.0,
            "errors": 0,
        }
        
        logger.info("MediaPipe Holistic 服务初始化完成")
    
    def extract_landmarks(self, image: np.ndarray) -> Dict:
        """
        从图像中提取关键点
        
        Args:
            image: BGR格式的图像数组
            
        Returns:
            包含所有关键点的字典
        """
        start_time = time.time()
        
        try:
            # 转换颜色空间 (BGR -> RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # 进行推理
            results = self.holistic.process(rgb_image)
            
            # 提取关键点
            landmarks_data = self._extract_all_landmarks(results)
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return {
                "success": True,
                "landmarks": landmarks_data,
                "processing_time": processing_time,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"关键点提取失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            }
    
    def _extract_all_landmarks(self, results) -> Dict[str, List[List[float]]]:
        """提取所有类型的关键点，增加稳定性和错误处理"""
        landmarks_data = {
            "pose": [],
            "left_hand": [],
            "right_hand": [],
            "face": [],
        }

        try:
            # 姿态关键点 (33个点)
            if results.pose_landmarks:
                landmarks_data["pose"] = [
                    [
                        float(lm.x), float(lm.y), float(lm.z),
                        float(getattr(lm, 'visibility', 1.0))
                    ]
                    for lm in results.pose_landmarks.landmark
                ]
            else:
                # 如果没有检测到姿态，填充零值
                landmarks_data["pose"] = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]

            # 左手关键点 (21个点)
            if results.left_hand_landmarks:
                landmarks_data["left_hand"] = [
                    [float(lm.x), float(lm.y), float(lm.z)]
                    for lm in results.left_hand_landmarks.landmark
                ]
            else:
                # 如果没有检测到左手，填充零值
                landmarks_data["left_hand"] = [[0.0, 0.0, 0.0] for _ in range(21)]

            # 右手关键点 (21个点)
            if results.right_hand_landmarks:
                landmarks_data["right_hand"] = [
                    [float(lm.x), float(lm.y), float(lm.z)]
                    for lm in results.right_hand_landmarks.landmark
                ]
            else:
                # 如果没有检测到右手，填充零值
                landmarks_data["right_hand"] = [[0.0, 0.0, 0.0] for _ in range(21)]

            # 面部关键点 (468个点) - 简化为关键区域
            if results.face_landmarks:
                # 只提取面部的关键点，减少数据量
                key_face_indices = list(range(0, 468, 10))  # 每10个点取一个
                landmarks_data["face"] = [
                    [float(lm.x), float(lm.y), float(lm.z)]
                    for i, lm in enumerate(results.face_landmarks.landmark)
                    if i in key_face_indices
                ]
            else:
                # 如果没有检测到面部，填充零值
                landmarks_data["face"] = [[0.0, 0.0, 0.0] for _ in range(47)]  # 468/10 ≈ 47

        except Exception as e:
            logger.error(f"关键点提取过程中出错: {e}")
            # 返回默认的零值数据
            landmarks_data = {
                "pose": [[0.0, 0.0, 0.0, 0.0] for _ in range(33)],
                "left_hand": [[0.0, 0.0, 0.0] for _ in range(21)],
                "right_hand": [[0.0, 0.0, 0.0] for _ in range(21)],
                "face": [[0.0, 0.0, 0.0] for _ in range(47)],
            }

        return landmarks_data
    
    def landmarks_to_array(self, landmarks_data: Dict) -> np.ndarray:
        """
        将关键点数据转换为固定长度的数组
        
        Returns:
            shape为(543, 3)的numpy数组，对应543个关键点的x,y,z坐标
        """
        # 预分配数组 (543个点 × 3个坐标)
        landmarks_array = np.zeros((543, 3), dtype=np.float32)
        
        offset = 0
        
        # 姿态关键点 (33个点，取x,y,z)
        if landmarks_data["pose"]:
            pose_points = np.array(landmarks_data["pose"])[:, :3]  # 忽略visibility
            landmarks_array[offset:offset+33] = pose_points
        offset += 33
        
        # 左手关键点 (21个点)
        if landmarks_data["left_hand"]:
            left_hand_points = np.array(landmarks_data["left_hand"])
            landmarks_array[offset:offset+21] = left_hand_points
        offset += 21
        
        # 右手关键点 (21个点)
        if landmarks_data["right_hand"]:
            right_hand_points = np.array(landmarks_data["right_hand"])
            landmarks_array[offset:offset+21] = right_hand_points
        offset += 21
        
        # 面部关键点 (468个点)
        if landmarks_data["face"]:
            face_points = np.array(landmarks_data["face"])
            landmarks_array[offset:offset+468] = face_points
        
        return landmarks_array

    def get_normalized_landmarks_for_cslr(self, landmarks_data: Dict) -> np.ndarray:
        """获取适用于CSLR模型的标准化关键点数据"""
        try:
            # 提取关键的关键点，减少维度
            pose_points = landmarks_data.get("pose", [])
            left_hand_points = landmarks_data.get("left_hand", [])
            right_hand_points = landmarks_data.get("right_hand", [])

            # 只使用上半身姿态关键点（肩膀、手肘、手腕等）
            key_pose_indices = [11, 12, 13, 14, 15, 16]  # 肩膀、手肘、手腕
            selected_pose = []
            for i in key_pose_indices:
                if i < len(pose_points):
                    selected_pose.extend(pose_points[i][:3])  # x, y, z
                else:
                    selected_pose.extend([0.0, 0.0, 0.0])

            # 合并所有关键点
            all_points = []
            all_points.extend(selected_pose)  # 6 * 3 = 18 维

            # 添加左手关键点
            for point in left_hand_points:
                all_points.extend(point[:3])  # 21 * 3 = 63 维

            # 添加右手关键点
            for point in right_hand_points:
                all_points.extend(point[:3])  # 21 * 3 = 63 维

            # 总维度: 18 + 63 + 63 = 144
            target_dim = 144
            if len(all_points) < target_dim:
                all_points.extend([0.0] * (target_dim - len(all_points)))
            elif len(all_points) > target_dim:
                all_points = all_points[:target_dim]

            # 转换为numpy数组
            landmarks_array = np.array(all_points, dtype=np.float32)

            # 归一化到[-1, 1]范围
            landmarks_array = np.clip((landmarks_array - 0.5) * 2.0, -1.0, 1.0)

            return landmarks_array

        except Exception as e:
            logger.error(f"CSLR关键点标准化失败: {e}")
            return np.zeros(144, dtype=np.float32)

    def draw_landmarks(self, image: np.ndarray, landmarks_data: Dict) -> np.ndarray:
        """
        在图像上绘制关键点
        
        Args:
            image: 原始图像
            landmarks_data: 关键点数据
            
        Returns:
            绘制了关键点的图像
        """
        annotated_image = image.copy()
        
        try:
            # 重新处理图像以获取MediaPipe结果对象
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_image)
            
            # 绘制姿态关键点
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # 绘制手部关键点
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # 绘制面部关键点 (可选，可能会很密集)
            # if results.face_landmarks:
            #     self.mp_drawing.draw_landmarks(
            #         annotated_image,
            #         results.face_landmarks,
            #         self.mp_holistic.FACEMESH_CONTOURS,
            #         landmark_drawing_spec=None,
            #         connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            #     )
            
        except Exception as e:
            logger.error(f"绘制关键点失败: {e}")
        
        return annotated_image
    
    def _update_stats(self, processing_time: float):
        """更新性能统计"""
        self.stats["total_frames_processed"] += 1
        self.stats["last_processing_time"] = processing_time
        
        # 计算平均处理时间 (使用指数移动平均)
        alpha = 0.1
        if self.stats["average_processing_time"] == 0:
            self.stats["average_processing_time"] = processing_time
        else:
            self.stats["average_processing_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats["average_processing_time"]
            )
    
    def get_stats(self) -> Dict:
        """获取性能统计信息"""
        fps = 1.0 / self.stats["average_processing_time"] if self.stats["average_processing_time"] > 0 else 0
        
        return {
            **self.stats,
            "average_fps": fps,
            "model_complexity": settings.MEDIAPIPE_MODEL_COMPLEXITY,
            "detection_confidence": settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            "tracking_confidence": settings.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        }
    
    async def cleanup(self):
        """清理资源"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
        logger.info("MediaPipe 服务已清理")
