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
        """提取所有类型的关键点"""
        landmarks_data = {
            "pose": [],
            "left_hand": [],
            "right_hand": [],
            "face": [],
        }
        
        # 姿态关键点 (33个点)
        if results.pose_landmarks:
            landmarks_data["pose"] = [
                [lm.x, lm.y, lm.z, lm.visibility] 
                for lm in results.pose_landmarks.landmark
            ]
        
        # 左手关键点 (21个点)
        if results.left_hand_landmarks:
            landmarks_data["left_hand"] = [
                [lm.x, lm.y, lm.z] 
                for lm in results.left_hand_landmarks.landmark
            ]
        
        # 右手关键点 (21个点)
        if results.right_hand_landmarks:
            landmarks_data["right_hand"] = [
                [lm.x, lm.y, lm.z] 
                for lm in results.right_hand_landmarks.landmark
            ]
        
        # 面部关键点 (468个点)
        if results.face_landmarks:
            landmarks_data["face"] = [
                [lm.x, lm.y, lm.z] 
                for lm in results.face_landmarks.landmark
            ]
        
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
