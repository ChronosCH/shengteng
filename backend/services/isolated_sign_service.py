"""
孤立手语识别服务 - 基于 mind_wl I3D 模型
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import mindspore
    from mindspore import Tensor, context
    from mindspore import load_checkpoint, load_param_into_net
    MS_AVAILABLE = True
except Exception as exc:
    MS_AVAILABLE = False
    mindspore = None
    Tensor = None
    context = None
    logging.getLogger(__name__).warning(
        "MindSpore 导入失败，孤立手语识别推理不可用: %s", exc
    )


@dataclass
class InferenceResult:
    """推理结果"""
    predicted_gloss: Optional[str]
    confidence: float
    top_k_predictions: Optional[List[Dict[str, any]]] = None


class IsolatedSignService:
    """
    孤立手语识别推理服务
    使用 mind_wl 文件夹下的 I3D 模型进行推理
    """

    def __init__(
        self,
        model_checkpoint: str = None,
        class_list_path: str = None,
        num_classes: int = 2000,
        target_size: int = 224,
        top_k: int = 10,
        device_target: str = "CPU",
    ) -> None:
        self.logger = logging.getLogger(__name__)
        
        # 获取 mind_wl 目录的路径
        project_root = Path(__file__).resolve().parents[2]
        mind_wl_dir = project_root / "mind_wl"
        
        # 设置默认路径
        if model_checkpoint is None:
            model_checkpoint = str(mind_wl_dir / "weights" / "i3d_wlasl2000.ckpt")
        if class_list_path is None:
            class_list_path = str(mind_wl_dir / "wlasl_class_list.txt")
        
        self.model_checkpoint = model_checkpoint
        self.class_list_path = class_list_path
        self.num_classes = num_classes
        self.target_size = target_size
        self.top_k = top_k
        self.device_target = device_target
        self.max_frames = 64
        self.min_frames = 16
        
        self._model = None
        self._class_names: Dict[int, str] = {}
        self._lock = asyncio.Lock()
        
        # 将 mind_wl 目录添加到 sys.path
        if str(mind_wl_dir) not in sys.path:
            sys.path.insert(0, str(mind_wl_dir))

    async def _lazy_initialize(self) -> None:
        """延迟初始化模型"""
        if self._model is not None:
            return
        
        if not MS_AVAILABLE:
            raise RuntimeError("MindSpore 不可用，无法加载孤立手语模型")
        
        # 设置 MindSpore 上下文
        context.set_context(mode=context.GRAPH_MODE, device_target=self.device_target)
        self.logger.info(f"使用设备: {self.device_target}")
        self.logger.info(f"MindSpore版本: {mindspore.__version__}")
        
        # 加载类别名称
        self._load_class_names()
        
        # 导入 I3D 模型
        try:
            from models.i3d_mindspore import InceptionI3d
        except ImportError as e:
            self.logger.error(f"无法导入 I3D 模型: {e}")
            raise RuntimeError("无法导入 I3D 模型，请确保 mind_wl 目录结构正确")
        
        # 加载模型
        self.logger.info(f"正在加载模型...")
        self.logger.info(f"  - 类别数: {self.num_classes}")
        
        self._model = InceptionI3d(num_classes=self.num_classes, in_channels=3)
        
        if not os.path.exists(self.model_checkpoint):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_checkpoint}")
        
        self.logger.info(f"  - 加载MindSpore权重: {self.model_checkpoint}")
        param_dict = load_checkpoint(self.model_checkpoint)
        load_param_into_net(self._model, param_dict)
        self._model.set_train(False)
        
        self.logger.info("✓ 模型加载成功!")

    def _load_class_names(self) -> None:
        """加载类别名称"""
        if not os.path.exists(self.class_list_path):
            raise FileNotFoundError(f"类别列表文件不存在: {self.class_list_path}")
        
        self.logger.info(f"加载类别名称: {self.class_list_path}")
        with open(self.class_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    idx, name = parts
                    if int(idx) < self.num_classes:
                        self._class_names[int(idx)] = name
        
        self.logger.info(f"✓ 加载了 {len(self._class_names)} 个类别")

    def _load_video_frames(self, video_path: str):
        """从视频文件加载RGB帧"""
        self.logger.info(f"正在加载视频: {video_path}")
        vidcap = cv2.VideoCapture(video_path)
        
        if not vidcap.isOpened():
            self.logger.warning(f"OpenCV 无法打开视频文件，尝试使用 imageio 解码: {video_path}")
            return self._load_video_frames_with_imageio(video_path)
        
        frames = []
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"  - 总帧数: {total_frames}")
        self.logger.info(f"  - FPS: {fps:.2f}")
        
        # 采样策略：最多 self.max_frames 帧
        sample_interval = max(1, total_frames // self.max_frames)
        frame_count = 0
        
        while True:
            success, img = vidcap.read()
            if not success:
                break
            
            if frame_count % sample_interval == 0:
                # 中心裁剪为正方形
                h, w, c = img.shape
                if h != w:
                    size = min(h, w)
                    start_h = (h - size) // 2
                    start_w = (w - size) // 2
                    img = img[start_h:start_h+size, start_w:start_w+size]
                
                # 调整大小
                img = cv2.resize(img, (self.target_size, self.target_size))
                
                # 归一化到 [-1, 1]
                img = (img / 255.0) * 2 - 1
                frames.append(img)
            
            frame_count += 1
        
        vidcap.release()
        sampled_frames = len(frames)
        self.logger.info(f"  - 成功采样 {sampled_frames} 帧 (间隔: {sample_interval})")
        
        if sampled_frames == 0 or sampled_frames < self.min_frames:
            self.logger.warning("OpenCV 未能提取到有效帧，尝试使用 imageio 解码")
            return self._load_video_frames_with_imageio(video_path)
        
        # 转换为 MindSpore tensor: (T, H, W, C) -> (1, C, T, H, W)
        frames_array = np.array(frames, dtype=np.float32)
        frames_tensor = frames_array.transpose(3, 0, 1, 2)  # (C, T, H, W)
        frames_tensor = np.expand_dims(frames_tensor, axis=0)  # (1, C, T, H, W)
        
        return Tensor(frames_tensor, mindspore.float32)

    def _load_video_frames_with_imageio(self, video_path: str):
        """使用 imageio 作为后备方案加载视频帧"""
        try:
            import imageio.v2 as imageio
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.error(f"imageio 未安装或导入失败，无法读取视频: {exc}")
            return None

        try:
            frames_raw = []
            with imageio.get_reader(video_path) as reader:
                for frame in reader:
                    frames_raw.append(frame)

            total_frames = len(frames_raw)
            if total_frames == 0:
                self.logger.error("imageio 也未能读取到任何帧")
                return None

            # 选择均匀采样的帧索引
            max_frames = min(self.max_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=np.int64)

            frames = []
            for idx in indices:
                img = frames_raw[int(idx)]
                h, w, c = img.shape
                if h != w:
                    size = min(h, w)
                    start_h = (h - size) // 2
                    start_w = (w - size) // 2
                    img = img[start_h:start_h+size, start_w:start_w+size]

                img = cv2.resize(img, (self.target_size, self.target_size))
                img = (img / 255.0) * 2 - 1
                frames.append(img.astype(np.float32))

            self.logger.info(
                "使用 imageio 解码，共读取 %s 帧，采样至 %s 帧",
                total_frames,
                len(frames),
            )

            if not frames:
                self.logger.error("imageio 解码后仍未获得有效帧")
                return None

            frames_array = np.stack(frames, axis=0)

            # 如果帧数不足最小需求，进行重复填充
            if frames_array.shape[0] < self.min_frames:
                repeat_times = int(np.ceil(self.min_frames / frames_array.shape[0]))
                frames_array = np.tile(frames_array, (repeat_times, 1, 1, 1))
                frames_array = frames_array[:self.min_frames]

            frames_tensor = frames_array.transpose(3, 0, 1, 2)  # (C, T, H, W)
            frames_tensor = np.expand_dims(frames_tensor, axis=0)

            return Tensor(frames_tensor, mindspore.float32)

        except Exception as exc:
            self.logger.error(f"使用 imageio 读取视频失败: {exc}")
            return None

    async def predict(self, video_path: str) -> InferenceResult:
        """
        对视频进行推理
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            InferenceResult: 推理结果
        """
        async with self._lock:
            await self._lazy_initialize()
        
        if self._model is None:
            raise RuntimeError("模型尚未初始化")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 加载视频帧
        frames = self._load_video_frames(video_path)
        if frames is None:
            raise RuntimeError("无法从视频中提取帧")
        
        self.logger.info(f"输入形状: {frames.shape}")
        
        # 推理
        self.logger.info("正在进行推理...")
        logits = self._model(frames)  # (1, num_classes, T)
        
        # 对时间维度取平均
        predictions = logits.mean(axis=2)[0]  # (num_classes,)
        
        # 计算softmax概率
        exp_pred = np.exp(predictions.asnumpy() - np.max(predictions.asnumpy()))
        probs = exp_pred / np.sum(exp_pred)
        
        # 获取top-k
        top_indices = np.argsort(probs)[-self.top_k:][::-1]
        top_probs = probs[top_indices]
        
        # 构建结果
        top_k_predictions = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            class_name = self._class_names.get(idx, f"未知({idx})")
            top_k_predictions.append({
                'rank': i + 1,
                'class_id': int(idx),
                'class_name': class_name,
                'confidence': float(prob)
            })
        
        # 获取 top-1 结果
        top1 = top_k_predictions[0]
        
        self.logger.info(f"识别结果: {top1['class_name']} (置信度: {top1['confidence']*100:.2f}%)")
        
        return InferenceResult(
            predicted_gloss=top1['class_name'],
            confidence=top1['confidence'],
            top_k_predictions=top_k_predictions,
        )


__all__ = ["IsolatedSignService", "InferenceResult"]
