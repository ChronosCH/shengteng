from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor
    MS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    MS_AVAILABLE = False
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    logging.getLogger(__name__).warning(
        "MindSpore 导入失败，孤立手语识别推理不可用: %s", exc
    )

_ISOLATED_CONFIG_DEFAULT = {
    "input_size": (160, 160),
    "sequence_length": 32,
    "use_gpu": False,
    "device_id": 0,
}


@dataclass
class InferenceResult:
    predicted_gloss: Optional[str]
    confidence: float
    logits: Optional[List[float]] = None


class IsolatedSignService:
    """孤立手语识别推理服务。"""

    def __init__(
        self,
        model_checkpoint: str,
        train_csv_path: str,
        config: Optional[Dict] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.model_checkpoint = model_checkpoint
        self.train_csv_path = train_csv_path
        self.config = {**_ISOLATED_CONFIG_DEFAULT, **(config or {})}
        self.mapping_path: Optional[str] = self.config.get("class_mapping_path")

        self._model = None
        self._class_to_idx: Dict[str, int] = {}
        self._idx_to_class: Dict[int, str] = {}
        self._processor = None
        self._lock = asyncio.Lock()

    async def _lazy_initialize(self) -> None:
        if self._model is not None:
            return
        if not MS_AVAILABLE:
            raise RuntimeError("MindSpore 不可用，无法加载孤立手语模型")
        from training_ASL.config import Config as ASLConfig
        from training_ASL.src.data_loader import get_class_mapping, VideoProcessor
        from training_ASL.src.model import ASLRecognitionModel

        if not os.path.exists(self.model_checkpoint):
            raise FileNotFoundError(
                f"模型检查点不存在: {self.model_checkpoint}"
            )

        if not os.path.exists(self.train_csv_path):
            alt_csv = os.path.join(os.path.dirname(self.train_csv_path), "train_subset.csv")
            if os.path.exists(alt_csv):
                self.logger.warning("使用 train_subset.csv 作为类别映射来源")
                self.train_csv_path = alt_csv
            elif self.mapping_path:
                self.logger.warning("train.csv 缺失，尝试从映射文件加载类别映射: %s", self.mapping_path)
            else:
                raise FileNotFoundError(
                    f"训练集CSV不存在: {self.train_csv_path}"
                )

        if self.mapping_path and os.path.exists(self.mapping_path):
            self._class_to_idx, self._idx_to_class = self._load_mapping_from_file(self.mapping_path)
        else:
            self._class_to_idx, self._idx_to_class = get_class_mapping(self.train_csv_path)

        num_classes = len(self._class_to_idx)
        if num_classes == 0:
            raise RuntimeError("训练集类别映射为空")

        # Video processor
        self._processor = VideoProcessor(
            target_size=self.config.get("input_size", ASLConfig.INPUT_SIZE),
            sequence_length=self.config.get("sequence_length", ASLConfig.SEQUENCE_LENGTH),
            sampling_mode=self.config.get("sampling_mode", "random_segment"),
            sampling_stride=int(self.config.get("sampling_stride", 1)),
            sampling_rand=bool(self.config.get("sampling_rand", True)),
            augment=False,
        )

        # MindSpore context
        device_target = "GPU" if self.config.get("use_gpu") else "CPU"
        ms.context.set_context(
            mode=ms.context.GRAPH_MODE,
            device_target=device_target,
            device_id=self.config.get("device_id", 0),
        )

        self._model = ASLRecognitionModel(
            num_classes=num_classes,
            sequence_length=self._processor.sequence_length,
            input_size=self._processor.target_size,
            base_channels=self.config.get("base_channels", 64),
        )
        param_dict = ms.load_checkpoint(self.model_checkpoint)
        ms.load_param_into_net(self._model, param_dict)
        self._model.set_train(False)
        self.logger.info(
            "孤立手语模型已加载，类别数=%s, checkpoint=%s",
            num_classes,
            self.model_checkpoint,
        )

    def _load_mapping_from_file(self, path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "class_to_idx" in data:
            class_to_idx = {str(k): int(v) for k, v in data["class_to_idx"].items()}
        else:
            class_to_idx = {str(k): int(v) for k, v in data.items()}
        idx_to_class = {idx: gloss for gloss, idx in class_to_idx.items()}
        self.logger.info("从映射文件加载 %s 个类别: %s", len(class_to_idx), os.path.basename(path))
        return class_to_idx, idx_to_class

    async def predict(self, video_path: str) -> InferenceResult:
        async with self._lock:
            await self._lazy_initialize()

        if self._model is None or self._processor is None:
            raise RuntimeError("模型尚未初始化")

        frames = self._processor.extract_frames(video_path)
        if frames is None:
            raise RuntimeError("无法从视频中提取帧")

        frames = np.transpose(frames, (3, 0, 1, 2))  # -> (C, T, H, W)
        frames = np.expand_dims(frames, axis=0)

        tensor = Tensor(frames, ms.float32)
        logits = self._model(tensor)
        probs = ms.ops.softmax(logits, axis=1)

        probabilities = probs.asnumpy()[0]
        idx = int(np.argmax(probabilities))
        confidence = float(probabilities[idx])
        gloss = self._idx_to_class.get(idx)

        return InferenceResult(
            predicted_gloss=gloss,
            confidence=confidence,
            logits=probabilities.tolist(),
        )


__all__ = ["IsolatedSignService", "InferenceResult"]
