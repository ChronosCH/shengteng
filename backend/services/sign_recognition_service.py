"""
离线/批处理 手语视频识别服务
实现流程:
1. 读取视频 -> 解码为 RGB 帧序列（必要时导出到 mind_vac 输出目录）
2. Mind-VAC CSLR 模型推理 -> 解码出 gloss 序列
3. 可选：通义千问 LLM 增强翻译 -> 生成自然语言文本
4. 保存结果 JSON + 任务状态管理
"""
from __future__ import annotations
import cv2
import os
import json
import uuid
import time
import math
import asyncio
import threading
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from utils.logger import setup_logger
from utils.config import get_settings

logger = setup_logger(__name__)

# 新增：引入 MindSpore 与训练模型定义（用于帧模型推理）
try:
    import mindspore as ms
    from mindspore import Tensor
    from mindspore import load_checkpoint, load_param_into_net
    from training.tfnet_model import TFNetModel
    from training.config_manager import ConfigManager
    MS_AVAILABLE = True
except Exception as e:
    MS_AVAILABLE = False
    logger.debug(f"MindSpore 帧模型依赖不可用: {e}")


@dataclass
class RecognitionSegment:
    gloss_sequence: List[str]
    start_frame: int
    end_frame: int
    confidence: float
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class RecognitionResult:
    task_id: str
    file_path: str
    gloss_sequence: List[str]
    text: str
    segments: List[RecognitionSegment]
    overall_confidence: float
    frame_count: int
    fps: float
    duration: float
    baseline_text: Optional[str] = None
    pipeline: str = "unknown"
    srt_path: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())
    raw_gloss_text: Optional[str] = None
    llm_result: Optional[Dict[str, Any]] = None
    frames_dir: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "file_path": self.file_path,
            "gloss_sequence": self.gloss_sequence,
            "text": self.text,
            "baseline_text": self.baseline_text,
            "pipeline": self.pipeline,
            "segments": [
                {
                    "gloss_sequence": seg.gloss_sequence,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "confidence": seg.confidence,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                } for seg in self.segments
            ],
            "overall_confidence": self.overall_confidence,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration": self.duration,
            "srt_path": self.srt_path,
            "created_at": self.created_at,
            "raw_gloss_text": self.raw_gloss_text,
            "llm_result": self.llm_result,
            "frames_dir": self.frames_dir,
            "extra": self.extra,
        }


class MindVacEngine:
    """Mind-VAC CSLR 推理引擎封装"""

    def __init__(self, settings):
        self.enabled = bool(getattr(settings, "MINDVAC_ENABLED", False))
        self.device = getattr(settings, "MINDVAC_DEVICE", "CPU")
        self.checkpoint_path = Path(getattr(settings, "MINDVAC_CHECKPOINT_PATH", "mind_vac/slr_mindspore.ckpt"))
        self.dict_path = Path(getattr(settings, "MINDVAC_DICT_PATH", "mind_vac/gloss_dict.npy"))
        self.output_dir = Path(getattr(settings, "MINDVAC_OUTPUT_DIR", "mind_vac/output_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = bool(getattr(settings, "MINDVAC_USE_LLM", True))
        self.qwen_model = getattr(settings, "MINDVAC_QWEN_MODEL", "qwen-plus")

        self.available = False
        self.initialized = False
        self.last_error: Optional[str] = None
        self._init_lock = threading.Lock()
        self.qwen_client = None
        self.model = None
        self.decoder = None
        self.gloss_dict: Optional[Dict[str, Any]] = None
        self.num_classes: int = 0

        if not self.enabled:
            logger.info("Mind-VAC 集成被禁用，使用备用识别管线")
            return

        try:
            import mindspore as ms
            from mindspore import context, Tensor, load_checkpoint, load_param_into_net
            from mind_vac.model import SLRModel
            from mind_vac.decoder import Decode, softmax
            from mind_vac.transforms import Compose, CenterCrop, ToTensor, normalize_video
            from mind_vac.qwen_api import QwenAPI
        except Exception as exc:
            self.last_error = f"Mind-VAC 依赖导入失败: {exc}"
            logger.warning(self.last_error)
            return

        if not self.checkpoint_path.exists():
            self.last_error = f"未找到 Mind-VAC 模型权重: {self.checkpoint_path}"
            logger.warning(self.last_error)
            return

        if not self.dict_path.exists():
            self.last_error = f"未找到 Mind-VAC 词典文件: {self.dict_path}"
            logger.warning(self.last_error)
            return

        self.ms = ms
        self.context = context
        self.Tensor = Tensor
        self.load_checkpoint = load_checkpoint
        self.load_param_into_net = load_param_into_net
        self.SLRModel = SLRModel
        self.Decode = Decode
        self.softmax = softmax
        self.Compose = Compose
        self.CenterCrop = CenterCrop
        self.ToTensor = ToTensor
        self.normalize_video = normalize_video
        self.QwenAPI = QwenAPI

        try:
            import torch
            self.torch = torch
        except Exception:
            self.torch = None

        self.device = self.device.upper()
        self.available = True

    def _ensure_initialized(self):
        if not self.available:
            raise RuntimeError(self.last_error or "Mind-VAC 引擎不可用")
        if self.initialized:
            return

        with self._init_lock:
            if self.initialized:
                return
            try:
                try:
                    self.context.set_context(mode=self.context.PYNATIVE_MODE, device_target=self.device)
                except Exception:
                    self.context.set_context(mode=self.context.GRAPH_MODE, device_target=self.device)

                # 加载词典
                self.gloss_dict = np.load(str(self.dict_path), allow_pickle=True).item()
                if not isinstance(self.gloss_dict, dict):
                    raise ValueError("Mind-VAC 词典格式无效")

                self.num_classes = len(self.gloss_dict) + 1

                # 创建模型
                self.model = self.SLRModel(
                    num_classes=self.num_classes,
                    hidden_size=1024,
                    conv_type=2,
                    use_bn=True,
                    weight_norm=False,
                    share_classifier=False,
                )

                # 加载权重
                if self.checkpoint_path.suffix == ".pt":
                    param_dict = self._convert_pytorch_to_mindspore(str(self.checkpoint_path), self.num_classes)
                else:
                    raw_params = self.load_checkpoint(str(self.checkpoint_path))
                    param_dict, _, _ = self._preprocess_checkpoint(raw_params)

                load_result = self.load_param_into_net(self.model, param_dict)
                if isinstance(load_result, tuple):
                    param_not_load, ckpt_not_load = load_result
                    if param_not_load:
                        logger.warning(f"Mind-VAC 模型存在未加载参数: {param_not_load}")
                    if ckpt_not_load:
                        logger.warning(f"Mind-VAC 权重存在未匹配项: {ckpt_not_load}")
                elif load_result:
                    logger.warning(f"Mind-VAC 模型未完全加载: {load_result}")

                self.model.set_train(False)

                # 创建解码器
                self.decoder = self.Decode(self.gloss_dict, self.num_classes, search_mode='beam')

                # 初始化Qwen客户端
                if self.use_llm:
                    api_key = os.environ.get('DASHSCOPE_API_KEY')
                    if api_key:
                        try:
                            self.qwen_client = self.QwenAPI(api_key=api_key, model=self.qwen_model)
                        except Exception as exc:
                            logger.warning(f"通义千问客户端初始化失败，将跳过LLM增强: {exc}")
                            self.qwen_client = None
                    else:
                        logger.warning("未检测到 DASHSCOPE_API_KEY，Mind-VAC LLM 功能已禁用")
                        self.qwen_client = None

                self.initialized = True
                logger.info("Mind-VAC 模型与资源加载完成")

            except Exception as exc:
                self.last_error = f"Mind-VAC 初始化失败: {exc}"
                logger.error(self.last_error)
                raise

    def _preprocess_checkpoint(self, param_dict):
        converted = {}
        skipped_keys = []
        reshaped_keys = []

        for name, param in param_dict.items():
            if name.endswith('num_batches_tracked'):
                skipped_keys.append(name)
                continue

            value = param
            if hasattr(value, 'asnumpy'):
                array = value.asnumpy()
                dtype = value.dtype if hasattr(value, 'dtype') else self.ms.float32
            else:
                array = np.array(value)
                dtype = self.ms.float32

            target_name = name

            if name.startswith('conv1d.temporal_conv.'):
                parts = name.split('.')
                if len(parts) >= 4:
                    layer_idx, param_name = parts[2], parts[3]
                    if layer_idx in {'1', '5'}:
                        if param_name == 'weight':
                            target_name = name.replace('.weight', '.gamma')
                        elif param_name == 'bias':
                            target_name = name.replace('.bias', '.beta')

            if name.startswith('conv1d.fc.'):
                target_name = name.replace('conv1d.', '', 1)

            if target_name in {'fc.weight', 'classifier.weight'} and array.ndim == 2:
                array = array.T

            tensor = self.Tensor(array, dtype=dtype)
            converted[target_name] = self.ms.Parameter(tensor, name=target_name)

        return converted, skipped_keys, reshaped_keys

    def _convert_pytorch_to_mindspore(self, pt_path: str, num_classes: int):
        if not self.torch:
            raise RuntimeError("需要安装 PyTorch 才能加载 Mind-VAC PyTorch 权重")

        state_dict = self.torch.load(pt_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        converted = {}
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name.endswith('num_batches_tracked'):
                continue

            array = param.cpu().numpy()
            target_name = name

            if 'running_mean' in target_name:
                target_name = target_name.replace('running_mean', 'moving_mean')
            if 'running_var' in target_name:
                target_name = target_name.replace('running_var', 'moving_variance')

            if any(tag in target_name for tag in ['bn', 'downsample.1']) and target_name.endswith('.weight'):
                target_name = target_name.replace('.weight', '.gamma')
            if any(tag in target_name for tag in ['bn', 'downsample.1']) and target_name.endswith('.bias'):
                target_name = target_name.replace('.bias', '.beta')

            if target_name.startswith('conv1d.temporal_conv.'):
                parts = target_name.split('.')
                if len(parts) >= 4:
                    layer_idx, param_name = parts[2], parts[3]
                    if layer_idx in {'1', '5'}:
                        if param_name == 'weight':
                            target_name = target_name.replace('.weight', '.gamma')
                        elif param_name == 'bias':
                            target_name = target_name.replace('.bias', '.beta')

            if target_name.startswith('conv1d.temporal_conv.') and target_name.endswith('.weight') and array.ndim == 3:
                array = np.expand_dims(array, axis=2)

            if target_name.startswith('conv1d.fc.'):
                target_name = target_name.replace('conv1d.', '', 1)

            if target_name.endswith('.weight') and array.ndim == 2 and target_name.startswith('fc.') and array.shape[0] != num_classes:
                array = array.T
            if target_name.endswith('.weight') and array.ndim == 2 and target_name.startswith('classifier.') and array.shape[0] != num_classes:
                array = array.T

            converted[target_name] = self.ms.Parameter(self.Tensor(array, dtype=self.ms.float32), name=target_name)

        return converted

    def _preprocess_video(self, frames: List[np.ndarray], crop_size: int = 224):
        if not frames:
            raise ValueError("Mind-VAC 预处理需要至少一帧图像")

        transform = self.Compose([
            self.CenterCrop(crop_size),
            self.ToTensor(),
        ])

        video_tensor, _ = transform(frames, None, None)
        video_tensor = self.normalize_video(video_tensor)

        video_array = video_tensor.asnumpy()

        total_frames = video_array.shape[0]
        left_pad = 6
        right_pad = int(np.ceil(total_frames / 4.0)) * 4 - total_frames + 6

        if left_pad > 0:
            pad_front = np.repeat(video_array[:1], left_pad, axis=0)
        else:
            pad_front = np.empty((0,) + video_array.shape[1:], dtype=video_array.dtype)

        if right_pad > 0:
            pad_back = np.repeat(video_array[-1:], right_pad, axis=0)
        else:
            pad_back = np.empty((0,) + video_array.shape[1:], dtype=video_array.dtype)

        padded_video = np.concatenate([pad_front, video_array, pad_back], axis=0)
        padded_video = np.expand_dims(padded_video, axis=0)

        tensor = self.Tensor(padded_video, dtype=self.ms.float32)
        seq_len = [padded_video.shape[1]]
        return tensor, seq_len

    def _feat_len_to_int(self, feat_len) -> int:
        try:
            if hasattr(feat_len, 'asnumpy'):
                data = feat_len.asnumpy()
            else:
                data = np.array(feat_len)
            if data.size == 0:
                return 0
            return int(data.reshape(-1)[0])
        except Exception:
            return 0

    def _compute_confidence(self, sequence_logits, feat_len) -> float:
        try:
            logits = sequence_logits.asnumpy() if hasattr(sequence_logits, 'asnumpy') else np.array(sequence_logits)
            if logits.ndim == 3:
                logits_btc = np.transpose(logits, (1, 0, 2))  # (B, T, C)
            elif logits.ndim == 2:
                logits_btc = logits[None, ...]
            else:
                return 0.0

            seq_length = self._feat_len_to_int(feat_len)
            probs = self.softmax(logits_btc, axis=-1)
            valid = probs[0, :seq_length] if seq_length > 0 else probs[0]
            if valid.size == 0:
                return 0.0
            best = valid.max(axis=-1)
            return float(best.mean())
        except Exception as exc:
            logger.warning(f"Mind-VAC 置信度计算失败: {exc}")
            return 0.0

    def _run_sync(self, frames: List[np.ndarray], use_llm_flag: bool) -> Dict[str, Any]:
        if not frames:
            raise ValueError("Mind-VAC 推理需要至少一帧图像")

        self._ensure_initialized()

        video_tensor, seq_len = self._preprocess_video(frames)
        outputs = self.model(video_tensor, seq_len)
        sequence_logits = outputs['sequence_logits']
        feat_len = outputs['feat_len']

        decoded = self.decoder.decode(sequence_logits, feat_len, batch_first=False, probs=False) if self.decoder else []
        recognized = decoded[0] if decoded else []
        gloss_sequence = [item[0] for item in recognized]
        gloss_sequence = [token for token in gloss_sequence if str(token).strip()]
        decoder_raw = [
            {
                "token": str(item[0]),
                "position": int(item[1]) if len(item) > 1 else idx,
            }
            for idx, item in enumerate(recognized)
        ] if recognized else []
        raw_gloss_text = " ".join(gloss_sequence)

        confidence = self._compute_confidence(sequence_logits, feat_len)

        llm_result = None
        if use_llm_flag and self.qwen_client and raw_gloss_text:
            try:
                llm_result = self.qwen_client.translate_gloss_to_sentence(raw_gloss_text)
            except Exception as exc:
                logger.warning(f"Mind-VAC 调用通义千问失败: {exc}")
                llm_result = {
                    "success": False,
                    "error": str(exc),
                }

        return {
            "gloss_sequence": gloss_sequence,
            "raw_gloss_text": raw_gloss_text,
            "confidence": confidence,
            "llm_result": llm_result,
            "decoder_raw": decoder_raw,
        }

    async def run_on_frames(self, frames: List[np.ndarray], use_llm: Optional[bool] = None) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.last_error or "Mind-VAC 引擎不可用")

        use_llm_flag = self.use_llm if use_llm is None else use_llm
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, frames, use_llm_flag)

class SignRecognitionService:
    def __init__(self, mediapipe_service, cslr_service, result_dir: str = "temp/sign_results"):
        self.mediapipe_service = mediapipe_service
        self.cslr_service = cslr_service
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.target_fps = 25
        self.window_length = getattr(self.cslr_service.config, "max_sequence_length", 64)
        self.window_overlap = 0.4  # 40% 重叠
        self.gloss_dict_path = os.path.join(self.result_dir, "gloss_dictionary.json")
        self.gloss_dict = self._load_or_create_gloss_dict()

        # 新增：帧模型推理相关
        # 只有在MindSpore可用时才启用帧模型
    # 由配置控制是否启用帧模型（默认 False，仅作为 Mind-VAC 备用方案）
        settings = None
        try:
            settings = get_settings()
        except Exception as cfg_exc:
            logger.warning(f"获取配置失败，使用默认设置: {cfg_exc}")

        self.use_frame_model = MS_AVAILABLE and bool(getattr(settings, 'USE_FRAME_MODEL', False)) if settings else False
        self.frame_image_size = (112, 112)
        self.frame_seq_len = 64
        if self.use_frame_model:
            self.window_length = self.frame_seq_len  # 统一窗口长度
            logger.info("✅ MindSpore 帧模型作为备用管线已启用")
        else:
            logger.info("ℹ️ MindSpore 帧模型未启用，Mind-VAC 将作为主要识别管线")
        self.frame_model = None
        self.frame_model_ready = False

        # Mind-VAC 引擎
        self.mind_vac_engine = MindVacEngine(settings)
        self.use_mind_vac = bool(getattr(self.mind_vac_engine, "enabled", False))
        if self.use_mind_vac:
            if getattr(self.mind_vac_engine, "available", False):
                logger.info("✅ Mind-VAC 连续手语识别已启用")
            else:
                logger.warning(f"⚠️ Mind-VAC 引擎尚未就绪: {self.mind_vac_engine.last_error}")
        else:
            logger.info("Mind-VAC 集成未启用")

        logger.info("SignRecognitionService 初始化完成")

    async def start_video_recognition(self, file_path: str) -> str:
        task_id = str(uuid.uuid4())
        async with self._lock:
            self.tasks[task_id] = {
                "status": "queued",
                "progress": 0.0,
                "file_path": file_path,
                "source": "video",
            }
        asyncio.create_task(self._process_task(task_id, file_path))
        return task_id

    async def create_frames_task(self, fps: float = 25.0) -> tuple[str, Path]:
        if not self.use_mind_vac:
            raise RuntimeError("Mind-VAC 管线未启用，无法处理帧序列")

        task_id = str(uuid.uuid4())
        frames_dir = self.mind_vac_engine.output_dir / task_id
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            self.tasks[task_id] = {
                "status": "queued",
                "progress": 0.0,
                "frames_dir": str(frames_dir),
                "fps": float(fps) if fps and fps > 0 else 25.0,
                "source": "frames",
            }

        return task_id, frames_dir

    async def start_frames_task(self, task_id: str) -> None:
        async with self._lock:
            task = self.tasks.get(task_id)
        if not task:
            raise RuntimeError(f"未找到任务 {task_id}")

        frames_dir = Path(task.get("frames_dir", ""))
        if not frames_dir.exists():
            raise RuntimeError("帧目录不存在或已被移除")

        fps = float(task.get("fps", 25.0) or 25.0)
        asyncio.create_task(self._process_frames_task(task_id, frames_dir, fps))

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self.tasks.get(task_id)

    async def _update_task(self, task_id: str, **kwargs):
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)

    async def _process_task(self, task_id: str, file_path: str):
        try:
            await self._update_task(task_id, status="processing", progress=0.01)
            result = await self._run_pipeline(task_id, file_path)
            # 保存结果 JSON
            result_path = os.path.join(self.result_dir, f"{task_id}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            await self._update_task(task_id, status="finished", progress=1.0, result=result.to_dict(), result_path=result_path)
        except Exception as e:
            logger.error(f"任务 {task_id} 处理失败: {e}")
            await self._update_task(task_id, status="error", progress=1.0, error=str(e))

    async def _process_frames_task(self, task_id: str, frames_dir: Path, fps: float):
        try:
            await self._update_task(task_id, status="processing", progress=0.02)
            result = await self._run_mind_vac_frameset_pipeline(task_id, frames_dir, fps)
            result_path = os.path.join(self.result_dir, f"{task_id}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            await self._update_task(
                task_id,
                status="finished",
                progress=1.0,
                result=result.to_dict(),
                result_path=result_path,
            )
        except Exception as e:
            logger.error(f"帧任务 {task_id} 处理失败: {e}")
            await self._update_task(task_id, status="error", progress=1.0, error=str(e))

    async def _run_pipeline(self, task_id: str, file_path: str) -> RecognitionResult:
        # Mind-VAC 是默认管线
        if self.use_mind_vac:
            try:
                logger.info("使用 Mind-VAC 管线进行连续手语识别")
                return await self._run_mind_vac_pipeline(task_id, file_path)
            except Exception as exc:
                detail = self.mind_vac_engine.last_error or str(exc)
                logger.error(f"Mind-VAC 管线处理失败: {detail}")
                if self.use_frame_model and MS_AVAILABLE:
                    logger.warning("Mind-VAC 失败，尝试 MindSpore 帧模型备用方案")
                    await self._ensure_frame_model_loaded()
                    return await self._run_frame_model_pipeline(task_id, file_path)
                raise RuntimeError(f"Mind-VAC 管线处理失败: {detail}") from exc

        # 若 Mind-VAC 未启用，仅当帧模型显式开启时才继续
        if self.use_frame_model and MS_AVAILABLE:
            logger.info("Mind-VAC 未启用，使用 MindSpore 帧模型进行推理")
            await self._ensure_frame_model_loaded()
            return await self._run_frame_model_pipeline(task_id, file_path)

        raise RuntimeError("Mind-VAC 管线未启用或初始化失败，请检查配置和依赖")

    async def _run_mind_vac_pipeline(self, task_id: str, file_path: str) -> RecognitionResult:
        if not self.mind_vac_engine or not self.use_mind_vac:
            raise RuntimeError("Mind-VAC 引擎未启用")
        if not getattr(self.mind_vac_engine, "available", False):
            raise RuntimeError(self.mind_vac_engine.last_error or "Mind-VAC 引擎未正确初始化")

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_rgb: List[np.ndarray] = []
        frame_index = 0

        try:
            await self._update_task(task_id, progress=0.05)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(rgb_frame)
                frame_index += 1

                if frame_index % 30 == 0:
                    progress = 0.05 + 0.4 * (frame_index / max(1, total_frames))
                    await self._update_task(task_id, progress=min(progress, 0.45))
        finally:
            cap.release()

        if not frames_rgb:
            raise RuntimeError("Mind-VAC 解析失败: 未读取到有效视频帧")

        duration = frame_index / fps if fps > 0 else 0.0

        # 将视频帧导出到 Mind-VAC 输出目录，确保与离线脚本兼容
        frame_output_dir = self.mind_vac_engine.output_dir / task_id
        if frame_output_dir.exists():
            shutil.rmtree(frame_output_dir, ignore_errors=True)
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(frames_rgb):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = frame_output_dir / f"{idx:05d}.png"
            cv2.imwrite(str(frame_path), frame_bgr)
            if (idx + 1) % 50 == 0:
                progress = 0.45 + 0.05 * ((idx + 1) / max(1, frame_index))
                await self._update_task(task_id, progress=min(progress, 0.5))

        await self._update_task(task_id, progress=0.5, frame_dir=str(frame_output_dir))

        inference_result = await self.mind_vac_engine.run_on_frames(frames_rgb)

        gloss_sequence = inference_result.get("gloss_sequence", []) or []
        raw_gloss_text = inference_result.get("raw_gloss_text", "") or ""
        llm_result = inference_result.get("llm_result")
        confidence = float(inference_result.get("confidence", 0.0) or 0.0)
        decoder_raw = inference_result.get("decoder_raw") or []

        logger.info("Mind-VAC 原始 gloss: %s", gloss_sequence if gloss_sequence else "<empty>")
        if llm_result and llm_result.get("success"):
            logger.info(
                "Mind-VAC LLM 输出: zh=%s en=%s",
                llm_result.get("chinese"),
                llm_result.get("english"),
            )
        elif llm_result and llm_result.get("error"):
            logger.warning("Mind-VAC LLM 失败: %s", llm_result.get("error"))

        await self._update_task(task_id, progress=0.8)

        # 文本后处理
        baseline_text = self._translate_gloss_to_text(gloss_sequence)
        translated_text = baseline_text
        if llm_result and llm_result.get("success"):
            translated_text = llm_result.get("chinese") or translated_text

        if not translated_text:
            translated_text = raw_gloss_text.replace(" ", "")
        if not baseline_text:
            baseline_text = raw_gloss_text.replace(" ", "")

        segments: List[RecognitionSegment] = []
        if gloss_sequence:
            segments.append(RecognitionSegment(
                gloss_sequence=gloss_sequence,
                start_frame=0,
                end_frame=max(frame_index - 1, 0),
                confidence=confidence,
                start_time=0.0,
                end_time=duration,
            ))

        srt_path = self._generate_srt(task_id, segments, translated_text)

        await self._update_task(task_id, progress=0.92)

        return RecognitionResult(
            task_id=task_id,
            file_path=file_path,
            gloss_sequence=gloss_sequence,
            text=translated_text,
            baseline_text=baseline_text,
            pipeline="mind_vac",
            segments=segments,
            overall_confidence=confidence,
            frame_count=frame_index,
            fps=fps,
            duration=duration,
            srt_path=srt_path,
            raw_gloss_text=raw_gloss_text,
            llm_result=llm_result,
            frames_dir=str(frame_output_dir),
            extra={
                "mind_vac": {
                    "decoder_raw": decoder_raw,
                    "frame_count": frame_index,
                    "fps": fps,
                }
            },
        )

    async def _run_mind_vac_frameset_pipeline(self, task_id: str, frames_dir: Path, fps: float) -> RecognitionResult:
        if not self.mind_vac_engine or not self.use_mind_vac:
            raise RuntimeError("Mind-VAC 引擎未启用")
        if not getattr(self.mind_vac_engine, "available", False):
            raise RuntimeError(self.mind_vac_engine.last_error or "Mind-VAC 引擎未正确初始化")

        await self._update_task(task_id, progress=0.1, frame_dir=str(frames_dir))

        frame_paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            frame_paths.extend(frames_dir.glob(pattern))
        unique_paths = {path: None for path in frame_paths}
        frame_paths = sorted(unique_paths.keys())
        if not frame_paths:
            raise RuntimeError("帧目录中未找到任何图像文件")

        frames_rgb: List[np.ndarray] = []
        for idx, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                logger.warning("跳过无法读取的帧: %s", frame_path)
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            if (idx + 1) % 50 == 0:
                progress = 0.1 + 0.3 * ((idx + 1) / max(1, len(frame_paths)))
                await self._update_task(task_id, progress=min(progress, 0.45))

        frame_count = len(frames_rgb)
        if frame_count == 0:
            raise RuntimeError("未成功读取任何有效帧")

        fps_value = fps if fps and fps > 0 else 25.0
        duration = frame_count / fps_value if fps_value > 0 else 0.0

        await self._update_task(task_id, progress=0.5)

        inference_result = await self.mind_vac_engine.run_on_frames(frames_rgb)

        gloss_sequence = inference_result.get("gloss_sequence", []) or []
        raw_gloss_text = inference_result.get("raw_gloss_text", "") or ""
        llm_result = inference_result.get("llm_result")
        confidence = float(inference_result.get("confidence", 0.0) or 0.0)
        decoder_raw = inference_result.get("decoder_raw") or []

        logger.info("Mind-VAC (frameset) 原始 gloss: %s", gloss_sequence if gloss_sequence else "<empty>")
        if llm_result and llm_result.get("success"):
            logger.info(
                "Mind-VAC LLM 输出: zh=%s en=%s",
                llm_result.get("chinese"),
                llm_result.get("english"),
            )
        elif llm_result and llm_result.get("error"):
            logger.warning("Mind-VAC LLM 失败: %s", llm_result.get("error"))

        await self._update_task(task_id, progress=0.8)

        baseline_text = self._translate_gloss_to_text(gloss_sequence)
        translated_text = baseline_text
        if llm_result and llm_result.get("success"):
            translated_text = llm_result.get("chinese") or translated_text

        if not translated_text:
            translated_text = raw_gloss_text.replace(" ", "")
        if not baseline_text:
            baseline_text = raw_gloss_text.replace(" ", "")

        segments: List[RecognitionSegment] = []
        if gloss_sequence:
            segments.append(RecognitionSegment(
                gloss_sequence=gloss_sequence,
                start_frame=0,
                end_frame=max(frame_count - 1, 0),
                confidence=confidence,
                start_time=0.0,
                end_time=duration,
            ))

        srt_path = self._generate_srt(task_id, segments, translated_text)

        await self._update_task(task_id, progress=0.92)

        return RecognitionResult(
            task_id=task_id,
            file_path=str(frames_dir),
            gloss_sequence=gloss_sequence,
            text=translated_text,
            baseline_text=baseline_text,
            pipeline="mind_vac_frameset",
            segments=segments,
            overall_confidence=confidence,
            frame_count=frame_count,
            fps=fps_value,
            duration=duration,
            srt_path=srt_path,
            raw_gloss_text=raw_gloss_text,
            llm_result=llm_result,
            frames_dir=str(frames_dir),
            extra={
                "mind_vac": {
                    "decoder_raw": decoder_raw,
                    "frame_count": frame_count,
                    "fps": fps_value,
                }
            },
        )

    async def _run_frame_model_pipeline(self, task_id: str, file_path: str) -> RecognitionResult:
        """使用 MindSpore 帧模型的推理流程"""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = int(max(1, round(fps / self.target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 读取并预处理为 (T,F) 序列
        frames_flat: List[np.ndarray] = []
        frame_indices: List[int] = []
        fid = 0
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fid % frame_interval == 0:
                flat = self._preprocess_frame_to_flat(frame)
                frames_flat.append(flat)
                frame_indices.append(fid)
                processed += 1
                if processed % 50 == 0:
                    await self._update_task(task_id, progress=min(0.8, 0.1 + 0.6 * processed / max(1, total_frames)))
            fid += 1
        cap.release()

        if not frames_flat:
            raise RuntimeError("未获取到有效帧")

        # 按窗口滑动并分类
        win = self.window_length
        step = int(win * (1 - self.window_overlap)) or 1
        T = len(frames_flat)
        idx = 0

        segments: List[RecognitionSegment] = []
        gloss_full: List[str] = []
        confidences: List[float] = []

        while idx < T:
            window_frames = frames_flat[idx: idx + win]
            if len(window_frames) < win:
                # 末尾不足则零填充
                pad = [np.zeros_like(window_frames[0])] * (win - len(window_frames))
                window_frames = window_frames + pad
            x = np.stack(window_frames, axis=0).astype(np.float32)  # (T,C,H,W)
            pred_label, prob = await self._predict_window_frames(x)

            # 忽略空白或仅空格的标签
            if pred_label and str(pred_label).strip():
                start_f = frame_indices[idx] if idx < len(frame_indices) else 0
                end_f = frame_indices[min(idx + win - 1, len(frame_indices) - 1)] if frame_indices else start_f
                segments.append(RecognitionSegment(
                    gloss_sequence=[pred_label],
                    start_frame=start_f,
                    end_frame=end_f,
                    confidence=float(prob),
                ))
                gloss_full.append(pred_label)
                confidences.append(float(prob))

            idx += step
            await self._update_task(task_id, progress=min(0.9, 0.8 + 0.1 * idx / max(1, T)))

        # 时间戳
        for seg in segments:
            seg.start_time = seg.start_frame / fps if fps > 0 else 0.0
            seg.end_time = seg.end_frame / fps if fps > 0 else seg.start_time

        # 合并相邻重复
        merged_gloss: List[str] = []
        for g in gloss_full:
            if not merged_gloss or merged_gloss[-1] != g:
                merged_gloss.append(g)

        text = self._translate_gloss_to_text(merged_gloss)
        overall_conf = float(np.mean(confidences)) if confidences else 0.0
        srt_path = self._generate_srt(task_id, segments, text)

        return RecognitionResult(
            task_id=task_id,
            file_path=file_path,
            gloss_sequence=merged_gloss,
            text=text,
            baseline_text=text,
            pipeline="frame_model",
            segments=segments,
            overall_confidence=overall_conf,
            frame_count=fid,
            fps=fps,
            duration=fid / fps if fps > 0 else 0.0,
            srt_path=srt_path,
        )

    async def _run_mediapipe_pipeline(self, task_id: str, file_path: str) -> RecognitionResult:
        """使用MediaPipe + CSLR的视频识别流程"""
        if self.mediapipe_service is None:
            raise RuntimeError("MediaPipe 服务未初始化")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = int(max(1, round(fps / self.target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        landmark_sequences: List[np.ndarray] = []
        frame_indices: List[int] = []

        frame_id = 0
        processed_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_interval == 0:
                    # 提取关键点
                    result = self.mediapipe_service.extract_landmarks(frame)
                    if result.get("success"):
                        # 使用标准化的关键点数据
                        landmarks_data = result["landmarks"]
                        normalized_landmarks = self.mediapipe_service.get_normalized_landmarks_for_cslr(landmarks_data)
                        landmark_sequences.append(normalized_landmarks)
                        frame_indices.append(frame_id)
                    else:
                        # 如果关键点提取失败，添加零向量
                        zero_landmarks = np.zeros(144, dtype=np.float32)
                        landmark_sequences.append(zero_landmarks)
                        frame_indices.append(frame_id)

                    processed_frames += 1
                    if processed_frames % 20 == 0:
                        progress = min(0.6, 0.1 + 0.5 * processed_frames / max(1, total_frames / frame_interval))
                        await self._update_task(task_id, progress=progress)

                frame_id += 1

        finally:
            cap.release()

        if not landmark_sequences:
            raise RuntimeError("未提取到任何有效的关键点数据")

        # 分窗口推理
        win = self.window_length
        step = int(win * (1 - self.window_overlap)) or 1
        segments: List[RecognitionSegment] = []
        gloss_full: List[str] = []
        confidences: List[float] = []
        T = len(landmark_sequences)
        idx = 0

        while idx < T:
            # 获取窗口序列
            end_idx = min(idx + win, T)
            window_landmarks = landmark_sequences[idx:end_idx]

            # 如果窗口不足，进行填充
            if len(window_landmarks) < win:
                padding_size = win - len(window_landmarks)
                zero_padding = [np.zeros(144, dtype=np.float32) for _ in range(padding_size)]
                window_landmarks.extend(zero_padding)

            # 转换为CSLR服务期望的格式
            window_data = [landmarks.tolist() for landmarks in window_landmarks]

            # 调用CSLR服务进行预测
            try:
                pred = await self.cslr_service.predict(window_data)
                if pred.status == "success" and pred.gloss_sequence:
                    start_frame = frame_indices[idx] if idx < len(frame_indices) else 0
                    end_frame = frame_indices[min(end_idx-1, len(frame_indices)-1)] if frame_indices else start_frame

                    segments.append(RecognitionSegment(
                        gloss_sequence=pred.gloss_sequence,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        confidence=pred.confidence,
                    ))
                    gloss_full.extend(pred.gloss_sequence)
                    confidences.append(pred.confidence)
            except Exception as e:
                logger.warning(f"窗口 {idx} 预测失败: {e}")

            idx += step
            progress = min(0.9, 0.6 + 0.3 * idx / T)
            await self._update_task(task_id, progress=progress)
        # 计算时间戳
        for seg in segments:
            seg.start_time = seg.start_frame / fps if fps > 0 else 0.0
            seg.end_time = seg.end_frame / fps if fps > 0 else seg.start_time

        # 合并重复 (简单相邻去重)
        merged_gloss: List[str] = []
        for g in gloss_full:
            if not merged_gloss or merged_gloss[-1] != g:
                merged_gloss.append(g)

        text = self._translate_gloss_to_text(merged_gloss)
        overall_conf = float(np.mean(confidences)) if confidences else 0.0
        # 生成 SRT
        srt_path = self._generate_srt(task_id, segments, text)
        return RecognitionResult(
            task_id=task_id,
            file_path=file_path,
            gloss_sequence=merged_gloss,
            text=text,
            baseline_text=text,
            pipeline="mediapipe",
            segments=segments,
            overall_confidence=overall_conf,
            frame_count=frame_id,
            fps=fps,
            duration=frame_id / fps if fps > 0 else 0.0,
            srt_path=srt_path,
        )

    def _load_or_create_gloss_dict(self) -> Dict[str, str]:
        if os.path.exists(self.gloss_dict_path):
            try:
                with open(self.gloss_dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        sample = {
            "我": "我", "你": "你", "他": "他", "她": "她",
            "学习": "学习", "工作": "工作", "医院": "医院", "学校": "学校",
            "谢谢": "谢谢", "你好": "你好", "再见": "再见", "今天": "今天",
            "昨天": "昨天", "明天": "明天", "想": "想", "去": "去", "吃": "吃",
            "喝": "喝", "家": "家", "是": "是", "不是": "不是"
        }
        try:
            with open(self.gloss_dict_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return sample

    def _translate_gloss_to_text(self, gloss_seq: List[str]) -> str:
        if not gloss_seq:
            return ""
        mapped = [self.gloss_dict.get(g, g) for g in gloss_seq]
        # 去除连续重复
        cleaned = []
        for g in mapped:
            if not cleaned or cleaned[-1] != g:
                cleaned.append(g)
        # 分句规则增强
        pronouns = {"我", "你", "他", "她"}
        verbs = {"学习", "工作", "吃", "喝", "去", "想", "睡觉"}
        time_words = {"今天", "昨天", "明天"}
        question_words = {"吗", "请问", "什么", "怎么", "为什么", "谁", "哪儿", "哪里"}
        exclam_words = {"啊", "呀", "哇", "太棒了", "太好了"}
        logic_words = {"因为", "所以", "但是", "如果", "然后"}
        polite_words = {"谢谢", "再见"}
        result = []
        sentence = []
        for i, word in enumerate(cleaned):
            sentence.append(word)
            # 分句条件：遇到时间词、逻辑词、礼貌词、问句/感叹词、主语后动宾结构
            if word in time_words or word in logic_words or word in polite_words:
                result.append(''.join(sentence))
                sentence = []
            elif word in question_words or (i+1 < len(cleaned) and cleaned[i+1] in question_words):
                result.append(''.join(sentence))
                sentence = []
            elif word in exclam_words or (i+1 < len(cleaned) and cleaned[i+1] in exclam_words):
                result.append(''.join(sentence))
                sentence = []
            elif word in pronouns and i+1 < len(cleaned) and cleaned[i+1] in verbs:
                # 主语+动词后如有宾语或时间词，分句
                if i+2 < len(cleaned) and (cleaned[i+2] not in pronouns and cleaned[i+2] not in verbs):
                    sentence.append(cleaned[i+2])
                    result.append(''.join(sentence))
                    sentence = []
        if sentence:
            result.append(''.join(sentence))
        # 标点插入
        sentences = []
        for s in result:
            if any(q in s for q in question_words):
                sentences.append(s + '？')
            elif any(e in s for e in exclam_words):
                sentences.append(s + '！')
            elif any(p in s for p in polite_words):
                sentences.append(s + '。')
            else:
                sentences.append(s + '。')
        text = ''.join(sentences)
        # 进一步去除多余标点
        text = text.replace('。。', '。').replace('！！', '！').replace('？？', '？')
        return text

    def _generate_srt(self, task_id: str, segments: List[RecognitionSegment], full_text: str) -> Optional[str]:
        if not segments:
            return None
        def format_ts(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start_ts = format_ts(seg.start_time)
            end_ts = format_ts(max(seg.end_time, seg.start_time + 0.04))
            line_text = "".join(seg.gloss_sequence)
            srt_lines.append(str(i))
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.append(line_text)
            srt_lines.append("")
        # 添加一条合并总句
        srt_lines.append(str(len(segments) + 1))
        srt_lines.append("00:00:00,000 --> 00:00:59,999")  # 粗略覆盖前一分钟
        srt_lines.append(full_text)
        srt_lines.append("")
        path = os.path.join(self.result_dir, f"{task_id}.srt")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("\n".join(srt_lines))
            return path
        except Exception as e:
            logger.error(f"写入SRT失败: {e}")
            return None

    async def get_status(self, task_id: str) -> Dict[str, Any]:
        task = await self.get_task(task_id)
        if not task:
            return {"status": "not_found"}
        data = {k: v for k, v in task.items() if k != "result"}
        return data

    async def get_result(self, task_id: str) -> Dict[str, Any]:
        task = await self.get_task(task_id)
        if not task:
            return {"status": "not_found"}
        if task.get("status") != "finished":
            return {"status": task.get("status"), "progress": task.get("progress")}
        return {"status": "finished", "result": task.get("result")}

    async def cleanup(self):
        # 未来可清理过期任务文件
        pass

    # 新增：帧模型工具函数
    async def _ensure_frame_model_loaded(self):
        if self.frame_model_ready:
            return
        # 复用 CSLR 的词表作为 idx 映射
        vocab_size = len(getattr(self.cslr_service, 'vocab', {}) or {})
        if vocab_size <= 0:
            try:
                if hasattr(self.cslr_service, 'load_model'):
                    await self.cslr_service.load_model()
                elif hasattr(self.cslr_service, '_load_vocabulary'):
                    await self.cslr_service._load_vocabulary()  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError(f"词表加载失败: {e}")
            vocab_size = len(getattr(self.cslr_service, 'vocab', {}) or {})
            if vocab_size <= 0:
                raise RuntimeError("词表未加载或为空")
        
        # 创建配置管理器并获取配置
        config_manager = ConfigManager()
        config = config_manager.config
        hidden_size = config.get('model', {}).get('hidden_size', 1024)
        
        try:
            # 设备设置（尽量兼容）
            if hasattr(ms, 'set_device'):
                try:
                    ms.set_device('CPU')
                except Exception:
                    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
            else:
                ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
        except Exception:
            pass
        # 构建网络并加载 ckpt
        self.frame_model = TFNetModel(hidden_size=hidden_size, word_set_num=vocab_size, device_target="CPU")
        ckpt_path = getattr(self.cslr_service.config, 'model_path', None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise RuntimeError(f"模型权重不存在: {ckpt_path}")
        params = load_checkpoint(ckpt_path)
        # 尝试从权重中推断 hidden_size，必要时重建网络以避免维度不匹配
        try:
            w = params.get('conv1d.temporal_conv.0.weight')
            if w is not None and hasattr(w, 'data') and hasattr(w.data, 'shape'):
                inferred_hidden = int(w.data.shape[0])
                if inferred_hidden > 0 and inferred_hidden != hidden_size:
                    hidden_size = inferred_hidden
                    self.frame_model = TFNetModel(hidden_size=hidden_size, word_set_num=vocab_size, device_target="CPU")
        except Exception:
            pass
        # 统计匹配度
        try:
            net_params = {p.name: p for p in self.frame_model.get_parameters()}
            total = len(net_params)
            matched = 0
            mismatched_shapes = []
            for name, tensor in params.items():
                if name in net_params:
                    try:
                        if tuple(net_params[name].shape) == tuple(tensor.data.shape):
                            matched += 1
                        else:
                            mismatched_shapes.append((name, tuple(net_params[name].shape), tuple(tensor.data.shape)))
                
                    except Exception:
                        pass
            load_param_into_net(self.frame_model, params)
            cover = (matched / max(1, total)) * 100.0
            if cover < 95.0:
                logger.warning(f"模型权重加载覆盖率偏低: {cover:.1f}% (匹配 {matched}/{total})")
                if mismatched_shapes:
                    head_mismatch = [n for n, s1, s2 in mismatched_shapes if 'classifier' in n]
                    if head_mismatch:
                        logger.warning(f"分类头权重维度不匹配，可能导致输出集中在少数类别: {head_mismatch[:3]} ...")
        except Exception:
            load_param_into_net(self.frame_model, params)
        self.frame_model.set_train(False)
        self.frame_model_ready = True
        logger.info("帧模型已加载并就绪")

    def _preprocess_frame_to_flat(self, frame: np.ndarray) -> np.ndarray:
         # BGR->RGB，缩放到 image_size，归一化到[0,1]，转为 (C,H,W) 再展平成 (F,)
         try:
             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         except Exception:
             img = frame
         img = cv2.resize(img, self.frame_image_size, interpolation=cv2.INTER_AREA)
         img = img.astype(np.float32)
         if img.max() > 1.0:
             img = img / 255.0
         # (H,W,C) -> (C,H,W)
         chw = np.transpose(img, (2, 0, 1))
         return chw

    async def _predict_window_frames(self, x_tf: np.ndarray) -> Tuple[str, float]:
         # x_tf: (T,C,H,W) -> (1,T,C,H,W)
         if not self.frame_model_ready:
             await self._ensure_frame_model_loaded()
         x = x_tf[None, ...]
         out = self.frame_model(Tensor(x, ms.float32))
         # 兼容返回元组的模型输出，取融合分类头 logits（index 4），否则取第一个
         if isinstance(out, (tuple, list)):
             logits = out[4] if len(out) > 4 else out[0]
         else:
             logits = out
         arr = logits.asnumpy()
         # 统一为 (classes,) 概率向量：对时间/批次维做平均
         if arr.ndim == 3:  # (T, B, C)
             vec = arr.mean(axis=(0, 1))
         elif arr.ndim == 2:  # (B, C)
             vec = arr.mean(axis=0)
         elif arr.ndim == 1:  # (C,)
             vec = arr
         else:
             # 回退：展平到 (C,)
             vec = arr.reshape(-1)
         vec = vec.astype(np.float64)
         # softmax 归一化
         vec = np.exp(vec - np.max(vec))
         denom = float(np.sum(vec)) if float(np.sum(vec)) > 0 else 1.0
         probs = vec / denom
         idx = int(np.argmax(probs))
         conf = float(probs[idx])
         # idx->词：优先使用 idx2word，回退 reverse_vocab
         idx2word = getattr(self.cslr_service, 'idx2word', None)
         if isinstance(idx2word, list) and idx < len(idx2word):
             label = idx2word[idx]
         else:
             label = self.cslr_service.reverse_vocab.get(idx, "")
         return label, conf
