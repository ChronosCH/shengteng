"""
Enhanced Sign Language Processing Service
集成TFNet手语识别和Diffusion手语生成的综合服务
基于华为昇腾AI处理器优化
"""

import asyncio
import logging
import time
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import mindspore as ms
    import mindspore.context as ms_context
    from mindspore import Tensor, load_checkpoint, load_param_into_net
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    logging.warning("MindSpore not available, using mock implementation")

# 导入TFNet相关模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "training"))

try:
    from tfnet_mindspore import TFNetMindSpore
    from tfnet_decoder import CTCDecoder, TFNetEvaluator
    from cecsl_data_processor import CECSLLabelProcessor
    TFNET_AVAILABLE = True
except ImportError as e:
    TFNET_AVAILABLE = False
    logging.warning(f"TFNet modules not available: {e}")

from utils.config import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class EmotionType(Enum):
    """情绪类型枚举"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    EXCITED = "excited"


class SigningSpeed(Enum):
    """手语速度枚举"""
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


@dataclass
class DiffusionConfig:
    """Diffusion 生成配置"""
    emotion: EmotionType = EmotionType.NEUTRAL
    speed: SigningSpeed = SigningSpeed.NORMAL
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    max_sequence_length: int = 200
    fps: int = 30


@dataclass
class SignSequence:
    """手语序列数据结构"""
    keypoints: np.ndarray  # Shape: (T, 543, 3) - 时间步 x 关键点 x 坐标
    timestamps: np.ndarray  # Shape: (T,) - 每帧时间戳
    confidence: float  # 生成置信度
    emotion: EmotionType
    speed: SigningSpeed
    text: str  # 原始文本
    duration: float  # 总时长(秒)


class DiffusionSLPService:
    """Diffusion 手语生成服务"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device_type = "cpu"  # 或 "ascend"
        
        # 统计信息
        self.stats = {
            "total_recognitions": 0,
            "total_generations": 0,
            "recognition_time": 0.0,
            "generation_time": 0.0,
            "average_recognition_time": 0.0,
            "average_generation_time": 0.0,
            "cache_hits": 0
        }
        
        # 生成缓存
        self.generation_cache = {}
        self.max_cache_size = 100
        
        # TFNet手语识别相关
        self.tfnet_model = None
        self.decoder = None
        self.evaluator = None
        
    async def initialize(self):
        """初始化服务"""
        try:
            logger.info("正在初始化 Enhanced Sign Language Processing 服务...")
            
            if MINDSPORE_AVAILABLE:
                await self._load_mindspore_model()
                if TFNET_AVAILABLE:
                    await self._load_tfnet_model()
            else:
                await self._load_mock_model()
                
            self.is_loaded = True
            logger.info("Enhanced Sign Language Processing 服务初始化完成")
            
        except Exception as e:
            logger.error(f"Enhanced Sign Language Processing 服务初始化失败: {e}")
            raise
    
    async def _load_mindspore_model(self):
        """加载 MindSpore Diffusion 模型"""
        try:
            # 设置MindSpore上下文
            if getattr(settings, 'USE_ASCEND', False):
                ms_context.set_context(mode=ms_context.GRAPH_MODE, device_target="Ascend")
                self.device_type = "ascend"
            else:
                ms_context.set_context(mode=ms_context.GRAPH_MODE, device_target="CPU")
                self.device_type = "cpu"
                
            # 对于开发环境，使用模拟实现
            logger.info("使用模拟推理模式（开发环境）")
            self.model = None  # 模拟模型
            
            # 加载文本编码器
            self._load_text_encoder()
            
            logger.info(f"MindSpore Diffusion 推理环境初始化成功 (设备: {self.device_type})")
            
        except Exception as e:
            logger.error(f"MindSpore Diffusion 模型加载失败: {e}")
            # 降级到模拟模式
            self.model = None
            self.device_type = "cpu"
            logger.info("降级到模拟推理模式")
    
    def _load_text_encoder(self):
        """加载文本编码器"""
        # 简化的文本编码器实现
        # 实际项目中应该使用预训练的 BERT 或类似模型
        self.vocab = {
            "你好": 0, "谢谢": 1, "再见": 2, "请": 3, "对不起": 4,
            "是": 5, "不是": 6, "好": 7, "不好": 8, "我": 9,
            "你": 10, "他": 11, "她": 12, "我们": 13, "你们": 14,
            "他们": 15, "什么": 16, "哪里": 17, "什么时候": 18, "为什么": 19,
            "怎么": 20, "多少": 21, "喜欢": 22, "不喜欢": 23, "想要": 24,
            "需要": 25, "可以": 26, "不可以": 27, "会": 28, "不会": 29,
            "有": 30, "没有": 31, "大": 32, "小": 33, "高": 34,
            "矮": 35, "快": 36, "慢": 37, "新": 38, "旧": 39,
            "热": 40, "冷": 41, "饿": 42, "渴": 43, "累": 44,
            "开心": 45, "难过": 46, "生气": 47, "害怕": 48, "惊讶": 49
        }
        self.max_vocab_size = len(self.vocab)
        
    async def _load_mock_model(self):
        """加载模拟模型 (用于开发测试)"""
        logger.warning("使用模拟 Diffusion SLP 模型")
        self.model = "mock_diffusion_model"
        self._load_text_encoder()
    
    async def generate_sign_sequence(self, text: str, config: DiffusionConfig = None) -> SignSequence:
        """
        从文本生成手语序列
        
        Args:
            text: 输入文本
            config: 生成配置
            
        Returns:
            生成的手语序列
        """
        if not self.is_loaded:
            raise RuntimeError("Diffusion SLP 服务未初始化")
            
        if config is None:
            config = DiffusionConfig()
            
        start_time = time.time()
        self.generation_stats["total_requests"] += 1
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(text, config)
            if cache_key in self.generation_cache:
                self.generation_stats["cache_hits"] += 1
                logger.info(f"从缓存返回结果: {text}")
                return self.generation_cache[cache_key]
            
            # 文本编码
            text_embedding = self._encode_text(text)
            
            # 情绪和速度编码
            emotion_embedding = self._encode_emotion(config.emotion)
            speed_embedding = self._encode_speed(config.speed)
            
            # Diffusion 生成
            if MINDSPORE_AVAILABLE and self.model != "mock_diffusion_model":
                keypoints = await self._diffusion_inference(
                    text_embedding, emotion_embedding, speed_embedding, config
                )
            else:
                keypoints = await self._mock_diffusion_inference(
                    text_embedding, emotion_embedding, speed_embedding, config
                )
            
            # 生成时间戳
            num_frames = keypoints.shape[0]
            duration = num_frames / config.fps
            timestamps = np.linspace(0, duration, num_frames)
            
            # 创建结果
            result = SignSequence(
                keypoints=keypoints,
                timestamps=timestamps,
                confidence=0.85,  # 模拟置信度
                emotion=config.emotion,
                speed=config.speed,
                text=text,
                duration=duration
            )
            
            # 更新缓存
            self._update_cache(cache_key, result)
            
            # 更新统计
            generation_time = time.time() - start_time
            self._update_stats(generation_time, success=True)
            
            logger.info(f"成功生成手语序列: {text} (耗时: {generation_time:.3f}s)")
            return result
            
        except Exception as e:
            self._update_stats(time.time() - start_time, success=False)
            logger.error(f"手语序列生成失败: {e}")
            raise
    
    def _encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        # 简化的文本编码实现
        words = text.split()
        encoded = []
        
        for word in words:
            if word in self.vocab:
                encoded.append(self.vocab[word])
            else:
                encoded.append(0)  # UNK token
                
        # 填充或截断到固定长度
        max_length = 50
        if len(encoded) < max_length:
            encoded.extend([0] * (max_length - len(encoded)))
        else:
            encoded = encoded[:max_length]
            
        return np.array(encoded, dtype=np.float32)
    
    def _encode_emotion(self, emotion: EmotionType) -> np.ndarray:
        """编码情绪为向量"""
        emotion_map = {
            EmotionType.NEUTRAL: [1, 0, 0, 0, 0, 0],
            EmotionType.HAPPY: [0, 1, 0, 0, 0, 0],
            EmotionType.SAD: [0, 0, 1, 0, 0, 0],
            EmotionType.ANGRY: [0, 0, 0, 1, 0, 0],
            EmotionType.SURPRISED: [0, 0, 0, 0, 1, 0],
            EmotionType.EXCITED: [0, 0, 0, 0, 0, 1],
        }
        return np.array(emotion_map[emotion], dtype=np.float32)
    
    def _encode_speed(self, speed: SigningSpeed) -> np.ndarray:
        """编码速度为向量"""
        speed_map = {
            SigningSpeed.SLOW: [1, 0, 0],
            SigningSpeed.NORMAL: [0, 1, 0],
            SigningSpeed.FAST: [0, 0, 1],
        }
        return np.array(speed_map[speed], dtype=np.float32)

    async def _diffusion_inference(self, text_emb: np.ndarray, emotion_emb: np.ndarray,
                                 speed_emb: np.ndarray, config: DiffusionConfig) -> np.ndarray:
        """使用 MindSpore 进行 Diffusion 推理"""
        try:
            # 准备输入
            batch_size = 1
            sequence_length = config.max_sequence_length
            num_keypoints = 543

            # 初始噪声
            if config.seed is not None:
                np.random.seed(config.seed)
            noise = np.random.randn(batch_size, sequence_length, num_keypoints, 3).astype(np.float32)

            # 条件向量
            condition = np.concatenate([text_emb, emotion_emb, speed_emb])
            condition = np.expand_dims(condition, 0)  # 添加 batch 维度

            # Diffusion 去噪过程
            x = noise
            for step in range(config.num_inference_steps):
                # 时间步编码
                t = np.array([step / config.num_inference_steps], dtype=np.float32)
                t = np.expand_dims(t, 0)

                # 模型推理
                inputs = self.model.get_inputs()
                inputs[0].set_data_from_numpy(x)
                inputs[1].set_data_from_numpy(condition)
                inputs[2].set_data_from_numpy(t)

                self.model.predict(inputs)

                outputs = self.model.get_outputs()
                noise_pred = outputs[0].get_data_to_numpy()

                # DDPM 更新步骤
                alpha = 1.0 - step / config.num_inference_steps
                x = x - alpha * noise_pred

            # 后处理
            keypoints = self._post_process_keypoints(x[0], config)
            return keypoints

        except Exception as e:
            logger.error(f"Diffusion 推理失败: {e}")
            raise

    async def _mock_diffusion_inference(self, text_emb: np.ndarray, emotion_emb: np.ndarray,
                                      speed_emb: np.ndarray, config: DiffusionConfig) -> np.ndarray:
        """模拟 Diffusion 推理"""
        # 模拟推理延迟
        await asyncio.sleep(0.1)

        # 根据文本长度和速度计算序列长度
        text_length = np.sum(text_emb > 0)  # 非零元素个数
        speed_factor = {
            SigningSpeed.SLOW: 1.5,
            SigningSpeed.NORMAL: 1.0,
            SigningSpeed.FAST: 0.7
        }[config.speed]

        sequence_length = int(text_length * 10 * speed_factor)  # 每个词约10帧
        sequence_length = min(sequence_length, config.max_sequence_length)
        sequence_length = max(sequence_length, 30)  # 最少30帧

        # 生成模拟关键点数据
        keypoints = self._generate_mock_keypoints(sequence_length, config)

        return keypoints

    def _generate_mock_keypoints(self, sequence_length: int, config: DiffusionConfig) -> np.ndarray:
        """生成模拟关键点数据"""
        num_keypoints = 543  # MediaPipe Holistic 总关键点数

        # 基础姿态
        base_pose = np.zeros((num_keypoints, 3))

        # 设置基础身体姿态
        # 面部关键点 (0-467)
        face_points = 468
        base_pose[:face_points, :] = self._get_base_face_pose()

        # 左手关键点 (468-488)
        left_hand_start = 468
        left_hand_end = 489
        base_pose[left_hand_start:left_hand_end, :] = self._get_base_hand_pose("left")

        # 右手关键点 (489-509)
        right_hand_start = 489
        right_hand_end = 510
        base_pose[right_hand_start:right_hand_end, :] = self._get_base_hand_pose("right")

        # 身体关键点 (510-542)
        body_start = 510
        base_pose[body_start:, :] = self._get_base_body_pose()

        # 生成动画序列
        keypoints_sequence = np.zeros((sequence_length, num_keypoints, 3))

        for t in range(sequence_length):
            # 添加时间变化
            time_factor = t / sequence_length

            # 情绪影响
            emotion_factor = self._get_emotion_factor(config.emotion, time_factor)

            # 速度影响
            speed_factor = self._get_speed_factor(config.speed, time_factor)

            # 生成当前帧
            current_pose = base_pose.copy()

            # 添加手部动作
            current_pose = self._add_hand_animation(current_pose, time_factor, emotion_factor)

            # 添加面部表情
            current_pose = self._add_facial_expression(current_pose, emotion_factor)

            # 添加身体动作
            current_pose = self._add_body_movement(current_pose, time_factor, speed_factor)

            keypoints_sequence[t] = current_pose

        return keypoints_sequence

    def _get_base_face_pose(self) -> np.ndarray:
        """获取基础面部姿态"""
        # 简化的面部关键点，实际应该使用真实的面部模型
        face_points = np.random.normal(0, 0.1, (468, 3))
        face_points[:, 2] = 0  # Z坐标设为0（正面）
        return face_points

    def _get_base_hand_pose(self, hand: str) -> np.ndarray:
        """获取基础手部姿态"""
        # 21个手部关键点
        hand_points = np.zeros((21, 3))

        # 设置基础手形（自然下垂）
        if hand == "left":
            hand_points[:, 0] = -0.3  # X坐标（左侧）
        else:
            hand_points[:, 0] = 0.3   # X坐标（右侧）

        hand_points[:, 1] = 0.0   # Y坐标（中间）
        hand_points[:, 2] = 0.0   # Z坐标（正面）

        return hand_points

    def _get_base_body_pose(self) -> np.ndarray:
        """获取基础身体姿态"""
        # 33个身体关键点
        body_points = np.zeros((33, 3))

        # 设置基础身体姿态（直立）
        body_points[:, 1] = np.linspace(1.0, -1.0, 33)  # Y坐标从头到脚

        return body_points

    def _get_emotion_factor(self, emotion: EmotionType, time_factor: float) -> float:
        """获取情绪影响因子"""
        emotion_intensity = {
            EmotionType.NEUTRAL: 0.0,
            EmotionType.HAPPY: 0.3,
            EmotionType.SAD: -0.2,
            EmotionType.ANGRY: 0.5,
            EmotionType.SURPRISED: 0.4,
            EmotionType.EXCITED: 0.6,
        }

        base_intensity = emotion_intensity[emotion]
        # 添加时间变化
        return base_intensity * (1.0 + 0.2 * np.sin(time_factor * 2 * np.pi))

    def _get_speed_factor(self, speed: SigningSpeed, time_factor: float) -> float:
        """获取速度影响因子"""
        speed_multiplier = {
            SigningSpeed.SLOW: 0.7,
            SigningSpeed.NORMAL: 1.0,
            SigningSpeed.FAST: 1.4,
        }
        return speed_multiplier[speed]

    def _add_hand_animation(self, pose: np.ndarray, time_factor: float, emotion_factor: float) -> np.ndarray:
        """添加手部动画"""
        # 左手关键点范围
        left_hand_start, left_hand_end = 468, 489
        right_hand_start, right_hand_end = 489, 510

        # 生成手部动作
        wave_amplitude = 0.1 + abs(emotion_factor) * 0.1
        wave_frequency = 2.0 + emotion_factor

        # 左手动作
        pose[left_hand_start:left_hand_end, 1] += wave_amplitude * np.sin(time_factor * wave_frequency * 2 * np.pi)
        pose[left_hand_start:left_hand_end, 0] += wave_amplitude * 0.5 * np.cos(time_factor * wave_frequency * 2 * np.pi)

        # 右手动作
        pose[right_hand_start:right_hand_end, 1] += wave_amplitude * np.sin(time_factor * wave_frequency * 2 * np.pi + np.pi/2)
        pose[right_hand_start:right_hand_end, 0] += wave_amplitude * 0.5 * np.cos(time_factor * wave_frequency * 2 * np.pi + np.pi/2)

        return pose

    def _add_facial_expression(self, pose: np.ndarray, emotion_factor: float) -> np.ndarray:
        """添加面部表情"""
        # 面部关键点范围 (0-467)
        face_end = 468

        # 根据情绪调整面部表情
        if emotion_factor > 0:  # 积极情绪
            # 嘴角上扬
            pose[61:68, 1] += emotion_factor * 0.02  # 上嘴唇
            pose[291:298, 1] += emotion_factor * 0.02  # 下嘴唇
        elif emotion_factor < 0:  # 消极情绪
            # 嘴角下垂
            pose[61:68, 1] += emotion_factor * 0.02
            pose[291:298, 1] += emotion_factor * 0.02

        return pose

    def _add_body_movement(self, pose: np.ndarray, time_factor: float, speed_factor: float) -> np.ndarray:
        """添加身体动作"""
        # 身体关键点范围 (510-542)
        body_start = 510

        # 添加轻微的身体摆动
        sway_amplitude = 0.02 * speed_factor
        sway_frequency = 1.0 * speed_factor

        pose[body_start:, 0] += sway_amplitude * np.sin(time_factor * sway_frequency * 2 * np.pi)

        return pose

    def _post_process_keypoints(self, keypoints: np.ndarray, config: DiffusionConfig) -> np.ndarray:
        """后处理关键点数据"""
        # 平滑处理
        keypoints = self._smooth_keypoints(keypoints)

        # 归一化坐标
        keypoints = self._normalize_keypoints(keypoints)

        # 确保关键点在合理范围内
        keypoints = np.clip(keypoints, -1.0, 1.0)

        return keypoints

    def _smooth_keypoints(self, keypoints: np.ndarray, window_size: int = 5) -> np.ndarray:
        """平滑关键点序列"""
        if keypoints.shape[0] < window_size:
            return keypoints

        # 简单的移动平均平滑
        smoothed = keypoints.copy()
        half_window = window_size // 2

        for i in range(half_window, keypoints.shape[0] - half_window):
            smoothed[i] = np.mean(keypoints[i-half_window:i+half_window+1], axis=0)

        return smoothed

    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """归一化关键点坐标"""
        # 以身体中心为原点进行归一化
        body_center_idx = 510 + 11  # 身体中心点索引

        if keypoints.shape[1] > body_center_idx:
            center = keypoints[:, body_center_idx:body_center_idx+1, :]
            keypoints = keypoints - center

        return keypoints

    def _get_cache_key(self, text: str, config: DiffusionConfig) -> str:
        """生成缓存键"""
        return f"{text}_{config.emotion.value}_{config.speed.value}_{config.seed}"

    def _update_cache(self, key: str, result: SignSequence):
        """更新缓存"""
        if len(self.generation_cache) >= self.max_cache_size:
            # 删除最旧的条目
            oldest_key = next(iter(self.generation_cache))
            del self.generation_cache[oldest_key]

        self.generation_cache[key] = result

    def _update_stats(self, generation_time: float, success: bool):
        """更新统计信息"""
        if success:
            self.generation_stats["successful_generations"] += 1

        # 更新平均生成时间
        total_successful = self.generation_stats["successful_generations"]
        if total_successful > 0:
            current_avg = self.generation_stats["average_generation_time"]
            new_avg = (current_avg * (total_successful - 1) + generation_time) / total_successful
            self.generation_stats["average_generation_time"] = new_avg

    async def get_stats(self) -> Dict:
        """获取服务统计信息"""
        return {
            **self.generation_stats,
            "cache_size": len(self.generation_cache),
            "is_loaded": self.is_loaded,
            "device_type": self.device_type
        }

    async def clear_cache(self):
        """清空缓存"""
        self.generation_cache.clear()
        logger.info("Diffusion SLP 缓存已清空")

    async def cleanup(self):
        """清理资源"""
        try:
            if self.model and MINDSPORE_AVAILABLE:
                # MindSpore Lite 模型清理
                pass

            self.generation_cache.clear()
            
            # 清理TFNet模型
            if self.tfnet_model:
                self.tfnet_model = None
            
            logger.info("Enhanced Sign Language Processing 服务清理完成")
            
        except Exception as e:
            logger.error(f"服务清理失败: {e}")

    async def _load_tfnet_model(self):
        """加载TFNet手语识别模型"""
        try:
            logger.info("正在加载TFNet手语识别模型...")
            
            # 模型配置
            tfnet_config = {
                "hidden_size": 512,
                "vocab_size": 1000,  # 将在加载词汇表后更新
                "module_choice": "TFNet",
                "dataset_name": "CE-CSL"
            }
            
            # 加载词汇表
            vocab_file = os.path.join(settings.MODEL_DIR, "vocab.json")
            if os.path.exists(vocab_file):
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                tfnet_config["vocab_size"] = vocab_data.get("vocab_size", 1000)
                
                # 初始化解码器
                self.decoder = CTCDecoder(vocab_file, blank_id=0)
                self.evaluator = TFNetEvaluator(vocab_file, blank_id=0)
            else:
                logger.warning(f"词汇表文件不存在: {vocab_file}")
                return
            
            # 创建模型
            self.tfnet_model = TFNetMindSpore(
                hidden_size=tfnet_config["hidden_size"],
                vocab_size=tfnet_config["vocab_size"],
                module_choice=tfnet_config["module_choice"],
                dataset_name=tfnet_config["dataset_name"]
            )
            
            # 加载预训练权重
            checkpoint_path = os.path.join(settings.MODEL_DIR, "tfnet_cecsl_best.ckpt")
            if os.path.exists(checkpoint_path):
                param_dict = load_checkpoint(checkpoint_path)
                load_param_into_net(self.tfnet_model, param_dict)
                logger.info(f"成功加载TFNet预训练模型: {checkpoint_path}")
            else:
                logger.warning(f"TFNet预训练模型不存在: {checkpoint_path}")
                logger.info("将使用随机初始化的模型")
            
            # 设置为推理模式
            self.tfnet_model.set_train(False)
            
            logger.info("TFNet手语识别模型加载完成")
            
        except Exception as e:
            logger.error(f"TFNet模型加载失败: {e}")
            self.tfnet_model = None
            self.decoder = None
            self.evaluator = None

    async def recognize_sign_language(self, video_frames: np.ndarray, 
                                    frame_rate: int = 25) -> Dict:
        """
        手语识别
        Args:
            video_frames: 视频帧数组 (seq_len, height, width, channels)
            frame_rate: 帧率
        Returns:
            识别结果
        """
        if not self.is_loaded or not self.tfnet_model:
            return {
                "success": False,
                "error": "TFNet模型未加载",
                "recognized_sequence": [],
                "confidence": 0.0
            }
        
        start_time = time.time()
        
        try:
            # 预处理视频帧
            processed_frames = await self._preprocess_video_frames(video_frames)
            
            # 模型推理
            with ms.no_grad():
                # 添加batch维度
                input_tensor = Tensor(processed_frames[np.newaxis, :], ms.float32)
                video_length = Tensor([len(processed_frames)], ms.int32)
                
                # 前向传播
                logits1, logits2, logits3, logits4, _, _, _, _ = self.tfnet_model(
                    input_tensor, video_length, is_train=False
                )
                
                # 使用第一个分支的输出进行解码
                log_probs = ms.ops.log_softmax(logits1, axis=-1)
                log_probs_np = log_probs.asnumpy()
            
            # CTC解码
            decoded_sequences = self.decoder.greedy_decode(
                log_probs_np, np.array([len(processed_frames)])
            )
            
            # 转换为词汇
            word_sequences = self.decoder.decode_to_words(decoded_sequences)
            recognized_words = word_sequences[0] if word_sequences else []
            
            # 计算置信度（简化版）
            confidence = float(np.exp(np.mean(log_probs_np)))
            
            # 更新统计信息
            recognition_time = time.time() - start_time
            self.stats["total_recognitions"] += 1
            self.stats["recognition_time"] += recognition_time
            self.stats["average_recognition_time"] = (
                self.stats["recognition_time"] / self.stats["total_recognitions"]
            )
            
            result = {
                "success": True,
                "recognized_sequence": recognized_words,
                "recognized_sentence": " ".join(recognized_words) if recognized_words else "",
                "confidence": confidence,
                "processing_time": recognition_time,
                "frame_count": len(video_frames),
                "frame_rate": frame_rate
            }
            
            logger.info(f"手语识别完成: {result['recognized_sentence']} "
                       f"(置信度: {confidence:.3f}, 耗时: {recognition_time:.3f}s)")
            
            return result
            
        except Exception as e:
            error_msg = f"手语识别失败: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "recognized_sequence": [],
                "confidence": 0.0
            }

    async def _preprocess_video_frames(self, video_frames: np.ndarray) -> np.ndarray:
        """
        预处理视频帧
        Args:
            video_frames: 原始视频帧 (seq_len, height, width, channels)
        Returns:
            预处理后的帧 (seq_len, channels, height, width)
        """
        try:
            # 调整大小到224x224
            target_size = (224, 224)
            processed_frames = []
            
            for frame in video_frames:
                if frame.shape[2] == 3:  # RGB
                    # 调整大小
                    import cv2
                    resized_frame = cv2.resize(frame, target_size)
                    
                    # 归一化到[0,1]
                    normalized_frame = resized_frame.astype(np.float32) / 255.0
                    
                    # 标准化（ImageNet统计）
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    standardized_frame = (normalized_frame - mean) / std
                    
                    # 转换维度顺序 (H, W, C) -> (C, H, W)
                    transposed_frame = np.transpose(standardized_frame, (2, 0, 1))
                    processed_frames.append(transposed_frame)
                else:
                    logger.warning(f"不支持的帧格式: {frame.shape}")
            
            if processed_frames:
                return np.array(processed_frames)
            else:
                # 返回空的帧数组
                return np.zeros((1, 3, 224, 224), dtype=np.float32)
                
        except Exception as e:
            logger.error(f"视频帧预处理失败: {e}")
            return np.zeros((1, 3, 224, 224), dtype=np.float32)

    async def batch_recognize_sign_language(self, video_batch: List[np.ndarray]) -> List[Dict]:
        """
        批量手语识别
        Args:
            video_batch: 视频帧数组列表
        Returns:
            识别结果列表
        """
        results = []
        
        for i, video_frames in enumerate(video_batch):
            try:
                result = await self.recognize_sign_language(video_frames)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"批量识别第{i}个视频失败: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "batch_index": i,
                    "recognized_sequence": [],
                    "confidence": 0.0
                })
        
        return results


# 全局服务实例
diffusion_slp_service = DiffusionSLPService()
