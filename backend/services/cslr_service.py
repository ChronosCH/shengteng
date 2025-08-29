"""
连续手语识别 (CSLR) 服务 - 优化版本
基于 ST-Transformer-CTC 模型和 MindSpore Lite
添加了更好的错误处理、性能优化和资源管理
"""

import json
import time
import asyncio
from collections import deque
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

# MindSpore 推理 (包含 Lite 功能)
try:
    import mindspore as ms
    import mindspore.context as ms_context
    from mindspore import Tensor
    # 对于推理，我们使用标准的 MindSpore API
    MINDSPORE_AVAILABLE = True
    print("MindSpore 模块导入成功，将使用内置推理功能")
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("警告: MindSpore 未安装，将使用模拟推理")

from utils.logger import setup_logger
from utils.config import get_settings

logger = setup_logger(__name__)


@dataclass
class PredictionResult:
    """预测结果数据类"""
    text: str
    confidence: float
    gloss_sequence: List[str]
    inference_time: float
    timestamp: float
    status: str
    error: Optional[str] = None


@dataclass
class CSLRConfig:
    """CSLR服务配置"""
    model_path: str
    vocab_path: str
    confidence_threshold: float
    max_sequence_length: int
    enable_cache: bool
    cache_size: int
    batch_size: int = 1
    use_threading: bool = True
    max_workers: int = 2


class ModelCache:
    """模型推理缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = deque()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict]:
        """获取缓存结果"""
        with self._lock:
            if key in self.cache:
                # 更新访问顺序
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        return None
    
    def put(self, key: str, value: Dict) -> None:
        """添加缓存结果"""
        with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的项
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self) -> int:
        """清空缓存"""
        with self._lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_order.clear()
            return count


class CSLRService:
    """连续手语识别服务 - 优化版本"""
    
    def __init__(self, config: Optional[CSLRConfig] = None):
        self.settings = get_settings()
        self.config = config or self._create_default_config()
        
        # 模型相关
        self.model = None
        self.vocab = {}
        self.reverse_vocab = {}
        self.is_loaded = False
        self._model_lock = threading.Lock()
        
        # 序列缓冲区
        self.sequence_buffer = deque(maxlen=self.config.max_sequence_length)
        self._buffer_lock = threading.Lock()
        
        # 缓存系统
        self.cache = ModelCache(self.config.cache_size) if self.config.enable_cache else None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers) if self.config.use_threading else None
        
        # 性能统计
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_inference_time": 0.0,
            "last_inference_time": 0.0,
            "model_warmup_completed": False,
        }
        
        # CTC解码器配置
        self.ctc_config = {
            "blank_id": 0,
            "beam_width": 5,
            "alpha": 0.8,
            "beta": 1.2,
        }
        
        logger.info(f"CSLR 服务初始化完成 - 配置: {self.config}")
    
    def _create_default_config(self) -> CSLRConfig:
        """创建默认配置"""
        return CSLRConfig(
            model_path=self.settings.CSLR_MODEL_PATH,
            vocab_path=self.settings.CSLR_VOCAB_PATH,
            confidence_threshold=self.settings.CSLR_CONFIDENCE_THRESHOLD,
            max_sequence_length=self.settings.CSLR_MAX_SEQUENCE_LENGTH,
            enable_cache=self.settings.CSLR_ENABLE_CACHE,
            cache_size=self.settings.CSLR_CACHE_SIZE,
            batch_size=self.settings.MODEL_BATCH_SIZE,
            use_threading=True,
            max_workers=2,
        )
    
    async def load_model(self) -> bool:
        """加载CSLR模型和词汇表"""
        try:
            logger.info("开始加载CSLR模型...")
            
            # 加载词汇表
            await self._load_vocabulary()
            
            # 加载模型
            if MINDSPORE_AVAILABLE:
                await self._load_mindspore_model()
            else:
                await self._load_mock_model()
            
            # 模型预热
            await self._warmup_model()
            
            self.is_loaded = True
            logger.info("CSLR 模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"CSLR 模型加载失败: {e}")
            self.is_loaded = False
            return False
    
    async def _load_vocabulary(self) -> None:
        """加载词汇表"""
        try:
            with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"词汇表加载成功，包含 {len(self.vocab)} 个词汇")
            
        except FileNotFoundError:
            logger.warning("词汇表文件不存在，创建默认词汇表")
            await self._create_default_vocabulary()
        except json.JSONDecodeError as e:
            logger.error(f"词汇表文件格式错误: {e}")
            raise
    
    async def _create_default_vocabulary(self) -> None:
        """创建默认词汇表"""
        default_vocab = {
            "<blank>": 0, "<unk>": 1, "你好": 2, "谢谢": 3, "再见": 4,
            "是": 5, "不是": 6, "好": 7, "不好": 8, "我": 9, "你": 10,
            "他": 11, "她": 12, "吃": 13, "喝": 14, "睡觉": 15,
            "工作": 16, "学习": 17, "家": 18, "学校": 19, "医院": 20,
        }
        
        self.vocab = default_vocab
        self.reverse_vocab = {v: k for k, v in default_vocab.items()}
        
        # 保存词汇表
        import os
        os.makedirs(os.path.dirname(self.config.vocab_path), exist_ok=True)
        with open(self.config.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(default_vocab, f, ensure_ascii=False, indent=2)
    
    async def _load_mindspore_model(self) -> None:
        """加载MindSpore模型"""
        try:
            # 设置MindSpore上下文
            ms_context.set_context(mode=ms_context.GRAPH_MODE, device_target="CPU")
            
            # 对于推理，我们使用模拟实现，因为真实的模型加载需要具体的模型文件
            logger.info("使用模拟推理模式（开发环境）")
            self.model = None  # 模拟模型
            
            logger.info(f"MindSpore 推理环境初始化成功")
            
        except Exception as e:
            logger.error(f"MindSpore 模型加载失败: {e}")
            # 降级到模拟模式
            self.model = None
            logger.info("降级到模拟推理模式")
    
    async def _load_mock_model(self) -> None:
        """加载模拟模型"""
        logger.warning("使用模拟CSLR模型")
        self.model = "mock_model"
    
    async def _warmup_model(self) -> None:
        """模型预热"""
        if not self.model:
            return
        
        try:
            logger.info("开始模型预热...")
            warmup_iterations = self.settings.MODEL_WARMUP_ITERATIONS
            
            for i in range(warmup_iterations):
                # 创建模拟输入
                dummy_input = np.random.randn(1, 50, 543*3).astype(np.float32)
                
                if MINDSPORE_AVAILABLE and self.model != "mock_model":
                    inputs = self.model.get_inputs()
                    inputs[0].set_data_from_numpy(dummy_input)
                    self.model.predict(inputs)
                else:
                    await asyncio.sleep(0.01)  # 模拟推理时间
            
            self.stats["model_warmup_completed"] = True
            logger.info(f"模型预热完成 ({warmup_iterations} 次迭代)")
            
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    async def predict(self, landmarks_sequence: List[List[float]]) -> PredictionResult:
        """对关键点序列进行手语识别"""
        start_time = time.time()
        
        try:
            # 检查模型是否已加载
            if not self.is_loaded:
                return PredictionResult(
                    text="", confidence=0.0, gloss_sequence=[],
                    inference_time=0.0, timestamp=time.time(),
                    status="model_not_loaded", error="模型未加载"
                )
            
            # 使用传入的窗口序列进行长度检查（避免依赖缓冲区）
            if landmarks_sequence is None or len(landmarks_sequence) < 10:
                return PredictionResult(
                    text="", confidence=0.0, gloss_sequence=[],
                    inference_time=0.0, timestamp=time.time(),
                    status="insufficient_frames"
                )
            
            # 准备输入数据（直接由传入序列构造）
            try:
                input_array = np.array(landmarks_sequence, dtype=np.float32)  # (T, 543*3)
                # 限长
                max_length = self.config.max_sequence_length
                if input_array.shape[0] > max_length:
                    input_array = input_array[-max_length:]
                # 添加batch维度: (1, T, 543*3)
                input_array = np.expand_dims(input_array, axis=0)
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"输入数据构造失败: {e}")
                return PredictionResult(
                    text="", confidence=0.0, gloss_sequence=[],
                    inference_time=time.time() - start_time, timestamp=time.time(),
                    status="error", error=f"输入数据构造失败: {e}"
                )
            
            # 缓存（基于传入窗口）
            cache_key = self._generate_cache_key(landmarks_sequence)
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return PredictionResult(**cached_result)
                else:
                    self.stats["cache_misses"] += 1
            
            # 模型推理
            if self.config.use_threading and self.executor:
                prediction = await self._threaded_inference(input_array)
            else:
                if MINDSPORE_AVAILABLE and self.model != "mock_model":
                    prediction = await self._mindspore_inference(input_array)
                else:
                    prediction = await self._mock_inference(input_array)
            
            # CTC解码
            decoded_result = self._ctc_decode(prediction)
            
            # 后处理
            result_dict = self._post_process(decoded_result)
            
            # 创建结果对象
            inference_time = time.time() - start_time
            result = PredictionResult(
                **result_dict,
                inference_time=inference_time,
                timestamp=time.time(),
                status="success"
            )
            
            # 更新缓存
            if self.cache and result.confidence > 0.3:
                self.cache.put(cache_key, result.__dict__)
            
            # 更新统计
            self._update_stats(inference_time, success=True)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"CSLR 推理失败: {e}")
            return PredictionResult(
                text="", confidence=0.0, gloss_sequence=[],
                inference_time=time.time() - start_time, timestamp=time.time(),
                status="error", error=str(e)
            )
    
    def _generate_cache_key(self, landmarks: List[List[float]]) -> str:
        """生成缓存键"""
        # 简化的哈希策略
        landmarks_array = np.array(landmarks)
        # 降采样并量化以减少缓存键的敏感性
        simplified = landmarks_array[::5, ::10]  # 每5帧取一帧，每10个特征取一个
        quantized = np.round(simplified, 2)
        return str(hash(quantized.tobytes()))
    
    async def _threaded_inference(self, input_data: np.ndarray) -> np.ndarray:
        """线程池推理"""
        loop = asyncio.get_event_loop()
        
        if MINDSPORE_AVAILABLE and self.model != "mock_model":
            return await loop.run_in_executor(
                self.executor, self._sync_mindspore_inference, input_data
            )
        else:
            return await loop.run_in_executor(
                self.executor, self._sync_mock_inference, input_data
            )
    
    def _sync_mindspore_inference(self, input_data: np.ndarray) -> np.ndarray:
        """同步MindSpore推理"""
        with self._model_lock:
            inputs = self.model.get_inputs()
            inputs[0].set_data_from_numpy(input_data)
            self.model.predict(inputs)
            outputs = self.model.get_outputs()
            return outputs[0].get_data_to_numpy()
    
    def _sync_mock_inference(self, input_data: np.ndarray) -> np.ndarray:
        """同步模拟推理"""
        time.sleep(0.02)  # 模拟推理时间
        T = input_data.shape[1]
        vocab_size = len(self.vocab)
        
        # 创建更真实的概率分布
        prediction = np.random.rand(1, T, vocab_size).astype(np.float32)
        
        # 为了模拟更真实的模型输出，增强某些类别的概率
        for t in range(T):
            # 每个时间步随机选择1-3个主要类别
            num_main_classes = np.random.randint(1, 4)
            main_classes = np.random.choice(vocab_size, num_main_classes, replace=False)
            
            # 给主要类别增加概率权重
            for cls in main_classes:
                prediction[0, t, cls] += np.random.uniform(2.0, 5.0)
        
        return np.exp(prediction) / np.sum(np.exp(prediction), axis=-1, keepdims=True)
    
    def _ctc_decode(self, prediction: np.ndarray) -> Dict:
        """CTC解码"""
        try:
            # 简化的贪心解码
            # 实际应用中可以使用beam search等更复杂的解码方法
            
            # 获取最大概率的类别
            predicted_ids = np.argmax(prediction[0], axis=-1)
            
            # 移除重复和空白标记
            decoded_ids = []
            prev_id = -1
            selected_probs = []  # 记录选中的概率
            
            for i, id in enumerate(predicted_ids):
                if id != prev_id and id != self.ctc_config["blank_id"]:
                    decoded_ids.append(id)
                    selected_probs.append(prediction[0, i, id])  # 记录该时刻该类别的概率
                prev_id = id
            
            # 计算更合理的置信度 - 使用选中词汇的概率
            if selected_probs:
                confidence = np.mean(selected_probs)
            else:
                # 如果没有选中任何词汇，使用所有时刻最大概率的平均值
                max_probs = np.max(prediction[0], axis=-1)
                confidence = np.mean(max_probs)
            
            return {
                "decoded_ids": decoded_ids,
                "confidence": float(confidence),
                "raw_prediction": prediction
            }
            
        except Exception as e:
            logger.error(f"CTC解码失败: {e}")
            return {
                "decoded_ids": [],
                "confidence": 0.0,
                "raw_prediction": prediction
            }
    
    def _post_process(self, decoded_result: Dict) -> Dict:
        """后处理解码结果"""
        decoded_ids = decoded_result["decoded_ids"]
        confidence = decoded_result["confidence"]
        
        # 将ID转换为词汇
        gloss_sequence = []
        for id in decoded_ids:
            if id in self.reverse_vocab:
                gloss_sequence.append(self.reverse_vocab[id])
            else:
                gloss_sequence.append("<unk>")
        
        # 组合成文本
        text = " ".join(gloss_sequence)
        
        # 过滤低置信度结果
        if confidence < self.config.confidence_threshold:
            text = ""
            gloss_sequence = []
        
        return {
            "text": text,
            "confidence": confidence,
            "gloss_sequence": gloss_sequence,
        }
    
    def _update_stats(self, inference_time: float, success: bool = True):
        """更新性能统计"""
        self.stats["total_predictions"] += 1
        self.stats["last_inference_time"] = inference_time
        
        if success:
            self.stats["successful_predictions"] += 1
        
        # 计算平均推理时间
        alpha = 0.1
        if self.stats["average_inference_time"] == 0:
            self.stats["average_inference_time"] = inference_time
        else:
            self.stats["average_inference_time"] = (
                alpha * inference_time + 
                (1 - alpha) * self.stats["average_inference_time"]
            )
    
    async def update_config(self, config_updates: Dict) -> bool:
        """动态更新配置"""
        try:
            if "confidence_threshold" in config_updates:
                self.config.confidence_threshold = config_updates["confidence_threshold"]
            
            if "ctc_config" in config_updates:
                self.ctc_config.update(config_updates["ctc_config"])
            
            if "cache_size" in config_updates and self.cache:
                # 重新创建缓存
                old_cache = self.cache.cache.copy()
                self.cache = ModelCache(config_updates["cache_size"])
                # 迁移部分缓存
                for key, value in list(old_cache.items())[:config_updates["cache_size"]]:
                    self.cache.put(key, value)
            
            logger.info("CSLR 配置更新成功")
            return True
            
        except Exception as e:
            logger.error(f"CSLR 配置更新失败: {e}")
            return False
    
    def get_comprehensive_stats(self) -> Dict:
        """获取综合统计信息"""
        success_rate = (
            self.stats["successful_predictions"] / self.stats["total_predictions"]
            if self.stats["total_predictions"] > 0 else 0
        )
        
        cache_hit_rate = (
            self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
            if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
        )
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "vocab_size": len(self.vocab),
            "sequence_buffer_size": len(self.sequence_buffer),
            "confidence_threshold": self.config.confidence_threshold,
            "is_loaded": self.is_loaded,
            "cache_size": len(self.cache.cache) if self.cache else 0,
            "config": self.config.__dict__,
        }
    
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            # 关闭线程池
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # 清理模型
            if MINDSPORE_AVAILABLE and hasattr(self, 'model') and self.model != "mock_model":
                # MindSpore Lite 模型清理
                self.model = None
            
            # 清理缓存
            if self.cache:
                cleared_count = self.cache.clear()
                logger.info(f"清理了 {cleared_count} 个缓存项")
            
            # 清理缓冲区
            with self._buffer_lock:
                self.sequence_buffer.clear()
            
            self.is_loaded = False
            logger.info("CSLR 服务资源清理完成")
            
        except Exception as e:
            logger.error(f"CSLR 服务清理失败: {e}")
    
    def _prepare_input_data(self) -> np.ndarray:
        """准备模型输入数据"""
        # 将序列缓冲区转换为numpy数组
        sequence_data = list(self.sequence_buffer)
        
        # 确保序列长度一致
        max_length = self.config.max_sequence_length
        if len(sequence_data) > max_length:
            sequence_data = sequence_data[-max_length:]
        
        # 转换为numpy数组并标准化
        input_array = np.array(sequence_data, dtype=np.float32)
        
        # 添加batch维度: (1, T, 543*3)
        input_array = np.expand_dims(input_array, axis=0)
        
        # 数据标准化 (可选)
        # input_array = (input_array - input_array.mean()) / (input_array.std() + 1e-8)
        
        return input_array
    
    async def _mindspore_inference(self, input_data: np.ndarray) -> np.ndarray:
        """MindSpore模型推理"""
        try:
            # 设置输入
            inputs = self.model.get_inputs()
            inputs[0].set_data_from_numpy(input_data)
            
            # 执行推理
            self.model.predict(inputs)
            
            # 获取输出
            outputs = self.model.get_outputs()
            prediction = outputs[0].get_data_to_numpy()
            
            return prediction
            
        except Exception as e:
            logger.error(f"MindSpore 推理失败: {e}")
            raise
    
    async def _mock_inference(self, input_data: np.ndarray) -> np.ndarray:
        """模拟推理 (用于开发测试)"""
        # 模拟推理延迟
        await asyncio.sleep(0.02)  # 20ms
        
        # 生成模拟输出 (T, vocab_size)
        T = input_data.shape[1]
        vocab_size = len(self.vocab)
        
        # 创建更真实的概率分布，不是完全随机
        prediction = np.random.rand(1, T, vocab_size).astype(np.float32)
        
        # 为了模拟更真实的模型输出，增强某些类别的概率
        # 随机选择几个"主要"类别，给它们更高的概率
        for t in range(T):
            # 每个时间步随机选择1-3个主要类别
            num_main_classes = np.random.randint(1, 4)
            main_classes = np.random.choice(vocab_size, num_main_classes, replace=False)
            
            # 给主要类别增加概率权重
            for cls in main_classes:
                prediction[0, t, cls] += np.random.uniform(2.0, 5.0)
        
        # 应用softmax
        prediction = np.exp(prediction) / np.sum(np.exp(prediction), axis=-1, keepdims=True)
        
        return prediction
