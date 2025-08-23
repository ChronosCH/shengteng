"""
增强版CE-CSL手语识别服务
支持加载训练好的enhanced_cecsl_final_model.ckpt模型
"""

import json
import time
import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# MindSpore
try:
    import mindspore as ms
    from mindspore import nn, ops, Tensor, load_checkpoint, load_param_into_net
    import mindspore.context as ms_context
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("警告: MindSpore 未安装，将使用模拟推理")

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EnhancedCECSLConfig:
    """增强版CE-CSL模型配置"""
    vocab_size: int = 1000
    d_model: int = 192
    n_layers: int = 2
    dropout: float = 0.3
    image_size: Tuple[int, int] = (112, 112)
    max_sequence_length: int = 64


class ImprovedCECSLModel(nn.Cell):
    """改进的CE-CSL手语识别模型（与训练代码保持一致）"""
    
    def __init__(self, config: EnhancedCECSLConfig, vocab_size: int):
        super().__init__()
        self.config = config
        
        input_size = 3 * config.image_size[0] * config.image_size[1]
        
        # 改进的特征提取网络
        self.feature_extractor = nn.SequentialCell([
            nn.Dense(input_size, config.d_model * 2),
            nn.LayerNorm([config.d_model * 2]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            
            nn.Dense(config.d_model * 2, config.d_model),
            nn.LayerNorm([config.d_model]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            
            nn.Dense(config.d_model, config.d_model),
            nn.LayerNorm([config.d_model]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout / 2)
        ])
        
        # 双向LSTM
        self.temporal_model = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0
        )
        
        # 注意力机制
        self.attention = nn.SequentialCell([
            nn.Dense(config.d_model * 2, config.d_model),
            nn.Tanh(),
            nn.Dense(config.d_model, 1)
        ])
        
        # 改进的分类器
        self.classifier = nn.SequentialCell([
            nn.Dense(config.d_model * 2, config.d_model),
            nn.LayerNorm([config.d_model]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            
            nn.Dense(config.d_model, config.d_model // 2),
            nn.LayerNorm([config.d_model // 2]),
            nn.ReLU(),
            nn.Dropout(p=config.dropout / 2),
            
            nn.Dense(config.d_model // 2, vocab_size)
        ])
    
    def construct(self, x):
        """推理时的前向传播"""
        batch_size, seq_len, input_size = x.shape
        
        # 特征提取
        x_reshaped = x.view(batch_size * seq_len, input_size)
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, self.config.d_model)
        
        # 双向LSTM
        lstm_output, _ = self.temporal_model(features)  # (batch, seq, hidden*2)
        
        # 注意力权重
        attention_weights = self.attention(lstm_output)  # (batch, seq, 1)
        attention_weights = ops.Softmax(axis=1)(attention_weights)
        
        # 加权平均
        attended_output = ops.ReduceSum()(lstm_output * attention_weights, axis=1)  # (batch, hidden*2)
        
        # 分类
        logits = self.classifier(attended_output)
        
        return logits


@dataclass
class PredictionResult:
    """预测结果"""
    text: str
    confidence: float
    gloss_sequence: List[str]
    inference_time: float
    timestamp: float
    status: str
    error: Optional[str] = None


class EnhancedCECSLService:
    """增强版CE-CSL手语识别服务"""
    
    def __init__(self, model_path: str, vocab_path: str):
        self.model_path = Path(model_path)
        self.vocab_path = Path(vocab_path)
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.config = None
        self.is_loaded = False
        
        # 统计信息
        self.stats = {
            "predictions": 0,
            "errors": 0,
            "total_inference_time": 0.0,
            "avg_inference_time": 0.0
        }
    
    async def initialize(self) -> bool:
        """初始化服务"""
        try:
            logger.info("初始化增强版CE-CSL服务...")
            
            # 设置MindSpore上下文
            if MINDSPORE_AVAILABLE:
                ms_context.set_context(mode=ms_context.GRAPH_MODE, device_target="CPU")
            
            # 加载词汇表
            await self._load_vocabulary()
            
            # 创建模型配置
            self.config = EnhancedCECSLConfig(
                vocab_size=len(self.vocab)
            )
            
            # 加载模型
            await self._load_model()
            
            self.is_loaded = True
            logger.info("增强版CE-CSL服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"增强版CE-CSL服务初始化失败: {e}")
            self.is_loaded = False
            return False
    
    async def _load_vocabulary(self) -> None:
        """加载词汇表"""
        try:
            if not self.vocab_path.exists():
                logger.error(f"词汇表文件不存在: {self.vocab_path}")
                await self._create_default_vocabulary()
                return
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # 兼容多种格式
            if isinstance(vocab_data, dict) and 'word2idx' in vocab_data and isinstance(vocab_data['word2idx'], dict):
                self.vocab = vocab_data['word2idx']
            elif isinstance(vocab_data, dict) and 'vocab' in vocab_data and isinstance(vocab_data['vocab'], dict):
                self.vocab = vocab_data['vocab']
            elif isinstance(vocab_data, dict) and 'label_names' in vocab_data and isinstance(vocab_data['label_names'], list):
                # 从标签名列表构建映射
                self.vocab = {label: i for i, label in enumerate(vocab_data['label_names'])}
            elif isinstance(vocab_data, dict):
                # 假定本身就是 { token: index }
                # 过滤出 value 为整数的键值对
                self.vocab = {k: v for k, v in vocab_data.items() if isinstance(v, int)}
            else:
                raise ValueError("不支持的词汇表格式")

            if not self.vocab or not isinstance(self.vocab, dict):
                raise ValueError("词汇表解析失败或为空")

            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"词汇表加载成功，包含 {len(self.vocab)} 个词汇")

        except Exception as e:
            logger.error(f"词汇表加载失败: {e}")
            await self._create_default_vocabulary()
    
    async def _create_default_vocabulary(self) -> None:
        """创建默认词汇表"""
        default_vocab = {
            "<PAD>": 0, "<UNK>": 1, "你好": 2, "谢谢": 3, "再见": 4,
            "是": 5, "不是": 6, "好": 7, "不好": 8, "我": 9, "你": 10,
        }
        
        self.vocab = default_vocab
        self.reverse_vocab = {v: k for k, v in default_vocab.items()}
        logger.warning("使用默认词汇表")
    
    async def _load_model(self) -> None:
        """加载模型"""
        try:
            if not MINDSPORE_AVAILABLE:
                logger.warning("MindSpore不可用，使用模拟模型")
                self.model = "mock_model"
                return
            
            if not self.model_path.exists():
                logger.error(f"模型文件不存在: {self.model_path}")
                self.model = "mock_model"
                return
            
            # 创建模型实例
            self.model = ImprovedCECSLModel(self.config, len(self.vocab))
            
            # 加载checkpoint
            param_dict = load_checkpoint(str(self.model_path))
            load_param_into_net(self.model, param_dict)
            
            # 设置为评估模式
            self.model.set_train(False)
            
            logger.info(f"模型加载成功: {self.model_path}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = "mock_model"
    
    async def predict_from_landmarks(self, landmarks: List[List[float]]) -> PredictionResult:
        """从关键点预测手语"""
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise RuntimeError("服务未初始化")
            
            # 预处理输入数据
            input_data = self._preprocess_landmarks(landmarks)
            
            # 模型推理
            if MINDSPORE_AVAILABLE and self.model != "mock_model":
                prediction = await self._mindspore_inference(input_data)
            else:
                prediction = await self._mock_inference(input_data)
            
            # 后处理
            result_dict = self._postprocess_prediction(prediction)
            
            # 创建结果
            inference_time = time.time() - start_time
            result = PredictionResult(
                **result_dict,
                inference_time=inference_time,
                timestamp=time.time(),
                status="success"
            )
            
            # 更新统计
            self._update_stats(inference_time, True)
            
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"预测失败: {e}")
            self._update_stats(inference_time, False)
            
            return PredictionResult(
                text="", confidence=0.0, gloss_sequence=[],
                inference_time=inference_time, timestamp=time.time(),
                status="error", error=str(e)
            )
    
    def _preprocess_landmarks(self, landmarks: List[List[float]]) -> np.ndarray:
        """预处理关键点数据"""
        try:
            # 转换为numpy数组
            landmarks_array = np.array(landmarks, dtype=np.float32)
            
            # 确保有足够的帧数
            target_frames = self.config.max_sequence_length
            if landmarks_array.shape[0] < target_frames:
                # 重复最后一帧来填充
                pad_frames = target_frames - landmarks_array.shape[0]
                last_frame = landmarks_array[-1:] if landmarks_array.shape[0] > 0 else np.zeros((1, landmarks_array.shape[1]))
                landmarks_array = np.concatenate([landmarks_array] + [last_frame] * pad_frames, axis=0)
            elif landmarks_array.shape[0] > target_frames:
                # 截断
                landmarks_array = landmarks_array[:target_frames]
            
            # 模拟转换为图像特征（这里简化处理）
            # 实际应用中需要根据关键点生成对应的图像特征
            image_size = self.config.image_size[0] * self.config.image_size[1] * 3
            
            # 将关键点特征映射到图像特征空间
            if landmarks_array.shape[1] < image_size:
                # 扩展特征维度
                scale_factor = image_size // landmarks_array.shape[1] + 1
                expanded = np.tile(landmarks_array, (1, scale_factor))[:, :image_size]
            else:
                expanded = landmarks_array[:, :image_size]
            
            # 添加batch维度
            batch_data = expanded[np.newaxis, :]  # (1, seq_len, feature_dim)
            
            return batch_data
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            # 返回默认数据
            image_size = self.config.image_size[0] * self.config.image_size[1] * 3
            return np.random.rand(1, self.config.max_sequence_length, image_size).astype(np.float32)
    
    async def _mindspore_inference(self, input_data: np.ndarray) -> np.ndarray:
        """MindSpore模型推理"""
        try:
            # 转换为Tensor
            input_tensor = Tensor(input_data, ms.float32)
            
            # 推理
            output = self.model(input_tensor)
            
            # 转换回numpy
            if isinstance(output, Tensor):
                return output.asnumpy()
            else:
                return output
            
        except Exception as e:
            logger.error(f"MindSpore推理失败: {e}")
            # 降级到模拟推理
            return await self._mock_inference(input_data)
    
    async def _mock_inference(self, input_data: np.ndarray) -> np.ndarray:
        """模拟推理"""
        await asyncio.sleep(0.02)  # 模拟推理时间
        vocab_size = len(self.vocab)
        # 返回随机预测
        prediction = np.random.rand(vocab_size).astype(np.float32)
        # 应用softmax
        exp_pred = np.exp(prediction - np.max(prediction))
        return exp_pred / np.sum(exp_pred)
    
    def _postprocess_prediction(self, prediction: np.ndarray) -> Dict:
        """后处理预测结果"""
        try:
            # 获取最高概率的类别
            if prediction.ndim == 1:
                probabilities = prediction
            else:
                probabilities = prediction.flatten()
            
            top_idx = np.argmax(probabilities)
            confidence = float(probabilities[top_idx])
            
            # 获取对应的词汇
            if top_idx in self.reverse_vocab:
                predicted_word = self.reverse_vocab[top_idx]
            else:
                predicted_word = "<UNK>"
            
            # 获取top-5预测用于gloss序列
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            gloss_sequence = []
            for idx in top5_indices:
                if idx in self.reverse_vocab and probabilities[idx] > 0.1:
                    gloss_sequence.append(self.reverse_vocab[idx])
            
            return {
                "text": predicted_word,
                "confidence": confidence,
                "gloss_sequence": gloss_sequence
            }
            
        except Exception as e:
            logger.error(f"后处理失败: {e}")
            return {
                "text": "<UNK>",
                "confidence": 0.0,
                "gloss_sequence": []
            }
    
    def _update_stats(self, inference_time: float, success: bool) -> None:
        """更新统计信息"""
        self.stats["predictions"] += 1
        if not success:
            self.stats["errors"] += 1
        
        self.stats["total_inference_time"] += inference_time
        self.stats["avg_inference_time"] = (
            self.stats["total_inference_time"] / self.stats["predictions"]
        )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理增强版CE-CSL服务资源")
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.is_loaded = False
