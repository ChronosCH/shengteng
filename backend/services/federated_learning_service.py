"""
Explainable Federated Learning Service for Sign Language Recognition
可解释的联邦学习手语识别服务
"""

import asyncio
import logging
import time
import numpy as np
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque

try:
    import mindspore as ms
    import mindspore.context as ms_context
    from mindspore import Tensor
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    logging.warning("MindSpore not available, using mock implementation")

from utils.config import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class ClientRole(Enum):
    """客户端角色"""
    TRAINER = "trainer"      # 训练客户端
    VALIDATOR = "validator"  # 验证客户端
    OBSERVER = "observer"    # 观察客户端


class AggregationMethod(Enum):
    """聚合方法"""
    FEDAVG = "fedavg"           # 联邦平均
    FEDPROX = "fedprox"         # 联邦近端
    SCAFFOLD = "scaffold"       # SCAFFOLD算法
    FEDNOVA = "fednova"         # FedNova算法


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str
    role: ClientRole
    data_size: int
    model_version: int
    last_update: float
    privacy_budget: float
    contribution_score: float


@dataclass
class ModelUpdate:
    """模型更新"""
    client_id: str
    round_number: int
    gradients: Dict[str, np.ndarray]
    loss: float
    accuracy: float
    data_size: int
    timestamp: float
    privacy_noise: float


@dataclass
class ExplanationData:
    """解释数据"""
    saliency_maps: Dict[str, np.ndarray]  # 显著性图
    attention_weights: Dict[str, np.ndarray]  # 注意力权重
    feature_importance: Dict[str, float]  # 特征重要性
    prediction_confidence: float
    explanation_confidence: float


@dataclass
class FederatedRound:
    """联邦学习轮次"""
    round_number: int
    participants: List[str]
    global_model_hash: str
    aggregated_loss: float
    aggregated_accuracy: float
    explanation_summary: Dict[str, Any]
    start_time: float
    end_time: float


class FederatedLearningService:
    """联邦学习服务"""
    
    def __init__(self):
        self.global_model = None
        self.local_model = None
        self.is_loaded = False
        self.is_training = False
        self.device_type = "cpu"
        
        # 联邦学习状态
        self.current_round = 0
        self.client_info = ClientInfo(
            client_id=self._generate_client_id(),
            role=ClientRole.TRAINER,
            data_size=0,
            model_version=0,
            last_update=time.time(),
            privacy_budget=1.0,
            contribution_score=0.0
        )
        
        # 模型更新历史
        self.update_history = deque(maxlen=100)
        self.round_history = deque(maxlen=50)
        
        # 解释性数据
        self.explanation_cache = {}
        self.saliency_computer = None
        
        # 隐私保护
        self.differential_privacy = True
        self.noise_multiplier = 1.0
        self.max_grad_norm = 1.0
        
        # 统计信息
        self.fl_stats = {
            "total_rounds": 0,
            "successful_updates": 0,
            "average_round_time": 0.0,
            "privacy_budget_used": 0.0,
            "explanation_requests": 0
        }
        
    def _generate_client_id(self) -> str:
        """生成客户端ID"""
        import uuid
        return f"client_{uuid.uuid4().hex[:8]}"
    
    async def initialize(self):
        """初始化服务"""
        try:
            logger.info("正在初始化联邦学习服务...")
            
            # 加载本地模型
            if MINDSPORE_AVAILABLE:
                await self._load_mindspore_model()
            else:
                await self._load_mock_model()
            
            # 初始化解释性组件
            await self._initialize_explainability()
            
            self.is_loaded = True
            logger.info("联邦学习服务初始化完成")
            
        except Exception as e:
            logger.error(f"联邦学习服务初始化失败: {e}")
            raise
    
    async def _load_mindspore_model(self):
        """加载 MindSpore 模型"""
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
            self.local_model = None  # 模拟模型
            
            logger.info(f"MindSpore 联邦学习推理环境初始化成功 (设备: {self.device_type})")
            
        except Exception as e:
            logger.error(f"MindSpore 联邦学习模型加载失败: {e}")
            # 降级到模拟模式
            self.local_model = None
            self.device_type = "cpu"
            logger.info("降级到模拟推理模式")
    
    async def _load_mock_model(self):
        """加载模拟模型"""
        logger.warning("使用模拟联邦学习模型")
        self.local_model = "mock_federated_model"
    
    async def _initialize_explainability(self):
        """初始化解释性组件"""
        try:
            # 初始化显著性计算器
            self.saliency_computer = SaliencyComputer()
            
            logger.info("解释性组件初始化完成")
            
        except Exception as e:
            logger.error(f"解释性组件初始化失败: {e}")
            raise
    
    async def start_federated_training(self, config: Dict = None) -> bool:
        """开始联邦学习训练"""
        if self.is_training:
            logger.warning("联邦学习训练已在进行中")
            return False
        
        try:
            self.is_training = True
            self.current_round += 1
            
            logger.info(f"开始联邦学习第 {self.current_round} 轮")
            
            # 模拟训练过程
            if MINDSPORE_AVAILABLE and self.local_model != "mock_federated_model":
                update = await self._mindspore_local_training()
            else:
                update = await self._mock_local_training()
            
            # 添加差分隐私噪声
            if self.differential_privacy:
                update = self._add_differential_privacy_noise(update)
            
            # 记录更新
            self.update_history.append(update)
            
            # 更新统计
            self.fl_stats["successful_updates"] += 1
            self.fl_stats["total_rounds"] = self.current_round
            
            logger.info(f"联邦学习第 {self.current_round} 轮完成")
            return True
            
        except Exception as e:
            logger.error(f"联邦学习训练失败: {e}")
            return False
        finally:
            self.is_training = False
    
    async def _mindspore_local_training(self) -> ModelUpdate:
        """MindSpore 本地训练"""
        try:
            # 模拟本地训练过程
            # 实际实现中应该使用真实的训练数据和损失函数
            
            # 获取模型参数
            gradients = {}
            
            # 模拟梯度计算
            for i in range(5):  # 假设5层网络
                layer_name = f"layer_{i}"
                grad_shape = (64, 32) if i < 4 else (32, 10)  # 最后一层输出10类
                gradients[layer_name] = np.random.normal(0, 0.1, grad_shape).astype(np.float32)
            
            # 计算损失和准确率
            loss = np.random.uniform(0.1, 0.5)
            accuracy = np.random.uniform(0.7, 0.95)
            
            update = ModelUpdate(
                client_id=self.client_info.client_id,
                round_number=self.current_round,
                gradients=gradients,
                loss=loss,
                accuracy=accuracy,
                data_size=self.client_info.data_size,
                timestamp=time.time(),
                privacy_noise=0.0
            )
            
            return update
            
        except Exception as e:
            logger.error(f"MindSpore 本地训练失败: {e}")
            raise
    
    async def _mock_local_training(self) -> ModelUpdate:
        """模拟本地训练"""
        # 模拟训练延迟
        await asyncio.sleep(0.5)
        
        # 生成模拟梯度
        gradients = {}
        for i in range(3):
            layer_name = f"mock_layer_{i}"
            gradients[layer_name] = np.random.normal(0, 0.1, (10, 10)).astype(np.float32)
        
        # 模拟训练指标
        loss = np.random.uniform(0.2, 0.6)
        accuracy = np.random.uniform(0.6, 0.9)
        
        update = ModelUpdate(
            client_id=self.client_info.client_id,
            round_number=self.current_round,
            gradients=gradients,
            loss=loss,
            accuracy=accuracy,
            data_size=100,  # 模拟数据大小
            timestamp=time.time(),
            privacy_noise=0.0
        )
        
        return update
    
    def _add_differential_privacy_noise(self, update: ModelUpdate) -> ModelUpdate:
        """添加差分隐私噪声"""
        try:
            # 计算梯度范数
            total_norm = 0.0
            for grad in update.gradients.values():
                total_norm += np.linalg.norm(grad) ** 2
            total_norm = np.sqrt(total_norm)
            
            # 梯度裁剪
            if total_norm > self.max_grad_norm:
                clip_factor = self.max_grad_norm / total_norm
                for layer_name in update.gradients:
                    update.gradients[layer_name] *= clip_factor
            
            # 添加高斯噪声
            noise_scale = self.noise_multiplier * self.max_grad_norm
            total_noise = 0.0
            
            for layer_name in update.gradients:
                noise = np.random.normal(0, noise_scale, update.gradients[layer_name].shape)
                update.gradients[layer_name] += noise.astype(np.float32)
                total_noise += np.linalg.norm(noise)
            
            update.privacy_noise = total_noise
            
            # 更新隐私预算
            epsilon_used = 1.0 / (self.noise_multiplier ** 2)
            self.client_info.privacy_budget -= epsilon_used
            self.fl_stats["privacy_budget_used"] += epsilon_used
            
            logger.debug(f"添加差分隐私噪声: 噪声规模={noise_scale:.4f}, 剩余隐私预算={self.client_info.privacy_budget:.4f}")
            
            return update
            
        except Exception as e:
            logger.error(f"添加差分隐私噪声失败: {e}")
            return update
    
    async def generate_explanation(self, input_data: np.ndarray, prediction: Dict) -> ExplanationData:
        """生成模型解释"""
        if not self.saliency_computer:
            raise RuntimeError("解释性组件未初始化")
        
        try:
            self.fl_stats["explanation_requests"] += 1
            
            # 计算显著性图
            saliency_maps = await self.saliency_computer.compute_saliency(
                input_data, self.local_model
            )
            
            # 计算注意力权重
            attention_weights = await self.saliency_computer.compute_attention(
                input_data, self.local_model
            )
            
            # 计算特征重要性
            feature_importance = await self.saliency_computer.compute_feature_importance(
                input_data, self.local_model
            )
            
            explanation = ExplanationData(
                saliency_maps=saliency_maps,
                attention_weights=attention_weights,
                feature_importance=feature_importance,
                prediction_confidence=prediction.get('confidence', 0.0),
                explanation_confidence=0.8  # 模拟解释置信度
            )
            
            # 缓存解释
            cache_key = hashlib.md5(input_data.tobytes()).hexdigest()
            self.explanation_cache[cache_key] = explanation
            
            logger.info("模型解释生成完成")
            return explanation
            
        except Exception as e:
            logger.error(f"生成模型解释失败: {e}")
            raise
    
    async def get_federated_stats(self) -> Dict:
        """获取联邦学习统计信息"""
        return {
            **self.fl_stats,
            "client_info": asdict(self.client_info),
            "current_round": self.current_round,
            "is_training": self.is_training,
            "is_loaded": self.is_loaded,
            "device_type": self.device_type,
            "update_history_size": len(self.update_history),
            "explanation_cache_size": len(self.explanation_cache),
            "differential_privacy": self.differential_privacy,
            "privacy_budget_remaining": self.client_info.privacy_budget
        }
    
    async def get_latest_update(self) -> Optional[ModelUpdate]:
        """获取最新的模型更新"""
        if self.update_history:
            return self.update_history[-1]
        return None
    
    async def get_explanation_summary(self) -> Dict:
        """获取解释摘要"""
        if not self.explanation_cache:
            return {"message": "暂无解释数据"}
        
        # 计算解释统计
        total_explanations = len(self.explanation_cache)
        avg_confidence = np.mean([
            exp.explanation_confidence for exp in self.explanation_cache.values()
        ])
        
        return {
            "total_explanations": total_explanations,
            "average_confidence": avg_confidence,
            "cache_size": total_explanations,
            "latest_explanation_time": time.time()
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            self.is_training = False
            
            if self.local_model and MINDSPORE_AVAILABLE:
                # MindSpore 模型清理
                pass
            
            self.update_history.clear()
            self.round_history.clear()
            self.explanation_cache.clear()
            
            self.is_loaded = False
            logger.info("联邦学习服务资源清理完成")
            
        except Exception as e:
            logger.error(f"联邦学习服务清理失败: {e}")


class SaliencyComputer:
    """显著性计算器"""
    
    async def compute_saliency(self, input_data: np.ndarray, model: Any) -> Dict[str, np.ndarray]:
        """计算显著性图"""
        # 简化实现：生成模拟显著性图
        saliency_maps = {}
        
        if len(input_data.shape) == 3:  # 假设输入是关键点序列 (T, N, 3)
            T, N, _ = input_data.shape
            
            # 为每个关键点生成显著性分数
            saliency_maps["keypoints"] = np.random.uniform(0, 1, (T, N))
            
            # 为时间步生成显著性分数
            saliency_maps["temporal"] = np.random.uniform(0, 1, T)
            
        return saliency_maps
    
    async def compute_attention(self, input_data: np.ndarray, model: Any) -> Dict[str, np.ndarray]:
        """计算注意力权重"""
        # 简化实现：生成模拟注意力权重
        attention_weights = {}
        
        if len(input_data.shape) == 3:
            T, N, _ = input_data.shape
            
            # 自注意力权重矩阵
            attention_weights["self_attention"] = np.random.uniform(0, 1, (T, T))
            
            # 关键点注意力权重
            attention_weights["keypoint_attention"] = np.random.uniform(0, 1, N)
            
        return attention_weights
    
    async def compute_feature_importance(self, input_data: np.ndarray, model: Any) -> Dict[str, float]:
        """计算特征重要性"""
        # 简化实现：生成模拟特征重要性分数
        feature_importance = {
            "hand_landmarks": np.random.uniform(0.7, 0.9),
            "face_landmarks": np.random.uniform(0.3, 0.6),
            "pose_landmarks": np.random.uniform(0.4, 0.7),
            "temporal_dynamics": np.random.uniform(0.5, 0.8),
            "spatial_relationships": np.random.uniform(0.6, 0.8)
        }
        
        return feature_importance


# 全局服务实例
federated_learning_service = FederatedLearningService()
