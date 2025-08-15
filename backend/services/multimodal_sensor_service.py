"""
Multimodal Sensor Service - EMG + IMU Fusion
多模态传感器服务 - 肌电信号与惯性测量单元融合
"""

import asyncio
import logging
import time
import numpy as np
import serial
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
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


class SensorType(Enum):
    """传感器类型"""
    EMG = "emg"          # 肌电信号
    IMU = "imu"          # 惯性测量单元
    VISUAL = "visual"    # 视觉关键点


class FusionMode(Enum):
    """融合模式"""
    EARLY_FUSION = "early"      # 早期融合
    LATE_FUSION = "late"        # 后期融合
    HYBRID_FUSION = "hybrid"    # 混合融合


@dataclass
class EMGData:
    """肌电信号数据"""
    channels: np.ndarray  # Shape: (16, samples) - 16通道肌电信号
    timestamp: float
    sampling_rate: int = 1000  # 采样率 Hz
    amplitude: float = 0.0  # 信号幅度
    frequency: float = 0.0  # 主频率


@dataclass
class IMUData:
    """IMU数据"""
    accelerometer: np.ndarray  # Shape: (3,) - 加速度计 [x, y, z]
    gyroscope: np.ndarray      # Shape: (3,) - 陀螺仪 [x, y, z]
    magnetometer: np.ndarray   # Shape: (3,) - 磁力计 [x, y, z]
    timestamp: float
    orientation: Optional[np.ndarray] = None  # 四元数姿态


@dataclass
class MultimodalData:
    """多模态数据"""
    emg: Optional[EMGData] = None
    imu: Optional[IMUData] = None
    visual_landmarks: Optional[np.ndarray] = None
    timestamp: float = 0.0
    confidence: float = 0.0


@dataclass
class SensorConfig:
    """传感器配置"""
    emg_enabled: bool = True
    imu_enabled: bool = True
    visual_enabled: bool = True
    fusion_mode: FusionMode = FusionMode.EARLY_FUSION
    emg_channels: int = 16
    imu_rate: int = 100  # Hz
    emg_rate: int = 1000  # Hz
    buffer_size: int = 100


class MultimodalSensorService:
    """多模态传感器服务"""
    
    def __init__(self):
        self.fusion_model = None
        self.is_loaded = False
        self.is_collecting = False
        self.device_type = "cpu"
        
        # 传感器连接
        self.emg_device = None
        self.imu_device = None
        
        # 数据缓冲区
        self.emg_buffer = deque(maxlen=1000)
        self.imu_buffer = deque(maxlen=1000)
        self.visual_buffer = deque(maxlen=100)
        
        # 数据收集线程
        self.collection_thread = None
        self.stop_collection = threading.Event()
        
        # 统计信息
        self.sensor_stats = {
            "total_samples": 0,
            "emg_samples": 0,
            "imu_samples": 0,
            "visual_samples": 0,
            "fusion_predictions": 0,
            "average_latency": 0.0
        }
        
        # 配置
        self.config = SensorConfig()
        
    async def initialize(self):
        """初始化服务"""
        try:
            logger.info("正在初始化多模态传感器服务...")
            
            # 初始化传感器设备
            await self._initialize_sensors()
            
            # 加载融合模型
            if MINDSPORE_AVAILABLE:
                await self._load_fusion_model()
            else:
                await self._load_mock_model()
                
            self.is_loaded = True
            logger.info("多模态传感器服务初始化完成")
            
        except Exception as e:
            logger.error(f"多模态传感器服务初始化失败: {e}")
            raise
    
    async def _initialize_sensors(self):
        """初始化传感器设备"""
        try:
            # 初始化EMG设备
            if self.config.emg_enabled:
                await self._initialize_emg_device()
            
            # 初始化IMU设备
            if self.config.imu_enabled:
                await self._initialize_imu_device()
                
            logger.info("传感器设备初始化完成")
            
        except Exception as e:
            logger.error(f"传感器设备初始化失败: {e}")
            # 使用模拟设备
            await self._initialize_mock_devices()
    
    async def _initialize_emg_device(self):
        """初始化EMG设备"""
        try:
            # 尝试连接EMG设备 (通过串口或USB)
            emg_port = getattr(settings, 'EMG_DEVICE_PORT', '/dev/ttyUSB0')
            self.emg_device = serial.Serial(emg_port, 115200, timeout=1)
            logger.info(f"EMG设备连接成功: {emg_port}")
        except Exception as e:
            logger.warning(f"EMG设备连接失败: {e}, 使用模拟设备")
            self.emg_device = "mock_emg"
    
    async def _initialize_imu_device(self):
        """初始化IMU设备"""
        try:
            # 尝试连接IMU设备
            imu_port = getattr(settings, 'IMU_DEVICE_PORT', '/dev/ttyUSB1')
            self.imu_device = serial.Serial(imu_port, 115200, timeout=1)
            logger.info(f"IMU设备连接成功: {imu_port}")
        except Exception as e:
            logger.warning(f"IMU设备连接失败: {e}, 使用模拟设备")
            self.imu_device = "mock_imu"
    
    async def _initialize_mock_devices(self):
        """初始化模拟设备"""
        self.emg_device = "mock_emg"
        self.imu_device = "mock_imu"
        logger.info("使用模拟传感器设备")
    
    async def _load_fusion_model(self):
        """加载MindSpore融合模型"""
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
            self.fusion_model = None  # 模拟模型
            
            logger.info(f"MindSpore 多模态融合推理环境初始化成功 (设备: {self.device_type})")
            
        except Exception as e:
            logger.error(f"MindSpore 多模态融合模型加载失败: {e}")
            # 降级到模拟模式
            self.fusion_model = None
            self.device_type = "cpu"
            logger.info("降级到模拟推理模式")
    
    async def _load_mock_model(self):
        """加载模拟模型"""
        logger.warning("使用模拟多模态融合模型")
        self.fusion_model = "mock_fusion_model"
    
    async def start_collection(self):
        """开始数据收集"""
        if self.is_collecting:
            logger.warning("数据收集已在进行中")
            return
            
        self.is_collecting = True
        self.stop_collection.clear()
        
        # 启动数据收集线程
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.start()
        
        logger.info("多模态数据收集已开始")
    
    async def stop_collection(self):
        """停止数据收集"""
        if not self.is_collecting:
            return
            
        self.stop_collection.set()
        self.is_collecting = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
            
        logger.info("多模态数据收集已停止")
    
    def _collection_loop(self):
        """数据收集循环"""
        while not self.stop_collection.is_set():
            try:
                current_time = time.time()
                
                # 收集EMG数据
                if self.config.emg_enabled:
                    emg_data = self._read_emg_data(current_time)
                    if emg_data:
                        self.emg_buffer.append(emg_data)
                        self.sensor_stats["emg_samples"] += 1
                
                # 收集IMU数据
                if self.config.imu_enabled:
                    imu_data = self._read_imu_data(current_time)
                    if imu_data:
                        self.imu_buffer.append(imu_data)
                        self.sensor_stats["imu_samples"] += 1
                
                self.sensor_stats["total_samples"] += 1
                
                # 控制采样率
                time.sleep(1.0 / self.config.imu_rate)
                
            except Exception as e:
                logger.error(f"数据收集错误: {e}")
                time.sleep(0.1)
    
    def _read_emg_data(self, timestamp: float) -> Optional[EMGData]:
        """读取EMG数据"""
        try:
            if self.emg_device == "mock_emg":
                return self._generate_mock_emg_data(timestamp)
            
            # 从真实设备读取数据
            if self.emg_device.in_waiting > 0:
                raw_data = self.emg_device.readline().decode('utf-8').strip()
                data = json.loads(raw_data)
                
                channels = np.array(data.get('channels', [0] * 16))
                
                return EMGData(
                    channels=channels.reshape(16, -1),
                    timestamp=timestamp,
                    amplitude=np.mean(np.abs(channels)),
                    frequency=self._estimate_frequency(channels)
                )
                
        except Exception as e:
            logger.debug(f"EMG数据读取错误: {e}")
            return None
    
    def _read_imu_data(self, timestamp: float) -> Optional[IMUData]:
        """读取IMU数据"""
        try:
            if self.imu_device == "mock_imu":
                return self._generate_mock_imu_data(timestamp)
            
            # 从真实设备读取数据
            if self.imu_device.in_waiting > 0:
                raw_data = self.imu_device.readline().decode('utf-8').strip()
                data = json.loads(raw_data)
                
                return IMUData(
                    accelerometer=np.array(data.get('accel', [0, 0, 0])),
                    gyroscope=np.array(data.get('gyro', [0, 0, 0])),
                    magnetometer=np.array(data.get('mag', [0, 0, 0])),
                    timestamp=timestamp
                )
                
        except Exception as e:
            logger.debug(f"IMU数据读取错误: {e}")
            return None
    
    def _generate_mock_emg_data(self, timestamp: float) -> EMGData:
        """生成模拟EMG数据"""
        # 模拟16通道肌电信号
        channels = np.random.normal(0, 0.1, (16, 10))  # 10个采样点
        
        # 添加手势相关的信号模式
        gesture_signal = np.sin(timestamp * 2 * np.pi) * 0.5
        channels[0:4] += gesture_signal  # 前4个通道模拟手部肌肉
        
        return EMGData(
            channels=channels,
            timestamp=timestamp,
            amplitude=np.mean(np.abs(channels)),
            frequency=50.0  # 模拟50Hz主频率
        )
    
    def _generate_mock_imu_data(self, timestamp: float) -> IMUData:
        """生成模拟IMU数据"""
        # 模拟手部运动的IMU数据
        accel = np.array([
            np.sin(timestamp * 2) * 2.0,  # X轴加速度
            np.cos(timestamp * 1.5) * 1.5,  # Y轴加速度
            9.8 + np.sin(timestamp * 0.5) * 0.5  # Z轴加速度（包含重力）
        ])
        
        gyro = np.array([
            np.cos(timestamp * 3) * 0.5,  # X轴角速度
            np.sin(timestamp * 2.5) * 0.3,  # Y轴角速度
            np.sin(timestamp * 1.8) * 0.2   # Z轴角速度
        ])
        
        mag = np.array([0.3, 0.1, 0.5])  # 模拟磁场
        
        return IMUData(
            accelerometer=accel,
            gyroscope=gyro,
            magnetometer=mag,
            timestamp=timestamp
        )
    
    def _estimate_frequency(self, signal: np.ndarray) -> float:
        """估计信号主频率"""
        # 简化的频率估计
        if len(signal) < 10:
            return 0.0
            
        # 使用FFT估计主频率
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1.0 / self.config.emg_rate)
        
        # 找到最大幅度对应的频率
        max_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        return abs(freqs[max_idx])
    
    async def add_visual_landmarks(self, landmarks: np.ndarray):
        """添加视觉关键点数据"""
        if self.config.visual_enabled:
            self.visual_buffer.append({
                'landmarks': landmarks,
                'timestamp': time.time()
            })
            self.sensor_stats["visual_samples"] += 1

    async def predict_multimodal(self, window_size: int = 30) -> Dict:
        """
        多模态融合预测

        Args:
            window_size: 时间窗口大小

        Returns:
            预测结果
        """
        if not self.is_loaded:
            raise RuntimeError("多模态传感器服务未初始化")

        start_time = time.time()

        try:
            # 获取最近的多模态数据
            multimodal_data = self._get_recent_multimodal_data(window_size)

            if not multimodal_data:
                return {
                    "prediction": "",
                    "confidence": 0.0,
                    "modalities_used": [],
                    "status": "insufficient_data"
                }

            # 特征提取
            features = self._extract_multimodal_features(multimodal_data)

            # 融合预测
            if MINDSPORE_AVAILABLE and self.fusion_model != "mock_fusion_model":
                prediction = await self._mindspore_fusion_inference(features)
            else:
                prediction = await self._mock_fusion_inference(features)

            # 更新统计
            inference_time = time.time() - start_time
            self.sensor_stats["fusion_predictions"] += 1
            self._update_latency_stats(inference_time)

            return {
                **prediction,
                "inference_time": inference_time,
                "timestamp": time.time(),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"多模态融合预测失败: {e}")
            raise

    def _get_recent_multimodal_data(self, window_size: int) -> List[MultimodalData]:
        """获取最近的多模态数据"""
        current_time = time.time()
        time_window = 1.0  # 1秒时间窗口

        multimodal_data = []

        # 获取时间窗口内的数据
        recent_emg = [data for data in self.emg_buffer
                     if current_time - data.timestamp <= time_window]
        recent_imu = [data for data in self.imu_buffer
                     if current_time - data.timestamp <= time_window]
        recent_visual = [data for data in self.visual_buffer
                        if current_time - data['timestamp'] <= time_window]

        # 时间对齐和数据融合
        for i in range(min(window_size, len(recent_emg), len(recent_imu))):
            emg_data = recent_emg[i] if i < len(recent_emg) else None
            imu_data = recent_imu[i] if i < len(recent_imu) else None
            visual_data = recent_visual[i]['landmarks'] if i < len(recent_visual) else None

            multimodal_data.append(MultimodalData(
                emg=emg_data,
                imu=imu_data,
                visual_landmarks=visual_data,
                timestamp=current_time - (window_size - i) * 0.01  # 10ms间隔
            ))

        return multimodal_data

    def _extract_multimodal_features(self, data_list: List[MultimodalData]) -> Dict[str, np.ndarray]:
        """提取多模态特征"""
        features = {}

        # EMG特征提取
        if any(d.emg for d in data_list):
            emg_features = self._extract_emg_features(data_list)
            features['emg'] = emg_features

        # IMU特征提取
        if any(d.imu for d in data_list):
            imu_features = self._extract_imu_features(data_list)
            features['imu'] = imu_features

        # 视觉特征提取
        if any(d.visual_landmarks is not None for d in data_list):
            visual_features = self._extract_visual_features(data_list)
            features['visual'] = visual_features

        return features

    def _extract_emg_features(self, data_list: List[MultimodalData]) -> np.ndarray:
        """提取EMG特征"""
        emg_signals = []

        for data in data_list:
            if data.emg:
                # 计算每个通道的RMS值
                rms_values = np.sqrt(np.mean(data.emg.channels ** 2, axis=1))
                emg_signals.append(rms_values)

        if not emg_signals:
            return np.zeros(16)  # 16个通道的零特征

        emg_matrix = np.array(emg_signals)

        # 提取统计特征
        features = np.concatenate([
            np.mean(emg_matrix, axis=0),  # 均值
            np.std(emg_matrix, axis=0),   # 标准差
            np.max(emg_matrix, axis=0),   # 最大值
            np.min(emg_matrix, axis=0)    # 最小值
        ])

        return features

    def _extract_imu_features(self, data_list: List[MultimodalData]) -> np.ndarray:
        """提取IMU特征"""
        accel_data = []
        gyro_data = []

        for data in data_list:
            if data.imu:
                accel_data.append(data.imu.accelerometer)
                gyro_data.append(data.imu.gyroscope)

        if not accel_data:
            return np.zeros(18)  # 9个加速度特征 + 9个陀螺仪特征

        accel_matrix = np.array(accel_data)
        gyro_matrix = np.array(gyro_data)

        # 提取统计特征
        accel_features = np.concatenate([
            np.mean(accel_matrix, axis=0),
            np.std(accel_matrix, axis=0),
            np.max(accel_matrix, axis=0) - np.min(accel_matrix, axis=0)  # 范围
        ])

        gyro_features = np.concatenate([
            np.mean(gyro_matrix, axis=0),
            np.std(gyro_matrix, axis=0),
            np.max(gyro_matrix, axis=0) - np.min(gyro_matrix, axis=0)
        ])

        return np.concatenate([accel_features, gyro_features])

    def _extract_visual_features(self, data_list: List[MultimodalData]) -> np.ndarray:
        """提取视觉特征"""
        landmarks_data = []

        for data in data_list:
            if data.visual_landmarks is not None:
                # 简化：只使用手部关键点 (468-509)
                if len(data.visual_landmarks) > 509:
                    hand_landmarks = data.visual_landmarks[468:510]
                    landmarks_data.append(hand_landmarks.flatten())

        if not landmarks_data:
            return np.zeros(126)  # 42个手部关键点 * 3坐标

        landmarks_matrix = np.array(landmarks_data)

        # 提取运动特征
        if len(landmarks_matrix) > 1:
            velocity = np.diff(landmarks_matrix, axis=0)
            features = np.concatenate([
                np.mean(landmarks_matrix, axis=0),
                np.std(landmarks_matrix, axis=0),
                np.mean(velocity, axis=0) if len(velocity) > 0 else np.zeros(landmarks_matrix.shape[1])
            ])
        else:
            features = landmarks_matrix[0]

        return features

    async def _mindspore_fusion_inference(self, features: Dict[str, np.ndarray]) -> Dict:
        """MindSpore融合推理"""
        try:
            # 准备输入数据
            input_tensor = self._prepare_fusion_input(features)

            # 模型推理
            inputs = self.fusion_model.get_inputs()
            inputs[0].set_data_from_numpy(input_tensor)

            self.fusion_model.predict(inputs)

            outputs = self.fusion_model.get_outputs()
            prediction = outputs[0].get_data_to_numpy()

            # 解析预测结果
            result = self._parse_fusion_output(prediction, features)

            return result

        except Exception as e:
            logger.error(f"MindSpore融合推理失败: {e}")
            raise

    async def _mock_fusion_inference(self, features: Dict[str, np.ndarray]) -> Dict:
        """模拟融合推理"""
        # 模拟推理延迟
        await asyncio.sleep(0.02)

        # 基于特征计算模拟预测
        modalities_used = list(features.keys())

        # 简单的规则基预测
        if 'emg' in features and 'imu' in features:
            # EMG + IMU 融合
            emg_activity = np.mean(features['emg'][:16])  # EMG活动强度
            imu_movement = np.linalg.norm(features['imu'][:3])  # IMU运动强度

            if emg_activity > 0.1 and imu_movement > 1.0:
                prediction = "手语动作"
                confidence = min(0.9, (emg_activity + imu_movement / 10) / 2)
            else:
                prediction = "静止"
                confidence = 0.7
        elif 'visual' in features:
            # 仅视觉
            prediction = "视觉手语"
            confidence = 0.6
        else:
            prediction = "未知"
            confidence = 0.3

        return {
            "prediction": prediction,
            "confidence": confidence,
            "modalities_used": modalities_used,
            "feature_summary": {mod: feat.shape for mod, feat in features.items()}
        }

    def _prepare_fusion_input(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """准备融合模型输入"""
        # 将所有特征拼接成单一向量
        feature_vector = []

        # 固定顺序拼接特征
        if 'emg' in features:
            feature_vector.extend(features['emg'])
        else:
            feature_vector.extend(np.zeros(64))  # EMG特征维度

        if 'imu' in features:
            feature_vector.extend(features['imu'])
        else:
            feature_vector.extend(np.zeros(18))  # IMU特征维度

        if 'visual' in features:
            feature_vector.extend(features['visual'])
        else:
            feature_vector.extend(np.zeros(126))  # 视觉特征维度

        # 添加批次维度
        return np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    def _parse_fusion_output(self, prediction: np.ndarray, features: Dict[str, np.ndarray]) -> Dict:
        """解析融合输出"""
        # 假设输出是分类概率
        if len(prediction.shape) > 1:
            prediction = prediction[0]

        # 获取最高概率的类别
        max_idx = np.argmax(prediction)
        confidence = prediction[max_idx]

        # 手语词汇映射
        vocab = ["你好", "谢谢", "再见", "请", "对不起", "是", "不是", "好", "不好"]

        if max_idx < len(vocab):
            predicted_word = vocab[max_idx]
        else:
            predicted_word = f"手语_{max_idx}"

        return {
            "prediction": predicted_word,
            "confidence": float(confidence),
            "modalities_used": list(features.keys()),
            "class_probabilities": prediction.tolist()
        }

    def _update_latency_stats(self, latency: float):
        """更新延迟统计"""
        total_predictions = self.sensor_stats["fusion_predictions"]
        if total_predictions > 0:
            current_avg = self.sensor_stats["average_latency"]
            new_avg = (current_avg * (total_predictions - 1) + latency) / total_predictions
            self.sensor_stats["average_latency"] = new_avg

    async def get_stats(self) -> Dict:
        """获取服务统计信息"""
        return {
            **self.sensor_stats,
            "is_loaded": self.is_loaded,
            "is_collecting": self.is_collecting,
            "device_type": self.device_type,
            "buffer_sizes": {
                "emg": len(self.emg_buffer),
                "imu": len(self.imu_buffer),
                "visual": len(self.visual_buffer)
            },
            "config": {
                "emg_enabled": self.config.emg_enabled,
                "imu_enabled": self.config.imu_enabled,
                "visual_enabled": self.config.visual_enabled,
                "fusion_mode": self.config.fusion_mode.value
            }
        }

    async def update_config(self, new_config: Dict):
        """更新配置"""
        if 'emg_enabled' in new_config:
            self.config.emg_enabled = new_config['emg_enabled']
        if 'imu_enabled' in new_config:
            self.config.imu_enabled = new_config['imu_enabled']
        if 'visual_enabled' in new_config:
            self.config.visual_enabled = new_config['visual_enabled']
        if 'fusion_mode' in new_config:
            self.config.fusion_mode = FusionMode(new_config['fusion_mode'])

        logger.info("多模态传感器配置已更新")

    async def cleanup(self):
        """清理资源"""
        try:
            # 停止数据收集
            self.stop_collection.set()
            
            # 等待一小段时间确保线程停止
            await asyncio.sleep(0.1)

            # 关闭设备连接
            if self.emg_device and self.emg_device != "mock_emg":
                self.emg_device.close()
            if self.imu_device and self.imu_device != "mock_imu":
                self.imu_device.close()

            # 清空缓冲区
            self.emg_buffer.clear()
            self.imu_buffer.clear()
            self.visual_buffer.clear()

            self.is_loaded = False
            logger.info("多模态传感器服务资源清理完成")

        except Exception as e:
            logger.error(f"多模态传感器服务清理失败: {e}")


# 全局服务实例
multimodal_sensor_service = MultimodalSensorService()
