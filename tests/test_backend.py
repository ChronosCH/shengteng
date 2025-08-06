"""
后端服务测试
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# 导入要测试的模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app
from services.mediapipe_service import MediaPipeService
from services.cslr_service import CSLRService


class TestBackendAPI:
    """后端API测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "SignAvatar Web API" in response.text
    
    def test_health_check(self):
        """测试健康检查端点"""
        response = self.client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "mediapipe" in data["services"]
        assert "cslr" in data["services"]
        assert "websocket" in data["services"]
    
    def test_websocket_connection(self):
        """测试WebSocket连接"""
        with self.client.websocket_connect("/ws/sign-recognition") as websocket:
            # 测试连接建立
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            
            # 测试发送配置消息
            config_message = {
                "type": "config",
                "payload": {"test": "value"}
            }
            websocket.send_json(config_message)
            
            response = websocket.receive_json()
            assert response["type"] == "config_updated"


class TestMediaPipeService:
    """MediaPipe服务测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.service = MediaPipeService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """测试服务初始化"""
        assert self.service.mp_holistic is not None
        assert self.service.holistic is not None
        assert self.service.stats["total_frames_processed"] == 0
    
    def test_landmarks_to_array(self):
        """测试关键点转换为数组"""
        # 模拟关键点数据
        mock_landmarks = {
            "pose": [[0.1, 0.2, 0.3, 0.9]] * 33,
            "left_hand": [[0.4, 0.5, 0.6]] * 21,
            "right_hand": [[0.7, 0.8, 0.9]] * 21,
            "face": [[0.1, 0.1, 0.1]] * 468,
        }
        
        result = self.service.landmarks_to_array(mock_landmarks)
        assert result.shape == (543, 3)
        assert result.dtype == 'float32'
    
    def test_stats_tracking(self):
        """测试性能统计"""
        initial_stats = self.service.get_stats()
        assert "total_frames_processed" in initial_stats
        assert "average_processing_time" in initial_stats
        assert "average_fps" in initial_stats
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """测试资源清理"""
        await self.service.cleanup()
        # 验证清理后的状态


class TestCSLRService:
    """CSLR服务测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.service = CSLRService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """测试服务初始化"""
        assert not self.service.is_loaded
        assert len(self.service.sequence_buffer) == 0
        assert self.service.stats["total_predictions"] == 0
    
    @pytest.mark.asyncio
    async def test_vocabulary_creation(self):
        """测试词汇表创建"""
        await self.service._create_default_vocabulary()
        assert len(self.service.vocab) > 0
        assert "<blank>" in self.service.vocab
        assert "<unk>" in self.service.vocab
    
    @pytest.mark.asyncio
    async def test_mock_inference(self):
        """测试模拟推理"""
        # 创建模拟输入数据
        input_data = [[0.1, 0.2, 0.3] * 543] * 10  # 10帧数据
        input_array = self.service._prepare_input_data()
        
        # 测试模拟推理
        result = await self.service._mock_inference(input_array)
        assert result.shape[0] == 1  # batch size
        assert result.shape[2] == len(self.service.vocab)  # vocab size
    
    def test_ctc_decode(self):
        """测试CTC解码"""
        # 创建模拟预测结果
        import numpy as np
        mock_prediction = np.random.rand(1, 20, 10).astype(np.float32)
        
        result = self.service._ctc_decode(mock_prediction)
        assert "decoded_ids" in result
        assert "confidence" in result
        assert isinstance(result["decoded_ids"], list)
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_predict_insufficient_frames(self):
        """测试帧数不足的情况"""
        landmarks = [[0.1, 0.2, 0.3] * 543]  # 只有1帧
        
        result = await self.service.predict(landmarks)
        assert result["status"] == "insufficient_frames"
        assert result["text"] == ""
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """测试资源清理"""
        await self.service.cleanup()
        assert len(self.service.sequence_buffer) == 0


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.client = TestClient(app)
    
    def test_full_recognition_pipeline(self):
        """测试完整的识别流程"""
        with self.client.websocket_connect("/ws/sign-recognition") as websocket:
            # 1. 建立连接
            connection_msg = websocket.receive_json()
            assert connection_msg["type"] == "connection_established"
            
            # 2. 发送关键点数据
            landmarks_msg = {
                "type": "landmarks",
                "payload": {
                    "landmarks": [[0.1, 0.2, 0.3] * 543] * 15,  # 15帧数据
                    "timestamp": 1234567890,
                    "frameId": 1
                }
            }
            websocket.send_json(landmarks_msg)
            
            # 3. 接收识别结果
            result_msg = websocket.receive_json()
            assert result_msg["type"] == "recognition_result"
            assert "text" in result_msg["payload"]
            assert "confidence" in result_msg["payload"]
    
    def test_error_handling(self):
        """测试错误处理"""
        with self.client.websocket_connect("/ws/sign-recognition") as websocket:
            # 发送无效消息
            invalid_msg = {
                "type": "invalid_type",
                "payload": {}
            }
            websocket.send_json(invalid_msg)
            
            # 应该收到错误响应
            error_msg = websocket.receive_json()
            assert error_msg["type"] == "error"


class TestPerformance:
    """性能测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.mediapipe_service = MediaPipeService()
        self.cslr_service = CSLRService()
    
    @pytest.mark.asyncio
    async def test_mediapipe_performance(self):
        """测试MediaPipe性能"""
        import numpy as np
        import time
        
        # 创建模拟图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试处理时间
        start_time = time.time()
        result = self.mediapipe_service.extract_landmarks(test_image)
        processing_time = time.time() - start_time
        
        # 验证性能要求 (应该在50ms以内)
        assert processing_time < 0.05
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_cslr_performance(self):
        """测试CSLR推理性能"""
        import time
        
        # 准备测试数据
        landmarks = [[0.1, 0.2, 0.3] * 543] * 30  # 30帧数据
        
        # 测试推理时间
        start_time = time.time()
        result = await self.cslr_service.predict(landmarks)
        inference_time = time.time() - start_time
        
        # 验证性能要求 (应该在100ms以内)
        assert inference_time < 0.1
        assert result["status"] in ["success", "insufficient_frames"]
    
    def test_end_to_end_latency(self):
        """测试端到端延迟"""
        with TestClient(app).websocket_connect("/ws/sign-recognition") as websocket:
            import time
            
            # 记录开始时间
            start_time = time.time()
            
            # 发送数据
            landmarks_msg = {
                "type": "landmarks",
                "payload": {
                    "landmarks": [[0.1, 0.2, 0.3] * 543] * 20,
                    "timestamp": start_time * 1000,
                    "frameId": 1
                }
            }
            websocket.send_json(landmarks_msg)
            
            # 接收结果
            websocket.receive_json()  # connection message
            result_msg = websocket.receive_json()
            
            # 计算延迟
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            
            # 验证延迟要求 (应该在150ms以内)
            assert latency < 150
            print(f"端到端延迟: {latency:.2f}ms")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
