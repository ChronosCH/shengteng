"""
Haptic Feedback Service for DeafBlind Users
为盲聋用户提供触觉反馈的服务
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


class HapticPattern(Enum):
    """触觉模式"""
    PULSE = "pulse"          # 脉冲
    VIBRATION = "vibration"  # 振动
    TAP = "tap"             # 轻拍
    WAVE = "wave"           # 波浪
    RHYTHM = "rhythm"       # 节奏


class HapticIntensity(Enum):
    """触觉强度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BrailleMode(Enum):
    """盲文模式"""
    STANDARD = "standard"    # 标准盲文
    CONTRACTED = "contracted"  # 缩写盲文
    COMPUTER = "computer"    # 计算机盲文


@dataclass
class HapticCommand:
    """触觉命令"""
    actuator_id: int  # 致动器ID (0-15)
    pattern: HapticPattern
    intensity: HapticIntensity
    duration: float  # 持续时间(秒)
    delay: float = 0.0  # 延迟时间(秒)


@dataclass
class BrailleCell:
    """盲文单元"""
    dots: List[bool]  # 6个点的状态 [1,2,3,4,5,6]
    character: str    # 对应字符
    duration: float = 1.0  # 显示持续时间


@dataclass
class HapticMessage:
    """触觉消息"""
    text: str
    commands: List[HapticCommand]
    braille_cells: List[BrailleCell]
    total_duration: float
    timestamp: float


class HapticService:
    """触觉反馈服务"""
    
    def __init__(self):
        self.haptic_device = None
        self.braille_device = None
        self.is_loaded = False
        self.is_active = False
        
        # 触觉致动器配置 (16个触点)
        self.num_actuators = 16
        self.actuator_positions = self._init_actuator_positions()
        
        # 盲文显示器配置 (6点盲文)
        self.braille_cells = 8  # 8个盲文单元
        
        # 消息队列
        self.message_queue = deque(maxlen=100)
        self.current_message = None
        
        # 播放线程
        self.playback_thread = None
        self.stop_playback = threading.Event()
        
        # 统计信息
        self.haptic_stats = {
            "total_messages": 0,
            "successful_outputs": 0,
            "average_latency": 0.0,
            "device_status": "disconnected"
        }
        
        # 语义映射
        self.semantic_patterns = self._init_semantic_patterns()
        self.braille_mapping = self._init_braille_mapping()
        
    def _init_actuator_positions(self) -> Dict[int, Tuple[float, float]]:
        """初始化致动器位置 (手套上的16个触点)"""
        positions = {}
        
        # 手指触点 (每个手指3个触点)
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, finger in enumerate(fingers):
            for j in range(3):  # 指尖、中段、根部
                actuator_id = i * 3 + j
                if actuator_id < 15:  # 前15个触点
                    positions[actuator_id] = (i * 0.2, j * 0.3)
        
        # 手掌中心触点
        positions[15] = (0.5, 0.5)
        
        return positions
    
    def _init_semantic_patterns(self) -> Dict[str, List[HapticCommand]]:
        """初始化语义触觉模式"""
        patterns = {
            # 基本情感
            "开心": [
                HapticCommand(0, HapticPattern.PULSE, HapticIntensity.MEDIUM, 0.2),
                HapticCommand(1, HapticPattern.PULSE, HapticIntensity.MEDIUM, 0.2, 0.1),
                HapticCommand(2, HapticPattern.PULSE, HapticIntensity.MEDIUM, 0.2, 0.2),
            ],
            "难过": [
                HapticCommand(15, HapticPattern.VIBRATION, HapticIntensity.LOW, 1.0),
            ],
            "生气": [
                HapticCommand(i, HapticPattern.TAP, HapticIntensity.HIGH, 0.1, i * 0.05)
                for i in range(5)
            ],
            
            # 基本动作
            "你好": [
                HapticCommand(0, HapticPattern.WAVE, HapticIntensity.MEDIUM, 0.5),
                HapticCommand(5, HapticPattern.WAVE, HapticIntensity.MEDIUM, 0.5, 0.2),
            ],
            "再见": [
                HapticCommand(i, HapticPattern.PULSE, HapticIntensity.MEDIUM, 0.1, i * 0.1)
                for i in range(10, 15)
            ],
            "谢谢": [
                HapticCommand(15, HapticPattern.RHYTHM, HapticIntensity.MEDIUM, 0.8),
            ],
            
            # 方向指示
            "左": [HapticCommand(0, HapticPattern.TAP, HapticIntensity.HIGH, 0.3)],
            "右": [HapticCommand(4, HapticPattern.TAP, HapticIntensity.HIGH, 0.3)],
            "上": [HapticCommand(15, HapticPattern.PULSE, HapticIntensity.HIGH, 0.3)],
            "下": [HapticCommand(10, HapticPattern.PULSE, HapticIntensity.HIGH, 0.3)],
            
            # 数字 (1-10)
            **{str(i): [HapticCommand(j, HapticPattern.TAP, HapticIntensity.MEDIUM, 0.1, j * 0.1) 
                       for j in range(i)] for i in range(1, 11)},
        }
        
        return patterns
    
    def _init_braille_mapping(self) -> Dict[str, List[bool]]:
        """初始化盲文映射 (简化版本)"""
        # 标准6点盲文映射
        braille_map = {
            'a': [True, False, False, False, False, False],
            'b': [True, True, False, False, False, False],
            'c': [True, False, False, True, False, False],
            'd': [True, False, False, True, True, False],
            'e': [True, False, False, False, True, False],
            'f': [True, True, False, True, False, False],
            'g': [True, True, False, True, True, False],
            'h': [True, True, False, False, True, False],
            'i': [False, True, False, True, False, False],
            'j': [False, True, False, True, True, False],
            'k': [True, False, True, False, False, False],
            'l': [True, True, True, False, False, False],
            'm': [True, False, True, True, False, False],
            'n': [True, False, True, True, True, False],
            'o': [True, False, True, False, True, False],
            'p': [True, True, True, True, False, False],
            'q': [True, True, True, True, True, False],
            'r': [True, True, True, False, True, False],
            's': [False, True, True, True, False, False],
            't': [False, True, True, True, True, False],
            'u': [True, False, True, False, False, True],
            'v': [True, True, True, False, False, True],
            'w': [False, True, False, True, True, True],
            'x': [True, False, True, True, False, True],
            'y': [True, False, True, True, True, True],
            'z': [True, False, True, False, True, True],
            ' ': [False, False, False, False, False, False],  # 空格
        }
        
        # 添加中文常用字符的简化映射
        chinese_map = {
            '你': [True, True, False, True, False, True],
            '好': [True, False, True, False, True, True],
            '我': [True, True, True, False, True, True],
            '是': [False, True, True, False, True, True],
            '的': [True, False, False, False, False, True],
            '在': [True, True, False, False, False, True],
            '有': [False, False, True, True, True, True],
            '不': [False, True, False, False, False, True],
            '了': [False, False, True, False, False, True],
            '人': [False, False, False, True, True, True],
        }
        
        braille_map.update(chinese_map)
        return braille_map
    
    async def initialize(self):
        """初始化服务"""
        try:
            logger.info("正在初始化触觉反馈服务...")
            
            # 初始化触觉设备
            await self._initialize_haptic_device()
            
            # 初始化盲文设备
            await self._initialize_braille_device()
            
            self.is_loaded = True
            logger.info("触觉反馈服务初始化完成")
            
        except Exception as e:
            logger.error(f"触觉反馈服务初始化失败: {e}")
            raise
    
    async def _initialize_haptic_device(self):
        """初始化触觉设备"""
        try:
            # 尝试连接触觉手套设备
            haptic_port = getattr(settings, 'HAPTIC_DEVICE_PORT', '/dev/ttyUSB2')
            self.haptic_device = serial.Serial(haptic_port, 115200, timeout=1)
            
            # 发送初始化命令
            init_command = {
                "type": "init",
                "actuators": self.num_actuators,
                "positions": self.actuator_positions
            }
            self.haptic_device.write(json.dumps(init_command).encode() + b'\n')
            
            logger.info(f"触觉设备连接成功: {haptic_port}")
            self.haptic_stats["device_status"] = "connected"
            
        except Exception as e:
            logger.warning(f"触觉设备连接失败: {e}, 使用模拟设备")
            self.haptic_device = "mock_haptic"
            self.haptic_stats["device_status"] = "mock"
    
    async def _initialize_braille_device(self):
        """初始化盲文设备"""
        try:
            # 尝试连接盲文显示器
            braille_port = getattr(settings, 'BRAILLE_DEVICE_PORT', '/dev/ttyUSB3')
            self.braille_device = serial.Serial(braille_port, 9600, timeout=1)
            
            # 发送初始化命令
            init_command = {
                "type": "init",
                "cells": self.braille_cells
            }
            self.braille_device.write(json.dumps(init_command).encode() + b'\n')
            
            logger.info(f"盲文设备连接成功: {braille_port}")
            
        except Exception as e:
            logger.warning(f"盲文设备连接失败: {e}, 使用模拟设备")
            self.braille_device = "mock_braille"
    
    async def send_haptic_message(self, text: str, use_braille: bool = True, 
                                use_haptic: bool = True) -> HapticMessage:
        """
        发送触觉消息
        
        Args:
            text: 要传达的文本
            use_braille: 是否使用盲文输出
            use_haptic: 是否使用触觉模式
            
        Returns:
            触觉消息对象
        """
        if not self.is_loaded:
            raise RuntimeError("触觉反馈服务未初始化")
            
        start_time = time.time()
        self.haptic_stats["total_messages"] += 1
        
        try:
            # 生成触觉命令
            haptic_commands = []
            if use_haptic:
                haptic_commands = self._generate_haptic_commands(text)
            
            # 生成盲文单元
            braille_cells = []
            if use_braille:
                braille_cells = self._generate_braille_cells(text)
            
            # 计算总持续时间
            haptic_duration = max([cmd.delay + cmd.duration for cmd in haptic_commands], default=0)
            braille_duration = sum([cell.duration for cell in braille_cells], default=0)
            total_duration = max(haptic_duration, braille_duration)
            
            # 创建触觉消息
            message = HapticMessage(
                text=text,
                commands=haptic_commands,
                braille_cells=braille_cells,
                total_duration=total_duration,
                timestamp=time.time()
            )
            
            # 添加到队列
            self.message_queue.append(message)
            
            # 如果没有正在播放的消息，立即开始播放
            if not self.is_active:
                await self._start_playback()
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_latency_stats(processing_time)
            
            logger.info(f"触觉消息已发送: {text}")
            return message
            
        except Exception as e:
            logger.error(f"发送触觉消息失败: {e}")
            raise
    
    def _generate_haptic_commands(self, text: str) -> List[HapticCommand]:
        """生成触觉命令"""
        commands = []
        
        # 检查是否有预定义的语义模式
        if text in self.semantic_patterns:
            commands.extend(self.semantic_patterns[text])
        else:
            # 为未知文本生成基础触觉模式
            for i, char in enumerate(text[:10]):  # 限制前10个字符
                actuator_id = i % self.num_actuators
                commands.append(HapticCommand(
                    actuator_id=actuator_id,
                    pattern=HapticPattern.PULSE,
                    intensity=HapticIntensity.MEDIUM,
                    duration=0.3,
                    delay=i * 0.2
                ))
        
        return commands
    
    def _generate_braille_cells(self, text: str) -> List[BrailleCell]:
        """生成盲文单元"""
        cells = []
        
        for char in text.lower():
            if char in self.braille_mapping:
                cell = BrailleCell(
                    dots=self.braille_mapping[char],
                    character=char,
                    duration=1.0
                )
                cells.append(cell)
            else:
                # 未知字符用空格表示
                cell = BrailleCell(
                    dots=[False] * 6,
                    character=' ',
                    duration=0.5
                )
                cells.append(cell)
        
        return cells

    async def _start_playback(self):
        """开始播放触觉消息"""
        if self.is_active:
            return

        self.is_active = True
        self.stop_playback.clear()

        # 启动播放线程
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.start()

        logger.info("触觉消息播放已开始")

    async def stop_playback(self):
        """停止播放触觉消息"""
        if not self.is_active:
            return

        self.stop_playback.set()
        self.is_active = False

        if self.playback_thread:
            self.playback_thread.join(timeout=5)

        logger.info("触觉消息播放已停止")

    def _playback_loop(self):
        """播放循环"""
        while not self.stop_playback.is_set() and self.message_queue:
            try:
                # 获取下一条消息
                message = self.message_queue.popleft()
                self.current_message = message

                # 并行播放触觉和盲文
                haptic_thread = threading.Thread(
                    target=self._play_haptic_commands,
                    args=(message.commands,)
                )
                braille_thread = threading.Thread(
                    target=self._play_braille_cells,
                    args=(message.braille_cells,)
                )

                haptic_thread.start()
                braille_thread.start()

                # 等待播放完成
                haptic_thread.join()
                braille_thread.join()

                # 消息间隔
                time.sleep(0.5)

                self.haptic_stats["successful_outputs"] += 1

            except Exception as e:
                logger.error(f"播放触觉消息错误: {e}")
                time.sleep(0.1)

        self.is_active = False
        self.current_message = None

    def _play_haptic_commands(self, commands: List[HapticCommand]):
        """播放触觉命令"""
        if not commands:
            return

        start_time = time.time()

        for command in commands:
            if self.stop_playback.is_set():
                break

            # 等待延迟时间
            elapsed = time.time() - start_time
            if command.delay > elapsed:
                time.sleep(command.delay - elapsed)

            # 发送触觉命令
            self._send_haptic_command(command)

    def _play_braille_cells(self, cells: List[BrailleCell]):
        """播放盲文单元"""
        if not cells:
            return

        for cell in cells:
            if self.stop_playback.is_set():
                break

            # 发送盲文单元
            self._send_braille_cell(cell)

            # 等待显示时间
            time.sleep(cell.duration)

    def _send_haptic_command(self, command: HapticCommand):
        """发送触觉命令到设备"""
        try:
            if self.haptic_device == "mock_haptic":
                self._mock_haptic_output(command)
                return

            # 构建设备命令
            device_command = {
                "type": "haptic",
                "actuator": command.actuator_id,
                "pattern": command.pattern.value,
                "intensity": self._get_intensity_value(command.intensity),
                "duration": int(command.duration * 1000)  # 转换为毫秒
            }

            # 发送到设备
            command_json = json.dumps(device_command) + '\n'
            self.haptic_device.write(command_json.encode())

        except Exception as e:
            logger.error(f"发送触觉命令失败: {e}")

    def _send_braille_cell(self, cell: BrailleCell):
        """发送盲文单元到设备"""
        try:
            if self.braille_device == "mock_braille":
                self._mock_braille_output(cell)
                return

            # 构建盲文命令
            device_command = {
                "type": "braille",
                "dots": cell.dots,
                "character": cell.character
            }

            # 发送到设备
            command_json = json.dumps(device_command) + '\n'
            self.braille_device.write(command_json.encode())

        except Exception as e:
            logger.error(f"发送盲文命令失败: {e}")

    def _get_intensity_value(self, intensity: HapticIntensity) -> int:
        """获取强度数值"""
        intensity_map = {
            HapticIntensity.LOW: 30,
            HapticIntensity.MEDIUM: 60,
            HapticIntensity.HIGH: 100
        }
        return intensity_map[intensity]

    def _mock_haptic_output(self, command: HapticCommand):
        """模拟触觉输出"""
        logger.debug(f"模拟触觉: 致动器{command.actuator_id}, "
                    f"模式{command.pattern.value}, "
                    f"强度{command.intensity.value}, "
                    f"持续{command.duration}s")

    def _mock_braille_output(self, cell: BrailleCell):
        """模拟盲文输出"""
        dots_str = ''.join(['●' if dot else '○' for dot in cell.dots])
        logger.debug(f"模拟盲文: {cell.character} -> {dots_str}")

    async def send_semantic_feedback(self, semantic_type: str, intensity: str = "medium"):
        """
        发送语义反馈

        Args:
            semantic_type: 语义类型 (emotion, action, direction, number等)
            intensity: 强度级别
        """
        try:
            intensity_enum = HapticIntensity(intensity)

            if semantic_type in self.semantic_patterns:
                # 调整强度
                commands = []
                for cmd in self.semantic_patterns[semantic_type]:
                    new_cmd = HapticCommand(
                        actuator_id=cmd.actuator_id,
                        pattern=cmd.pattern,
                        intensity=intensity_enum,
                        duration=cmd.duration,
                        delay=cmd.delay
                    )
                    commands.append(new_cmd)

                # 创建消息
                message = HapticMessage(
                    text=semantic_type,
                    commands=commands,
                    braille_cells=[],
                    total_duration=max([cmd.delay + cmd.duration for cmd in commands]),
                    timestamp=time.time()
                )

                self.message_queue.append(message)

                if not self.is_active:
                    await self._start_playback()

                logger.info(f"语义反馈已发送: {semantic_type}")

            else:
                logger.warning(f"未知语义类型: {semantic_type}")

        except Exception as e:
            logger.error(f"发送语义反馈失败: {e}")
            raise

    async def send_emergency_alert(self):
        """发送紧急警报"""
        try:
            # 紧急警报模式：所有致动器高强度脉冲
            commands = []
            for i in range(self.num_actuators):
                commands.append(HapticCommand(
                    actuator_id=i,
                    pattern=HapticPattern.PULSE,
                    intensity=HapticIntensity.HIGH,
                    duration=0.2,
                    delay=i * 0.05
                ))

            # 重复3次
            for repeat in range(3):
                for cmd in commands:
                    new_cmd = HapticCommand(
                        actuator_id=cmd.actuator_id,
                        pattern=cmd.pattern,
                        intensity=cmd.intensity,
                        duration=cmd.duration,
                        delay=cmd.delay + repeat * 1.0
                    )
                    commands.append(new_cmd)

            message = HapticMessage(
                text="EMERGENCY",
                commands=commands,
                braille_cells=[],
                total_duration=3.0,
                timestamp=time.time()
            )

            # 清空队列，优先播放紧急消息
            self.message_queue.clear()
            self.message_queue.append(message)

            if not self.is_active:
                await self._start_playback()

            logger.warning("紧急警报已发送")

        except Exception as e:
            logger.error(f"发送紧急警报失败: {e}")
            raise

    def _update_latency_stats(self, latency: float):
        """更新延迟统计"""
        total_messages = self.haptic_stats["total_messages"]
        if total_messages > 0:
            current_avg = self.haptic_stats["average_latency"]
            new_avg = (current_avg * (total_messages - 1) + latency) / total_messages
            self.haptic_stats["average_latency"] = new_avg

    async def get_stats(self) -> Dict:
        """获取服务统计信息"""
        return {
            **self.haptic_stats,
            "is_loaded": self.is_loaded,
            "is_active": self.is_active,
            "queue_size": len(self.message_queue),
            "current_message": self.current_message.text if self.current_message else None,
            "device_config": {
                "num_actuators": self.num_actuators,
                "braille_cells": self.braille_cells,
                "semantic_patterns": len(self.semantic_patterns),
                "braille_characters": len(self.braille_mapping)
            }
        }

    async def test_devices(self) -> Dict[str, bool]:
        """测试设备连接"""
        results = {}

        # 测试触觉设备
        try:
            test_command = HapticCommand(
                actuator_id=0,
                pattern=HapticPattern.PULSE,
                intensity=HapticIntensity.LOW,
                duration=0.1
            )
            self._send_haptic_command(test_command)
            results["haptic_device"] = True
        except Exception as e:
            logger.error(f"触觉设备测试失败: {e}")
            results["haptic_device"] = False

        # 测试盲文设备
        try:
            test_cell = BrailleCell(
                dots=[True, False, False, False, False, False],
                character='a',
                duration=0.1
            )
            self._send_braille_cell(test_cell)
            results["braille_device"] = True
        except Exception as e:
            logger.error(f"盲文设备测试失败: {e}")
            results["braille_device"] = False

        return results

    async def cleanup(self):
        """清理资源"""
        try:
            # 停止播放
            self.stop_playback.set()
            
            # 等待一小段时间确保线程停止
            await asyncio.sleep(0.1)

            # 关闭设备连接
            if self.haptic_device and self.haptic_device != "mock_haptic":
                self.haptic_device.close()
            if self.braille_device and self.braille_device != "mock_braille":
                self.braille_device.close()

            # 清空队列
            self.message_queue.clear()

            self.is_loaded = False
            logger.info("触觉反馈服务资源清理完成")

        except Exception as e:
            logger.error(f"触觉反馈服务清理失败: {e}")


# 全局服务实例
haptic_service = HapticService()
