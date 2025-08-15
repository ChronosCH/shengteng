"""
训练配置加载器
从JSON配置文件加载预定义的训练配置
支持配置合并和自定义覆盖
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置加载器
        
        Args:
            config_file: 配置文件路径，默认为当前目录下的training_configs.json
        """
        if config_file is None:
            config_file = Path(__file__).parent / "configs" / "training_configs.json"
        
        self.config_file = Path(config_file)
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict:
        """加载配置文件"""
        try:
            if not self.config_file.exists():
                logger.warning(f"配置文件不存在: {self.config_file}")
                return {}
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            
            logger.info(f"配置文件加载成功: {self.config_file}")
            return configs
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return {}
    
    def list_training_configs(self) -> List[str]:
        """列出所有可用的训练配置"""
        return list(self.configs.get('training_configs', {}).keys())
    
    def list_device_configs(self) -> List[str]:
        """列出所有可用的设备配置"""
        return list(self.configs.get('device_configs', {}).keys())
    
    def list_preprocessing_configs(self) -> List[str]:
        """列出所有可用的预处理配置"""
        return list(self.configs.get('data_preprocessing', {}).keys())
    
    def list_training_presets(self) -> List[str]:
        """列出所有可用的训练预设"""
        return list(self.configs.get('training_presets', {}).keys())
    
    def get_training_config(self, config_name: str) -> Dict:
        """获取训练配置"""
        training_configs = self.configs.get('training_configs', {})
        if config_name not in training_configs:
            available = list(training_configs.keys())
            raise ValueError(f"训练配置 '{config_name}' 不存在。可用配置: {available}")
        
        return training_configs[config_name].copy()
    
    def get_device_config(self, config_name: str) -> Dict:
        """获取设备配置"""
        device_configs = self.configs.get('device_configs', {})
        if config_name not in device_configs:
            available = list(device_configs.keys())
            raise ValueError(f"设备配置 '{config_name}' 不存在。可用配置: {available}")
        
        return device_configs[config_name].copy()
    
    def get_preprocessing_config(self, config_name: str) -> Dict:
        """获取预处理配置"""
        preprocessing_configs = self.configs.get('data_preprocessing', {})
        if config_name not in preprocessing_configs:
            available = list(preprocessing_configs.keys())
            raise ValueError(f"预处理配置 '{config_name}' 不存在。可用配置: {available}")
        
        return preprocessing_configs[config_name].copy()
    
    def get_training_preset(self, preset_name: str) -> Dict:
        """获取训练预设"""
        presets = self.configs.get('training_presets', {})
        if preset_name not in presets:
            available = list(presets.keys())
            raise ValueError(f"训练预设 '{preset_name}' 不存在。可用预设: {available}")
        
        preset = presets[preset_name].copy()
        
        # 如果预设指定了基础配置，则加载并合并
        if 'config' in preset:
            base_config = self.get_training_config(preset['config'])
            # 从预设中移除config字段，避免冲突
            base_config_name = preset.pop('config')
            # 合并配置（预设优先级更高）
            merged_config = self._merge_configs(base_config, preset)
            return merged_config
        
        return preset
    
    def get_common_paths(self) -> Dict[str, str]:
        """获取通用路径配置"""
        return self.configs.get('common_paths', {}).copy()
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """合并配置，override_config优先级更高"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 递归合并字典
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # 直接覆盖
                merged[key] = value
        
        return merged
    
    def create_full_config(self, 
                          training_config: str = "tfnet_default",
                          device_config: str = "ascend_910",
                          preprocessing_config: str = "ce_csl_standard",
                          custom_overrides: Dict = None) -> Dict:
        """
        创建完整的配置
        
        Args:
            training_config: 训练配置名称
            device_config: 设备配置名称
            preprocessing_config: 预处理配置名称
            custom_overrides: 自定义覆盖配置
        
        Returns:
            合并后的完整配置
        """
        # 获取基础配置
        train_cfg = self.get_training_config(training_config)
        device_cfg = self.get_device_config(device_config)
        preprocess_cfg = self.get_preprocessing_config(preprocessing_config)
        paths_cfg = self.get_common_paths()
        
        # 合并配置
        full_config = {
            'training': train_cfg,
            'device': device_cfg,
            'preprocessing': preprocess_cfg,
            'paths': paths_cfg,
            'metadata': {
                'training_config_name': training_config,
                'device_config_name': device_config,
                'preprocessing_config_name': preprocessing_config
            }
        }
        
        # 应用自定义覆盖
        if custom_overrides:
            full_config = self._merge_configs(full_config, custom_overrides)
        
        # 设备特定的优化建议
        self._apply_device_optimizations(full_config)
        
        return full_config
    
    def _apply_device_optimizations(self, config: Dict):
        """根据设备类型应用优化建议"""
        device_cfg = config.get('device', {})
        train_cfg = config.get('training', {})
        
        device_target = device_cfg.get('device_target', 'Ascend')
        recommended_batch_size = device_cfg.get('recommended_batch_size')
        max_workers = device_cfg.get('max_workers')
        
        # 根据设备调整批次大小（如果未手动设置）
        if recommended_batch_size and 'batch_size' in train_cfg.get('training_config', {}):
            current_batch_size = train_cfg['training_config']['batch_size']
            if current_batch_size > recommended_batch_size:
                logger.warning(
                    f"当前批次大小 {current_batch_size} 超过设备推荐值 {recommended_batch_size}，"
                    f"建议调整以避免内存溢出"
                )
        
        # 设置最大工作线程数
        if max_workers:
            config.setdefault('system', {})['max_workers'] = max_workers
        
        # 特定设备的优化建议
        if device_target == "Ascend":
            # 昇腾特定优化
            config.setdefault('optimization_hints', {}).update({
                'use_graph_kernel': True,
                'enable_amp': True,
                'dataset_sink_mode': True,
                'recommended_amp_level': 'O1'
            })
        elif device_target == "GPU":
            # GPU特定优化
            config.setdefault('optimization_hints', {}).update({
                'use_mixed_precision': True,
                'enable_cudnn_benchmark': True,
                'recommended_amp_level': 'O2'
            })
        elif device_target == "CPU":
            # CPU特定优化
            config.setdefault('optimization_hints', {}).update({
                'disable_amp': True,
                'use_openmp': True,
                'recommended_batch_size': 4
            })
    
    def save_config(self, config: Dict, output_path: str):
        """保存配置到文件"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            raise
    
    def validate_config(self, config: Dict) -> bool:
        """验证配置的完整性和合理性"""
        try:
            # 检查必需的配置部分
            required_sections = ['training', 'device', 'preprocessing']
            for section in required_sections:
                if section not in config:
                    logger.error(f"缺少必需的配置部分: {section}")
                    return False
            
            # 检查训练配置
            train_cfg = config['training']
            required_train_fields = ['model_config', 'training_config', 'data_config']
            for field in required_train_fields:
                if field not in train_cfg:
                    logger.error(f"训练配置缺少字段: {field}")
                    return False
            
            # 检查模型配置
            model_cfg = train_cfg['model_config']
            if 'model_type' not in model_cfg:
                logger.error("模型配置缺少model_type字段")
                return False
            
            # 检查数值合理性
            training_cfg = train_cfg['training_config']
            if training_cfg.get('batch_size', 0) <= 0:
                logger.error("批次大小必须大于0")
                return False
            
            if training_cfg.get('learning_rate', 0) <= 0:
                logger.error("学习率必须大于0")
                return False
            
            if training_cfg.get('epochs', 0) <= 0:
                logger.error("训练轮数必须大于0")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def get_config_summary(self, config: Dict) -> str:
        """生成配置摘要"""
        try:
            summary_lines = ["配置摘要:"]
            summary_lines.append("=" * 50)
            
            # 模型信息
            model_cfg = config['training']['model_config']
            summary_lines.append(f"模型类型: {model_cfg.get('model_type', 'Unknown')}")
            summary_lines.append(f"词汇表大小: {model_cfg.get('vocab_size', 'Unknown')}")
            summary_lines.append(f"模型维度: {model_cfg.get('d_model', 'Unknown')}")
            
            # 训练信息
            train_cfg = config['training']['training_config']
            summary_lines.append(f"批次大小: {train_cfg.get('batch_size', 'Unknown')}")
            summary_lines.append(f"学习率: {train_cfg.get('learning_rate', 'Unknown')}")
            summary_lines.append(f"训练轮数: {train_cfg.get('epochs', 'Unknown')}")
            
            # 设备信息
            device_cfg = config['device']
            summary_lines.append(f"设备类型: {device_cfg.get('device_target', 'Unknown')}")
            summary_lines.append(f"最大内存: {device_cfg.get('max_device_memory', 'Unknown')}")
            
            # 数据配置
            data_cfg = config['training']['data_config']
            summary_lines.append(f"图像尺寸: {data_cfg.get('image_size', 'Unknown')}")
            summary_lines.append(f"序列长度: {data_cfg.get('max_sequence_length', 'Unknown')}")
            summary_lines.append(f"关键点提取: {data_cfg.get('enable_keypoints', 'Unknown')}")
            
            # 优化配置
            opt_cfg = config['training'].get('optimization_config', {})
            summary_lines.append(f"混合精度: {opt_cfg.get('enable_amp', 'Unknown')}")
            summary_lines.append(f"AMP级别: {opt_cfg.get('amp_level', 'Unknown')}")
            summary_lines.append(f"图内核优化: {opt_cfg.get('enable_graph_kernel', 'Unknown')}")
            
            summary_lines.append("=" * 50)
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"生成配置摘要失败: {e}")
            return "配置摘要生成失败"

def load_config_from_file(config_file: str) -> Dict:
    """从文件加载配置的便捷函数"""
    loader = ConfigLoader(config_file)
    return loader.configs

def create_training_config(training_config: str = "tfnet_default",
                          device_config: str = "ascend_910",
                          preprocessing_config: str = "ce_csl_standard",
                          custom_overrides: Dict = None) -> Dict:
    """创建训练配置的便捷函数"""
    loader = ConfigLoader()
    return loader.create_full_config(
        training_config=training_config,
        device_config=device_config,
        preprocessing_config=preprocessing_config,
        custom_overrides=custom_overrides
    )

def create_config_from_preset(preset_name: str,
                             custom_overrides: Dict = None) -> Dict:
    """从预设创建配置的便捷函数"""
    loader = ConfigLoader()
    preset_config = loader.get_training_preset(preset_name)
    
    if custom_overrides:
        preset_config = loader._merge_configs(preset_config, custom_overrides)
    
    return preset_config

# 使用示例
if __name__ == "__main__":
    # 创建配置加载器
    loader = ConfigLoader()
    
    # 列出可用配置
    print("可用训练配置:", loader.list_training_configs())
    print("可用设备配置:", loader.list_device_configs())
    print("可用预处理配置:", loader.list_preprocessing_configs())
    print("可用训练预设:", loader.list_training_presets())
    
    # 创建完整配置
    config = loader.create_full_config(
        training_config="tfnet_default",
        device_config="ascend_910",
        preprocessing_config="ce_csl_standard"
    )
    
    # 验证配置
    if loader.validate_config(config):
        print("配置验证通过")
        print(loader.get_config_summary(config))
    
    # 保存配置
    loader.save_config(config, "output_config.json")
