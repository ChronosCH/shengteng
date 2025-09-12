import json
import os
from typing import Dict, Any

class ConfigManager:
    """训练参数配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            # 数据集配置
            "dataset": {
                "name": "CE-CSL",
                "train_data_path": "data/CE-CSL/video/train",
                "valid_data_path": "data/CE-CSL/video/dev", 
                "test_data_path": "data/CE-CSL/video/test",
                "train_label_path": "data/CE-CSL/label/train.csv",
                "valid_label_path": "data/CE-CSL/label/dev.csv",
                "test_label_path": "data/CE-CSL/label/test.csv",
                "crop_size": 224,
                "max_frames": 300
            },
            
            # 模型配置
            "model": {
                "name": "TFNet",
                "hidden_size": 1024,
                "device_target": "CPU",
                "device_id": 0,
                "enable_graph_kernel": False,
                "enable_reduce_precision": False,
                "enable_auto_mixed_precision": False
            },
            
            # 训练配置
            "training": {
                "batch_size": 2,
                "learning_rate": 0.0001,
                "num_epochs": 55,
                "num_workers": 1,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
                "save_interval": 5,
                "eval_interval": 1,
                "early_stopping_patience": 10,
                "prefetch_size": 2,
                "max_rowsize": 32,
                "enable_data_sink": False
            },
            
            # 优化器配置
            "optimizer": {
                "type": "Adam",
                "lr_scheduler": {
                    "type": "MultiStepLR",
                    "milestones": [35, 45],
                    "gamma": 0.2
                }
            },
            
            # 损失配置
            "loss": {
                "ctc_blank_id": 0,
                "ctc_reduction": "mean",
                "kd_temperature": 8,
                "kd_weight": 25.0
            },
            
            # 路径配置
            "paths": {
                "checkpoint_dir": "training/checkpoints",
                "log_dir": "training/logs",
                "output_dir": "training/output",
                "best_model_path": "training/checkpoints/best_model.ckpt",
                "current_model_path": "training/checkpoints/current_model.ckpt"
            },
            
            # 日志配置
            "logging": {
                "level": "INFO",
                "save_logs": True,
                "print_interval": 10
            },
            
            # GPU优化配置
            "gpu_optimization": {
                "enable_graph_mode": False,
                "enable_mem_reuse": False,
                "max_device_memory": "8GB",
                "enable_profiling": False,
                "enable_dump": False
            }
        }
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # 与默认配置合并
            self._merge_config(self.config, loaded_config)
            print(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
    
    def _merge_config(self, default_config: Dict, loaded_config: Dict):
        """Recursively merge loaded config with default config"""
        for key, value in loaded_config.items():
            if key in default_config:
                if isinstance(value, dict) and isinstance(default_config[key], dict):
                    self._merge_config(default_config[key], value)
                else:
                    default_config[key] = value
            else:
                default_config[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'model.hidden_size')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        # 导航到父字典
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
    
    def update_paths_for_dataset(self, dataset_name: str):
        """Update paths based on dataset name"""
        if dataset_name == "CE-CSL":
            self.set("dataset.train_data_path", "data/CE-CSL/video/train")
            self.set("dataset.valid_data_path", "data/CE-CSL/video/dev")
            self.set("dataset.test_data_path", "data/CE-CSL/video/test")
            self.set("dataset.train_label_path", "data/CE-CSL/label/train.csv")
            self.set("dataset.valid_label_path", "data/CE-CSL/label/dev.csv")
            self.set("dataset.test_label_path", "data/CE-CSL/label/test.csv")
        
        self.set("dataset.name", dataset_name)
    
    def create_directories(self):
        """Create necessary directories with improved error handling"""
        dirs_to_create = [
            self.get("paths.checkpoint_dir"),
            self.get("paths.log_dir"),
            self.get("paths.output_dir")
        ]

        created_dirs = []
        failed_dirs = []

        for dir_path in dirs_to_create:
            if dir_path:
                try:
                    # 为跨平台兼容性标准化路径
                    normalized_path = os.path.normpath(dir_path)
                    os.makedirs(normalized_path, exist_ok=True)
                    created_dirs.append(normalized_path)
                    print(f"✓ Directory ready: {normalized_path}")
                except Exception as e:
                    failed_dirs.append((normalized_path, str(e)))
                    print(f"✗ Failed to create directory {normalized_path}: {e}")

        if failed_dirs:
            print(f"Warning: Failed to create {len(failed_dirs)} directories")
            for dir_path, error in failed_dirs:
                print(f"  - {dir_path}: {error}")
            return False

        print(f"✓ All {len(created_dirs)} directories are ready")
        return True

    def ensure_directory_exists(self, dir_path):
        """Ensure a single directory exists, create if necessary"""
        if not dir_path:
            return False

        try:
            normalized_path = os.path.normpath(dir_path)
            os.makedirs(normalized_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
            return False

    def get_safe_path(self, key_path, create_if_missing=True):
        """Get path and optionally create directory if it doesn't exist"""
        path = self.get(key_path)
        if path and create_if_missing:
            # If it's a file path, create parent directory
            if key_path.endswith('_path') and not key_path.endswith('_dir'):
                parent_dir = os.path.dirname(path)
                if parent_dir:
                    self.ensure_directory_exists(parent_dir)
            # If it's a directory path, create the directory
            elif key_path.endswith('_dir'):
                self.ensure_directory_exists(path)
        return path
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Check required paths
        required_paths = [
            "dataset.train_data_path",
            "dataset.train_label_path"
        ]
        
        for path_key in required_paths:
            path = self.get(path_key)
            if not path or not os.path.exists(path):
                errors.append(f"Required path not found: {path_key} = {path}")
        
        # Check numeric parameters
        if self.get("training.batch_size", 0) <= 0:
            errors.append("batch_size must be positive")
        
        if self.get("training.learning_rate", 0) <= 0:
            errors.append("learning_rate must be positive")
        
        if self.get("model.hidden_size", 0) <= 0:
            errors.append("hidden_size must be positive")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed")
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            "batch_size": self.get("training.batch_size"),
            "learning_rate": self.get("training.learning_rate"),
            "num_epochs": self.get("training.num_epochs"),
            "weight_decay": self.get("training.weight_decay"),
            "gradient_clip_norm": self.get("training.gradient_clip_norm"),
            "device_target": self.get("model.device_target")
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "hidden_size": self.get("model.hidden_size"),
            "device_target": self.get("model.device_target"),
            "dataset_name": self.get("dataset.name")
        }
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset-specific configuration"""
        return {
            "name": self.get("dataset.name"),
            "train_data_path": self.get("dataset.train_data_path"),
            "valid_data_path": self.get("dataset.valid_data_path"),
            "test_data_path": self.get("dataset.test_data_path"),
            "train_label_path": self.get("dataset.train_label_path"),
            "valid_label_path": self.get("dataset.valid_label_path"),
            "test_label_path": self.get("dataset.test_label_path"),
            "crop_size": self.get("dataset.crop_size"),
            "max_frames": self.get("dataset.max_frames")
        }
