# ASL项目配置

from __future__ import annotations
import os


class Config:
    """
    项目全局配置（UTF-8，无乱码）。
    注意：保持与现有代码字段名一致，避免引入兼容性问题。
    """

    # ===== 路径配置 =====
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ASL_Citizen", "ASL_Citizen")
    VIDEO_DIR = os.path.join(DATA_DIR, "videos")
    SPLITS_DIR = os.path.join(DATA_DIR, "splits")

    # 预提取帧（用于极致加速）
    USE_PREEXTRACTED_FRAMES = True
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")  # 每个视频一个子目录，存放 000.jpg, 001.jpg, ...
    FRAME_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

    @classmethod
    def validate_paths(cls):
        """校验关键路径是否存在。"""
        paths_to_check = [
            (cls.PROJECT_ROOT, "项目根目录"),
            (cls.DATA_DIR, "数据根目录"),
            (cls.VIDEO_DIR, "视频目录"),
            (cls.SPLITS_DIR, "数据分割目录"),
        ]
        missing = []
        for path, desc in paths_to_check:
            if not os.path.exists(path):
                missing.append((path, desc))
                print(f"[缺失] {desc}: {path}")
            else:
                print(f"[存在] {desc}: {path}")
        return len(missing) == 0

    # ===== 输出目录 =====
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # ===== 数据与增强（极速配置） =====
    INPUT_SIZE = (160, 160)       # (H, W)
    SEQUENCE_LENGTH = 8

    USE_DATA_AUGMENTATION = False
    HORIZONTAL_FLIP_PROB = 0.0
    ROTATION_RANGE = 0
    BRIGHTNESS_RANGE = 0
    CONTRAST_RANGE = 0
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # ===== 模型与正则化 =====
    DROPOUT_RATE = 0.3
    LABEL_SMOOTHING = 0.1

    # ===== 训练超参 =====
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    TOP_K_GLOSSES: int | None = None       # 仅保留出现频率最高的前K个词汇
    MAX_SAMPLES_PER_GLOSS: int | None = None  # 每个类别最多使用的样本数

    # 学习率调度（none/cosine/exponential）
    LR_SCHEDULER = 'cosine'
    WARMUP_EPOCHS = 0
    MIN_LR_RATIO = 0.1
    USE_LR_SCHEDULER = True
    LR_DECAY_STEPS = 10
    LR_DECAY_RATE = 0.8

    OPTIMIZER = 'Adam'    # 'Adam', 'SGD', 'AdamW'
    MOMENTUM = 0.9

    # AMP/混合精度与梯度
    USE_AMP = True
    AMP_LEVEL = 'O2'
    LOSS_SCALE = 1024.0
    GRAD_CLIP_NORM = None

    # ===== 评估/早停/日志 =====
    EVAL_INTERVAL = 1
    EARLY_STOPPING_PATIENCE = 10

    SAVE_CHECKPOINT_STEPS = 1000
    KEEP_CHECKPOINT_MAX = 5
    SAVE_BEST_ONLY = True

    PRINT_STEPS = 100
    LOG_INTERVAL = 50

    # ===== 设备与性能 =====
    USE_GPU = True
    DEVICE_ID = 0
    INFERENCE_BATCH_SIZE = 16
    NUM_WORKERS = 8
    SEED = 42
    PREFETCH_BUFFER_SIZE = 2
    SHUFFLE_BUFFER_SIZE = 1024
    DATASET_SINK_MODE = False

    # 数据集后端行为控制
    DATASET_USE_PYTHON_MP = False   # 关闭 GeneratorDataset python_multiprocessing，避免卡住
    DATASET_SAFE_GET_SIZE = True    # 若 True，跳过 get_dataset_size()，用样本数+batch 估算

    # ===== 实验信息 =====
    EXPERIMENT_NAME = "asl_recognition"
    VERSION = "v1.0"

    # 模型变体
    MODEL_VARIANTS = {
        'small': {
            'base_channels': 32,
            'dropout_rate': 0.2,
            'batch_size': 32,
        },
        'medium': {
            'base_channels': 64,
            'dropout_rate': 0.3,
            'batch_size': 16,
        },
        'large': {
            'base_channels': 128,
            'dropout_rate': 0.4,
            'batch_size': 4,
        },
    }

    # ===== 缓存与高效解码配置 =====
    CACHE_ENABLED = False                       # 关闭帧级缓存
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    CACHE_MAX_SIZE_GB = 50
    CACHE_FORMAT = "webp"
    CACHE_QUALITY = 90

    DECODER_BACKEND = "auto"  # auto/decord/pyav/opencv

    # ===== 帧采样策略 =====
    FRAME_SAMPLING_MODE = 'stride'
    FRAME_SAMPLING_STRIDE = 2
    FRAME_SAMPLING_RANDOM_OFFSET = False

    @classmethod
    def setup_directories(cls):
        """确保输出目录存在。"""
        for d in (cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.RESULTS_DIR):
            os.makedirs(d, exist_ok=True)
            print(f"[OK] 目录已就绪: {d}")
        if getattr(cls, 'CACHE_ENABLED', False):
            os.makedirs(cls.CACHE_DIR, exist_ok=True)
            print(f"[OK] 目录已就绪: {cls.CACHE_DIR}")
        if getattr(cls, 'USE_PREEXTRACTED_FRAMES', False):
            os.makedirs(cls.FRAMES_DIR, exist_ok=True)
            print(f"[OK] 目录已就绪: {cls.FRAMES_DIR}")

    @classmethod
    def get_model_config(cls, variant: str = 'medium'):
        """获取指定变体的模型配置。"""
        if variant not in cls.MODEL_VARIANTS:
            variant = 'medium'
        cfg = cls.MODEL_VARIANTS[variant].copy()
        cfg.update({
            'input_size': cls.INPUT_SIZE,
            'sequence_length': cls.SEQUENCE_LENGTH,
            'dropout_rate': cfg.get('dropout_rate', cls.DROPOUT_RATE),
        })
        return cfg

    @classmethod
    def get_train_config(cls, variant: str = 'medium'):
        """返回训练用的完整配置（基础版，可在上层覆盖）。"""
        m = cls.get_model_config(variant)
        return {
            'data_dir': cls.DATA_DIR,
            'checkpoint_dir': cls.CHECKPOINT_DIR,
            'log_dir': cls.LOG_DIR,

            'input_size': cls.INPUT_SIZE,
            'sequence_length': cls.SEQUENCE_LENGTH,
            'batch_size': m.get('batch_size', cls.BATCH_SIZE),
            'base_channels': m.get('base_channels', 64),

            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'label_smoothing': cls.LABEL_SMOOTHING,

            'num_workers': cls.NUM_WORKERS,
            'seed': cls.SEED,

            'eval_interval': cls.EVAL_INTERVAL,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            'use_lr_scheduler': cls.USE_LR_SCHEDULER,
            'lr_scheduler': cls.LR_SCHEDULER,
            'warmup_epochs': cls.WARMUP_EPOCHS,
            'min_lr_ratio': cls.MIN_LR_RATIO,
            'lr_decay_steps': cls.LR_DECAY_STEPS,
            'lr_decay_rate': cls.LR_DECAY_RATE,
            'save_steps': cls.SAVE_CHECKPOINT_STEPS,
            'keep_checkpoint_max': cls.KEEP_CHECKPOINT_MAX,
            'print_steps': cls.PRINT_STEPS,

            'optimizer': cls.OPTIMIZER,
            'momentum': cls.MOMENTUM,
            'use_amp': cls.USE_AMP,
            'amp_level': cls.AMP_LEVEL,
            'loss_scale': cls.LOSS_SCALE,
            'grad_clip_norm': cls.GRAD_CLIP_NORM,

            'use_gpu': cls.USE_GPU,
            'device_id': cls.DEVICE_ID,
            'dataset_sink_mode': cls.DATASET_SINK_MODE,

            'mean': cls.MEAN,
            'std': cls.STD,
            'use_data_augmentation': cls.USE_DATA_AUGMENTATION,
            'horizontal_flip_prob': cls.HORIZONTAL_FLIP_PROB,
            'rotation_range': cls.ROTATION_RANGE,
            'brightness_range': cls.BRIGHTNESS_RANGE,
            'contrast_range': cls.CONTRAST_RANGE,
            'frame_sampling_mode': cls.FRAME_SAMPLING_MODE,
            'frame_sampling_stride': cls.FRAME_SAMPLING_STRIDE,
            'frame_sampling_random_offset': cls.FRAME_SAMPLING_RANDOM_OFFSET,
            'prefetch_buffer_size': cls.PREFETCH_BUFFER_SIZE,
            'shuffle_buffer_size': cls.SHUFFLE_BUFFER_SIZE,
            'top_k_glosses': cls.TOP_K_GLOSSES,
            'max_samples_per_class': cls.MAX_SAMPLES_PER_GLOSS,
        }

    @classmethod
    def print_config(cls):
        """打印关键配置。"""
        print("=== ASL 训练配置 ===")
        print(f"项目根目录: {cls.PROJECT_ROOT}")
        print(f"数据目录  : {cls.DATA_DIR}")
        print(f"检查点目录: {cls.CHECKPOINT_DIR}")
        print(f"输入尺寸  : {cls.INPUT_SIZE}")
        print(f"序列长度  : {cls.SEQUENCE_LENGTH}")
        print(f"batch size: {cls.BATCH_SIZE}")
        print(f"epochs    : {cls.EPOCHS}")
        print(f"学习率    : {cls.LEARNING_RATE}")
        print(f"使用GPU   : {cls.USE_GPU}")


# 预设实验配置
EXPERIMENT_CONFIGS = {
    'quick_test': {
        'epochs': 5,
        'batch_size': 4,
        'eval_interval': 1,
        'print_steps': 10,
        'save_steps': 100,
    },
    'full_training': {
        'epochs': 50,
        'batch_size': 8,
        'eval_interval': 1,
        'print_steps': 100,
        'save_steps': 1000,
    },
    'large_model': {
        'epochs': 100,
        'batch_size': 4,
        'learning_rate': 5e-5,
        'eval_interval': 2,
        'print_steps': 50,
        'save_steps': 500,
    },
    'ultra_fast': {
        'epochs': 3,
        'batch_size': 16,
        'eval_interval': 1,
        'print_steps': 20,
        'save_steps': 200,
        # 新增：使用很小的子集加速验证
        'subset_train': 256,
        'subset_val': 128,
    },
    # ===== 新增：更稳健与自检用的预设 =====
    'quick_warmup': {
        # 加暖启 + 稍高学习率，禁用AMP，提升前期下降速度
        'epochs': 8,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'use_lr_scheduler': True,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 2,
        'min_lr_ratio': 0.1,
        'label_smoothing': 0.05,
        'weight_decay': 5e-5,
        'use_amp': False,
        'eval_interval': 1,
        'print_steps': 20,
        'save_steps': 200,
    },
    'overfit_probe': {
        # 小子集快速过拟合自检：应在数个epoch内显著下降
        'epochs': 10,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'use_lr_scheduler': False,
        'label_smoothing': 0.0,
        'weight_decay': 0.0,
        'use_amp': False,
        'subset_train': 128,
        'subset_val': 128,
        'eval_interval': 1,
        'print_steps': 5,
        'save_steps': 100,
        # MindSpore 内部 Accuracy/TopK 指标在小子集上可能触发 gather 越界
        # 依靠 ValidationMonitor 输出更完整的验证指标即可
        'disable_internal_metrics': True,
    },
    'amp_o1_warmup': {
        # 混合精度 O1 + 暖启，兼顾速度与稳定性
        'epochs': 8,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'use_lr_scheduler': True,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 2,
        'min_lr_ratio': 0.1,
        'label_smoothing': 0.05,
        'weight_decay': 5e-5,
        'use_amp': True,
        'amp_level': 'O1',
        'eval_interval': 1,
        'print_steps': 20,
        'save_steps': 200,
    },
    'common_signs_fast': {
        # 高频词汇快速实验：聚焦前 300 个类别，每类最多 40 条样本
        'epochs': 20,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'use_lr_scheduler': True,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 2,
        'min_lr_ratio': 0.2,
        'label_smoothing': 0.05,
        'weight_decay': 1e-4,
        'use_amp': False,
        'top_k_glosses': 300,
        'max_samples_per_class': 40,
        'subset_train': 12000,
        'subset_val': 3000,
        'subset_test': 3000,
        'early_stopping_patience': 20,
        'eval_interval': 1,
        'print_steps': 20,
        'save_steps': 200,
    },
    'common_signs_aug150': {
        # 高频词汇 + 强数据增强 + 更高样本密度
        'epochs': 200,
        'batch_size': 16,
        'learning_rate': 5e-4,
        'use_lr_scheduler': True,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 3,
        'min_lr_ratio': 0.15,
        'label_smoothing': 0.0,
        'weight_decay': 5e-5,
        'use_amp': True,
        'amp_level': 'O1',
        'top_k_glosses': 150,
        'max_samples_per_class': 60,
        'subset_train': 9000,
        'subset_val': 1800,
        'subset_test': 1800,
        'use_data_augmentation': True,
        'horizontal_flip_prob': 0.5,
        'rotation_range': 15,
        'brightness_range': 0.25,
        'contrast_range': 0.25,
        'frame_sampling_mode': 'random_segment',
        'frame_sampling_stride': 2,
        'frame_sampling_random_offset': True,
        'early_stopping_patience': 25,
        'eval_interval': 1,
        'print_steps': 20,
        'save_steps': 200,
    },
}


if __name__ == "__main__":
    Config.setup_directories()
    Config.print_config()
    print("\n=== 路径校验 ===")
    Config.validate_paths()