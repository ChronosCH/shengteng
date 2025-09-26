from __future__ import annotations
import os
import time
import math
import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import nn, context, Model, Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.optim import Adam, SGD, AdamWeightDecay
from mindspore.nn.metrics import Accuracy, Top5CategoricalAccuracy
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net

# 稳健导入：优先相对，失败则回退绝对
try:
    from .data_loader import ASLDataset, VideoProcessor, create_mindspore_dataset, get_class_mapping
    from .model import ASLRecognitionModel, ASLLoss
    from .validation_monitor import ValidationMonitor
    from .checkpoint_alias import CheckpointAliasCallback
    from .optimizer_state_cb import OptimizerStateCallback
except Exception:
    import sys as _sys
    _PARENT = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_PARENT)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from src.data_loader import ASLDataset, VideoProcessor, create_mindspore_dataset, get_class_mapping
    from src.model import ASLRecognitionModel, ASLLoss
    from src.validation_monitor import ValidationMonitor
    from src.checkpoint_alias import CheckpointAliasCallback
    from src.optimizer_state_cb import OptimizerStateCallback

from config import Config


def _vprint(*a, **k):
    if os.environ.get('ASL_VERBOSE') == '1' or k.pop('_force', False):
        print(*a, **k)


class ASLTrainer:
    """ASL手语识别训练器（支持断点续训）"""
    
    def __init__(self, config, resume_from=None):
        self.config = config
        self.resume_from = resume_from
        self.start_epoch = 0
        self.best_accuracy = 0.0
        self._resume_global_step = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epochs': []
        }
        self.allowed_glosses: set[str] | None = None
        self.max_samples_per_class: int | None = None
        self._drop_unknown = False
        self._subset_test: int | None = None
        if config.get('verbose'):
            os.environ['ASL_VERBOSE'] = '1'
        _vprint("[VERBOSE] 初始化训练器")
        
        # 设置MindSpore环境
        device_target = "GPU" if config.get('use_gpu', True) else "CPU"
        _vprint(f"[VERBOSE] 设置 MindSpore 上下文: target={device_target}, device_id={config.get('device_id', 0)}")
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target=device_target,
            device_id=config.get('device_id', 0)
        )
        
        # 设置随机种子
        set_seed(config['seed'])
        np.random.seed(config['seed'])
        _vprint(f"[VERBOSE] 种子: {config['seed']}")
        
        # 初始化组件
        t0=time.time(); self.setup_data(); _vprint(f"[VERBOSE] 数据设置完成，用时 {time.time()-t0:.2f}s")
        t0=time.time(); self.setup_model(); _vprint(f"[VERBOSE] 模型设置完成，用时 {time.time()-t0:.2f}s")
        t0=time.time(); self.setup_training(); _vprint(f"[VERBOSE] 训练设置完成，用时 {time.time()-t0:.2f}s")
        
        # 如果指定了恢复路径，加载检查点
        if self.resume_from:
            _vprint(f"[VERBOSE] 尝试从检查点恢复: {self.resume_from}")
            self.load_checkpoint()
            # 断点后对齐学习率
            self._adjust_lr_after_resume()

    def setup_data(self):
        """设置数据"""
        print("正在设置数据...")
        _vprint(f"[VERBOSE] 数据目录: {self.config['data_dir']}")
        
        # 数据路径
        data_dir = self.config['data_dir']
        video_dir = os.path.join(data_dir, "videos")
        splits_dir = os.path.join(data_dir, "splits")
        _vprint(f"[VERBOSE] splits: {splits_dir}")
        
        # 创建视频处理器（训练/验证分离增强）
        common_kwargs = dict(
            target_size=self.config['input_size'],
            sequence_length=self.config['sequence_length']
        )
        augment_flag = bool(self.config.get('use_data_augmentation', getattr(Config, 'USE_DATA_AUGMENTATION', False)))
        hflip_prob = self.config.get('horizontal_flip_prob', getattr(Config, 'HORIZONTAL_FLIP_PROB', 0.0))
        rotation_range = self.config.get('rotation_range', getattr(Config, 'ROTATION_RANGE', 0.0))
        brightness_range = self.config.get('brightness_range', getattr(Config, 'BRIGHTNESS_RANGE', 0.0))
        contrast_range = self.config.get('contrast_range', getattr(Config, 'CONTRAST_RANGE', 0.0))
        mean = self.config.get('mean', getattr(Config, 'MEAN', (0.485, 0.456, 0.406)))
        std = self.config.get('std', getattr(Config, 'STD', (0.229, 0.224, 0.225)))
        sampling_mode_train = self.config.get('frame_sampling_mode', getattr(Config, 'FRAME_SAMPLING_MODE', 'uniform'))
        sampling_stride = self.config.get('frame_sampling_stride', getattr(Config, 'FRAME_SAMPLING_STRIDE', 1))
        sampling_rand = self.config.get('frame_sampling_random_offset', getattr(Config, 'FRAME_SAMPLING_RANDOM_OFFSET', False))

        _vprint(f"[VERBOSE] 处理器参数: size={self.config['input_size']}, seq_len={self.config['sequence_length']}, augment={augment_flag}")
        self.train_processor = VideoProcessor(
            **common_kwargs,
            augment=augment_flag,
            hflip_prob=hflip_prob,
            rotation_range=rotation_range,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            mean=mean,
            std=std,
            sampling_mode=sampling_mode_train,
            sampling_stride=sampling_stride,
            sampling_rand=sampling_rand
        )
        self.val_processor = VideoProcessor(
            **common_kwargs,
            augment=False,
            hflip_prob=0.0,
            rotation_range=0.0,
            brightness_range=0.0,
            contrast_range=0.0,
            mean=mean,
            std=std,
            sampling_mode='uniform',
            sampling_stride=sampling_stride,
            sampling_rand=False
        )

        # 类别映射（支持高频词汇与子集重映射）
        train_csv = os.path.join(splits_dir, "train.csv")
        val_csv = os.path.join(splits_dir, "val.csv")
        subset_train = int(self.config.get('subset_train', 0) or 0)
        subset_val = int(self.config.get('subset_val', 0) or 0)
        subset_test = int(self.config.get('subset_test', 0) or 0)
        top_k_glosses = int(self.config.get('top_k_glosses', 0) or 0)
        max_per_class_cfg = int(self.config.get('max_samples_per_class', 0) or 0)
        self.max_samples_per_class = max_per_class_cfg if max_per_class_cfg > 0 else None
        self._subset_test = subset_test if subset_test > 0 else None

        mapping_classes: list[str] | None = None
        if top_k_glosses > 0:
            try:
                train_df_full = pd.read_csv(train_csv)
                gloss_counts = train_df_full['Gloss'].astype(str).value_counts()
                top_series = gloss_counts.head(top_k_glosses)
                mapping_classes = top_series.index.tolist()
                self.allowed_glosses = set(mapping_classes)
                covered = int(top_series.sum())
                total = int(gloss_counts.sum())
                _vprint(f"[VERBOSE] 仅保留前 {len(mapping_classes)} 个高频类别 (覆盖 {covered}/{total} 样本)")
            except Exception as e:
                self.allowed_glosses = None
                mapping_classes = None
                print(f"[WARN] 统计高频类别失败: {e}，将使用完整类别集")
        elif subset_train > 0 or subset_val > 0:
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            train_subset_df = train_df.head(subset_train) if subset_train > 0 else train_df
            val_subset_df = val_df.head(subset_val) if subset_val > 0 else val_df
            mapping_classes = sorted(set(train_subset_df['Gloss'].astype(str)) | set(val_subset_df['Gloss'].astype(str)))
            self.allowed_glosses = set(mapping_classes)
            _vprint(f"[VERBOSE] 子集类别数: {len(mapping_classes)}")
        else:
            self.allowed_glosses = None

        if mapping_classes:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(mapping_classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self.num_classes = len(self.class_to_idx)
        else:
            self.class_to_idx, self.idx_to_class = get_class_mapping(train_csv)
            self.num_classes = len(self.class_to_idx)
        _vprint(f"[VERBOSE] 使用类别数: {self.num_classes}")

        self._drop_unknown = bool(mapping_classes)
        drop_unknown = self._drop_unknown
        if self.max_samples_per_class:
            _vprint(f"[VERBOSE] 每类样本将限制为最多 {self.max_samples_per_class} 条")

        # 创建数据集
        train_dataset_py = ASLDataset(
            csv_path=train_csv,
            video_dir=video_dir,
            processor=self.train_processor,
            class_to_idx=self.class_to_idx,
            max_samples=(subset_train if subset_train > 0 else None),
            drop_unknown=drop_unknown,
            max_per_class=self.max_samples_per_class
        )
        _vprint(f"[VERBOSE] 训练样本数: {len(train_dataset_py)}")
        
        val_dataset_py = ASLDataset(
            csv_path=val_csv,
            video_dir=video_dir,
            processor=self.val_processor,
            class_to_idx=self.class_to_idx,
            max_samples=(subset_val if subset_val > 0 else None),
            drop_unknown=drop_unknown,
            max_per_class=self.max_samples_per_class
        )
        _vprint(f"[VERBOSE] 验证样本数: {len(val_dataset_py)}")
        
        # 保存样本数供 steps 估算
        self._train_len = len(train_dataset_py)
        self._val_len = len(val_dataset_py)
        
        # MindSpore 数据集
        self.train_dataset = create_mindspore_dataset(
            train_dataset_py,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_parallel_workers=self.config['num_workers']
        )
        self.val_dataset = create_mindspore_dataset(
            val_dataset_py,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_parallel_workers=self.config['num_workers']
        )
        
        print(f"训练集大小: {self._train_len}")
        print(f"验证集大小: {self._val_len}")
        try:
            safe_get_size = bool(getattr(Config, 'DATASET_SAFE_GET_SIZE', False))
            if not safe_get_size:
                _vprint(f"[VERBOSE] train_dataset size (batches): {self.train_dataset.get_dataset_size()}")
                _vprint(f"[VERBOSE] val_dataset size (batches): {self.val_dataset.get_dataset_size()}")
            else:
                _vprint("[VERBOSE] 跳过 get_dataset_size（DATASET_SAFE_GET_SIZE=True）")
        except Exception:
            pass
    
    def setup_model(self):
        """设置模型"""
        print("正在设置模型...")
        
        # 创建模型
        self.model = ASLRecognitionModel(
            num_classes=self.num_classes,
            sequence_length=self.config['sequence_length'],
            input_size=self.config['input_size'],
            base_channels=self.config.get('base_channels', 64)
        )
        _vprint("[VERBOSE] 模型已构建")
        
        # 损失函数
        self.loss_fn = ASLLoss(
            num_classes=self.num_classes,
            label_smoothing=self.config['label_smoothing']
        )
        
        print(f"模型参数数量: {sum(p.size for p in self.model.get_parameters())}")
    
    def _build_lr(self):
        """构建学习率序列（更稳健）"""
        steps_per_epoch = int(self.steps_per_epoch or 0)
        total_epochs = int(self.config['epochs'])
        base_lr = float(self.config['learning_rate'])
        scheduler = self.config.get('lr_scheduler', 'cosine') if self.config.get('use_lr_scheduler', True) else 'none'
        warmup_epochs = int(self.config.get('warmup_epochs', 0))
        min_lr_ratio = float(self.config.get('min_lr_ratio', 0.1))
        decay_steps_ep = int(self.config.get('lr_decay_steps', 10))
        gamma = float(self.config.get('lr_decay_rate', 0.8))
        total_steps = steps_per_epoch * total_epochs
        warmup_steps = max(0, warmup_epochs * steps_per_epoch)
        
        if steps_per_epoch == 0 or total_steps == 0:
            _vprint("[VERBOSE] 无法构建学习率计划（steps_per_epoch=0）")
            return None
        _vprint(f"[VERBOSE] 学习率计划: scheduler={scheduler}, total_steps={total_steps}, warmup={warmup_steps}")

        lr_list = []
        if scheduler == 'cosine':
            min_lr = base_lr * min_lr_ratio
            for global_step in range(total_steps):
                if global_step < warmup_steps and warmup_steps > 0:
                    lr = base_lr * (global_step + 1) / warmup_steps
                else:
                    t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))
                lr_list.append(lr)
        elif scheduler == 'exponential':
            decay_steps = max(1, decay_steps_ep * steps_per_epoch)
            for global_step in range(total_steps):
                if global_step < warmup_steps and warmup_steps > 0:
                    lr = base_lr * (global_step + 1) / warmup_steps
                else:
                    lr = base_lr * (gamma ** (global_step // decay_steps))
                lr_list.append(lr)
        else:
            lr_list = [base_lr] * total_steps
        return Tensor(np.array(lr_list, dtype=np.float32))

    def setup_training(self):
        """设置训练"""
        print("正在设置训练...")
        opt_name = str(self.config.get('optimizer', 'Adam')).lower()
        weight_decay = float(self.config['weight_decay'])
        
        # 记录每轮步数，优先 get_dataset_size，失败时用样本数估算
        steps_per_epoch = None
        safe_get_size = bool(getattr(Config, 'DATASET_SAFE_GET_SIZE', False))
        if not safe_get_size:
            try:
                steps_per_epoch = int(self.train_dataset.get_dataset_size())
                _vprint(f"[VERBOSE] steps_per_epoch(get_dataset_size)={steps_per_epoch}")
            except Exception as e:
                _vprint(f"[VERBOSE] get_dataset_size 失败: {e}")
        if not steps_per_epoch:
            bs = max(1, int(self.config['batch_size']))
            steps_per_epoch = max(1, math.ceil(self._train_len / bs))
            _vprint(f"[VERBOSE] steps_per_epoch(估算)={steps_per_epoch}, train_len={self._train_len}, bs={bs}")
        self.steps_per_epoch = steps_per_epoch
        
        # 学习率
        lr_tensor = self._build_lr()
        if lr_tensor is None:
            lr_tensor = Tensor(np.array([self.config['learning_rate']], dtype=np.float32))
        
        # 优化器
        if opt_name == 'sgd':
            self.optimizer = SGD(self.model.trainable_params(), learning_rate=lr_tensor,
                                 momentum=float(self.config.get('momentum', 0.9)), weight_decay=weight_decay)
        elif opt_name == 'adamw':
            self.optimizer = AdamWeightDecay(self.model.trainable_params(), learning_rate=lr_tensor, weight_decay=weight_decay)
        else:
            self.optimizer = Adam(self.model.trainable_params(), learning_rate=lr_tensor, weight_decay=weight_decay)
        _vprint(f"[VERBOSE] 优化器: {opt_name}, weight_decay={weight_decay}")
        
        # AMP 设置
        self.loss_scale_manager = None
        use_amp = bool(self.config.get('use_amp', False))
        self._use_amp = use_amp
        if use_amp:
            ls = self.config.get('loss_scale', 1024.0)
            if isinstance(ls, (int, float)):
                self.loss_scale_manager = FixedLossScaleManager(ls, False)
        # 记录用户期望的 amp_level，但在构建 Model 前统一决策
        self._amp_level_cfg = str(self.config.get('amp_level', 'O0'))

        # 训练/评估网络与指标
        self.train_net = self.model
        self.eval_net = nn.WithEvalCell(self.model, self.loss_fn)
        disable_metrics = bool(self.config.get('disable_internal_metrics', False))
        if disable_metrics:
            _vprint("[VERBOSE] 禁用 MindSpore 内置指标，改用 ValidationMonitor 统计")
            self.metrics = {}
        else:
            self.metrics = {
                'accuracy': Accuracy(),
                'top5_accuracy': Top5CategoricalAccuracy()
            }

    def _adjust_lr_after_resume(self):
        try:
            steps_per_epoch = int(self.steps_per_epoch or 0)
            if steps_per_epoch <= 0:
                return
            if self._resume_global_step is not None:
                offset_steps = int(self._resume_global_step)
            else:
                done_epochs = max(0, int(self.start_epoch) - 1)
                offset_steps = max(0, done_epochs * steps_per_epoch)
            lr = getattr(self.optimizer, 'learning_rate', None)
            if lr is not None and isinstance(lr, Tensor) and getattr(lr, 'ndim', 0) > 0:
                total = lr.shape[0]
                if offset_steps >= total:
                    offset_steps = total - 1
                new_seq = lr.asnumpy()[offset_steps:]
                self.optimizer.learning_rate = Tensor(new_seq.astype(np.float32))
                print(f"[LR] 恢复后截断学习率序列，跳过 {offset_steps} 步")
        except Exception as e:
            print(f"[LR] 学习率恢复调整失败: {e}")

    def load_checkpoint(self):
        if not self.resume_from or not os.path.exists(self.resume_from):
            print(f"检查点文件不存在: {self.resume_from}")
            return False
        print(f"正在从检查点恢复训练: {self.resume_from}")
        try:
            param_dict = load_checkpoint(self.resume_from)
            load_param_into_net(self.model, param_dict)
            print("✅ 模型参数加载成功")
            return True
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            print("将从头开始训练...")
            self.start_epoch = 0
            return False
    
    def train(self):
        print("开始训练...")
        if self.start_epoch > 0:
            print(f"🔄 从第 {self.start_epoch} 个epoch恢复训练")
        _vprint(f"[VERBOSE] 准备检查点与回调")
        
        # 检查点配置
        config_ck = CheckpointConfig(
            save_checkpoint_steps=self.config['save_steps'],
            keep_checkpoint_max=self.config['keep_checkpoint_max']
        )
        checkpoint_cb = ModelCheckpoint(
            prefix="asl_model",
            directory=self.config['checkpoint_dir'],
            config=config_ck
        )
        alias_cb = CheckpointAliasCallback(
            checkpoint_dir=self.config['checkpoint_dir'],
            save_interval_steps=self.config['save_steps'],
            latest_name='latest.ckpt'
        )
        opt_state_cb = OptimizerStateCallback(
            optimizer=self.optimizer,
            alias_callback=alias_cb,
            checkpoint_dir=self.config['checkpoint_dir']
        )
        validation_monitor = ValidationMonitor(
            model=self.model,
            val_dataset=self.val_dataset,
            eval_interval=self.config.get('eval_interval', 1),
            save_dir=os.path.join(self.config['checkpoint_dir'], "validation_results"),
            early_stopping_patience=self.config.get('early_stopping_patience', 10)
        )
        if hasattr(validation_monitor, 'attach_alias_callback'):
            try:
                validation_monitor.attach_alias_callback(alias_cb)
            except Exception:
                pass

        metrics_arg = self.metrics if self.metrics else None

        model = Model(
            self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            metrics=metrics_arg,
            amp_level=(self._amp_level_cfg if self._use_amp else 'O0'),
            loss_scale_manager=(self.loss_scale_manager if self._use_amp else None)
        )
        remaining_epochs = max(0, self.config['epochs'] - self.start_epoch)
        if remaining_epochs <= 0:
            print(f"训练已完成！当前epoch ({self.start_epoch}) >= 目标epochs ({self.config['epochs']})")
            return
        print(f"剩余训练轮数: {remaining_epochs}")
        _vprint(f"[VERBOSE] 将开始 model.train: remaining_epochs={remaining_epochs}, steps_per_epoch={self.steps_per_epoch}")
        
        class _StepTracker(Callback):
            def __init__(self):
                self.last_step = 0
            def step_end(self, run_context):
                cb = run_context.original_args()
                cur_step = getattr(cb, 'cur_step_num', None)
                if cur_step is not None:
                    self.last_step = int(cur_step)
        step_tracker = _StepTracker()
        
        model.train(
            epoch=remaining_epochs,
            train_dataset=self.train_dataset,
            callbacks=[
                LossMonitor(per_print_times=self.config['print_steps']),
                TimeMonitor(),
                checkpoint_cb,
                alias_cb,
                opt_state_cb,
                validation_monitor,
                step_tracker,
            ],
            dataset_sink_mode=bool(self.config.get('dataset_sink_mode', False))
        )
        print("训练完成!")
    
    def evaluate(self, test_csv_path=None):
        if test_csv_path is None:
            test_csv_path = os.path.join(self.config['data_dir'], "splits", "test.csv")
        print("正在评估测试集...")
        test_dataset_py = ASLDataset(
            csv_path=test_csv_path,
            video_dir=os.path.join(self.config['data_dir'], "videos"),
            processor=self.val_processor,
            class_to_idx=self.class_to_idx,
            max_samples=(self._subset_test if self._subset_test else None),
            drop_unknown=self._drop_unknown,
            max_per_class=self.max_samples_per_class
        )
        test_ds = create_mindspore_dataset(
            test_dataset_py,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_parallel_workers=self.config['num_workers']
        )
        metrics_arg = self.metrics if self.metrics else None
        if metrics_arg:
            model = Model(self.model, loss_fn=self.loss_fn, metrics=metrics_arg)
        else:
            model = Model(self.model)
        result = model.eval(test_ds, dataset_sink_mode=False)
        print("测试结果:")
        for name, value in result.items():
            try:
                print(f"{name}: {value:.4f}")
            except Exception:
                print(f"{name}: {value}")
        return result