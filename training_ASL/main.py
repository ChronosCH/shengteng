#!/usr/bin/env python3
"""
ASL手语识别项目主入口脚本
"""
from __future__ import annotations

import os
import sys
import argparse

# 确保优先导入本地 config.py（避免与 site-packages 中的同名包冲突）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config


def _ensure_src_on_path():
    """将项目 src 目录置于 sys.path 前端，避免与第三方同名包冲突。"""
    src_dir = os.path.join(Config.PROJECT_ROOT, 'src')
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def setup_environment():
    """设置项目环境"""
    project_root = Config.PROJECT_ROOT
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # 确保 src 在路径前端
    _ensure_src_on_path()
    Config.setup_directories()
    print("项目环境设置完成!")


def _pick_checkpoint(model_path_arg: str | None) -> str | None:
    """统一的 checkpoint 选择逻辑。"""
    if model_path_arg and os.path.exists(model_path_arg):
        return model_path_arg
    ckpt_dir = Config.CHECKPOINT_DIR
    if not os.path.isdir(ckpt_dir):
        return None
    best_ckpt = os.path.join(ckpt_dir, 'best.ckpt')
    if os.path.exists(best_ckpt):
        return best_ckpt
    latest_ckpt = os.path.join(ckpt_dir, 'latest.ckpt')
    if os.path.exists(latest_ckpt):
        return latest_ckpt
    ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if ckpts:
        ckpts.sort(key=lambda p: os.path.getmtime(p))
        return ckpts[-1]
    return None


def run_data_analysis():
    print("运行数据分析...")
    from utils.visualization import generate_dataset_report
    generate_dataset_report(Config.DATA_DIR)


def run_training(variant='medium', experiment='full_training', resume_from=None,
                 overrides: dict | None = None):
    print(f"开始训练 - 模型变体: {variant}, 实验配置: {experiment}")
    if resume_from:
        print(f"从检查点恢复训练: {resume_from}")
    _ensure_src_on_path()
    try:
        # 优先无前缀导入（使用本地 src 目录）
        from train import ASLTrainer  # type: ignore
    except Exception:
        # 回退到包前缀导入
        from src.train import ASLTrainer  # type: ignore
    from config import EXPERIMENT_CONFIGS
    config = Config.get_train_config(variant)
    if experiment in EXPERIMENT_CONFIGS:
        config.update(EXPERIMENT_CONFIGS[experiment])
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v
    if config.get('use_gpu') is False:
        config['device_id'] = 0
    trainer = ASLTrainer(config, resume_from=resume_from)
    trainer.train()
    trainer.evaluate()


def run_inference(model_path=None):
    model_path = _pick_checkpoint(model_path)
    if model_path is None or not os.path.exists(model_path):
        print("错误: 找不到模型文件，请先运行训练")
        return
    print(f"使用模型: {model_path}")
    _ensure_src_on_path()
    try:
        from inference import ASLDemo  # type: ignore
    except Exception:
        from src.inference import ASLDemo  # type: ignore
    config = Config.get_train_config()
    demo = ASLDemo(model_path, config)
    demo.demo_random_samples(10)
    demo.run_interactive_demo()


def run_evaluation(model_path=None):
    model_path = _pick_checkpoint(model_path)
    if model_path is None or not os.path.exists(model_path):
        print("错误: 找不到模型文件，请先运行训练")
        return
    print(f"评估模型: {model_path}")
    _ensure_src_on_path()
    try:
        from inference import ASLPredictor  # type: ignore
    except Exception:
        from src.inference import ASLPredictor  # type: ignore
    import pandas as pd
    config = Config.get_train_config()
    predictor = ASLPredictor(model_path, config)
    test_csv = os.path.join(Config.DATA_DIR, "splits", "test.csv")
    video_dir = Config.VIDEO_DIR
    predictions, accuracy = predictor.evaluate_predictions(test_csv, video_dir)
    results_df = pd.DataFrame(predictions)
    results_path = os.path.join(Config.RESULTS_DIR, "test_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"评估完成! 准确率: {accuracy:.4f}")
    print(f"详细结果已保存到: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="ASL手语识别项目")
    parser.add_argument('command', choices=['setup', 'analyze', 'train', 'inference', 'eval'],
                        help='要执行的命令')
    parser.add_argument('--variant', default='medium', choices=['small', 'medium', 'large'],
                        help='模型变体 (默认: medium)')
    # 动态读取实验配置可选项
    try:
        from config import EXPERIMENT_CONFIGS as _EXPCFGS
        _experiment_choices = sorted(list(_EXPCFGS.keys()))
    except Exception:
        _experiment_choices = ['quick_test', 'full_training', 'large_model', 'ultra_fast']
    parser.add_argument('--experiment', default='full_training',
                        choices=_experiment_choices,
                        help='实验配置 (默认: full_training)')
    parser.add_argument('--model-path', type=str, help='模型文件路径 (用于inference和eval)')
    parser.add_argument('--resume', type=str, help='从指定检查点恢复训练')
    # 训练超参覆盖
    parser.add_argument('--batch-size', type=int, dest='batch_size', help='训练批次大小')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', help='学习率')
    parser.add_argument('--epochs', type=int, dest='epochs', help='训练轮数')
    parser.add_argument('--workers', type=int, dest='num_workers', help='数据加载进程数')
    parser.add_argument('--eval-interval', type=int, dest='eval_interval', help='验证间隔（单位：epoch）')
    parser.add_argument('--subset-train', type=int, dest='subset_train', help='限制训练集使用的样本总数')
    parser.add_argument('--subset-val', type=int, dest='subset_val', help='限制验证集使用的样本总数')
    parser.add_argument('--subset-test', type=int, dest='subset_test', help='限制测试集使用的样本总数')
    parser.add_argument('--top-k-glosses', type=int, dest='top_k_glosses',
                        help='仅保留训练集中出现频率最高的前K个词汇')
    parser.add_argument('--max-per-class', type=int, dest='max_samples_per_class',
                        help='每个类别最多使用的样本数')
    parser.add_argument('--use-augmentation', dest='use_data_augmentation', action='store_true',
                        help='启用训练时的数据增强')
    parser.add_argument('--hflip-prob', type=float, dest='horizontal_flip_prob',
                        help='水平翻转概率 (0-1)')
    parser.add_argument('--rotation-range', type=float, dest='rotation_range',
                        help='旋转范围（度）')
    parser.add_argument('--brightness-range', type=float, dest='brightness_range',
                        help='亮度扰动范围 (0-1)')
    parser.add_argument('--contrast-range', type=float, dest='contrast_range',
                        help='对比度扰动范围 (0-1)')
    parser.add_argument('--frame-sampling-mode', type=str, dest='frame_sampling_mode',
                        choices=['uniform', 'stride', 'random_segment'],
                        help='帧采样模式')
    parser.add_argument('--frame-sampling-stride', type=int, dest='frame_sampling_stride',
                        help='帧采样步长')
    parser.add_argument('--frame-sampling-rand', dest='frame_sampling_random_offset', action='store_true',
                        help='启用帧采样随机偏移')
    parser.add_argument('--early-stopping-patience', type=int, dest='early_stopping_patience',
                        help='验证指标多少次无提升后触发早停（<=0 表示禁用早停）')
    parser.add_argument('--no-early-stopping', dest='no_early_stopping', action='store_true',
                        help='禁用早停（等价于 --early-stopping-patience 0）')
    parser.set_defaults(use_data_augmentation=None,
                        frame_sampling_random_offset=None,
                        early_stopping_patience=None,
                        no_early_stopping=False)
    # 设备
    parser.add_argument('--device-id', type=int, dest='device_id', help='设备ID（GPU）')
    parser.add_argument('--cpu', action='store_true', help='使用CPU训练（覆盖GPU设置）')
    # 详细日志
    parser.add_argument('--verbose', action='store_true', help='输出更详细的过程日志')
    # 新增：每多少步打印一次
    parser.add_argument('--print-steps', type=int, dest='print_steps', help='每多少步打印一次训练日志（Loss等）')
    args = parser.parse_args()

    if args.verbose:
        os.environ['ASL_VERBOSE'] = '1'
        try:
            import importlib
            cfg_mod = importlib.import_module('config')
            print(f"[VERBOSE] Using config module: {getattr(cfg_mod, '__file__', str(cfg_mod))}")
        except Exception:
            pass

    setup_environment()

    if args.command == 'setup':
        print("项目设置完成!")
        Config.print_config()
        return
    if args.command == 'analyze':
        run_data_analysis()
        return
    if args.command == 'train':
        overrides = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'num_workers': args.num_workers,
            'eval_interval': args.eval_interval,
            'subset_train': args.subset_train,
            'subset_val': args.subset_val,
            'subset_test': args.subset_test,
            'top_k_glosses': args.top_k_glosses,
            'max_samples_per_class': args.max_samples_per_class,
            'use_data_augmentation': args.use_data_augmentation,
            'horizontal_flip_prob': args.horizontal_flip_prob,
            'rotation_range': args.rotation_range,
            'brightness_range': args.brightness_range,
            'contrast_range': args.contrast_range,
            'frame_sampling_mode': args.frame_sampling_mode,
            'frame_sampling_stride': args.frame_sampling_stride,
            'frame_sampling_random_offset': args.frame_sampling_random_offset,
            'early_stopping_patience': (0 if args.no_early_stopping else args.early_stopping_patience),
            'device_id': args.device_id,
            'use_gpu': (False if args.cpu else None),
            'verbose': (True if args.verbose else None),
            'print_steps': args.print_steps,
        }
        # verbose 打开但未显式设置时，给一个更频繁的打印频率
        if args.verbose and (args.print_steps is None):
            overrides['print_steps'] = 20
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if args.cpu:
            overrides['use_gpu'] = False
        run_training(args.variant, args.experiment, args.resume, overrides)
        return
    if args.command == 'inference':
        run_inference(args.model_path)
        return
    if args.command == 'eval':
        run_evaluation(args.model_path)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
