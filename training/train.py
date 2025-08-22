#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 训练入口：统一调度不同训练器，默认走真实数据集

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))


def run_enhanced_cecsl(data_root: Path, epochs: int = None):
    from enhanced_cecsl_trainer import EnhancedCECSLConfig, EnhancedCECSLTrainer

    cfg = EnhancedCECSLConfig()
    # 使用真实数据集 CS-CSL，若不存在自动回退 CE-CSL
    cfg.data_root = str(data_root)
    if not Path(cfg.data_root).exists():
        alt = Path(str(data_root).replace('CS-CSL', 'CE-CSL'))
        if alt.exists():
            print(f"[warn] 未找到 {cfg.data_root} ，回退到 {alt}")
            cfg.data_root = str(alt)
    if epochs is not None and epochs > 0:
        cfg.epochs = int(epochs)

    trainer = EnhancedCECSLTrainer(cfg)
    trainer.load_data()
    trainer.build_model()

    print("[info] 开始训练(增强真实数据训练器)...")
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        # 简化单轮：遍历一个 epoch
        total, correct, loss_sum, batches = 0, 0, 0.0, 0
        trainer.model.set_train(True)
        for data, labels in trainer.train_dataset.create_tuple_iterator():
            loss, logits = trainer.train_step(data, labels)
            loss_sum += float(loss.asnumpy())
            batches += 1
        # 验证
        eval_res = trainer.evaluate()
        best_acc = max(best_acc, eval_res.get('accuracy', 0.0))
        print(f"epoch {epoch+1}/{cfg.epochs} - val_acc={eval_res.get('accuracy', 0.0):.4f} val_loss={eval_res.get('loss', 0.0):.4f}")

    print(f"[done] 训练完成，最佳验证准确率: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='CE-CSL/CS-CSL 训练入口')
    parser.add_argument('--model', choices=['enhanced', 'optimal'], default='enhanced')
    parser.add_argument('--data_root', default=str(ROOT.parent / 'data' / 'CS-CSL'))
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.model in ('enhanced', 'optimal'):
        # 两者均走真实数据增强训练器
        run_enhanced_cecsl(data_root, epochs=args.epochs)
    else:
        print(f"不支持的模型: {args.model}")
        sys.exit(2)


if __name__ == '__main__':
    main()
