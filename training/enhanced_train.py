#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版CE-CSL手语识别训练启动脚本
解决准确率过低和训练时间过短的问题
"""

import sys
import argparse
from pathlib import Path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版CE-CSL手语识别训练启动器')
    
    # 基础参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数（增加到100）')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小（减小适应小数据集）')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率（降低学习率）')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='CPU', help='设备类型')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_layers', type=int, default=3, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率（增加防止过拟合）')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='../data/CE-CSL', help='数据根目录')
    parser.add_argument('--max_seq_len', type=int, default=100, help='最大序列长度')
    
    # 数据增强参数
    parser.add_argument('--augment_factor', type=int, default=10, help='数据增强倍数')
    parser.add_argument('--noise_std', type=float, default=0.01, help='噪声标准差')
    
    # 训练策略参数
    parser.add_argument('--warmup_epochs', type=int, default=10, help='学习率预热轮数')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减')
    
    args = parser.parse_args()
    
    # 验证数据目录
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        print("请确保CE-CSL数据集已正确放置在指定目录")
        return False
    
    # 导入增强版训练器
    try:
        from enhanced_cecsl_trainer import EnhancedCECSLTrainer, EnhancedCECSLConfig
    except ImportError as e:
        print(f"❌ 导入增强版训练器失败: {e}")
        return False
    
    # 创建增强版配置
    config = EnhancedCECSLConfig(
        # 模型配置
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        
        # 训练配置
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        
        # 数据配置
        data_root=args.data_root,
        max_sequence_length=args.max_seq_len,
        
        # 数据增强配置
        augment_factor=args.augment_factor,
        noise_std=args.noise_std,
        
        # 设备配置
        device_target=args.device
    )
    
    print("🚀 增强版CE-CSL手语识别训练启动")
    print("🔧 优化改进:")
    print("  ✓ 数据增强: 每个样本增强{}倍，解决数据不足问题".format(args.augment_factor))
    print("  ✓ 改进模型: 双向LSTM + 注意力机制 + 更深的网络")
    print("  ✓ 训练策略: 学习率调度 + 早停 + 更多正则化")
    print("  ✓ 增加训练轮数: {} epochs，确保充分训练".format(args.epochs))
    print()
    print(f"📊 详细配置:")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 权重衰减: {args.weight_decay}")
    print(f"  - 设备: {args.device}")
    print(f"  - 模型维度: {args.d_model}")
    print(f"  - LSTM层数: {args.n_layers}")
    print(f"  - Dropout率: {args.dropout}")
    print(f"  - 数据目录: {args.data_root}")
    print(f"  - 数据增强倍数: {args.augment_factor}")
    print(f"  - 预热轮数: {args.warmup_epochs}")
    print(f"  - 早停耐心值: {args.patience}")
    print()
    
    try:
        # 创建增强版训练器
        trainer = EnhancedCECSLTrainer(config)
        
        # 加载数据
        print("📊 加载数据（包含数据增强）...")
        trainer.load_data()
        
        # 构建模型
        print("🧠 构建增强版模型...")
        trainer.build_model()
        
        # 开始训练
        print("🎯 开始增强版训练...")
        print("⏱️  预计训练时间会比之前长很多，这是正常的！")
        print()
        model = trainer.train()
        
        # 保存最终模型
        print("💾 保存最终模型...")
        trainer.save_model("./output/enhanced_cecsl_final_model.ckpt")
        
        print("🎉 增强版训练完成！")
        print("📁 模型已保存到: ./output/enhanced_cecsl_final_model.ckpt")
        print("📊 训练历史已保存到: ./output/enhanced_training_history.json")
        print()
        print("✨ 主要改进效果:")
        print("  ✓ 大幅增加训练数据量（通过数据增强）")
        print("  ✓ 显著延长训练时间（确保充分学习）")
        print("  ✓ 提升模型复杂度（更好的特征学习能力）")
        print("  ✓ 增强训练稳定性（学习率调度和早停）")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强版训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
