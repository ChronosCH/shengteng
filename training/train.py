#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CE-CSL手语识别训练启动脚本
快速启动训练，无需复杂配置
"""

import sys
import argparse
from pathlib import Path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CE-CSL手语识别训练启动器')
    
    # 基础参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='CPU', help='设备类型')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='../data/CE-CSL', help='数据根目录')
    parser.add_argument('--max_seq_len', type=int, default=100, help='最大序列长度')
    
    args = parser.parse_args()
    
    # 验证数据目录
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        print("请确保CE-CSL数据集已正确放置在指定目录")
        return False
    
    # 导入训练器
    try:
        from cecsl_real_trainer import CECSLTrainer, CECSLTrainingConfig
    except ImportError as e:
        print(f"❌ 导入训练器失败: {e}")
        return False
    
    # 创建配置
    config = CECSLTrainingConfig(
        vocab_size=1000,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_sequence_length=args.max_seq_len,
        image_size=(224, 224),
        device_target=args.device,
        data_root=args.data_root
    )
    
    print("🚀 CE-CSL手语识别训练启动")
    print(f"📊 配置信息:")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 设备: {args.device}")
    print(f"  - 模型维度: {args.d_model}")
    print(f"  - LSTM层数: {args.n_layers}")
    print(f"  - 数据目录: {args.data_root}")
    print()
    
    try:
        # 创建训练器
        trainer = CECSLTrainer(config)
        
        # 加载数据
        print("📊 加载数据...")
        trainer.load_data()
        
        # 构建模型
        print("🧠 构建模型...")
        trainer.build_model()
        
        # 开始训练
        print("🎯 开始训练...")
        model = trainer.train()
        
        # 保存最终模型
        print("💾 保存最终模型...")
        trainer.save_model("./output/cecsl_final_model.ckpt")
        
        print("🎉 训练完成！")
        print("📁 模型已保存到: ./output/cecsl_final_model.ckpt")
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
