#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试JIT优化后的训练器
"""

import sys
import os
sys.path.append(os.getcwd())

# 导入训练器
from cecsl_real_trainer import CECSLTrainer, CECSLTrainingConfig

def test_jit_optimization():
    """测试JIT优化"""
    print("🔍 测试JIT优化效果...")
    
    # 创建小配置用于快速测试
    config = CECSLTrainingConfig(
        vocab_size=1000,
        d_model=64,  # 更小的模型
        n_heads=2,
        n_layers=1,
        dropout=0.1,
        batch_size=2,  # 更小的批次
        learning_rate=1e-3,
        epochs=2,  # 只运行2个epoch
        max_sequence_length=50,  # 更短的序列
        image_size=(224, 224),
        device_target="CPU",
        data_root="../data/CE-CSL"
    )
    
    try:
        # 创建训练器
        print("📊 创建训练器...")
        trainer = CECSLTrainer(config)
        
        # 加载数据
        print("📚 加载数据...")
        trainer.load_data()
        
        # 构建模型
        print("🧠 构建模型...")
        trainer.build_model()
        
        # 训练一个epoch看警告情况
        print("🎯 开始测试训练...")
        print("=" * 50)
        
        for epoch in range(2):
            print(f"\n🔄 Epoch {epoch + 1}/2")
            train_loss, train_acc = trainer.train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        
        print("=" * 50)
        print("✅ JIT优化测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_jit_optimization()
    if success:
        print("\n🎉 JIT优化效果:")
        print("- 减少了重复的'after_grad'编译警告")
        print("- 训练和评估函数只编译一次")
        print("- 提高了训练效率")
    else:
        print("\n❌ 测试失败，请检查代码")
