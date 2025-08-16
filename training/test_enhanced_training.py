#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版训练效果
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

def test_data_augmentation():
    """测试数据增强效果"""
    print("🧪 测试数据增强效果...")
    
    try:
        from enhanced_cecsl_trainer import EnhancedCECSLConfig, EnhancedCECSLDataset
        
        config = EnhancedCECSLConfig(
            data_root='../data/CE-CSL',
            augment_factor=5,  # 测试用较小的增强倍数
            batch_size=2,
            epochs=5  # 测试用较少轮数
        )
        
        # 测试原始数据集
        print("  📊 加载原始训练数据...")
        original_dataset = EnhancedCECSLDataset(config, 'train', use_augmentation=False)
        print(f"  原始训练样本数: {len(original_dataset)}")
        
        # 测试增强数据集
        print("  🔄 加载增强训练数据...")
        augmented_dataset = EnhancedCECSLDataset(config, 'train', use_augmentation=True)
        print(f"  增强后训练样本数: {len(augmented_dataset)}")
        
        improvement = len(augmented_dataset) / len(original_dataset)
        print(f"  ✅ 数据增强倍数: {improvement:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据增强测试失败: {e}")
        return False

def test_model_complexity():
    """测试模型复杂度"""
    print("🧪 测试模型复杂度...")
    
    try:
        from enhanced_cecsl_trainer import EnhancedCECSLConfig, ImprovedCECSLModel
        from cecsl_real_trainer import CECSLTrainingConfig, CECSLModel
        import mindspore as ms
        
        ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
        
        # 原始模型
        old_config = CECSLTrainingConfig()
        old_model = CECSLModel(old_config, vocab_size=10)
        old_params = sum(p.size for p in old_model.get_parameters())
        
        # 增强模型
        new_config = EnhancedCECSLConfig()
        new_model = ImprovedCECSLModel(new_config, vocab_size=10)
        new_params = sum(p.size for p in new_model.get_parameters())
        
        print(f"  原始模型参数量: {old_params:,}")
        print(f"  增强模型参数量: {new_params:,}")
        print(f"  ✅ 模型复杂度提升: {new_params/old_params:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型复杂度测试失败: {e}")
        return False

def test_training_pipeline():
    """测试训练流程"""
    print("🧪 测试训练流程...")
    
    try:
        from enhanced_cecsl_trainer import EnhancedCECSLTrainer, EnhancedCECSLConfig
        
        # 创建测试配置
        config = EnhancedCECSLConfig(
            data_root='../data/CE-CSL',
            augment_factor=3,  # 较小的增强倍数用于测试
            batch_size=2,
            epochs=3,  # 只训练3轮用于测试
            learning_rate=1e-3,
            patience=10
        )
        
        print("  📊 初始化训练器...")
        trainer = EnhancedCECSLTrainer(config)
        
        print("  🔄 加载数据...")
        trainer.load_data()
        
        print("  🧠 构建模型...")
        trainer.build_model()
        
        print("  🎯 开始测试训练（3轮）...")
        start_time = time.time()
        model = trainer.train()
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"  ✅ 测试训练完成，耗时: {training_time:.1f}秒")
        print(f"  ⏱️  平均每轮训练时间: {training_time/3:.1f}秒")
        
        if training_time > 30:  # 如果训练时间超过30秒，说明训练量确实增加了
            print("  ✅ 训练时间显著增加，数据增强生效")
        else:
            print("  ⚠️  训练时间较短，可能需要进一步调整")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔬 增强版CE-CSL训练系统测试")
    print("=" * 50)
    
    test_results = []
    
    # 测试数据增强
    test_results.append(test_data_augmentation())
    print()
    
    # 测试模型复杂度
    test_results.append(test_model_complexity())
    print()
    
    # 测试训练流程
    test_results.append(test_training_pipeline())
    print()
    
    # 总结测试结果
    print("📋 测试结果总结:")
    print("=" * 30)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    test_names = [
        "数据增强测试",
        "模型复杂度测试", 
        "训练流程测试"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！增强版训练系统准备就绪")
        print("\n🚀 可以运行以下命令开始正式训练:")
        print("python enhanced_train.py")
    else:
        print("⚠️  部分测试失败，请检查环境和依赖")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
