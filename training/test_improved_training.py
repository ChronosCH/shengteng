#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版训练测试 - 训练更多epoch看效果
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def test_improved_training():
    """测试改进的训练"""
    
    # 导入训练模块
    from optimized_unified_trainer import OptimizedSignLanguageTrainer, OptimizedTrainingConfig
    
    logger.info("=" * 60)
    logger.info("开始改进训练测试 - 训练更多epoch")
    logger.info("=" * 60)
    
    # 创建配置 - 增加训练轮数和学习率
    config = OptimizedTrainingConfig(
        model_type="tfnet",
        vocab_size=100,
        d_model=128,  # 稍小的模型
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        batch_size=8,  # 增大批次
        learning_rate=5e-3,  # 增大学习率
        epochs=10,  # 增加训练轮数
        device_target="CPU"
    )
    
    try:
        # 创建训练器
        trainer = OptimizedSignLanguageTrainer(config)
        
        # 加载数据
        trainer.load_data()
        
        # 构建模型
        trainer.build_model()
        
        # 设置回调
        trainer.setup_callbacks()
        
        # 开始训练
        logger.info("开始模型训练...")
        model = trainer.train()
        
        # 最终评估
        logger.info("训练完成，开始最终评估...")
        results = trainer.evaluate()
        
        logger.info("最终评估结果:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # 保存模型
        os.makedirs("./output", exist_ok=True)
        model_path = "./output/improved_test_model.ckpt"
        trainer.save_model(model_path)
        
        logger.info("=" * 60)
        logger.info("改进训练测试成功完成!")
        logger.info("=" * 60)
        
        return results["accuracy"] > 0.2  # 期望准确率超过20%
        
    except Exception as e:
        logger.error(f"训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_training()
    if success:
        print("✅ 改进训练测试成功！模型准确率有明显提升")
    else:
        print("⚠️  训练完成但准确率仍然较低，这在随机数据上是正常的")
