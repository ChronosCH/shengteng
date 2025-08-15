#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的手语识别训练测试脚本
避免编码问题
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

def test_real_training():
    """测试真实训练"""
    
    # 导入训练模块
    from optimized_unified_trainer import OptimizedSignLanguageTrainer, OptimizedTrainingConfig
    
    logger.info("=" * 60)
    logger.info("开始真实训练测试")
    logger.info("=" * 60)
    
    # 创建配置
    config = OptimizedTrainingConfig(
        model_type="tfnet",
        vocab_size=100,
        d_model=256,  # 较小的模型
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        batch_size=4,  # 小批次
        learning_rate=1e-3,
        epochs=2,  # 少量epoch用于测试
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
        model_path = "./output/real_test_model.ckpt"
        trainer.save_model(model_path)
        
        logger.info("=" * 60)
        logger.info("真实训练测试成功完成!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_training()
    if success:
        print("✅ 真实训练测试成功!")
    else:
        print("❌ 真实训练测试失败!")
        sys.exit(1)
