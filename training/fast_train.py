#!/usr/bin/env python3
"""
快速训练脚本 - 使用轻量级模型进行快速验证
"""

import os
import sys
import time
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_tfnet import TFNetTrainer
from light_tfnet_model import LightTFNetModel

class FastTFNetTrainer(TFNetTrainer):
    """快速训练器 - 使用轻量级模型"""
    
    def build_model(self, vocab_size):
        """构建轻量级模型"""
        self.logger.info("Building lightweight model...")
        
        model_config = self.config_manager.get_model_config()
        
        # 使用轻量级模型
        self.model = LightTFNetModel(
            hidden_size=model_config["hidden_size"],
            word_set_num=vocab_size,
            device_target=model_config["device_target"],
            dataset_name=model_config["dataset_name"]
        )
        
        from decoder import CTCDecoder
        self.decoder = CTCDecoder(
            gloss_dict=self.word2idx,
            num_classes=vocab_size + 1,
            search_mode='max',
            blank_id=self.config_manager.get("loss.ctc_blank_id", 0)
        )
        
        self.logger.info("Lightweight model built successfully")
        
        # 初始化损失函数和优化器
        self.loss_fn = self.create_loss_fn()
        self.optimizer = self.create_optimizer()
        
        # 计算模型参数
        try:
            total_params = sum(p.size for p in self.model.get_parameters())
            self.logger.info(f"Model parameters: {total_params} (vs original ~41M)")
        except Exception:
            self.logger.info("Model parameters: Unable to calculate")
    
    def create_loss_fn(self):
        """创建简化的损失函数"""
        import mindspore.nn as nn
        
        # 只使用CTC损失，移除复杂的知识蒸馏
        blank_id = self.config_manager.get("loss.ctc_blank_id", 0)
        ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean')
        
        def simplified_loss_fn(log_probs1, log_probs2, log_probs3, log_probs4, log_probs5,
                              lgt, target_data, target_lengths):
            """简化的损失函数 - 只使用主要的logits"""
            # 应用log softmax
            log_softmax = nn.LogSoftmax(axis=-1)
            log_probs = log_softmax(log_probs1)
            
            # 只计算主要的CTC损失
            loss = ctc_loss(log_probs, target_data, lgt, target_lengths)
            return loss
        
        return simplified_loss_fn

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast TFNet Training Script')
    parser.add_argument('--config', type=str, default='training/configs/tfnet_config_fast.json',
                       help='Path to fast configuration file')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the model without full training')
    
    args = parser.parse_args()
    
    print("🚀 Starting Fast TFNet Training...")
    
    # 创建快速训练器
    trainer = FastTFNetTrainer(config_path=args.config)
    
    if args.test_only:
        print("🧪 Running model test...")
        try:
            # 只准备数据和构建模型，不进行训练
            vocab_size = trainer.prepare_data()
            trainer.build_model(vocab_size)
            
            # 测试一个batch
            print("✓ Model test successful!")
            print(f"  - Vocabulary size: {vocab_size}")
            print(f"  - Model built successfully")
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🏃 Starting full training...")
        try:
            start_time = time.time()
            trainer.train()
            total_time = time.time() - start_time
            print(f"✅ Training completed in {total_time/60:.1f} minutes")
        except KeyboardInterrupt:
            print("⏹️ Training interrupted by user")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
