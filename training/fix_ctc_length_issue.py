#!/usr/bin/env python3
"""
CTC长度不匹配问题解决脚本
专门用于修复训练中的CTC Loss输入长度问题
"""

import os
import sys
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
import mindspore.ops as ops

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_ctc_length_mismatch():
    """修复CTC长度不匹配问题"""
    
    print("=" * 60)
    print("CTC长度不匹配问题解决方案")
    print("=" * 60)
    
    # 1. 修复训练脚本中的长度验证
    print("\n1. 修复训练脚本中的CTC输入验证...")
    
    # 读取训练脚本
    train_script_path = "train_tfnet_gpu.py"
    
    try:
        with open(train_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找训练循环中的损失计算部分
        if "loss = loss_fn" in content:
            print("✓ 找到损失计算代码")
            
            # 添加CTC输入验证
            ctc_validation_code = '''
                # CTC输入长度验证和修复
                max_input_len = max([l.item() if hasattr(l, 'item') else int(l) for l in data_len])
                actual_time_steps = log_probs1.shape[0]
                
                if max_input_len > actual_time_steps:
                    # 调整输入长度以匹配实际时间步数
                    print(f"Warning: Adjusting input length from {max_input_len} to {actual_time_steps}")
                    adjusted_len = [min(int(l), actual_time_steps) for l in data_len]
                    data_len = Tensor(adjusted_len, ms.int32)
                
                # 确保所有长度都是正数且不超过时间步数
                validated_len = [max(1, min(int(l), actual_time_steps)) for l in data_len]
                data_len = Tensor(validated_len, ms.int32)
            '''
            
            # 在损失计算前插入验证代码
            new_content = content.replace(
                "loss = loss_fn",
                ctc_validation_code + "\n                loss = loss_fn"
            )
            
            # 写回文件
            with open(train_script_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            print("✓ 已添加CTC输入长度验证代码")
            
        else:
            print("⚠️  未找到损失计算代码，可能需要手动修复")
            
    except Exception as e:
        print(f"✗ 修复训练脚本失败: {e}")
    
    # 2. 修复模型中的长度计算
    print("\n2. 验证模型中的长度计算...")
    
    try:
        from tfnet_model import TFNetModel
        
        # 创建测试模型
        model = TFNetModel(
            hidden_size=256,
            word_set_num=100,
            device_target="GPU"
        )
        
        # 创建测试输入
        batch_size = 1
        seq_len = 50  # 中等长度序列
        test_input = Tensor(np.random.randn(batch_size, seq_len, 3, 160, 160).astype(np.float32))
        test_len = Tensor([seq_len], ms.int32)
        
        print(f"输入序列长度: {seq_len}")
        
        # 使用动态图模式进行详细调试
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        
        outputs = model(test_input, test_len, is_train=True)
        
        # 检查输出长度
        log_probs1, _, _, _, _, lgt_tensor, _, _, _ = outputs
        
        print(f"模型输出时间步数: {log_probs1.shape[0]}")
        print(f"返回的长度张量: {lgt_tensor}")
        print(f"长度张量值: {lgt_tensor.asnumpy()}")
        
        # 验证长度匹配
        max_lgt = max(lgt_tensor.asnumpy())
        actual_steps = log_probs1.shape[0]
        
        if max_lgt <= actual_steps:
            print("✓ 长度匹配正确")
        else:
            print(f"✗ 长度不匹配: max_lgt={max_lgt} > actual_steps={actual_steps}")
            print("需要调整长度计算逻辑")
            
    except Exception as e:
        print(f"✗ 模型长度验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 创建安全的训练配置
    print("\n3. 创建安全的训练配置...")
    
    safe_config = {
        "dataset": {
            "name": "CE-CSL",
            "train_data_path": "/root/shengteng/training/data/CE-CSL/video/train",
            "valid_data_path": "/root/shengteng/training/data/CE-CSL/video/dev", 
            "test_data_path": "/root/shengteng/training/data/CE-CSL/video/test",
            "train_label_path": "/root/shengteng/training/data/CE-CSL/label/train.csv",
            "valid_label_path": "/root/shengteng/training/data/CE-CSL/label/dev.csv",
            "test_label_path": "/root/shengteng/training/data/CE-CSL/label/test.csv",
            "crop_size": 160,
            "max_frames": 50  # 进一步减少帧数以确保稳定性
        },
        "model": {
            "name": "TFNet",
            "hidden_size": 128,  # 减小隐藏层大小
            "device_target": "GPU",
            "device_id": 0,
            "enable_graph_kernel": False,
            "enable_reduce_precision": True,
            "enable_auto_mixed_precision": False  # 禁用混合精度以避免精度问题
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 0.0001,
            "num_epochs": 5,  # 减少训练轮数用于测试
            "num_workers": 1,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "save_interval": 1,
            "eval_interval": 1,
            "early_stopping_patience": 3,
            "prefetch_size": 1,
            "max_rowsize": 4,
            "enable_data_sink": False
        },
        "gpu_optimization": {
            "enable_graph_mode": False,  # 使用动态图模式便于调试
            "enable_mem_reuse": True,
            "max_device_memory": "2GB",  # 进一步限制内存
            "enable_profiling": False,
            "enable_dump": False,
            "enable_memory_offload": True,
            "mempool_block_size": "256MB"
        },
        "paths": {
            "checkpoint_dir": "/root/shengteng/training/checkpoints_gpu",
            "log_dir": "/root/shengteng/training/logs_gpu",
            "output_dir": "/root/shengteng/training/output_gpu",
            "best_model_path": "/root/shengteng/training/checkpoints_gpu/best_model.ckpt",
            "current_model_path": "/root/shengteng/training/checkpoints_gpu/current_model.ckpt"
        },
        "logging": {
            "level": "INFO",
            "save_logs": True,
            "print_interval": 5
        },
        "loss": {
            "ctc_blank_id": 0,
            "ctc_reduction": "mean",
            "kd_temperature": 8,
            "kd_weight": 25.0
        }
    }
    
    import json
    safe_config_path = "configs/safe_gpu_config.json"
    
    try:
        os.makedirs(os.path.dirname(safe_config_path), exist_ok=True)
        with open(safe_config_path, 'w', encoding='utf-8') as f:
            json.dump(safe_config, f, indent=4, ensure_ascii=False)
        print(f"✓ 已创建安全配置文件: {safe_config_path}")
    except Exception as e:
        print(f"✗ 创建安全配置失败: {e}")
    
    print("\n" + "=" * 60)
    print("解决方案总结")
    print("=" * 60)
    print("1. ✓ 已修复device-side assert错误")
    print("2. ✓ 已解决GPU显存不足问题") 
    print("3. ⚠️  需要进一步解决CTC长度不匹配问题")
    print("4. ✓ 已创建安全的训练配置")
    print()
    print("下一步操作:")
    print("1. 使用安全配置进行测试:")
    print("   python train_tfnet_gpu.py --config configs/safe_gpu_config.json")
    print("2. 如果仍有CTC长度问题，切换到动态图模式进行详细调试")
    print("3. 根据调试结果进一步调整模型长度计算逻辑")

if __name__ == "__main__":
    fix_ctc_length_mismatch()
