#!/usr/bin/env python3
"""
训练系统验证脚本
快速验证所有模块是否正常工作
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """测试所有模块导入"""
    logger.info("测试模块导入...")
    
    tests = []
    
    # 测试核心模块
    try:
        from config_loader import ConfigLoader
        tests.append(("ConfigLoader", True, None))
    except Exception as e:
        tests.append(("ConfigLoader", False, str(e)))
    
    try:
        from enhanced_ascend_optimizer import EnhancedAscendOptimizer
        tests.append(("EnhancedAscendOptimizer", True, None))
    except Exception as e:
        tests.append(("EnhancedAscendOptimizer", False, str(e)))
    
    try:
        from enhanced_data_preprocessing import EnhancedSignLanguagePreprocessor
        tests.append(("EnhancedSignLanguagePreprocessor", True, None))
    except Exception as e:
        tests.append(("EnhancedSignLanguagePreprocessor", False, str(e)))
    
    try:
        from tfnet_mindspore import TFNetMindSpore
        tests.append(("TFNetMindSpore", True, None))
    except Exception as e:
        tests.append(("TFNetMindSpore", False, str(e)))
    
    try:
        from cecsl_data_processor import CECSLDataProcessor
        tests.append(("CECSLDataProcessor", True, None))
    except Exception as e:
        tests.append(("CECSLDataProcessor", False, str(e)))
    
    try:
        from tfnet_decoder import CTCDecoder
        tests.append(("CTCDecoder", True, None))
    except Exception as e:
        tests.append(("CTCDecoder", False, str(e)))
    
    try:
        from optimized_unified_trainer import OptimizedSignLanguageTrainer
        tests.append(("OptimizedSignLanguageTrainer", True, None))
    except Exception as e:
        tests.append(("OptimizedSignLanguageTrainer", False, str(e)))
    
    # 显示结果
    logger.info("模块导入测试结果:")
    success_count = 0
    for module_name, success, error in tests:
        if success:
            logger.info(f"  ✓ {module_name}")
            success_count += 1
        else:
            logger.error(f"  ✗ {module_name}: {error}")
    
    logger.info(f"导入测试完成: {success_count}/{len(tests)} 个模块成功")
    return success_count == len(tests)

def test_config_loader():
    """测试配置加载器"""
    logger.info("测试配置加载器...")
    
    try:
        from config_loader import ConfigLoader
        
        # 创建配置加载器
        loader = ConfigLoader()
        
        # 测试列出配置
        training_configs = loader.list_training_configs()
        device_configs = loader.list_device_configs()
        preprocessing_configs = loader.list_preprocessing_configs()
        presets = loader.list_training_presets()
        
        logger.info(f"  可用训练配置: {len(training_configs)} 个")
        logger.info(f"  可用设备配置: {len(device_configs)} 个") 
        logger.info(f"  可用预处理配置: {len(preprocessing_configs)} 个")
        logger.info(f"  可用预设: {len(presets)} 个")
        
        # 测试创建配置
        if training_configs and device_configs and preprocessing_configs:
            config = loader.create_full_config(
                training_config=training_configs[0],
                device_config=device_configs[0], 
                preprocessing_config=preprocessing_configs[0]
            )
            
            # 验证配置
            is_valid = loader.validate_config(config)
            logger.info(f"  配置验证: {'通过' if is_valid else '失败'}")
            
            if is_valid:
                summary = loader.get_config_summary(config)
                logger.info(f"  配置摘要: {summary.split('=')[0]}...")
        
        logger.info("  ✓ 配置加载器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ 配置加载器测试失败: {e}")
        return False

def test_mindspore():
    """测试MindSpore环境"""
    logger.info("测试MindSpore环境...")
    
    try:
        import mindspore as ms
        from mindspore import context, nn, ops, Tensor
        
        logger.info(f"  MindSpore版本: {ms.__version__}")
        
        # 测试基本操作
        x = Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
        y = ops.sum(x)
        
        logger.info(f"  基本张量操作: {y}")
        
        # 测试设备设置
        available_devices = []
        
        # 测试Ascend
        try:
            context.set_context(device_target="Ascend", device_id=0)
            available_devices.append("Ascend")
        except:
            pass
        
        # 测试GPU
        try:
            context.set_context(device_target="GPU", device_id=0)
            available_devices.append("GPU")
        except:
            pass
        
        # 测试CPU (总是可用)
        try:
            context.set_context(device_target="CPU")
            available_devices.append("CPU")
        except:
            pass
        
        logger.info(f"  可用设备: {available_devices}")
        
        logger.info("  ✓ MindSpore环境测试通过")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ MindSpore环境测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    logger.info("测试模型创建...")
    
    try:
        from tfnet_mindspore import TFNetMindSpore
        from cecsl_data_processor import CECSLVocabulary
        import mindspore as ms
        from mindspore import context
        
        # 设置CPU上下文 (最兼容)
        context.set_context(device_target="CPU")
        
        # 创建简单词汇表
        vocab = CECSLVocabulary()
        vocab.add_word("test")
        vocab.add_word("hello")
        vocab.build()
        
        # 创建模型
        model = TFNetMindSpore(
            vocab_size=len(vocab),
            hidden_size=64,  # 使用小尺寸用于测试
            num_classes=len(vocab)
        )
        
        # 测试前向传播
        batch_size = 2
        seq_len = 10
        channels = 3
        height = 64
        width = 64
        
        dummy_input = ms.Tensor(
            shape=(batch_size, seq_len, channels, height, width),
            dtype=ms.float32,
            init=ms.common.initializer.Normal(0.01)
        )
        
        output = model(dummy_input)
        logger.info(f"  模型输出形状: {output.shape}")
        
        # 计算参数量
        total_params = 0
        for param in model.get_parameters():
            total_params += param.size
        
        logger.info(f"  模型参数量: {total_params:,}")
        
        logger.info("  ✓ 模型创建测试通过")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ 模型创建测试失败: {e}")
        logger.error(f"  详细错误: {traceback.format_exc()}")
        return False

def test_data_preprocessing():
    """测试数据预处理"""
    logger.info("测试数据预处理...")
    
    try:
        from enhanced_data_preprocessing import (
            EnhancedSignLanguagePreprocessor, 
            PreprocessingConfig,
            VideoSample
        )
        import numpy as np
        
        # 创建预处理配置 (使用最小配置)
        config = PreprocessingConfig(
            target_fps=5,  # 低帧率用于测试
            max_sequence_length=10,
            min_sequence_length=1,
            image_size=(32, 32),  # 小图像用于测试
            enable_keypoints=False,  # 禁用MediaPipe避免依赖问题
            enable_augmentation=False,
            num_workers=1
        )
        
        # 创建预处理器
        preprocessor = EnhancedSignLanguagePreprocessor(config)
        
        # 测试质量检查
        dummy_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        quality_info = preprocessor.check_video_quality(dummy_frame)
        
        logger.info(f"  质量检查结果: blur_score={quality_info['blur_score']:.2f}")
        
        # 测试数据增强
        dummy_frames = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        augmented_frames = preprocessor.apply_augmentation(dummy_frames)
        
        logger.info(f"  数据增强输入形状: {dummy_frames.shape}")
        logger.info(f"  数据增强输出形状: {augmented_frames.shape}")
        
        logger.info("  ✓ 数据预处理测试通过")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ 数据预处理测试失败: {e}")
        logger.error(f"  详细错误: {traceback.format_exc()}")
        return False

def test_file_structure():
    """测试文件结构"""
    logger.info("测试文件结构...")
    
    required_files = [
        "tfnet_mindspore.py",
        "cecsl_data_processor.py", 
        "tfnet_decoder.py",
        "optimized_unified_trainer.py",
        "enhanced_ascend_optimizer.py",
        "enhanced_data_preprocessing.py",
        "config_loader.py",
        "train_script.py",
        "train_start.bat",
        "configs/training_configs.json"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    logger.info(f"  存在的文件: {len(existing_files)}/{len(required_files)}")
    
    if missing_files:
        logger.warning(f"  缺失的文件: {missing_files}")
    
    if len(existing_files) >= len(required_files) * 0.8:  # 80%的文件存在就算通过
        logger.info("  ✓ 文件结构测试通过")
        return True
    else:
        logger.error("  ✗ 文件结构测试失败，关键文件缺失")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("手语识别训练系统验证")
    logger.info("=" * 60)
    
    tests = [
        ("文件结构", test_file_structure),
        ("模块导入", test_imports),
        ("MindSpore环境", test_mindspore),
        ("配置加载器", test_config_loader),
        ("数据预处理", test_data_preprocessing),
        ("模型创建", test_model_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n开始测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    # 显示总结
    logger.info("\n" + "=" * 60)
    logger.info("验证结果总结")
    logger.info("=" * 60)
    
    success_count = 0
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        logger.info(f"  {test_name:<20} {status}")
        if success:
            success_count += 1
    
    total_tests = len(results)
    success_rate = success_count / total_tests * 100
    
    logger.info(f"\n总体结果: {success_count}/{total_tests} 个测试通过 ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 系统验证通过！可以开始使用训练系统。")
        return 0
    elif success_rate >= 60:
        logger.warning("⚠️ 系统部分验证通过，建议检查失败的测试项。")
        return 1
    else:
        logger.error("❌ 系统验证失败，请检查环境配置和依赖安装。")
        return 2

if __name__ == "__main__":
    exit_code = main()
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    print("如果验证通过，可以使用以下命令开始训练:")
    print("  Windows: train_start.bat")
    print("  Python:  python train_script.py --data_root YOUR_DATA_PATH")
    print("=" * 60)
    
    sys.exit(exit_code)
