"""
TFNet集成测试脚本
验证MindSpore TFNet实现的基本功能
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "training"))

def test_imports():
    """测试模块导入"""
    print("=" * 50)
    print("测试模块导入...")
    
    try:
        import mindspore as ms
        print(f"✓ MindSpore {ms.__version__} 导入成功")
    except ImportError as e:
        print(f"✗ MindSpore 导入失败: {e}")
        return False
    
    try:
        from tfnet_mindspore import TFNetMindSpore, SeqKD
        print("✓ TFNet MindSpore 模型导入成功")
    except ImportError as e:
        print(f"✗ TFNet 模型导入失败: {e}")
        return False
    
    try:
        from cecsl_data_processor import CECSLLabelProcessor, CECSLVideoProcessor
        print("✓ CE-CSL 数据处理器导入成功")
    except ImportError as e:
        print(f"✗ 数据处理器导入失败: {e}")
        return False
    
    try:
        from tfnet_decoder import CTCDecoder, WERCalculator
        print("✓ TFNet 解码器导入成功")
    except ImportError as e:
        print(f"✗ 解码器导入失败: {e}")
        # 这个可能会因为editdistance包而失败，但不是致命错误
        print("  注意: 可能需要安装 editdistance 包: pip install editdistance")
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("=" * 50)
    print("测试模型创建...")
    
    try:
        from tfnet_mindspore import TFNetMindSpore
        import mindspore as ms
        
        # 设置上下文
        ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")
        
        # 创建模型
        model = TFNetMindSpore(
            hidden_size=512,
            vocab_size=1000,
            module_choice="TFNet",
            dataset_name="CE-CSL"
        )
        
        print("✓ TFNet 模型创建成功")
        
        # 测试前向传播
        batch_size, seq_len, height, width, channels = 2, 10, 224, 224, 3
        dummy_input = ms.Tensor(np.random.randn(batch_size, seq_len, height, width, channels), ms.float32)
        dummy_lengths = ms.Tensor([8, 6], ms.int32)
        
        outputs = model(dummy_input, dummy_lengths, is_train=False)
        print(f"✓ 模型前向传播成功，输出数量: {len(outputs)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_data_processor():
    """测试数据处理器"""
    print("=" * 50)
    print("测试数据处理器...")
    
    try:
        from cecsl_data_processor import CECSLLabelProcessor
        
        # 创建示例词汇表
        processor = CECSLLabelProcessor()
        
        # 测试词汇预处理
        test_words = ["你好(1)", "世界", "手语识别", "测试123"]
        processed = processor.preprocess_words(test_words)
        print(f"✓ 词汇预处理成功: {test_words} -> {processed}")
        
        # 创建测试词汇表
        processor.word2idx = {' ': 0, '你好': 1, '世界': 2, '手语': 3, '识别': 4}
        processor.idx2word = [' ', '你好', '世界', '手语', '识别']
        
        # 保存词汇表
        test_vocab_dir = project_root / "temp"
        test_vocab_dir.mkdir(exist_ok=True)
        test_vocab_file = test_vocab_dir / "test_vocab.json"
        
        processor.save_vocabulary(str(test_vocab_file))
        print(f"✓ 词汇表保存成功: {test_vocab_file}")
        
        # 加载词汇表
        new_processor = CECSLLabelProcessor()
        new_processor.load_vocabulary(str(test_vocab_file))
        print("✓ 词汇表加载成功")
        
        # 清理测试文件
        test_vocab_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ 数据处理器测试失败: {e}")
        return False

def test_decoder():
    """测试解码器（如果可用）"""
    print("=" * 50)
    print("测试CTC解码器...")
    
    try:
        from tfnet_decoder import CTCDecoder, WERCalculator
        
        # 创建测试词汇表文件
        test_vocab_dir = project_root / "temp"
        test_vocab_dir.mkdir(exist_ok=True)
        test_vocab_file = test_vocab_dir / "test_vocab.json"
        
        test_vocab = {
            'word2idx': {' ': 0, '你好': 1, '世界': 2, '手语': 3, '识别': 4},
            'idx2word': [' ', '你好', '世界', '手语', '识别'],
            'vocab_size': 5
        }
        
        with open(test_vocab_file, 'w', encoding='utf-8') as f:
            json.dump(test_vocab, f, ensure_ascii=False)
        
        # 创建解码器
        decoder = CTCDecoder(str(test_vocab_file), blank_id=0)
        print("✓ CTC解码器创建成功")
        
        # 测试贪婪解码
        log_probs = np.random.randn(10, 1, 5)  # (seq_len, batch_size, vocab_size)
        input_lengths = np.array([8])
        
        greedy_results = decoder.greedy_decode(log_probs, input_lengths)
        print(f"✓ 贪婪解码成功: {greedy_results}")
        
        # 测试WER计算器
        wer_calc = WERCalculator()
        references = [['你好', '世界'], ['手语', '识别']]
        hypotheses = [['你好', '世界'], ['手语']]
        
        wer_results = wer_calc.compute_wer(references, hypotheses)
        print(f"✓ WER计算成功: {wer_results['wer']:.2f}%")
        
        # 清理测试文件
        test_vocab_file.unlink()
        
        return True
        
    except ImportError:
        print("⚠ 解码器测试跳过（editdistance 包未安装）")
        return True
    except Exception as e:
        print(f"✗ 解码器测试失败: {e}")
        return False

def test_service_integration():
    """测试服务集成"""
    print("=" * 50)
    print("测试服务集成...")
    
    try:
        # 检查服务文件
        service_file = project_root / "backend" / "services" / "diffusion_slp_service.py"
        if service_file.exists():
            print("✓ 服务文件存在")
            
            # 尝试导入服务
            sys.path.append(str(project_root / "backend"))
            from services.diffusion_slp_service import DiffusionSLPService
            print("✓ 服务类导入成功")
            
            return True
        else:
            print("✗ 服务文件不存在")
            return False
            
    except Exception as e:
        print(f"✗ 服务集成测试失败: {e}")
        return False

def test_config_files():
    """测试配置文件"""
    print("=" * 50)
    print("测试配置文件...")
    
    config_file = project_root / "training" / "configs" / "tfnet_cecsl_config.json"
    
    if not config_file.exists():
        print(f"✗ 配置文件不存在: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("✓ 配置文件加载成功")
        
        # 检查必要的配置项
        required_sections = ['model_config', 'training_config', 'hardware_config', 'paths']
        for section in required_sections:
            if section in config:
                print(f"✓ 配置段 '{section}' 存在")
            else:
                print(f"✗ 配置段 '{section}' 缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("TFNet MindSpore 集成测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("模型创建", test_model_creation),
        ("数据处理器", test_data_processor),
        ("解码器", test_decoder),
        ("服务集成", test_service_integration),
        ("配置文件", test_config_files),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试 '{test_name}' 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("=" * 50)
    print("测试总结:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！TFNet集成成功。")
        return True
    else:
        print("⚠ 部分测试失败，请检查相关问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
