"""
使用示例: 展示如何使用LLM集成功能

这个脚本展示了如何:
1. 单独使用通义千问API
2. 测试不同的提示词
3. 批量处理多个手语识别结果
"""

import os
from qwen_api import QwenAPI


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "="*80)
    print("示例1: 基本使用")
    print("="*80)
    
    # 初始化API客户端
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("请先设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    client = QwenAPI(api_key=api_key, model='qwen-plus')
    
    # 测试手语词汇
    gloss = "__ON__ LIEB ZUSCHAUER ABEND WINTER NULL loc-REGION UEBERSCHWEMMUNG AMERIKA"
    
    print(f"\n输入词汇: {gloss}")
    print("\n正在调用API...")
    
    result = client.translate_gloss_to_sentence(gloss)
    
    if result['success']:
        print(f"\n✓ 翻译成功!")
        print(f"中文: {result['chinese']}")
        print(f"英文: {result['english']}")
        print(f"置信度: {result['confidence']}")
        print(f"说明: {result['explanation']}")
    else:
        print(f"\n✗ 翻译失败: {result.get('error', '未知错误')}")


def example_2_different_models():
    """示例2: 测试不同模型"""
    print("\n" + "="*80)
    print("示例2: 测试不同模型")
    print("="*80)
    
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("请先设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    gloss = "HEUTE WETTER SCHOEN WARM SONNE"
    models = ['qwen-turbo', 'qwen-plus']
    
    for model in models:
        print(f"\n--- 使用模型: {model} ---")
        client = QwenAPI(api_key=api_key, model=model)
        result = client.translate_gloss_to_sentence(gloss)
        
        if result['success']:
            print(f"中文: {result['chinese']}")
            print(f"英文: {result['english']}")
        else:
            print(f"失败: {result.get('error', '未知错误')}")


def example_3_batch_processing():
    """示例3: 批量处理"""
    print("\n" + "="*80)
    print("示例3: 批量处理")
    print("="*80)
    
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("请先设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    client = QwenAPI(api_key=api_key, model='qwen-plus')
    
    # 多个手语词汇序列
    gloss_list = [
        "__ON__ LIEB ZUSCHAUER ABEND",
        "HEUTE WETTER SCHOEN",
        "WINTER SCHNEE KALT",
    ]
    
    print(f"\n批量处理 {len(gloss_list)} 个句子...")
    
    results = client.batch_translate(gloss_list)
    
    for i, (gloss, result) in enumerate(zip(gloss_list, results), 1):
        print(f"\n--- 句子 {i} ---")
        print(f"输入: {gloss}")
        if result['success']:
            print(f"中文: {result['chinese']}")
            print(f"英文: {result['english']}")
        else:
            print(f"失败: {result.get('error', '未知错误')}")


def example_4_custom_temperature():
    """示例4: 调整生成参数"""
    print("\n" + "="*80)
    print("示例4: 调整生成参数(temperature)")
    print("="*80)
    
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("请先设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    client = QwenAPI(api_key=api_key, model='qwen-plus')
    gloss = "HEUTE WETTER SCHOEN WARM"
    
    temperatures = [0.3, 0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        result = client.translate_gloss_to_sentence(gloss, temperature=temp)
        
        if result['success']:
            print(f"中文: {result['chinese']}")
            print(f"英文: {result['english']}")
        else:
            print(f"失败: {result.get('error', '未知错误')}")


def example_5_error_handling():
    """示例5: 错误处理"""
    print("\n" + "="*80)
    print("示例5: 错误处理")
    print("="*80)
    
    # 测试无效API密钥
    print("\n测试1: 无效API密钥")
    try:
        client = QwenAPI(api_key="invalid_key")
        result = client.translate_gloss_to_sentence("TEST")
        print(f"结果: {result}")
    except Exception as e:
        print(f"捕获异常: {type(e).__name__}: {e}")
    
    # 测试空词汇
    print("\n测试2: 空词汇序列")
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if api_key:
        client = QwenAPI(api_key=api_key)
        result = client.translate_gloss_to_sentence("")
        print(f"成功: {result['success']}")
        if not result['success']:
            print(f"错误: {result.get('error', '未知')}")


def main():
    """运行所有示例"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "通义千问API使用示例" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    
    # 检查API密钥
    if not os.environ.get('DASHSCOPE_API_KEY'):
        print("\n⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请先运行: export DASHSCOPE_API_KEY='your_api_key'")
        print("或运行配置助手: ./setup_api.sh")
        return
    
    try:
        # 运行示例
        example_1_basic_usage()
        
        print("\n\n按Enter继续下一个示例...")
        input()
        
        example_2_different_models()
        
        print("\n\n按Enter继续下一个示例...")
        input()
        
        example_3_batch_processing()
        
        print("\n\n按Enter继续下一个示例...")
        input()
        
        example_4_custom_temperature()
        
        print("\n\n按Enter继续最后一个示例...")
        input()
        
        example_5_error_handling()
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("所有示例完成!")
    print("="*80)


if __name__ == '__main__':
    main()
