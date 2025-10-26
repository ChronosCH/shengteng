"""
PyTorch权重转换为MindSpore格式的工具脚本
"""

import os
import sys


def convert_pytorch_to_mindspore(pytorch_path, mindspore_path):
    """
    将PyTorch的.pt权重转换为MindSpore的.ckpt格式
    
    注意: 这需要同时安装PyTorch和MindSpore
    """
    print("="*60)
    print("PyTorch -> MindSpore 权重转换工具")
    print("="*60)
    
    # 检查PyTorch
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
    except ImportError:
        print("✗ 错误: 未安装PyTorch")
        print("请先安装: pip install torch")
        return False
    
    # 检查MindSpore
    try:
        import mindspore
        from mindspore import save_checkpoint
        print(f"✓ MindSpore版本: {mindspore.__version__}")
    except ImportError:
        print("✗ 错误: 未安装MindSpore")
        print("请先安装: pip install mindspore")
        return False

    from utils.ckpt_adapter import build_checkpoint_payload, convert_pytorch_state_dict
    
    # 加载PyTorch权重
    if not os.path.exists(pytorch_path):
        print(f"\n✗ 错误: PyTorch权重文件不存在: {pytorch_path}")
        return False
    
    print(f"\n加载PyTorch权重: {pytorch_path}")
    try:
        pt_state_dict = torch.load(pytorch_path, map_location='cpu')
        print(f"✓ 成功加载 {len(pt_state_dict)} 个参数")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 转换参数
    print(f"\n转换参数...")
    converted_params, skipped = convert_pytorch_state_dict(pt_state_dict)

    print(f"✓ 转换完成，共 {len(converted_params)} 个参数")
    if skipped:
        print(f"  - 跳过 {len(skipped)} 个辅助参数 (例如 BatchNorm 的 num_batches_tracked)")
    
    # 保存MindSpore权重
    print(f"\n保存MindSpore权重: {mindspore_path}")
    try:
        os.makedirs(os.path.dirname(mindspore_path), exist_ok=True)
        save_checkpoint(build_checkpoint_payload(converted_params), mindspore_path)
        print(f"✓ 保存成功!")
        return True
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python convert_weights.py <pytorch_path> <mindspore_path>")
        print("\n示例:")
        print("  python convert_weights.py \\")
        print("    ../code/I3D/archived/asl2000/FINAL_nslt_2000_*.pt \\")
        print("    weights/i3d_wlasl2000.ckpt")
        return
    
    pytorch_path = sys.argv[1]
    mindspore_path = sys.argv[2]
    
    success = convert_pytorch_to_mindspore(pytorch_path, mindspore_path)
    
    if success:
        print("\n"+"="*60)
        print("✓ 转换完成!")
        print("="*60)
        print(f"\nMindSpore权重已保存到: {mindspore_path}")
        print(f"现在可以运行推理: python inference_mindspore.py")
    else:
        print("\n"+"="*60)
        print("✗ 转换失败")
        print("="*60)


if __name__ == '__main__':
    main()
