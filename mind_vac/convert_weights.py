"""
将PyTorch模型权重转换为MindSpore格式
"""
import torch
import numpy as np
import mindspore as ms
from collections import OrderedDict


def convert_pytorch_to_mindspore(pytorch_ckpt_path, output_path):
    """
    将PyTorch checkpoint转换为MindSpore checkpoint
    
    Args:
        pytorch_ckpt_path: PyTorch模型权重路径
        output_path: 输出的MindSpore权重路径
    """
    # 加载PyTorch权重
    print(f"Loading PyTorch checkpoint from {pytorch_ckpt_path}")
    pt_state_dict = torch.load(pytorch_ckpt_path, map_location='cpu')
    
    if 'model_state_dict' in pt_state_dict:
        pt_state_dict = pt_state_dict['model_state_dict']
    
    # 转换权重
    ms_params = []
    
    for name, param in pt_state_dict.items():
        # 移除'.module'前缀
        name = name.replace('.module', '')
        
        # 转换参数名称映射
        # PyTorch -> MindSpore 命名转换
        name = convert_param_name(name)
        
        # 转换numpy数组
        param_data = param.cpu().numpy()
        
        # 特殊处理BatchNorm的weight/bias -> gamma/beta
        if '.weight' in name and ('.bn' in name or 'downsample.1.weight' in name):
            # BatchNorm的weight映射到gamma
            name = name.replace('.weight', '.gamma')
            print(f"Converted BatchNorm weight->gamma: {name}, shape: {param_data.shape}")
        elif '.bias' in name and ('.bn' in name or 'downsample.1.bias' in name):
            # BatchNorm的bias映射到beta
            name = name.replace('.bias', '.beta')
            print(f"Converted BatchNorm bias->beta: {name}, shape: {param_data.shape}")
        # 特殊处理Conv1d权重: PyTorch是3D, MindSpore需要4D
        elif 'conv1d.temporal_conv' in name and '.weight' in name and len(param_data.shape) == 3:
            # (out_channels, in_channels, kernel_size) -> (out_channels, in_channels, 1, kernel_size)
            param_data = np.expand_dims(param_data, axis=2)
            print(f"Converted Conv1d: {name}, shape: {param.shape} -> {param_data.shape}")
        # 特殊处理Linear/Dense权重: PyTorch是(out, in), MindSpore需要(in, out)
        elif ('.fc.weight' in name or 'classifier.weight' in name) and len(param_data.shape) == 2:
            # (out_features, in_features) -> (in_features, out_features)
            param_data = param_data.T
            print(f"Converted Dense: {name}, shape: {param.shape} -> {param_data.shape}")
        else:
            # 其他参数正常输出
            if not ('num_batches_tracked' in name):
                print(f"Converted: {name}, shape: {param_data.shape}")
        
        # 创建MindSpore Parameter
        ms_param = {
            'name': name,
            'data': ms.Tensor(param_data)
        }
        ms_params.append(ms_param)
    
    # 保存MindSpore checkpoint
    print(f"\nSaving MindSpore checkpoint to {output_path}")
    ms.save_checkpoint(ms_params, output_path)
    print("Conversion completed!")
    
    return ms_params


def convert_param_name(pt_name):
    """
    转换PyTorch参数名称到MindSpore格式
    
    Args:
        pt_name: PyTorch参数名
    Returns:
        ms_name: MindSpore参数名
    """
    # 基本映射规则
    ms_name = pt_name
    
    # BatchNorm参数转换 - MindSpore使用gamma/beta而不是weight/bias
    # 但是在某些情况下也使用weight/bias,这里保持weight/bias命名
    # 因为我们使用nn.BatchNorm2d/nn.BatchNorm1d,它们使用weight/bias
    
    # 运行时统计量命名转换
    if 'running_mean' in ms_name:
        ms_name = ms_name.replace('running_mean', 'moving_mean')
    if 'running_var' in ms_name:
        ms_name = ms_name.replace('running_var', 'moving_variance')
    
    # LSTM参数保持原样,MindSpore的LSTM命名与PyTorch兼容
    
    return ms_name


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to MindSpore')
    parser.add_argument('--pytorch-ckpt', type=str, required=True,
                       help='Path to PyTorch checkpoint file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for MindSpore checkpoint')
    
    args = parser.parse_args()
    
    convert_pytorch_to_mindspore(args.pytorch_ckpt, args.output)
