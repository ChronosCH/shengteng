"""
MindSpore版本的单视频推理脚本
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
import json

import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net
import torch

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind_vac.model import SLRModel
from mind_vac.decoder import Decode
from mind_vac.transforms import Compose, CenterCrop, ToTensor, normalize_video
from mind_vac.qwen_api import QwenAPI


def preprocess_checkpoint(param_dict):
    """Adjust checkpoint tensors so they match the MindSpore model definition."""
    converted = {}
    skipped_keys = []
    reshaped_keys = []

    for name, param in param_dict.items():
        # MindSpore BatchNorm layers do not track batch statistics counters
        if name.endswith('num_batches_tracked'):
            skipped_keys.append(name)
            continue

        value = param
        if hasattr(value, 'asnumpy'):
            array = value.asnumpy()
            dtype = value.dtype if hasattr(value, 'dtype') else ms.float32
        else:
            array = np.array(value)
            dtype = ms.float32

        target_name = name

        # Align BatchNorm naming for temporal conv blocks (weight->gamma, bias->beta)
        if name.startswith('conv1d.temporal_conv.'):
            parts = name.split('.')
            if len(parts) >= 4:
                layer_idx, param_name = parts[2], parts[3]
                if layer_idx in {'1', '5'}:
                    if param_name == 'weight':
                        target_name = name.replace('.weight', '.gamma')
                    elif param_name == 'bias':
                        target_name = name.replace('.bias', '.beta')

        # Merge shared classifier weights
        if name.startswith('conv1d.fc.'):
            target_name = name.replace('conv1d.', '', 1)

        if target_name in {'fc.weight', 'classifier.weight'} and array.ndim == 2:
            array = array.T

        tensor = Tensor(array, dtype=dtype)
        converted[target_name] = ms.Parameter(tensor, name=target_name)

    return converted, skipped_keys, reshaped_keys


def convert_pytorch_to_mindspore(pt_path, num_classes):
    """Convert a PyTorch checkpoint to a MindSpore parameter dict."""
    state_dict = torch.load(pt_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    converted = {}
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if name.endswith('num_batches_tracked'):
            continue

        array = param.cpu().numpy()
        target_name = name

        if 'running_mean' in target_name:
            target_name = target_name.replace('running_mean', 'moving_mean')
        if 'running_var' in target_name:
            target_name = target_name.replace('running_var', 'moving_variance')

        if any(tag in target_name for tag in ['bn', 'downsample.1']) and target_name.endswith('.weight'):
            target_name = target_name.replace('.weight', '.gamma')
        if any(tag in target_name for tag in ['bn', 'downsample.1']) and target_name.endswith('.bias'):
            target_name = target_name.replace('.bias', '.beta')

        if target_name.startswith('conv1d.temporal_conv.'):
            parts = target_name.split('.')
            if len(parts) >= 4:
                layer_idx, param_name = parts[2], parts[3]
                if layer_idx in {'1', '5'}:
                    if param_name == 'weight':
                        target_name = target_name.replace('.weight', '.gamma')
                    elif param_name == 'bias':
                        target_name = target_name.replace('.bias', '.beta')

        if target_name.startswith('conv1d.temporal_conv.') and target_name.endswith('.weight') and array.ndim == 3:
            array = np.expand_dims(array, axis=2)

        if target_name.startswith('conv1d.fc.'):
            target_name = target_name.replace('conv1d.', '', 1)
        if target_name.startswith('classifier.'):
            target_name = target_name  # keep as-is

        if target_name.endswith('.weight') and array.ndim == 2 and array.shape[0] != num_classes and target_name.startswith('fc.'):
            # Ensure Dense weights follow (out, in)
            array = array.T
        if target_name.endswith('.weight') and array.ndim == 2 and target_name.startswith('classifier.') and array.shape[0] != num_classes:
            array = array.T

        converted[target_name] = ms.Parameter(Tensor(array, dtype=ms.float32), name=target_name)

    return converted


def load_gloss_dict(dict_path):
    """加载手语词汇字典"""
    gloss_dict = np.load(dict_path, allow_pickle=True).item()
    print(f"Loaded gloss dictionary with {len(gloss_dict)} entries")
    return gloss_dict


def load_video_frames(video_path):
    """
    从文件夹加载视频帧
    
    Args:
        video_path: 视频帧文件夹路径
    Returns:
        frames: list of images (H, W, C) in RGB format
    """
    # 查找所有图像文件
    img_list = sorted(glob.glob(os.path.join(video_path, "*.png")))
    if not img_list:
        img_list = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    
    if not img_list:
        raise ValueError(f"No images found in {video_path}")
    
    print(f"Found {len(img_list)} frames")
    
    # 读取所有图像
    frames = []
    for img_path in img_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_path}")
            continue
        # 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    return frames


def preprocess_video(frames, crop_size=224):
    """
    预处理视频帧
    
    Args:
        frames: list of images
        crop_size: 裁剪大小
    Returns:
        video_tensor: (1, T, C, H, W)
        seq_len: 序列长度
    """
    # 数据增强
    transform = Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])
    
    # 应用变换
    video_tensor, _ = transform(frames, None, None)
    
    # 标准化
    video_tensor = normalize_video(video_tensor)
    
    # video_tensor已经是 (T, C, H, W)
    video_array = video_tensor.asnumpy()

    # 对时间维度进行左右padding, 与PyTorch数据管道保持一致
    total_frames = video_array.shape[0]
    left_pad = 6
    right_pad = int(np.ceil(total_frames / 4.0)) * 4 - total_frames + 6

    if left_pad > 0:
        pad_front = np.repeat(video_array[:1], left_pad, axis=0)
    else:
        pad_front = np.empty((0,) + video_array.shape[1:], dtype=video_array.dtype)

    if right_pad > 0:
        pad_back = np.repeat(video_array[-1:], right_pad, axis=0)
    else:
        pad_back = np.empty((0,) + video_array.shape[1:], dtype=video_array.dtype)

    padded_video = np.concatenate([pad_front, video_array, pad_back], axis=0)

    # 添加batch维度: (T, C, H, W) -> (1, T, C, H, W)
    padded_video = np.expand_dims(padded_video, axis=0)
    
    video_tensor = Tensor(padded_video, dtype=ms.float32)
    seq_len = [padded_video.shape[1]]
    
    return video_tensor, seq_len


def inference(model, decoder, video_tensor, seq_len):
    """
    执行推理
    
    Args:
        model: SLR模型
        decoder: 解码器
        video_tensor: 视频tensor (1, T, C, H, W)
        seq_len: 序列长度
    Returns:
        recognized: 识别结果
    """
    # 推理
    outputs = model(video_tensor, seq_len)
    
    # 解码
    sequence_logits = outputs['sequence_logits']
    feat_len = outputs['feat_len']
    
    # 解码结果
    recognized = decoder.decode(
        sequence_logits,
        feat_len,
        batch_first=False,
        probs=False
    )
    
    return recognized[0] if recognized else []


def main():
    parser = argparse.ArgumentParser(description='MindSpore SLR Inference')
    parser.add_argument('--video-path', type=str, required=True,
                       help='Path to video frames folder')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to MindSpore checkpoint')
    parser.add_argument('--dict-path', type=str, 
                       default='./preprocess/phoenix2014/gloss_dict.npy',
                       help='Path to gloss dictionary')
    parser.add_argument('--device', type=str, default='CPU',
                       choices=['CPU', 'GPU', 'Ascend'],
                       help='Device type')
    parser.add_argument('--output', type=str, default='./mind_inference_output',
                       help='Output directory')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM (Qwen API) to generate complete sentences')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Qwen API key (or set DASHSCOPE_API_KEY env var)')
    parser.add_argument('--qwen-model', type=str, default='qwen-plus',
                       help='Qwen model to use (default: qwen-plus)')
    
    args = parser.parse_args()
    
    # 设置运行环境
    if args.device == 'CPU':
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    elif args.device == 'GPU':
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    
    print(f"Using device: {args.device}")
    print(f"MindSpore version: {ms.__version__}")
    
    # 加载词汇字典
    gloss_dict = load_gloss_dict(args.dict_path)
    num_classes = len(gloss_dict) + 1
    
    # 创建模型
    print("\nCreating model...")
    model = SLRModel(
        num_classes=num_classes,
        hidden_size=1024,
        conv_type=2,
        use_bn=True,
        weight_norm=False,
        share_classifier=False
    )
    
    # 加载权重
    print(f"Loading checkpoint from {args.checkpoint}")
    if args.checkpoint.endswith('.pt'):
        param_dict = convert_pytorch_to_mindspore(args.checkpoint, num_classes)
        skipped_keys, reshaped_keys = [], []
    else:
        raw_params = load_checkpoint(args.checkpoint)
        param_dict, skipped_keys, reshaped_keys = preprocess_checkpoint(raw_params)
    load_result = load_param_into_net(model, param_dict)
    param_not_load = []
    ckpt_not_load = []
    if isinstance(load_result, tuple):
        param_not_load, ckpt_not_load = load_result
    else:
        param_not_load = load_result

    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys not used by MindSpore BN (e.g. {skipped_keys[0]})")
    if reshaped_keys:
        print(f"Reshaped Conv1d kernels for {len(reshaped_keys)} layers")
    if param_not_load:
        print(f"Warning: parameters not loaded -> {param_not_load}")
    if ckpt_not_load:
        print(f"Warning: unused checkpoint entries -> {ckpt_not_load}")

    model.set_train(False)
    print("Model loaded successfully!")
    
    # 创建解码器
    decoder = Decode(gloss_dict, num_classes, search_mode='beam')
    
    # 加载视频
    print(f"\nLoading video from: {args.video_path}")
    frames = load_video_frames(args.video_path)
    
    # 预处理
    print("Preprocessing video...")
    video_tensor, seq_len = preprocess_video(frames)
    
    # 推理
    print("Running inference...")
    recognized = inference(model, decoder, video_tensor, seq_len)
    
    # 显示结果
    result_text = ' '.join([word for word, _ in recognized])
    
    print("\n" + "="*60)
    print("Inference Result:")
    print("="*60)
    print(f"Video: {os.path.basename(args.video_path)}")
    print(f"Frames: {len(frames)}")
    print(f"Recognized Gloss: {result_text}")
    
    # 如果启用LLM,调用通义千问API生成完整句子
    llm_result = None
    if args.use_llm:
        print("\n" + "-"*60)
        print("Calling Qwen API to generate complete sentences...")
        print("-"*60)
        try:
            # 初始化API客户端
            qwen_client = QwenAPI(
                api_key=args.api_key or os.environ.get('DASHSCOPE_API_KEY'),
                model=args.qwen_model
            )
            
            # 调用API
            llm_result = qwen_client.translate_gloss_to_sentence(result_text)
            
            if llm_result['success']:
                print("\n完整句子翻译:")
                print(f"中文: {llm_result['chinese']}")
                print(f"English: {llm_result['english']}")
                print(f"置信度: {llm_result['confidence']}")
                print(f"说明: {llm_result['explanation']}")
            else:
                print(f"\nLLM翻译失败: {llm_result.get('error', '未知错误')}")
                if llm_result.get('raw_response'):
                    print(f"原始响应: {llm_result['raw_response']}")
        except Exception as e:
            print(f"\nLLM调用出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "inference_result.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Video: {args.video_path}\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Recognized Gloss: {result_text}\n")
        
        if llm_result and llm_result['success']:
            f.write("\n" + "="*60 + "\n")
            f.write("LLM Generated Complete Sentences:\n")
            f.write("="*60 + "\n")
            f.write(f"中文: {llm_result['chinese']}\n")
            f.write(f"English: {llm_result['english']}\n")
            f.write(f"置信度: {llm_result['confidence']}\n")
            f.write(f"说明: {llm_result['explanation']}\n")
    
    # 同时保存JSON格式的结果
    json_output_file = os.path.join(args.output, "inference_result.json")
    json_data = {
        "video_path": args.video_path,
        "frames": len(frames),
        "recognized_gloss": result_text,
        "gloss_words": [word for word, _ in recognized],
    }
    
    if llm_result:
        json_data["llm_result"] = llm_result
    
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {json_output_file}")



if __name__ == '__main__':
    main()
