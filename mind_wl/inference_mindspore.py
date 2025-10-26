"""
视频推理脚本 - MindSpore版本
对test文件夹中的视频进行手语识别推理
"""

import os
import sys
import numpy as np
import cv2
import mindspore
from mindspore import Tensor, context, save_checkpoint
from mindspore import load_checkpoint, load_param_into_net

# 设置MindSpore为CPU模式
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
print(f"使用设备: CPU")
print(f"MindSpore版本: {mindspore.__version__}")

# 导入I3D模型
from models.i3d_mindspore import InceptionI3d
from utils.ckpt_adapter import (
    build_checkpoint_payload,
    convert_pytorch_state_dict,
    needs_remap,
    remap_mindspore_param_dict,
)


def load_class_names(class_file, num_classes=2000):
    """加载类别名称"""
    class_names = {}
    with open(class_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                idx, name = parts
                if int(idx) < num_classes:
                    class_names[int(idx)] = name
    return class_names


def load_rgb_frames_from_video(video_path, target_size=224):
    """从视频文件加载RGB帧"""
    print(f"正在加载视频: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return None
    
    frames = []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    print(f"  - 总帧数: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    
    # 采样策略：最多64帧
    sample_interval = max(1, total_frames // 64)
    frame_count = 0
    
    while True:
        success, img = vidcap.read()
        if not success:
            break
        
        if frame_count % sample_interval == 0:
            # 中心裁剪为正方形
            h, w, c = img.shape
            if h != w:
                size = min(h, w)
                start_h = (h - size) // 2
                start_w = (w - size) // 2
                img = img[start_h:start_h+size, start_w:start_w+size]
            
            # 调整大小
            img = cv2.resize(img, (target_size, target_size))
            
            # 归一化到 [-1, 1]
            img = (img / 255.0) * 2 - 1
            frames.append(img)
        
        frame_count += 1
    
    vidcap.release()
    sampled_frames = len(frames)
    print(f"  - 成功采样 {sampled_frames} 帧 (间隔: {sample_interval})")
    
    if sampled_frames == 0:
        return None
    
    # 转换为 MindSpore tensor: (T, H, W, C) -> (1, C, T, H, W)
    frames_array = np.array(frames, dtype=np.float32)
    frames_tensor = frames_array.transpose(3, 0, 1, 2)  # (C, T, H, W)
    frames_tensor = np.expand_dims(frames_tensor, axis=0)  # (1, C, T, H, W)
    
    return Tensor(frames_tensor, mindspore.float32)


def load_model(weights_path, num_classes=2000, pytorch_weights_path=None):
    """加载I3D模型"""
    print(f"\n正在加载模型...")
    print(f"  - 类别数: {num_classes}")

    model = InceptionI3d(num_classes=num_classes, in_channels=3)
    load_success = False

    if os.path.exists(weights_path):
        print(f"  - 加载MindSpore权重: {weights_path}")
        try:
            param_dict = load_checkpoint(weights_path)
            if not param_dict:
                print("  - 警告: MindSpore权重文件为空，忽略")
            else:
                remapped = False
                if needs_remap(param_dict.keys()):
                    print("  - 检测到BatchNorm命名不兼容，正在重映射参数...")
                    param_dict, skipped = remap_mindspore_param_dict(param_dict)
                    remapped = True
                    if skipped:
                        print(f"  - 跳过 {len(skipped)} 个辅助参数 (num_batches_tracked)")
                load_result = load_param_into_net(model, param_dict)
                if isinstance(load_result, tuple):
                    not_loaded, not_exist = load_result
                else:
                    not_loaded, not_exist = load_result, []

                missing_total = len(not_loaded) + len(not_exist)
                if missing_total:
                    print(f"  - 警告: 有 {missing_total} 个参数未能加载，将尝试从PyTorch权重转换")
                    if not_loaded:
                        print(f"    未加载参数: {not_loaded}")
                    if not_exist:
                        print(f"    网络缺少参数: {not_exist}")
                else:
                    print("✓ 模型加载成功!")
                    model.set_train(False)
                    load_success = True
                    if remapped:
                        try:
                            save_checkpoint(build_checkpoint_payload(param_dict), weights_path)
                            print("  - 已修复MindSpore权重命名并更新缓存")
                        except Exception as cache_err:
                            print(f"  - 警告: 无法更新MindSpore权重缓存: {cache_err}")
        except Exception as e:
            print(f"警告: 加载MindSpore权重失败: {e}")

    if load_success:
        return model

    if pytorch_weights_path and os.path.exists(pytorch_weights_path):
        print("\n提示: 使用PyTorch权重重新构建MindSpore模型...")
        print(f"  - PyTorch权重: {pytorch_weights_path}")
        try:
            import torch
        except ImportError:
            print("✗ 错误: 需要安装PyTorch才能转换权重 (pip install torch)")
            return None

        try:
            pt_state_dict = torch.load(pytorch_weights_path, map_location='cpu')
            print(f"  - PyTorch权重包含 {len(pt_state_dict)} 个参数")
        except Exception as e:
            print(f"✗ 错误: 加载PyTorch权重失败: {e}")
            return None

        print("  - 转换参数命名以适配MindSpore...")
        converted_params, skipped = convert_pytorch_state_dict(pt_state_dict)
        if skipped:
            print(f"  - 跳过 {len(skipped)} 个辅助参数 (num_batches_tracked)")

        load_result = load_param_into_net(model, converted_params)
        if isinstance(load_result, tuple):
            not_loaded, not_exist = load_result
        else:
            not_loaded, not_exist = load_result, []

        missing_total = len(not_loaded) + len(not_exist)
        if missing_total:
            print(f"✗ 错误: 转换后仍有 {missing_total} 个参数无法加载")
            if not_loaded:
                print(f"    未加载参数: {not_loaded}")
            if not_exist:
                print(f"    网络缺少参数: {not_exist}")
            return None

        print("✓ 从PyTorch权重转换并加载成功!")
        model.set_train(False)

        if weights_path:
            try:
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                save_checkpoint(build_checkpoint_payload(converted_params), weights_path)
                print(f"  - 已缓存MindSpore权重到: {weights_path}")
            except Exception as cache_err:
                print(f"  - 警告: 无法写入MindSpore权重缓存: {cache_err}")

        return model

    print(f"\n✗ 错误: 未找到任何可用的模型权重")
    print(f"  MindSpore权重: {weights_path} ({'存在' if os.path.exists(weights_path) else '不存在'})")
    if pytorch_weights_path:
        print(f"  PyTorch权重: {pytorch_weights_path} ({'存在' if os.path.exists(pytorch_weights_path) else '不存在'})")
    return None


def inference_video(model, video_path, class_names, top_k=10):
    """对单个视频进行推理"""
    print(f"\n{'='*60}")
    print(f"推理视频: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # 加载视频帧
    frames = load_rgb_frames_from_video(video_path)
    if frames is None:
        print("错误: 无法加载视频")
        return None
    
    print(f"输入形状: {frames.shape}")
    
    # 推理
    print("正在进行推理...")
    logits = model(frames)  # (1, num_classes, T)
    
    # 对时间维度取平均
    predictions = logits.mean(axis=2)[0]  # (num_classes,)
    
    # 计算softmax概率
    exp_pred = np.exp(predictions.asnumpy() - np.max(predictions.asnumpy()))
    probs = exp_pred / np.sum(exp_pred)
    
    # 获取top-k
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    
    # 显示结果
    print(f"\n{'='*60}")
    print(f"Top-{top_k} 预测结果:")
    print(f"{'='*60}")
    
    results = []
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        class_name = class_names.get(idx, f"未知({idx})")
        print(f"{i+1}. {class_name:20s} - 置信度: {prob*100:.2f}%")
        results.append({
            'rank': i+1,
            'class_id': int(idx),
            'class_name': class_name,
            'confidence': float(prob)
        })
    
    return results


def test_video_loading_only(video_dir):
    """仅测试视频加载（不需要模型）"""
    print("="*60)
    print("MindSpore 视频加载测试")
    print("="*60)
    
    if not os.path.exists(video_dir):
        print(f"\n错误: 视频目录不存在: {video_dir}")
        return
    
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    
    if not video_files:
        print(f"\n错误: 在 {video_dir} 目录下未找到 .mp4 视频文件")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件:")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf}")
    
    # 测试第一个视频
    print(f"\n{'='*60}")
    print("测试第一个视频的加载...")
    print(f"{'='*60}")
    
    video_path = os.path.join(video_dir, video_files[0])
    frames = load_rgb_frames_from_video(video_path)
    
    if frames is not None:
        print(f"\n✓ 视频加载成功!")
        print(f"  - 张量形状: {frames.shape}")
        print(f"  - 张量类型: {frames.dtype}")
        print(f"  - 值范围: [{float(frames.min().asnumpy()):.3f}, {float(frames.max().asnumpy()):.3f}]")
        
        # 创建简单测试模型
        print(f"\n{'='*60}")
        print("测试简单推理...")
        print(f"{'='*60}")
        
        try:
            from mindspore import nn
            simple_model = nn.SequentialCell([
                nn.Conv3d(3, 16, kernel_size=3, pad_mode='same'),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Dense(16, 10)
            ])
            simple_model.set_train(False)
            
            output = simple_model(frames)
            print(f"✓ 简单模型推理成功!")
            print(f"  - 输出形状: {output.shape}")
            
            # Softmax
            exp_out = np.exp(output.asnumpy()[0] - np.max(output.asnumpy()[0]))
            probs = exp_out / np.sum(exp_out)
            top_idx = np.argmax(probs)
            
            print(f"  - Top-1类别: {top_idx}")
            print(f"  - Top-1置信度: {probs[top_idx]*100:.2f}%")
            
        except Exception as e:
            print(f"简单模型测试失败: {e}")
        
        print(f"\n✓ 测试完成!")
        print(f"\n下一步: 准备MindSpore格式的预训练模型，然后运行完整推理")


def main():
    """主函数"""
    print("="*60)
    print("WLASL 视频推理 - MindSpore版本 (CPU)")
    print("="*60)
    
    # 配置参数
    NUM_CLASSES = 2000
    VIDEO_DIR = 'test'
    CLASS_FILE = 'wlasl_class_list.txt'
    WEIGHTS_PATH = 'weights/i3d_wlasl2000.ckpt'  # MindSpore格式
    PYTORCH_WEIGHTS_PATH = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'  # PyTorch格式
    TOP_K = 10
    
    # 检查是否只是测试加载
    if len(sys.argv) > 1 and sys.argv[1] == '--test-only':
        test_video_loading_only(VIDEO_DIR)
        return
    
    # 检查权重文件
    has_mindspore_weights = os.path.exists(WEIGHTS_PATH)
    has_pytorch_weights = os.path.exists(PYTORCH_WEIGHTS_PATH)
    
    if not has_mindspore_weights and not has_pytorch_weights:
        print(f"\n✗ 错误: 未找到任何权重文件")
        print(f"\nMindSpore权重: {WEIGHTS_PATH} - 不存在")
        print(f"PyTorch权重: {PYTORCH_WEIGHTS_PATH} - 不存在")
        print(f"\n请下载预训练模型:")
        print(f"  下载链接: https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view")
        print(f"  解压到: ../code/I3D/archived/asl2000/")
        print(f"\n或运行测试模式（仅视频加载）:")
        print(f"  python inference_mindspore.py --test-only")
        return
    
    if has_pytorch_weights and not has_mindspore_weights:
        print(f"\n提示: 找到PyTorch权重，将自动转换为MindSpore格式")
        print(f"  PyTorch: {PYTORCH_WEIGHTS_PATH}")
        print(f"  这需要安装PyTorch: pip install torch")
    
    # 加载类别
    print(f"\n加载类别名称: {CLASS_FILE}")
    class_names = load_class_names(CLASS_FILE, NUM_CLASSES)
    print(f"✓ 加载了 {len(class_names)} 个类别")
    
    # 加载模型
    model = load_model(WEIGHTS_PATH, NUM_CLASSES, PYTORCH_WEIGHTS_PATH)
    if model is None:
        print(f"\n✗ 模型加载失败，无法继续推理")
        return
    
    # 获取视频列表
    if not os.path.exists(VIDEO_DIR):
        print(f"\n错误: 视频目录不存在: {VIDEO_DIR}")
        return
    
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    if not video_files:
        print(f"\n错误: 在 {VIDEO_DIR} 目录下未找到 .mp4 视频文件")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    # 推理所有视频
    all_results = {}
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        results = inference_video(model, video_path, class_names, TOP_K)
        if results:
            all_results[video_file] = results
    
    # 总结
    print(f"\n{'='*60}")
    print("推理完成总结")
    print(f"{'='*60}")
    print(f"总共处理: {len(all_results)} 个视频")
    
    print("\n所有视频的Top-1预测:")
    for video_file, results in all_results.items():
        if results:
            top1 = results[0]
            print(f"  {video_file:40s} -> {top1['class_name']:20s} ({top1['confidence']*100:.2f}%)")
    
    print("\n✓ 推理完成!")


if __name__ == '__main__':
    main()
