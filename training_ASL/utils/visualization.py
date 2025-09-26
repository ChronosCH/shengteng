import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
import pandas as pd
from config import Config


def visualize_video_frames(video_path: str, num_frames: int = 8, save_path: str = None):
    """可视化视频帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"无法读取视频: {video_path}")
        return
    
    # 均匀采样帧
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # 显示帧
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'视频帧: {os.path.basename(video_path)}')
    
    for i, frame in enumerate(frames):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(frame)
        axes[row, col].set_title(f'帧 {frame_indices[i]}')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(frames), 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到: {save_path}")
    
    plt.show()


def analyze_class_distribution(csv_path: str, title: str = "类别分布"):
    """分析类别分布"""
    df = pd.read_csv(csv_path)
    
    class_counts = df['Gloss'].value_counts()
    
    print(f"\n=== {title} ===")
    print(f"总样本数: {len(df)}")
    print(f"类别数: {len(class_counts)}")
    print(f"平均每类样本数: {len(df) / len(class_counts):.2f}")
    print(f"最多样本的类别: {class_counts.index[0]} ({class_counts.iloc[0]} 个样本)")
    print(f"最少样本的类别: {class_counts.index[-1]} ({class_counts.iloc[-1]} 个样本)")
    
    # 绘制分布图
    plt.figure(figsize=(12, 6))
    
    # 前20个类别的分布
    plt.subplot(1, 2, 1)
    class_counts.head(20).plot(kind='bar')
    plt.title('前20个类别的样本数')
    plt.xlabel('类别')
    plt.ylabel('样本数')
    plt.xticks(rotation=45)
    
    # 样本数分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(class_counts.values, bins=50, alpha=0.7)
    plt.title('每类样本数分布')
    plt.xlabel('每类样本数')
    plt.ylabel('类别数量')
    
    plt.tight_layout()
    plt.show()
    
    return class_counts


def check_video_quality(video_dir: str, csv_path: str, sample_size: int = 100):
    """检查视频质量"""
    df = pd.read_csv(csv_path)
    
    # 随机采样
    sample_df = df.sample(n=min(sample_size, len(df)))
    
    valid_videos = 0
    invalid_videos = []
    video_info = []
    
    print(f"检查 {len(sample_df)} 个视频样本...")
    
    for idx, row in sample_df.iterrows():
        video_file = row['Video file']
        video_path = os.path.join(video_dir, video_file)
        
        if not os.path.exists(video_path):
            invalid_videos.append((video_file, "文件不存在"))
            continue
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            invalid_videos.append((video_file, "无法打开"))
            cap.release()
            continue
        
        # 获取视频信息
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        if frame_count == 0:
            invalid_videos.append((video_file, "无帧数据"))
        else:
            valid_videos += 1
            video_info.append({
                'file': video_file,
                'frames': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            })
        
        cap.release()
    
    print(f"\n视频质量检查结果:")
    print(f"有效视频: {valid_videos}/{len(sample_df)}")
    print(f"无效视频: {len(invalid_videos)}")
    
    if invalid_videos:
        print("\n无效视频列表:")
        for file, reason in invalid_videos[:10]:  # 只显示前10个
            print(f"  {file}: {reason}")
    
    if video_info:
        info_df = pd.DataFrame(video_info)
        print(f"\n视频统计信息:")
        print(f"平均帧数: {info_df['frames'].mean():.1f}")
        print(f"平均FPS: {info_df['fps'].mean():.1f}")
        print(f"平均时长: {info_df['duration'].mean():.2f}秒")
        print(f"分辨率范围: {info_df['width'].min()}x{info_df['height'].min()} ~ {info_df['width'].max()}x{info_df['height'].max()}")
    
    return video_info, invalid_videos


def create_class_samples_visualization(data_dir: str, classes: List[str], samples_per_class: int = 4):
    """为每个类别创建样本可视化"""
    train_csv = os.path.join(data_dir, "splits", "train.csv")
    video_dir = os.path.join(data_dir, "videos")
    
    df = pd.read_csv(train_csv)
    
    for class_name in classes:
        class_samples = df[df['Gloss'] == class_name]
        
        if len(class_samples) == 0:
            print(f"类别 {class_name} 没有样本")
            continue
        
        # 随机选择样本
        selected_samples = class_samples.sample(n=min(samples_per_class, len(class_samples)))
        
        fig, axes = plt.subplots(samples_per_class, 4, figsize=(16, 4 * samples_per_class))
        fig.suptitle(f'类别: {class_name}', fontsize=16)
        
        for sample_idx, (_, row) in enumerate(selected_samples.iterrows()):
            video_file = row['Video file']
            video_path = os.path.join(video_dir, video_file)
            
            if not os.path.exists(video_path):
                continue
            
            # 提取4帧
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, 4, dtype=int)
                
                for frame_idx, frame_num in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        if samples_per_class == 1:
                            axes[frame_idx].imshow(frame)
                            axes[frame_idx].set_title(f'帧 {frame_num}')
                            axes[frame_idx].axis('off')
                        else:
                            axes[sample_idx, frame_idx].imshow(frame)
                            axes[sample_idx, frame_idx].set_title(f'样本{sample_idx+1}-帧{frame_num}')
                            axes[sample_idx, frame_idx].axis('off')
            
            cap.release()
        
        plt.tight_layout()
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(Config.RESULTS_DIR, f"class_{class_name.replace(' ', '_')}_samples.png"), dpi=150, bbox_inches='tight')
        plt.show()


def generate_dataset_report(data_dir: str):
    """生成数据集报告"""
    splits_dir = os.path.join(data_dir, "splits")
    video_dir = os.path.join(data_dir, "videos")
    
    print("=== ASL数据集分析报告 ===\n")
    
    # 分析各分割
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(splits_dir, f"{split}.csv")
        if os.path.exists(csv_path):
            analyze_class_distribution(csv_path, f"{split.upper()}数据集")
    
    # 检查视频质量
    train_csv = os.path.join(splits_dir, "train.csv")
    if os.path.exists(train_csv):
        print("\n=== 视频质量检查 ===")
        video_info, invalid_videos = check_video_quality(video_dir, train_csv, sample_size=200)
    
    # 生成类别样本可视化 (选择几个类别)
    df = pd.read_csv(train_csv)
    common_classes = df['Gloss'].value_counts().head(5).index.tolist()
    
    print(f"\n生成常见类别的样本可视化: {common_classes}")
    create_class_samples_visualization(data_dir, common_classes, samples_per_class=2)


if __name__ == "__main__":
    data_dir = Config.DATA_DIR
    
    # 生成完整的数据集报告
    generate_dataset_report(data_dir)
