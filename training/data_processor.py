import os
import csv
import cv2
import numpy as np
import mindspore as ms
from mindspore import dataset as ds
from mindspore.dataset import vision, transforms
import json

PAD = ' '
# 通过移除括号和数字来预处理单词列表
def preprocess_words(words):
    """通过移除括号和数字来预处理单词列表"""
    for i in range(len(words)):
        word = words[i]
        
        n = 0
        sub_flag = False
        word_list = list(word)
        for j in range(len(word)):
            if word[j] in "({[（":
                sub_flag = True
            
            if sub_flag:
                word_list.pop(j - n)
                n = n + 1
            
            if word[j] in ")}]）":
                sub_flag = False
        
        word = "".join(word_list)
        
        if word and word[-1].isdigit():
            if not word[0].isdigit():
                word_list = list(word)
                word_list.pop(len(word) - 1)
                word = "".join(word_list)
        
        if word and word[0] in ",，":
            word_list = list(word)
            word_list[0] = '，'
            word = ''.join(word_list)
        
        if word and word[0] in "?？":
            word_list = list(word)
            word_list[0] = '？'
            word = ''.join(word_list)
        
        if word.isdigit():
            word = str(int(word))
        
        words[i] = word
    
    return words

def build_vocabulary(train_label_path, valid_label_path, test_label_path, dataset_name):
    """从标签文件构建词汇表"""
    word_list = []
    
    if dataset_name == "CE-CSL":
        # 处理CE-CSL数据集
        for label_path in [train_label_path, valid_label_path, test_label_path]:
            with open(label_path, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n != 0:  # 跳过表头
                        words = row[3].split("/")
                        words = preprocess_words(words)
                        word_list += words
    
    # 构建词汇表
    idx2word = [PAD]
    set2list = sorted(list(set(word_list)))
    idx2word.extend(set2list)
    
    word2idx = {w: i for i, w in enumerate(idx2word)}
    
    return word2idx, len(idx2word) - 1, idx2word

class VideoTransform:
    """用于数据增强的视频转换"""
    def __init__(self, is_train=True, crop_size=224):
        self.is_train = is_train
        self.crop_size = crop_size
    
    def __call__(self, video_frames):
        """对视频帧应用转换"""
        # 如果需要，转换为numpy数组
        if isinstance(video_frames, list):
            video_frames = np.array(video_frames)
        
        # 调整帧大小
        resized_frames = []
        for frame in video_frames:
            if self.is_train:
                # 训练时随机裁剪
                h, w = frame.shape[:2]
                if h > self.crop_size and w > self.crop_size:
                    top = np.random.randint(0, h - self.crop_size)
                    left = np.random.randint(0, w - self.crop_size)
                    frame = frame[top:top+self.crop_size, left:left+self.crop_size]
                else:
                    frame = cv2.resize(frame, (self.crop_size, self.crop_size))
                
                # 随机水平翻转
                if np.random.random() > 0.5:
                    frame = cv2.flip(frame, 1)
            else:
                # 验证/测试时中心裁剪
                frame = cv2.resize(frame, (self.crop_size, self.crop_size))
            
            resized_frames.append(frame)
        
        # 转换为张量格式 (T, H, W, C) -> (T, C, H, W)
        video_tensor = np.array(resized_frames)

        # 调试：检查张量形状
        if len(video_tensor.shape) != 4:
            print(f"Warning: Unexpected video tensor shape: {video_tensor.shape}")
            print(f"Expected 4D tensor (T, H, W, C), got {len(video_tensor.shape)}D")
            # 通过添加通道维度处理灰度图像
            if len(video_tensor.shape) == 3:
                video_tensor = np.expand_dims(video_tensor, axis=-1)
                print(f"添加了通道维度，新形状：{video_tensor.shape}")

        # 确保有正确的维度数量用于转置
        if len(video_tensor.shape) == 4:
            video_tensor = np.transpose(video_tensor, (0, 3, 1, 2))
        else:
            raise ValueError(f"无法转置形状为{video_tensor.shape}的张量")

        # 填充或截断到固定长度以便批处理
        max_frames = 300  # 来自配置
        current_frames = video_tensor.shape[0]

        if current_frames > max_frames:
            # 截断
            video_tensor = video_tensor[:max_frames]
        elif current_frames < max_frames:
            # 用零填充
            pad_frames = max_frames - current_frames
            pad_shape = (pad_frames,) + video_tensor.shape[1:]
            pad_tensor = np.zeros(pad_shape, dtype=video_tensor.dtype)
            video_tensor = np.concatenate([video_tensor, pad_tensor], axis=0)
        
        # 标准化到[0, 1]
        video_tensor = video_tensor.astype(np.float32) / 255.0
        
        return video_tensor

class CECSLDataset:
    """用于手语识别的CE-CSL数据集"""
    
    def __init__(self, data_path, label_path, word2idx, dataset_name, is_train=False, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.word2idx = word2idx
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.transform = transform or VideoTransform(is_train=is_train)
        
        # Load labels
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load samples from label file"""
        samples = []
        
        with open(self.label_path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:  # Skip header
                    video_name = row[0]
                    translator = row[1]  # Get translator (A, B, C, etc.)
                    words = row[3].split("/")
                    words = preprocess_words(words)

                    # Convert words to indices
                    label_indices = []
                    for word in words:
                        if word in self.word2idx:
                            label_indices.append(self.word2idx[word])

                    if label_indices:  # Only add if we have valid labels
                        # Construct proper video path: data_path/translator/video_name.mp4
                        video_path = os.path.join(self.data_path, translator, f"{video_name}.mp4")
                        samples.append({
                            'video_name': video_name,
                            'label': label_indices,
                            'video_path': video_path
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video frames
        video_frames = self._load_video(sample['video_path'])
        original_length = len(video_frames)

        # Apply transformations
        if self.transform:
            video_frames = self.transform(video_frames)

        # Pad labels to fixed length for batching
        max_label_length = 50  # Reasonable max label length
        label = sample['label']
        if len(label) > max_label_length:
            label = label[:max_label_length]
        else:
            # Pad with zeros (assuming 0 is padding token)
            label = label + [0] * (max_label_length - len(label))

        return {
            'video': video_frames,
            'label': label,
            'video_length': original_length,  # Use original length before padding
            'info': sample['video_name']
        }
    
    def _load_video(self, video_path):
        """Load video frames from directory"""
        frames = []

        # Debug: Print video path (only for first few videos)
        if len(frames) == 0:  # Only print for first video
            print(f"Loading video from: {video_path}")
            print(f"Path exists: {os.path.exists(video_path)}")
            print(f"Is directory: {os.path.isdir(video_path)}")

        if os.path.isdir(video_path):
            # Load from image directory
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
            print(f"Found {len(frame_files)} frame files")
            if len(frame_files) > 0:
                print(f"First few files: {frame_files[:5]}")

            for frame_file in frame_files:
                frame_path = os.path.join(video_path, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    print(f"Failed to load frame: {frame_path}")
        else:
            # Load from video file
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

        print(f"Loaded {len(frames)} frames")
        
        # Convert to numpy array and ensure proper dtype
        if frames:
            frames = np.asarray(frames, dtype=np.float32)
            # Normalize to [0, 1] range
            frames = frames / 255.0
        else:
            # Return empty array with proper shape if no frames loaded
            frames = np.empty((0, 224, 224, 3), dtype=np.float32)
            
        return frames

def collate_fn(videos, labels, video_lengths, infos):
    """Collate function for batching samples - MindSpore style"""
    # Convert to numpy arrays for processing
    videos = [np.array(v) for v in videos]
    video_lengths = [int(vl) for vl in video_lengths]

    # Find max video length in batch
    max_length = max(video_lengths)

    # Pad videos to max length
    padded_videos = []
    for i, video in enumerate(videos):
        video_length = video_lengths[i]

        # Pad video to max length
        if video_length < max_length:
            pad_length = max_length - video_length
            pad_shape = (pad_length,) + video.shape[1:]
            pad_frames = np.zeros(pad_shape, dtype=video.dtype)
            video = np.concatenate([video, pad_frames], axis=0)

        padded_videos.append(video)

    return (
        np.array(padded_videos),
        labels,
        video_lengths,
        infos
    )

def create_dataset(data_path, label_path, word2idx, dataset_name, 
                  batch_size=2, is_train=True, num_workers=1, 
                  prefetch_size=2, max_rowsize=32):
    """Create MindSpore dataset with GPU optimizations"""
    
    # Create custom dataset
    dataset = CECSLDataset(
        data_path=data_path,
        label_path=label_path,
        word2idx=word2idx,
        dataset_name=dataset_name,
        is_train=is_train
    )
    
    # Convert to MindSpore dataset
    def generator():
        for i in range(len(dataset)):
            sample = dataset[i]
            # Ensure all data is properly formatted with simple data types
            video = sample['video']
            if isinstance(video, list):
                video = np.array(video, dtype=np.float32)
            elif isinstance(video, np.ndarray):
                video = video.astype(np.float32)
            
            # Make sure video has the right shape (T, H, W, C)
            if video.ndim == 4:
                pass  # Already correct shape
            elif video.ndim == 3:
                video = np.expand_dims(video, axis=0)  # Add time dimension
            
            label = np.array(sample['label'], dtype=np.int32)
            video_length = int(sample['video_length'])  # Simple integer
            label_length = len(sample['label'])  # Label length as integer
            
            yield (video, label, video_length, label_length)
    
    # Create dataset with GPU optimizations
    ms_dataset = ds.GeneratorDataset(
        generator,
        column_names=['video', 'label', 'videoLength', 'labelLength'],
        shuffle=is_train,
        num_parallel_workers=num_workers,
        max_rowsize=max_rowsize  # GPU optimization for memory efficiency
    )
    
    # Apply data augmentation operations if training
    if is_train:
        # Add random operations for data augmentation
        # Note: Some operations might need to be moved to custom transforms
        pass
    
    # Set prefetch for better GPU utilization (before batching)
    try:
        ms_dataset = ms_dataset.prefetch(buffer_size=prefetch_size)
        print(f"✓ Prefetch enabled with buffer size: {prefetch_size}")
    except AttributeError:
        print(f"Warning: Prefetch not available, skipping prefetch optimization")
    
    # Batch dataset with optimizations
    ms_dataset = ms_dataset.batch(
        batch_size=batch_size,
        drop_remainder=True
    )
    
    # Enable repeat for training (helps with GPU utilization)
    if is_train:
        ms_dataset = ms_dataset.repeat()
    
    return ms_dataset
