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
    def __init__(self, is_train=True, crop_size=224, max_frames=150):
        self.is_train = is_train
        self.crop_size = crop_size
        self.max_frames = max_frames
    
    def __call__(self, video_frames):
        """对视频帧应用转换"""
        # 如果需要，转换为numpy数组
        if isinstance(video_frames, list):
            video_frames = np.array(video_frames)
        
        # 限制最大帧数以减少内存使用
        if len(video_frames) > self.max_frames:
            # 均匀采样帧
            indices = np.linspace(0, len(video_frames) - 1, self.max_frames, dtype=int)
            video_frames = video_frames[indices]
        
        # 调整帧大小
        resized_frames = []
        for frame in video_frames:
            # 确保帧已经是正确的大小 (224x224)
            if frame.shape[:2] != (self.crop_size, self.crop_size):
                frame = cv2.resize(frame, (self.crop_size, self.crop_size))
            
            if self.is_train:
                # 随机水平翻转 (减少其他增强以节省内存)
                if np.random.random() > 0.5:
                    frame = cv2.flip(frame, 1)
            
            resized_frames.append(frame)
        
        # 转换为张量格式 (T, H, W, C) -> (T, C, H, W)
        video_tensor = np.array(resized_frames, dtype=np.float32)

        # 调试：检查张量形状
        if len(video_tensor.shape) != 4:
            print(f"警告：意外的视频张量形状：{video_tensor.shape}")
            print(f"期望4维张量 (T, H, W, C)，得到{len(video_tensor.shape)}维")
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
        current_frames = video_tensor.shape[0]

        if current_frames > self.max_frames:
            # 截断
            video_tensor = video_tensor[:self.max_frames]
        elif current_frames < self.max_frames:
            # 用零填充
            pad_frames = self.max_frames - current_frames
            pad_shape = (pad_frames,) + video_tensor.shape[1:]
            pad_tensor = np.zeros(pad_shape, dtype=video_tensor.dtype)
            video_tensor = np.concatenate([video_tensor, pad_tensor], axis=0)
        
        # 标准化到[0, 1]
        video_tensor = video_tensor.astype(np.float32) / 255.0
        
        return video_tensor

class CECSLDataset:
    """用于手语识别的CE-CSL数据集"""
    
    def __init__(self, data_path, label_path, word2idx, dataset_name, is_train=False, transform=None, crop_size=224, max_frames=150):
        self.data_path = data_path
        self.label_path = label_path
        self.word2idx = word2idx
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.crop_size = crop_size
        self.max_frames = max_frames
        self.transform = transform or VideoTransform(is_train=is_train, crop_size=crop_size, max_frames=max_frames)
        
        # 加载标签
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """从标签文件加载样本"""
        samples = []
        
        with open(self.label_path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:  # 跳过表头
                    video_name = row[0]
                    translator = row[1]  # 获取翻译者 (A, B, C, 等)
                    words = row[3].split("/")
                    words = preprocess_words(words)

                    # 将单词转换为索引
                    label_indices = []
                    for word in words:
                        if word in self.word2idx:
                            label_indices.append(self.word2idx[word])

                    if label_indices:  # 只有在有有效标签时才添加
                        # 构建正确的视频路径：data_path/translator/video_name.mp4
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

        # 加载视频帧
        video_frames = self._load_video(sample['video_path'])
        original_length = len(video_frames)

        # 应用转换
        if self.transform:
            video_frames = self.transform(video_frames)

        # 记录真实标签长度(填充前)
        max_label_length = 50  # 合理的最大标签长度
        raw_label = sample['label']
        true_label_length = min(len(raw_label), max_label_length)

        # 填充或截断
        if len(raw_label) > max_label_length:
            padded_label = raw_label[:max_label_length]
        else:
            padded_label = raw_label + [0] * (max_label_length - len(raw_label))

        return {
            'video': video_frames,
            'label': padded_label,            # [S_max]
            'label_length': true_label_length, # 真实长度，用于CTC
            'video_length': original_length,   # 使用填充前的原始视频帧数
            'info': sample['video_name']
        }
    
    def _load_video(self, video_path):
        """从目录加载视频帧，进行内存优化"""
        frames = []
        max_frames = 150  # 限制最大帧数以减少内存使用

        # 调试：打印视频路径（仅对前几个视频）
        debug_print = len(getattr(self, '_debug_count', [])) < 3
        if debug_print:
            if not hasattr(self, '_debug_count'):
                self._debug_count = []
            self._debug_count.append(1)
            print(f"从以下位置加载视频：{video_path}")
            print(f"路径存在：{os.path.exists(video_path)}")
            print(f"是目录：{os.path.isdir(video_path)}")

        if os.path.isdir(video_path):
            # 从图像目录加载
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
            if debug_print:
                print(f"找到{len(frame_files)}个帧文件")
                if len(frame_files) > 0:
                    print(f"前几个文件：{frame_files[:5]}")

            # 限制帧数以避免内存问题
            if len(frame_files) > max_frames:
                # 在视频中均匀采样帧
                step = len(frame_files) // max_frames
                frame_files = frame_files[::step][:max_frames]

            for frame_file in frame_files:
                frame_path = os.path.join(video_path, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # 调整帧大小以减少内存使用
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    if debug_print:
                        print(f"加载帧失败：{frame_path}")
        else:
            # 从视频文件加载
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 如果视频太长，计算采样帧的步长
            step = max(1, total_frames // max_frames) if total_frames > max_frames else 1
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % step == 0:
                    # 调整帧大小以减少内存使用
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    
                frame_count += 1
                if len(frames) >= max_frames:
                    break
                    
            cap.release()

        if debug_print:
            print(f"加载了{len(frames)}帧")
        
        # 转换为numpy数组并确保正确的数据类型
        if frames:
            frames = np.asarray(frames, dtype=np.float32)
            # 标准化到[0, 1]范围
            frames = frames / 255.0
        else:
            # 如果没有加载到帧，返回正确形状的空数组
            frames = np.empty((0, 224, 224, 3), dtype=np.float32)
            
        return frames

def collate_fn(videos, labels, video_lengths, infos):
    """用于批处理样本的整理函数 - MindSpore风格"""
    # 转换为numpy数组进行处理
    videos = [np.array(v) for v in videos]
    video_lengths = [int(vl) for vl in video_lengths]

    # 找到批次中的最大视频长度
    max_length = max(video_lengths)

    # 将视频填充到最大长度
    padded_videos = []
    for i, video in enumerate(videos):
        video_length = video_lengths[i]

        # 将视频填充到最大长度
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
                  prefetch_size=1, max_rowsize=16, crop_size=224, max_frames=150):
    """创建带有内存优化的MindSpore数据集"""
    
    # 创建带有内存优化的自定义数据集
    dataset = CECSLDataset(
        data_path=data_path,
        label_path=label_path,
        word2idx=word2idx,
        dataset_name=dataset_name,
        is_train=is_train,
        crop_size=crop_size,
        max_frames=max_frames
    )
    
    # 使用完整数据集大小（对于小内存测试，可以添加：min(len(dataset), 100)）
    dataset_size = len(dataset)
    
    # 转换为MindSpore数据集
    def generator():
        for i in range(dataset_size):
            sample = dataset[i]
            # 确保所有数据都使用简单数据类型正确格式化
            video = sample['video']
            if isinstance(video, list):
                video = np.array(video, dtype=np.float32)
            elif isinstance(video, np.ndarray):
                video = video.astype(np.float32)
            
            # 确保视频有正确的形状 (T, H, W, C) 或 (T, C, H, W)
            if video.ndim == 4:
                pass
            else:
                raise ValueError(f"视频张量维度不正确: {video.shape}")
            
            # 限制每个视频的最大帧数以减少内存
            if video.shape[0] > max_frames:
                indices = np.linspace(0, video.shape[0] - 1, max_frames, dtype=int)
                video = video[indices]
            
            label = np.array(sample['label'], dtype=np.int32)
            # 使用真实标签长度(未填充部分)
            label_length = int(sample.get('label_length', np.count_nonzero(label)))
            # 保障最小为1（CTC允许空吗? 为安全若为0则置1并用blank填充）
            if label_length == 0:
                label_length = 1
            
            video_length = min(int(sample['video_length']), max_frames)  # 限制视频长度
            
            yield (video, label, video_length, label_length)
    
    # 创建带有内存优化的数据集
    ms_dataset = ds.GeneratorDataset(
        generator,
        column_names=['video', 'label', 'videoLength', 'labelLength'],
        shuffle=is_train,
        num_parallel_workers=min(num_workers, 2),  # 限制工作进程数以节省内存
        max_rowsize=max_rowsize  # GPU优化以提高内存效率
    )
    
    # 如果是训练则应用数据增强操作
    if is_train:
        # 为数据增强添加随机操作
        # 注意：一些操作可能需要移动到自定义转换中
        pass
    
    # 设置预取以提高GPU利用率（在批处理之前）
    try:
        ms_dataset = ms_dataset.prefetch(buffer_size=prefetch_size)
        print(f"✓ 预取已启用，缓冲区大小：{prefetch_size}")
    except AttributeError:
        print(f"警告：预取不可用，跳过预取优化")
    
    # 使用优化进行批处理数据集
    ms_dataset = ms_dataset.batch(
        batch_size=batch_size,
        drop_remainder=True  # 丢弃不完整的批次以保持一致的内存使用
    )
    
    # 为训练启用重复（有助于GPU利用率）
    if is_train:
        ms_dataset = ms_dataset.repeat()
    
    return ms_dataset
