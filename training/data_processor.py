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
import re
import string

# 常见中文标点 + 英文标点
CHS_PUNCTS = "，。？！、；：‘’“”（）【】《》〈〉「」『』—…·﹏·【】［］｛｝～｜·"
ALL_PUNCTS = string.punctuation + CHS_PUNCTS + "-–—‒―"  # 补充几种连字符

# 清除一段括号（中英文各种括号）的正则；重复应用可清除多段
BRACKET_RE = re.compile(r'[\(\[\{（【［]\s*[^)\]\}】］）]*[\)\]\}】］）]')

def preprocess_words(words):
    """移除括号内内容、去掉所有标点；保留你原来的数字处理规则"""
    out = []
    for w in words:
        w = str(w).strip()

        # 反复清除括号及其中内容（若有多段/嵌套的简单场景）
        prev = None
        while prev != w:
            prev = w
            w = BRACKET_RE.sub('', w)

        # 去掉所有标点符号
        w = w.translate(str.maketrans('', '', ALL_PUNCTS))

        # ——以下沿用你原来的数字处理规则——
        # 若末尾是数字且首字符不是数字：仅去掉最后一个数字
        if w and w[-1].isdigit() and not w[0].isdigit():
            w = w[:-1]

        # 如果全是数字，则转成 int 再转回字符串（去前导零）
        if w.isdigit():
            w = str(int(w))

        out.append(w)
    return out

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
    """用于数据增强的视频转换，优化内存使用"""
    def __init__(self, is_train=True, crop_size=224, max_frames=150, use_cache=True, prefetch_buffer=True):
        self.is_train = is_train
        self.crop_size = crop_size
        self.max_frames = max_frames
        self.use_cache = use_cache
        self.prefetch_buffer = prefetch_buffer
        
        # 内存优化配置
        self._cache = {} if use_cache else None
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 500  # 减少缓存以控制内存使用
        
        # 预分配缓冲区以减少内存分配
        if prefetch_buffer:
            self._buffer = np.zeros((max_frames, 3, crop_size, crop_size), dtype=np.float32)
        else:
            self._buffer = None
    
    def __call__(self, video_frames):
        """对视频帧应用转换，带缓存和内存优化"""
        # 缓存检查
        cache_key = None
        if self.use_cache and isinstance(video_frames, np.ndarray) and video_frames.size > 0:
            # 使用形状和内容哈希作为缓存键
            cache_key = hash((video_frames.shape, str(video_frames.data[:min(1024, video_frames.size * video_frames.itemsize)])))
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key].copy()
            else:
                self._cache_misses += 1
        
        # 如果需要，转换为numpy数组
        if isinstance(video_frames, list):
            video_frames = np.array(video_frames, dtype=np.uint8)  # 指定dtype减少内存转换
        
        # 限制最大帧数以减少内存使用
        if len(video_frames) > self.max_frames:
            # 均匀采样帧
            indices = np.linspace(0, len(video_frames) - 1, self.max_frames, dtype=int)
            video_frames = video_frames[indices]
        
        # 使用预分配的缓冲区或批量调整大小
        if self._buffer is not None:
            # 使用预分配缓冲区
            result_buffer = self._buffer[:len(video_frames)]
        else:
            result_buffer = np.zeros((len(video_frames), 3, self.crop_size, self.crop_size), dtype=np.float32)
        
        # 批量处理帧以提高效率
        for i, frame in enumerate(video_frames):
            # 确保帧已经是正确的大小
            if frame.shape[:2] != (self.crop_size, self.crop_size):
                frame = cv2.resize(frame, (self.crop_size, self.crop_size))
            
            if self.is_train:
                # 随机水平翻转 (减少其他增强以节省内存)
                if np.random.random() > 0.5:
                    frame = cv2.flip(frame, 1)
            
            # 直接写入结果缓冲区，减少内存拷贝
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # (H, W, C) -> (C, H, W)
                result_buffer[i] = frame.transpose(2, 0, 1)
            elif len(frame.shape) == 2:
                # 灰度图转RGB
                result_buffer[i, 0] = frame
                result_buffer[i, 1] = frame  
                result_buffer[i, 2] = frame
        
        # 转换为张量格式
        video_tensor = result_buffer[:len(video_frames)]

        # 检查和修复张量形状
        if len(video_tensor.shape) != 4:
            print(f"警告：意外的视频张量形状：{video_tensor.shape}")
            if len(video_tensor.shape) == 0 or video_tensor.size == 0:
                print("警告：检测到空视频张量，创建默认视频数据")
                video_tensor = np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
            else:
                print(f"错误：无法处理形状为{video_tensor.shape}的张量，创建默认视频数据")
                video_tensor = np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
        
        # 填充或截断到固定长度以便批处理
        current_frames = video_tensor.shape[0]
        
        if current_frames == 0:
            print("警告：视频张量为空，创建默认视频数据")
            video_tensor = np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
        elif current_frames > self.max_frames:
            # 截断
            video_tensor = video_tensor[:self.max_frames]
        elif current_frames < self.max_frames:
            # 如果使用预分配缓冲区，直接使用它
            if self._buffer is not None:
                self._buffer[current_frames:self.max_frames] = 0  # 清零填充部分
                video_tensor = self._buffer[:self.max_frames].copy()
            else:
                # 用零填充
                pad_frames = self.max_frames - current_frames
                pad_shape = (pad_frames,) + video_tensor.shape[1:]
                pad_tensor = np.zeros(pad_shape, dtype=video_tensor.dtype)
                video_tensor = np.concatenate([video_tensor, pad_tensor], axis=0)
        
        # 标准化到[0, 1] - 原地操作节省内存
        if video_tensor.max() > 1.0:
            np.divide(video_tensor, 255.0, out=video_tensor)
        
        # 缓存结果（限制缓存大小）
        if self.use_cache and cache_key is not None:
            if len(self._cache) >= self._max_cache_size:
                # 删除最旧的缓存项
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = video_tensor.copy()
        
        return video_tensor
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses, 
            'hit_rate': hit_rate,
            'cache_size': len(self._cache) if self._cache else 0
        }
    
    def clear_cache(self):
        """清理缓存释放内存"""
        if self._cache:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

def _normalize_mempool_size(value: str) -> str:
    """（若被其他模块需要可放这里）将 '256MB' 等转为 MindSpore 支持的 '0.25GB'."""
    if not isinstance(value, str):
        return value
    v = value.strip().upper()
    if v.endswith("MB"):
        try:
            mb = float(v[:-2])
            gb = mb / 1024.0
            # MindSpore 需要如 0.25GB / 1GB / 2GB
            return f"{gb:.3f}GB"
        except:
            return value
    return value

# 将帧列表转换为 (T, C, H, W) float32 ndarray。
# 处理截断与零填充，确保返回 numpy 基础数组，避免 MindSpore 内部 copy=False 触发异常。
def safe_stack_frames(frames, max_frames=None, target_size=(160, 160)):
    """
    统一将输入转换为 (T, C, H, W) float32
    支持:
      - list/tuple: 元素为 (H,W,C)、(C,H,W)、灰度( H,W )
      - ndarray: (T,H,W,C)、(T,C,H,W)、(H,W,C)、(C,H,W)
    填充/截断到 max_frames（若给定），返回: video_array, seq_len(int32)
    """
    if frames is None:
        return np.zeros((1, 3, target_size[0], target_size[1]), dtype=np.float32), np.int32(1)

    frame_list = []

    # 若是 ndarray
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.ndim == 4:
            # (T,H,W,C) 或 (T,C,H,W)
            if arr.shape[-1] == 3:          # (T,H,W,C)
                arr = np.transpose(arr, (0, 3, 1, 2))
            elif arr.shape[1] == 3:         # (T,C,H,W)
                pass
            else:
                # 形状不符合，返回占位
                return np.zeros((1, 3, target_size[0], target_size[1]), dtype=np.float32), np.int32(1)
            for i in range(arr.shape[0]):
                frame_list.append(arr[i].astype(np.float32, copy=False))
        elif arr.ndim == 3:
            # (H,W,C) 或 (C,H,W)
            if arr.shape[-1] == 3 and arr.shape[0] != 3:  # (H,W,C)
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.shape[0] == 3:
                pass
            else:
                return np.zeros((1, 3, target_size[0], target_size[1]), dtype=np.float32), np.int32(1)
            frame_list.append(arr.astype(np.float32, copy=False))
        else:
            return np.zeros((1, 3, target_size[0], target_size[1]), dtype=np.float32), np.int32(1)

    else:
        # 可迭代对象
        for f in frames:
            if f is None:
                continue
            af = np.asarray(f)
            if af.ndim == 2:  # 灰度补3通道
                af = np.repeat(af[:, :, None], 3, axis=2)
            if af.ndim != 3:
                continue
            if af.shape[-1] == 3 and af.shape[0] != 3:  # (H,W,C)
                af = np.transpose(af, (2, 0, 1))
            elif af.shape[0] == 3:
                pass
            else:
                continue
            frame_list.append(af.astype(np.float32, copy=False))

    if len(frame_list) == 0:
        return np.zeros((1, 3, target_size[0], target_size[1]), dtype=np.float32), np.int32(1)

    original_len = len(frame_list)

    # 截断 / 填充
    if max_frames is not None:
        if original_len > max_frames:
            frame_list = frame_list[:max_frames]
        elif original_len < max_frames:
            c, h, w = frame_list[0].shape
            pad_n = max_frames - original_len
            pad_block = np.zeros((pad_n, c, h, w), dtype=np.float32)
            for i in range(pad_n):
                frame_list.append(pad_block[i])

    video = np.stack(frame_list, axis=0)  # (T,C,H,W)
    video /= 255.0
    seq_len = min(original_len, video.shape[0])
    return np.ascontiguousarray(video, dtype=np.float32), np.int32(seq_len)

# 假设存在 CECSLDataset（示例修改）
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
                    video_name = row[0].strip()
                    translator = row[1].strip()  # 获取翻译者 (A, B, C, 等) 并去掉空格
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
                        
                        # 检查视频文件是否存在
                        if os.path.exists(video_path):
                            samples.append({
                                'video_name': video_name,
                                'label': label_indices,
                                'video_path': video_path
                            })
                        else:
                            print(f"警告：视频文件不存在，跳过: {video_path}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]

            # 加载视频帧
            video_frames = self._load_video(sample['video_path'])
            original_length = len(video_frames) if len(video_frames.shape) > 0 else 0

            # 应用转换
            if self.transform:
                video_frames = self.transform(video_frames)
        except Exception as e:
            print(f"警告：处理样本 {idx} 时出错: {e}")
            # 返回默认的安全数据
            crop_size = getattr(self, 'crop_size', 112)
            max_frames = getattr(self, 'max_frames', 30)
            video_frames = np.zeros((max_frames, 3, crop_size, crop_size), dtype=np.float32)
            sample = {'label': [0], 'video_path': 'dummy'}  # 默认标签

        # 记录真实标签长度(填充前)
        max_label_length = 50  # 合理的最大标签长度
        raw_label = sample['label']
        true_label_length = min(len(raw_label), max_label_length)

        # 填充或截断
        if len(raw_label) > max_label_length:
            padded_label = raw_label[:max_label_length]
        else:
            padded_label = raw_label + [0] * (max_label_length - len(raw_label))

        # 假设 self.max_frames 已从外部传入
        if video_frames is None or (isinstance(video_frames, (list, tuple)) and len(video_frames) == 0):
            # 处理空
            video_array = np.zeros((1,3,160,160), dtype=np.float32)
            seq_len = np.int32(1)
        else:
            video_array, seq_len = safe_stack_frames(video_frames, max_frames=getattr(self, "max_frames", None))
        label_ids = np.asarray(padded_label, dtype=np.int32)
        label_len = np.int32(len(padded_label))
        return video_array, label_ids, seq_len, label_len
    
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
            # 如果没有加载到帧，创建默认的单帧视频
            print(f"警告：视频文件 {video_path} 没有有效帧，创建默认帧")
            frames = np.zeros((1, 224, 224, 3), dtype=np.float32)
            
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
                  prefetch_size=1, max_rowsize=16, crop_size=224, max_frames=150, 
                  dtype='float32', enable_cache=True, memory_optimize=True):
    """创建高性能内存优化的MindSpore数据集
    - dtype: 'float32' 或 'float16'，用于减少内存占用
    - enable_cache: 启用数据缓存提高性能
    - memory_optimize: 启用内存优化特性
    - 针对31GB内存环境优化的配置
    """
    
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
        for i in range(len(dataset)):
            video_array, label_ids, seq_len, label_len = dataset[i]
            # 根据dtype强制类型转换，float16可显著降低内存占用
            if dtype == 'float16':
                video_array = np.asarray(video_array, dtype=np.float16)
            else:
                video_array = np.asarray(video_array, dtype=np.float32)
            label_ids = np.asarray(label_ids, dtype=np.int32)
            seq_len = np.int32(seq_len)
            label_len = np.int32(label_len)
            yield video_array, label_ids, seq_len, label_len
    
    # 创建高性能内存优化的数据集
    ms_dataset = ds.GeneratorDataset(
        generator,
        column_names=['video', 'label', 'videoLength', 'labelLength'],
        shuffle=is_train,
        num_parallel_workers=min(num_workers, 8),  # 充分利用多核CPU
        max_rowsize=max_rowsize
    )
    
    # 内存优化配置
    if memory_optimize:
        # 启用数据缓存（利用大内存）
        if enable_cache and is_train:
            try:
                cache_map = {}
                ms_dataset = ms_dataset.map(
                    operations=lambda x, y, z, w: (x, y, z, w), 
                    input_columns=['video', 'label', 'videoLength', 'labelLength'],
                    cache=cache_map
                )
                print(f"✓ 数据缓存已启用")
            except Exception as e:
                print(f"警告：无法启用数据缓存: {e}")
        
        # 设置更大的预取缓冲区充分利用内存
        prefetch_size = max(prefetch_size, 32 if is_train else 16)
    
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
    
    return ms_dataset
