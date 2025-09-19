import os
import csv
import cv2
import numpy as np
import mindspore as ms
from mindspore import dataset as ds
from mindspore.dataset import vision, transforms
import json

PAD = '<PAD>'  # 使用特殊标记而不是空格
# 通过移除括号和数字来预处理单词列表
import re
import string

# 常见中文标点 + 英文标点
CHS_PUNCTS = "，。？！、；：''""（）【】《》〈〉「」『』—…·﹏·【】［］｛｝～｜·"
ALL_PUNCTS = string.punctuation + CHS_PUNCTS + "-–—‒―"  # 补充几种连字符

# 清除一段括号（中英文各种括号）的正则；重复应用可清除多段
BRACKET_RE = re.compile(r'[\(\[\{（【［]\s*[^)\]\}】］）]*[\)\]\}】］）]')

# 定义低质量词汇集合（无意义的单字符标点、空格等）
LOW_QUALITY_WORDS = {
    '', ' ', '.', ',', '?', '!', ';', ':', '(', ')', '[', ']', '{', '}', 
    '"', "'", '-', '_', '=', '+', '*', '/', '', '|', '~', '`', '@', '#', 
    '$', '%', '^', '&', '<', '>', '，', '。', '？', '！', '、', '；', '：',
    '（', '）', '【', '】', '《', '》', '〈', '〉', '「', '」', '『', '』', 
    '—', '…', '·', '﹏', '［', '］', '｛', '｝', '～', '｜'
}

def preprocess_words(words):
    """移除括号内内容、去掉所有标点，清除低质量词汇，去除多余空格"""
    out = []
    for w in words:
        # 确保输入是字符串并去除首尾空格
        w = str(w).strip()
        
        # 如果已经是空字符串，直接跳过
        if not w:
            continue
        
        # 先删除多余空格（连续空格变为单个空格）
        w = re.sub(r'\s+', ' ', w)

        # 反复清除括号及其中内容（若有多段/嵌套的简单场景）
        prev = None
        while prev != w:
            prev = w
            w = BRACKET_RE.sub('', w)

        # 去掉所有标点符号
        w = w.translate(str.maketrans('', '', ALL_PUNCTS))
        
        # 再次删除清理后的多余空格
        w = w.strip()

        # 严格检查：跳过空字符串、纯空格、或低质量词汇
        if (not w or 
            w.isspace() or 
            w in LOW_QUALITY_WORDS or 
            len(w) == 0):
            continue

        # ——以下沿用你原来的数字处理规则——
        # 若末尾是数字且首字符不是数字：仅去掉最后一个数字
        if w and w[-1].isdigit() and not w[0].isdigit():
            w = w[:-1]

        # 如果全是数字，则转成 int 再转回字符串（去前导零）
        if w.isdigit():
            w = str(int(w))

        # 最终检查：确保处理后的结果有效
        w_final = w.strip()
        if (w_final and 
            not w_final.isspace() and 
            w_final not in LOW_QUALITY_WORDS and 
            len(w_final) > 0):
            out.append(w_final)
    
    return out

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
    
    # 对word_list进行最终过滤，确保没有空字符串、空格或其他无效词汇
    filtered_words = []
    for word in word_list:
        word_clean = str(word).strip()
        # 严格过滤：必须有实际内容，不能是空字符串、纯空格或低质量词汇
        if (word_clean and 
            word_clean not in LOW_QUALITY_WORDS and 
            len(word_clean) > 0 and 
            not word_clean.isspace() and
            word_clean != ''):
            filtered_words.append(word_clean)
    
    print(f"词汇过滤: {len(word_list)} -> {len(filtered_words)}")
    
    # 构建词汇表 - 注意PAD应该是空格，但我们要确保它不会与实际的空格词汇冲突
    # 使用一个特殊的PAD标记而不是空格
    PAD_TOKEN = '<PAD>'
    idx2word = [PAD_TOKEN]  # 使用特殊PAD标记而不是空格
    
    # 去重并排序
    unique_words = sorted(list(set(filtered_words)))
    
    # 再次确保没有无效词汇
    final_words = []
    for word in unique_words:
        if (word and 
            word.strip() and 
            word not in LOW_QUALITY_WORDS and
            word != PAD_TOKEN and  # 避免与PAD标记冲突
            len(word.strip()) > 0):
            final_words.append(word)
    
    idx2word.extend(final_words)
    
    print(f"词汇表构建完成: {len(idx2word)} 个词汇 (含PAD)")
    
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
        # 统一空输入处理：直接返回零填充张量，避免后续转置报错
        if video_frames is None:
            return np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
        if isinstance(video_frames, (list, tuple)) and len(video_frames) == 0:
            return np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
        if isinstance(video_frames, np.ndarray):
            if video_frames.size == 0 or (video_frames.ndim >= 1 and video_frames.shape[0] == 0):
                return np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
        
        # 如果需要，转换为numpy数组
        if isinstance(video_frames, list):
            video_frames = np.array(video_frames)
        
        # 限制最大帧数以减少内存使用（均匀采样）
        if len(video_frames) > self.max_frames:
            indices = np.linspace(0, len(video_frames) - 1, self.max_frames, dtype=int)
            video_frames = video_frames[indices]
        
        # 调整帧大小
        resized_frames = []
        for frame in video_frames:
            if frame is None:
                continue
            frame = np.asarray(frame)
            # 灰度转3通道
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            # 保障有通道维
            if frame.ndim != 3:
                continue
            # 统一为 (H, W, C)
            if frame.shape[0] == 3 and frame.shape[-1] != 3:  # (C,H,W) -> (H,W,C)
                frame = np.transpose(frame, (1, 2, 0))
            # 确保大小
            if frame.shape[:2] != (self.crop_size, self.crop_size):
                frame = cv2.resize(frame, (self.crop_size, self.crop_size))
            if self.is_train and np.random.random() > 0.5:
                frame = cv2.flip(frame, 1)
            resized_frames.append(frame)
        
        # 若所有帧都无效，回退为零张量
        if len(resized_frames) == 0:
            return np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
        
        # (T, H, W, C) -> (T, C, H, W)
        video_tensor = np.array(resized_frames)
        if video_tensor.ndim == 4:
            if video_tensor.shape[-1] == 3:
                video_tensor = np.transpose(video_tensor, (0, 3, 1, 2))
        else:
            # 无法转置则回退
            return np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)

        # 截断/填充到固定长度
        current_frames = video_tensor.shape[0]
        if current_frames > self.max_frames:
            video_tensor = video_tensor[:self.max_frames]
        elif current_frames < self.max_frames:
            pad_frames = self.max_frames - current_frames
            pad_shape = (pad_frames,) + video_tensor.shape[1:]
            pad_tensor = np.zeros(pad_shape, dtype=video_tensor.dtype)
            video_tensor = np.concatenate([video_tensor, pad_tensor], axis=0)
        
        # 条件归一化，避免重复/二次归一化
        video_tensor = video_tensor.astype(np.float32)
        if np.nanmax(video_tensor) > 1.0:
            video_tensor = video_tensor / 255.0
        
        return video_tensor

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
                frame_list.append(np.asarray(arr[i], dtype=np.float32))
        elif arr.ndim == 3:
            # (H,W,C) 或 (C,H,W)
            if arr.shape[-1] == 3 and arr.shape[0] != 3:  # (H,W,C)
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.shape[0] == 3:
                pass
            else:
                return np.zeros((1, 3, target_size[0], target_size[1]), dtype=np.float32), np.int32(1)
            frame_list.append(np.asarray(arr, dtype=np.float32))
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
            frame_list.append(np.asarray(af, dtype=np.float32))

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
    # 条件归一化，避免重复归一化
    if np.nanmax(video) > 1.0:
        video = video / 255.0
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
                    video_name = str(row[0]).strip()
                    translator = str(row[1]).strip()  # 去除前后空白，避免路径中出现空格
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
        original_length = int(video_frames.shape[0]) if isinstance(video_frames, np.ndarray) else len(video_frames)

        # 若无帧，直接返回占位视频，避免进入 transform
        if original_length == 0:
            video_array = np.zeros((self.max_frames, 3, self.crop_size, self.crop_size), dtype=np.float32)
            seq_len = np.int32(1)
        else:
            # 应用转换（已包含截断/填充和条件归一化）
            if self.transform:
                video_frames = self.transform(video_frames)
            # 确保返回 (T,C,H,W)
            if not isinstance(video_frames, np.ndarray) or video_frames.ndim != 4:
                # 兜底处理
                video_array, seq_len = safe_stack_frames(video_frames, max_frames=self.max_frames, target_size=(self.crop_size, self.crop_size))
            else:
                video_array = np.asarray(video_frames, dtype=np.float32)
                seq_len = np.int32(min(original_length, self.max_frames))

        # 记录真实标签长度(填充前)
        max_label_length = 50  # 合理的最大标签长度
        raw_label = sample['label']
        true_label_length = min(len(raw_label), max_label_length)

        # 填充或截断
        if len(raw_label) > max_label_length:
            padded_label = raw_label[:max_label_length]
        else:
            padded_label = raw_label + [0] * (max_label_length - len(raw_label))

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
        
        # 转换为numpy数组并确保正确的数据类型（此处不归一化，交由 transform/safe_stack 处理）
        if frames:
            frames = np.asarray(frames, dtype=np.float32)
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
        for i in range(len(dataset)):
            video_array, label_ids, seq_len, label_len = dataset[i]
            # 强制类型转换，确保都是 numpy 基础类型
            video_array = np.asarray(video_array, dtype=np.float32)
            label_ids = np.asarray(label_ids, dtype=np.int32)
            seq_len = np.int32(seq_len)
            label_len = np.int32(label_len)
            yield video_array, label_ids, seq_len, label_len
    
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
    
    # 不使用无限repeat，让每个epoch自然结束
    # 这样可以正确进行epoch间的验证和检查点保存
    
    return ms_dataset




