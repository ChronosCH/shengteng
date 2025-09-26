from __future__ import annotations

import os
import sys
from typing import Tuple, List, Dict, Optional
import warnings
import hashlib
import time
import threading
import json

# fcntl（非 Windows）可选
try:
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except Exception:
    fcntl = None
    _HAS_FCNTL = False

# 全局配置（可选）
try:
    from config import Config
except Exception:
    Config = None  # type: ignore

# 依赖
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

# 高效解码后端
try:
    import decord
    from decord import VideoReader
    from decord import cpu as decord_cpu
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

try:
    import av  # PyAV
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

# MindSpore
try:
    import mindspore as ms
    from mindspore import Tensor
    import mindspore.dataset as ds
except Exception:
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    ds = None  # type: ignore

_cache_lock = threading.Lock()


def _ensure_pkg(pkg, hint: str):
    if pkg is None:
        raise ImportError(hint)


def _select_backend() -> str:
    if Config and getattr(Config, 'DECODER_BACKEND', 'auto') != 'auto':
        return Config.DECODER_BACKEND
    if _HAS_DECORD:
        return 'decord'
    if _HAS_PYAV:
        return 'pyav'
    if cv2 is not None:
        return 'opencv'
    raise ImportError("No video backend available. Please install decord or av or opencv-python.")


# manifest/lock
def _manifest_path() -> str:
    base = Config.CACHE_DIR if (Config and getattr(Config, 'CACHE_DIR', None)) else os.path.join(os.getcwd(), 'cache')
    return os.path.join(base, 'manifest.json')


def _lock_path() -> str:
    base = Config.CACHE_DIR if (Config and getattr(Config, 'CACHE_DIR', None)) else os.path.join(os.getcwd(), 'cache')
    return os.path.join(base, '.lock')


class _FileLock:
    def __enter__(self):
        self.fd = None
        if not (Config and getattr(Config, 'CACHE_ENABLED', False)):
            return self
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        if _HAS_FCNTL:
            self.fd = open(_lock_path(), 'a+')
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
        else:
            _cache_lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not (Config and getattr(Config, 'CACHE_ENABLED', False)):
            return False
        if _HAS_FCNTL and self.fd is not None:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            finally:
                self.fd.close()
        else:
            _cache_lock.release()
        return False


def _manifest_load() -> Dict[str, Dict]:
    p = _manifest_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _manifest_save(m: Dict[str, Dict]):
    p = _manifest_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False)
    os.replace(tmp, p)


def _video_cache_key(video_path: str, target_size: Tuple[int, int], seq_len: int, sampling: Dict) -> str:
    h = hashlib.sha256()
    h.update(video_path.encode('utf-8'))
    h.update(str(target_size).encode('utf-8'))
    h.update(str(seq_len).encode('utf-8'))
    h.update(str(sampling.get('mode')).encode('utf-8'))
    h.update(str(sampling.get('stride')).encode('utf-8'))
    h.update(str(sampling.get('rand')).encode('utf-8'))
    return h.hexdigest()


def _make_indices(total_frames: int, seq_len: int, sampling: Dict) -> List[int]:
    import numpy as _np
    mode = sampling.get('mode', 'uniform')
    stride = int(sampling.get('stride', 1))
    rand = bool(sampling.get('rand', False))
    if total_frames <= 0:
        return []
    if mode == 'stride':
        if rand and total_frames > seq_len:
            start_max = max(0, total_frames - seq_len * stride)
            start = int(_np.random.randint(0, start_max + 1)) if start_max > 0 else 0
        else:
            start = 0
        return [min(start + i * stride, total_frames - 1) for i in range(seq_len)]
    if mode == 'random_segment':
        bounds = _np.linspace(0, total_frames, seq_len + 1, dtype=int)
        idx = []
        for i in range(seq_len):
            a, b = bounds[i], max(bounds[i] + 1, bounds[i + 1])
            j = int(_np.random.randint(a, b)) if b > a else a
            idx.append(min(j, total_frames - 1))
        return idx
    if rand and total_frames > seq_len:
        span = total_frames - 1
        offset = int(_np.random.randint(0, max(1, span // seq_len)))
        lin = _np.linspace(0 + offset, span, seq_len, dtype=int)
    else:
        lin = _np.linspace(0, total_frames - 1, seq_len, dtype=int)
    return lin.tolist()


def _sampling_deterministic(sampling: Dict) -> bool:
    mode = sampling.get('mode', 'uniform')
    rand = bool(sampling.get('rand', False))
    if mode == 'random_segment':
        return False
    if rand:
        return False
    return True


def _cache_dir_for_key(key: str) -> str:
    base = Config.CACHE_DIR if (Config and getattr(Config, 'CACHE_DIR', None)) else os.path.join(os.getcwd(), 'cache')
    return os.path.join(base, key[:2], key[2:4], key)


def _cache_meta_path(dirpath: str) -> str:
    return os.path.join(dirpath, 'meta.json')


def _cache_frame_path(dirpath: str, index: int, ext: str) -> str:
    return os.path.join(dirpath, f"{index:03d}.{ext}")


def _now_ts() -> float:
    return time.time()


def _touch_dir(dirpath: str):
    ts = _now_ts()
    try:
        os.utime(dirpath, (ts, ts))
    except Exception:
        pass


def _imwrite(image, path: str, ext: str, quality: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if ext.lower() == 'webp':
        cv2.imwrite(path, image[:, :, ::-1], [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    elif ext.lower() in ('jpg', 'jpeg'):
        cv2.imwrite(path, image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        cv2.imwrite(path, image[:, :, ::-1])


def _imread(path: str):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Failed to read cached image: {path}")
    return img[:, :, ::-1]


class VideoProcessor:
    """视频预处理类，支持预提取帧目录读取"""

    def __init__(self, target_size=(224, 224), sequence_length=16, augment: bool = False,
                 hflip_prob: float | None = None, rotation_range: float | None = None,
                 brightness_range: float | None = None, contrast_range: float | None = None,
                 mean=None, std=None, sampling_mode: str | None = None,
                 sampling_stride: int | None = None, sampling_rand: bool | None = None):
        self.target_size = target_size
        self.sequence_length = sequence_length
        self.augment = augment
        if Config is not None:
            default_hflip = getattr(Config, 'HORIZONTAL_FLIP_PROB', 0.0)
            default_rot = getattr(Config, 'ROTATION_RANGE', 10)
            default_bright = getattr(Config, 'BRIGHTNESS_RANGE', 0.1)
            default_contrast = getattr(Config, 'CONTRAST_RANGE', 0.1)
            default_mean = getattr(Config, 'MEAN', (0.485, 0.456, 0.406))
            default_std = getattr(Config, 'STD', (0.229, 0.224, 0.225))
            default_sampling_mode = getattr(Config, 'FRAME_SAMPLING_MODE', 'uniform')
            default_sampling_stride = getattr(Config, 'FRAME_SAMPLING_STRIDE', 1)
            default_sampling_rand = getattr(Config, 'FRAME_SAMPLING_RANDOM_OFFSET', False)
        else:
            default_hflip = 0.0
            default_rot = 10
            default_bright = 0.1
            default_contrast = 0.1
            default_mean = (0.485, 0.456, 0.406)
            default_std = (0.229, 0.224, 0.225)
            default_sampling_mode = 'uniform'
            default_sampling_stride = 1
            default_sampling_rand = False

        self.hflip_prob = float(default_hflip if hflip_prob is None else hflip_prob)
        self.rotation_range = float(default_rot if rotation_range is None else rotation_range)
        self.brightness_range = float(default_bright if brightness_range is None else brightness_range)
        self.contrast_range = float(default_contrast if contrast_range is None else contrast_range)
        self.mean = np.array(default_mean if mean is None else mean, dtype=np.float32)
        self.std = np.array(default_std if std is None else std, dtype=np.float32)
        self.sampling_mode = (sampling_mode if sampling_mode is not None else default_sampling_mode)
        self.sampling_stride = int(sampling_stride if sampling_stride is not None else default_sampling_stride)
        self.sampling_rand = bool(default_sampling_rand if sampling_rand is None else sampling_rand)
        _ensure_pkg(np, "NumPy is required. pip install numpy")
        _ensure_pkg(cv2, "OpenCV is required. pip install opencv-python")
        self.backend = _select_backend()

    # 预提取帧支持
    def _frames_dir_for(self, video_path: str) -> Optional[str]:
        if Config is None or not getattr(Config, 'USE_PREEXTRACTED_FRAMES', False):
            return None
        base = os.path.splitext(os.path.basename(video_path))[0]
        frames_root = getattr(Config, 'FRAMES_DIR', None)
        if not frames_root:
            return None
        d = os.path.join(frames_root, base)
        return d if os.path.isdir(d) else None

    def _list_pre_frames(self, frames_dir: str) -> List[str]:
        exts = set([e.lower() for e in getattr(Config, 'FRAME_IMAGE_EXTS', ('.jpg', '.jpeg', '.png', '.webp'))])
        files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                 if os.path.splitext(f)[1].lower() in exts]
        if not files:
            return []
        try:
            import re
            def key_fn(p):
                n = os.path.splitext(os.path.basename(p))[0]
                m = re.match(r'^(\d+)$', n)
                return (0, int(m.group(1))) if m else (1, n)
            files.sort(key=key_fn)
        except Exception:
            files.sort()
        return files

    def _decode_from_pre_frames(self, video_path: str, indices: List[int]):
        frames_dir = self._frames_dir_for(video_path)
        if not frames_dir:
            return None
        files = self._list_pre_frames(frames_dir)
        if not files:
            return None
        max_idx = len(files) - 1
        out = []
        for i in indices:
            j = min(max(0, int(i)), max_idx)
            img = cv2.imread(files[j])
            if img is None:
                img = (np.zeros((*self.target_size, 3), dtype=np.uint8))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if (img.shape[1], img.shape[0]) != self.target_size:
                    img = cv2.resize(img, self.target_size)
            out.append(img.astype(np.float32) / 255.0)
        return np.stack(out, axis=0)

    # 常规增广/归一化/解码
    def _augment_clip(self, clip):
        if np is None or cv2 is None:
            return clip
        T, H, W, C = clip.shape
        out = clip.copy()
        if self.augment and self.hflip_prob > 0 and np.random.rand() < self.hflip_prob:
            out = out[:, :, ::-1, :]
        angle = float(np.random.uniform(-self.rotation_range, self.rotation_range)) if self.augment else 0.0
        if abs(angle) > 1e-3:
            center = (W / 2.0, H / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = []
            for i in range(T):
                fr = (out[i] * 255.0).astype(np.uint8)
                fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                rotated.append(fr.astype(np.float32) / 255.0)
            out = np.stack(rotated, axis=0)
        if self.augment:
            c_delta = float(np.random.uniform(-self.contrast_range, self.contrast_range))
            b_delta = float(np.random.uniform(-self.brightness_range, self.brightness_range))
            out = np.clip(out * (1.0 + c_delta) + b_delta, 0.0, 1.0)
        return out

    def _normalize(self, clip):
        if clip is None:
            return None
        return (clip - self.mean.reshape(1, 1, 1, 3)) / self.std.reshape(1, 1, 1, 3)

    def _decode_with_decord(self, video_path: str, indices: List[int]):
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        frames = vr.get_batch(indices).asnumpy()
        out = [cv2.resize(fr, self.target_size).astype(np.float32) / 255.0 for fr in frames]
        return np.stack(out, axis=0)

    def _decode_with_pyav(self, video_path: str, indices: List[int]):
        container = av.open(video_path)
        stream = container.streams.video[0]
        result = []
        wanted = set(indices)
        cur = 0
        for frame in container.decode(stream):
            if cur in wanted:
                img = frame.to_ndarray(format='rgb24')
                img = cv2.resize(img, self.target_size)
                result.append(img.astype(np.float32) / 255.0)
                if len(result) == len(indices):
                    break
            cur += 1
        container.close()
        if len(result) < len(indices):
            while len(result) < len(indices):
                result.append(result[-1] if result else np.zeros((*self.target_size, 3), dtype=np.float32))
        return np.stack(result, axis=0)

    def _decode_with_opencv(self, video_path: str, indices: List[int]):
        cap = cv2.VideoCapture(video_path)
        out = []
        last = None
        cur = 0
        target_set = set(indices)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cur in target_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                last = frame
                out.append(frame.astype(np.float32) / 255.0)
                if len(out) == len(indices):
                    break
            cur += 1
        cap.release()
        if len(out) < len(indices):
            while len(out) < len(indices):
                out.append(last if last is not None else np.zeros((*self.target_size, 3), dtype=np.float32))
        return np.stack(out, axis=0)

    def _total_frames(self, video_path: str) -> int:
        try:
            frames_dir = self._frames_dir_for(video_path)
            if frames_dir:
                files = self._list_pre_frames(frames_dir)
                if files:
                    return len(files)
        except Exception:
            pass
        try:
            if _HAS_DECORD and self.backend == 'decord':
                vr = VideoReader(video_path, ctx=decord_cpu(0))
                return len(vr)
            if _HAS_PYAV and self.backend == 'pyav':
                container = av.open(video_path)
                stream = container.streams.video[0]
                frames = int(getattr(stream, 'frames', 0) or 0)
                if frames > 0:
                    container.close()
                    return frames
                rate = float(stream.average_rate) if stream.average_rate else 0.0
                duration = (stream.duration * float(stream.time_base)) if stream.duration and stream.time_base else 0.0
                est = int(duration * rate) if rate > 0 and duration > 0 else 0
                container.close()
                return est
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return total
        except Exception:
            return 0

    def _decode_clip(self, video_path: str):
        total_frames = self._total_frames(video_path)
        if total_frames <= 0:
            return None
        sampling = {
            'mode': self.sampling_mode,
            'stride': self.sampling_stride,
            'rand': self.sampling_rand,
        }
        indices = _make_indices(total_frames, self.sequence_length, sampling)
        pre = self._decode_from_pre_frames(video_path, indices)
        if pre is not None:
            return pre
        if _HAS_DECORD and self.backend == 'decord':
            return self._decode_with_decord(video_path, indices)
        if _HAS_PYAV and self.backend == 'pyav':
            return self._decode_with_pyav(video_path, indices)
        return self._decode_with_opencv(video_path, indices)

    def extract_frames(self, video_path: str):
        sampling = {
            'mode': self.sampling_mode,
            'stride': self.sampling_stride,
            'rand': self.sampling_rand,
        }
        deterministic = _sampling_deterministic(sampling)
        use_cache_global = bool(Config and getattr(Config, 'CACHE_ENABLED', False))
        use_cache = use_cache_global and deterministic
        key = _video_cache_key(video_path, self.target_size, self.sequence_length, sampling)
        clip = self._decode_clip(video_path)
        if clip is None:
            return None
        if self.augment:
            clip = self._augment_clip(clip)
        clip = self._normalize(clip)
        return clip


class ASLDataset:
    """ASL 数据集"""
    def __init__(self, csv_path: str, video_dir: str, processor: VideoProcessor,
                 class_to_idx: Dict[str, int] = None, max_samples: Optional[int] = None,
                 drop_unknown: bool = False, max_per_class: Optional[int] = None):
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.processor = processor
        self.data_df = pd.read_csv(csv_path) if pd is not None else None
        if self.data_df is None:
            raise RuntimeError("pandas 未可用，无法读取 CSV")
        # 可选：过滤掉不在映射内的类别
        if class_to_idx is not None and drop_unknown:
            before = len(self.data_df)
            self.data_df = self.data_df[self.data_df['Gloss'].astype(str).isin(class_to_idx.keys())].reset_index(drop=True)
            removed = before - len(self.data_df)
            if removed > 0:
                print(f"[INFO] 过滤掉 {removed} 个未在映射中的样本: {os.path.basename(csv_path)}")
        # 每类裁切（保持原始顺序）
        if isinstance(max_per_class, int) and max_per_class > 0:
            before = len(self.data_df)
            try:
                self.data_df = (
                    self.data_df
                    .groupby(self.data_df['Gloss'].astype(str), group_keys=False)
                    .head(max_per_class)
                    .reset_index(drop=True)
                )
                removed = before - len(self.data_df)
                print(f"[VERBOSE] 每类最多 {max_per_class} 个样本: 剪裁 {removed} 条 (保留 {len(self.data_df)}) - {os.path.basename(csv_path)}")
            except Exception as e:
                print(f"[WARN] 每类样本数限制失败({e})，继续使用完整数据")

        self._max_per_class = max_per_class if isinstance(max_per_class, int) and max_per_class > 0 else None

        # 子集裁切（在每类限制之后执行，避免提前截断长尾类别）
        if isinstance(max_samples, int) and max_samples > 0:
            before = len(self.data_df)
            self.data_df = self.data_df.head(int(max_samples))
            removed = before - len(self.data_df)
            print(f"[VERBOSE] 使用前 {len(self.data_df)} 个样本: 剪裁 {removed} 条 - {os.path.basename(csv_path)}")
        if class_to_idx is None:
            unique_classes = sorted(self.data_df['Gloss'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        print(f"数据集大小: {len(self.data_df)}")
        print(f"类别数量: {self.num_classes}")
        self._warned_oob = False

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        video_file = row['Video file']
        gloss = row['Gloss']
        video_path = os.path.join(self.video_dir, video_file)
        frames = self.processor.extract_frames(video_path)
        if frames is None:
            frames = np.zeros((self.processor.sequence_length, *self.processor.target_size, 3), dtype=np.float32)
        # (T,H,W,C)->(C,T,H,W)
        frames = np.transpose(frames, (3, 0, 1, 2))
        label = self.class_to_idx.get(gloss, -1)
        # 类型与范围
        try:
            label = int(label)
        except Exception:
            label = -1
        if label < 0 or label >= self.num_classes:
            if not self._warned_oob:
                print(f"[WARN] 标签越界，已裁剪（示例）: gloss={gloss}, raw_label={label}, num_classes={self.num_classes}")
                self._warned_oob = True
            label = max(0, min(self.num_classes - 1, label))
        label = np.int32(label)
        return frames, label


def create_mindspore_dataset(dataset: ASLDataset, batch_size: int = 32,
                             shuffle: bool = True, num_parallel_workers: int = 4):
    """创建 MindSpore 数据集"""
    def generator():
        for i in range(len(dataset)):
            frames, label = dataset[i]
            # 再防御一次
            try:
                ncls = int(dataset.num_classes)
            except Exception:
                ncls = None
            if ncls and (int(label) < 0 or int(label) >= ncls):
                if not getattr(dataset, '_warned_oob', False):
                    print(f"[WARN] 生成器检测到越界标签，裁剪: label={int(label)}, num_classes={ncls}")
                    dataset._warned_oob = True
                label = np.int32(max(0, min(ncls - 1, int(label))))
            else:
                label = np.int32(label)
            yield frames, label

    # 仅尝试提供 column_types（部分版本不支持 column_shapes）
    try:
        column_types = (ms.float32, ms.int32) if ms is not None else None
    except Exception:
        column_types = None

    mp = bool(getattr(Config, 'DATASET_USE_PYTHON_MP', False)) if Config else False

    # 规范化并校正并行度（MindSpore 要求 1..cpu_count）
    try:
        import os
        cpu_n = max(1, os.cpu_count() or 1)
    except Exception:
        cpu_n = 1
    try:
        req_npw = int(num_parallel_workers)
    except Exception:
        req_npw = 1
    npw = 1 if req_npw is None or req_npw < 1 else min(req_npw, cpu_n)

    base_kwargs = {
        'source': generator,
        'column_names': ["frames", "label"],
        'num_parallel_workers': npw,
        'shuffle': shuffle,
        'python_multiprocessing': mp,
    }

    if ds is None:
        raise ImportError("MindSpore dataset 未安装或不可用")

    print(f"[VERBOSE] 正在构建 MindSpore GeneratorDataset... num_parallel_workers={npw} (requested={req_npw}, cpu={cpu_n}), mp={mp}, shuffle={shuffle}")

    # 优先带 column_types，失败则回退
    try:
        if column_types is not None:
            kw = dict(base_kwargs)
            kw['column_types'] = column_types
            ms_dataset = ds.GeneratorDataset(**kw)
        else:
            ms_dataset = ds.GeneratorDataset(**base_kwargs)
    except TypeError as e:
        if 'column_types' in str(e):
            # 版本不支持 column_types，回退
            ms_dataset = ds.GeneratorDataset(**base_kwargs)
        else:
            raise

    try:
        if shuffle:
            buf = int(getattr(Config, 'SHUFFLE_BUFFER_SIZE', 1024)) if Config else 1024
            ms_dataset = ms_dataset.shuffle(buf)
    except Exception:
        pass

    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)

    try:
        prefetch = int(getattr(Config, 'PREFETCH_BUFFER_SIZE', 2)) if Config else 2
        ms_dataset = ms_dataset.prefetch(prefetch)
    except Exception:
        pass

    return ms_dataset


def get_class_mapping(train_csv_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique_classes: List[str] = []
    use_pandas = pd is not None
    if use_pandas:
        try:
            train_df = pd.read_csv(train_csv_path)
            unique_classes = sorted(map(str, train_df['Gloss'].unique().tolist()))
        except Exception as e:
            warnings.warn(f"pandas 读取失败，使用 csv 回退解析: {e}")
            use_pandas = False
    if not use_pandas:
        import csv
        uniq = set()
        with open(train_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and 'Gloss' in reader.fieldnames:
                for row in reader:
                    val = (row.get('Gloss') or '').strip()
                    if val:
                        uniq.add(val)
            else:
                f.seek(0)
                r2 = csv.reader(f)
                header = next(r2, [])
                gloss_idx = 2 if len(header) > 2 else None
                for row in r2:
                    if gloss_idx is not None and len(row) > gloss_idx:
                        val = (row[gloss_idx] or '').strip()
                        if val:
                            uniq.add(val)
        unique_classes = sorted(list(uniq))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class