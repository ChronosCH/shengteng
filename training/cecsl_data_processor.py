"""
CE-CSL数据集处理模块（适配给定目录结构）
- 将视频预处理为 .npy 帧序列，写入 data/CE-CSL/processed/{split}/
- 生成每个 split 的 metadata.json 和 stats.json
- 支持两类标签：
    1) label/{split}.csv 中的 Gloss（构建手语词汇表）
    2) {split}.corpus.csv 中的分段标注（video_id/start/end/label），用于训练片段分类/识别
- 提供基于分段标注（corpus）的数据集与 DataLoader（MindSpore）
"""

import os
import csv
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2
import imageio
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset as ds

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAD = ' '

def seed_everything(seed: int = 42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        ms.set_seed(seed)

# ---------------------- 视频预处理 ----------------------

class CECSLVideoProcessor:
        """
        将 data/CE-CSL/video/{split}/{translator}/*.mp4|*.avi|*.mov|*.mkv
        预处理为 data/CE-CSL/processed/{split}/{split}_video_{idx:03d}_frames.npy
        并生成:
            - {split}_metadata.json: 列表，每条包含 video_id、translator、源文件、fps、分辨率、帧计数等
            - {split}_stats.json: 简要统计
        约定：按 translator 目录字母序，目录内视频文件名字典序排序并编号，确保与 {split}.corpus.csv 中的 video_id 对齐
        """

        def __init__(self, target_size: Tuple[int, int] = (256, 256), sample_strategy: str = "uniform"):
                self.target_size = target_size
                self.sample_strategy = sample_strategy

        def _read_video_to_array(self, video_path: str, max_frames: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
                """
                读取视频为 numpy 数组 (T, H, W, 3)，RGB，uint8，按 target_size 缩放
                """
                try:
                        vid = imageio.get_reader(video_path)
                        meta = vid.get_meta_data()
                        nframes = meta.get('nframes') or meta.get('nb_frames') or None
                        if nframes is None:
                                # 一些容器取不到 nframes，用 count_frames
                                try:
                                        nframes = vid.count_frames()
                                except Exception:
                                        nframes = None

                        fps = meta.get('fps', None)
                        duration = meta.get('duration', None)
                        size = meta.get('size', None)  # (W,H)

                        frames = []
                        idx_iter = range(nframes) if nframes is not None else None

                        if idx_iter is None:
                                # 逐帧读取直到异常
                                i = 0
                                while True:
                                        try:
                                                frame = vid.get_data(i)  # imageio 为 RGB
                                                frame = cv2.resize(frame, self.target_size)  # HxW 不依赖通道顺序
                                                frames.append(frame)
                                                i += 1
                                                if max_frames and i >= max_frames:
                                                        break
                                        except Exception:
                                                break
                        else:
                                # 可计算帧数
                                if max_frames and nframes > max_frames:
                                        # 均匀采样
                                        indices = np.linspace(0, nframes - 1, max_frames, dtype=int)
                                else:
                                        indices = np.arange(nframes)
                                for i in indices:
                                        try:
                                                frame = vid.get_data(i)  # RGB
                                                frame = cv2.resize(frame, self.target_size)
                                                frames.append(frame)
                                        except Exception as e:
                                                logger.warning(f"读取帧失败 {video_path} @ {i}: {e}")

                        vid.close()

                        frames_np = np.array(frames, dtype=np.uint8)  # (T, H, W, 3)
                        info = {
                                "fps": fps,
                                "duration": duration,
                                "original_resolution": size[::-1] if size else None,  # 转为 (H,W)
                                "target_resolution": self.target_size,
                                "total_frames": int(frames_np.shape[0])
                        }
                        return frames_np, info
                except Exception as e:
                        logger.error(f"处理视频失败 {video_path}: {e}")
                        return np.zeros((0, *self.target_size, 3), dtype=np.uint8), {
                                "fps": None, "duration": None, "original_resolution": None,
                                "target_resolution": self.target_size, "total_frames": 0
                        }

        def batch_process(self, root: str, splits: List[str], max_frames: Optional[int] = None):
                """
                root: data/CE-CSL
                输入:  root/video/{split}/{translator}/*.mp4|*.avi|*.mov|*.mkv
                输出:  root/processed/{split}/{split}_video_{idx:03d}_frames.npy
                """
                video_root = Path(root) / "video"
                processed_root = Path(root) / "processed"
                processed_root.mkdir(parents=True, exist_ok=True)

                for split in splits:
                        split_video_dir = video_root / split
                        split_out_dir = processed_root / split
                        split_out_dir.mkdir(parents=True, exist_ok=True)

                        translators = [d for d in sorted(os.listdir(split_video_dir)) if (split_video_dir / d).is_dir()]
                        logger.info(f"[{split}] 发现译员目录: {translators}")

                        meta_list = []
                        stats = {
                                "split": split,
                                "total_videos": 0,
                                "success_videos": 0,
                                "failed_videos": 0,
                                "frames": [],
                                "fps": []
                        }

                        idx = 0
                        for tr in translators:
                                tr_dir = split_video_dir / tr
                                # 允许多种视频后缀
                                files = [f for f in sorted(os.listdir(tr_dir)) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                                for fname in tqdm(files, desc=f"处理 {split}/{tr}"):
                                        src = tr_dir / fname
                                        video_id = f"{split}_video_{idx:03d}"
                                        out_npy = split_out_dir / f"{video_id}_frames.npy"

                                        frames, info = self._read_video_to_array(str(src), max_frames=max_frames)

                                        stats["total_videos"] += 1
                                        if frames.shape[0] > 0:
                                                np.save(out_npy.as_posix(), frames)
                                                stats["success_videos"] += 1
                                                stats["frames"].append(info["total_frames"])
                                                if info["fps"] is not None:
                                                        stats["fps"].append(info["fps"])
                                        else:
                                                stats["failed_videos"] += 1

                                        meta_list.append({
                                                "video_id": video_id,
                                                "translator": tr,
                                                "file_name": fname,
                                                "source_path": src.as_posix(),
                                                "npy_path": out_npy.as_posix(),
                                                **info
                                        })
                                        idx += 1

                        # 写 metadata 与 stats
                        meta_path = split_out_dir / f"{split}_metadata.json"
                        with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump(meta_list, f, indent=2, ensure_ascii=False)

                        stat_json = {
                                **stats,
                                "frames_min": int(min(stats["frames"])) if stats["frames"] else 0,
                                "frames_max": int(max(stats["frames"])) if stats["frames"] else 0,
                                "frames_mean": float(np.mean(stats["frames"])) if stats["frames"] else 0.0,
                                "fps_min": float(min(stats["fps"])) if stats["fps"] else 0.0,
                                "fps_max": float(max(stats["fps"])) if stats["fps"] else 0.0,
                                "fps_mean": float(np.mean(stats["fps"])) if stats["fps"] else 0.0,
                        }
                        stat_path = split_out_dir / f"{split}_stats.json"
                        with open(stat_path, "w", encoding="utf-8") as f:
                                json.dump(stat_json, f, indent=2, ensure_ascii=False)

                        logger.info(f"[{split}] 处理完成: 共{stats['total_videos']} 成功{stats['success_videos']} 失败{stats['failed_videos']}")
                        logger.info(f"[{split}] metadata: {meta_path}")
                        logger.info(f"[{split}] stats: {stat_path}")

# ---------------------- 标签处理（Gloss 词表） ----------------------

class CECSLLabelProcessor:
        """
        用于处理 label/{split}.csv 中 Gloss 列，构建词表（用于序列级任务）。
        """
        def __init__(self):
                self.idx2word = [PAD]
                self.word2idx = {PAD: 0}

        def preprocess_words(self, words: List[str]) -> List[str]:
                processed = []
                for word in words:
                        n = 0
                        sub_flag = False
                        word_list = list(word)
                        for j in range(len(word)):
                                if word[j] in "({[（":
                                        sub_flag = True
                                if sub_flag:
                                        if j - n < len(word_list):
                                                word_list.pop(j - n)
                                                n += 1
                                if word[j] in ")}]）":
                                        sub_flag = False
                        word = "".join(word_list)
                        if word and word[-1].isdigit() and not word[0].isdigit():
                                word = word[:-1]
                        if word and word[0] in ",，":
                                word = "，" + word[1:]
                        if word and word[0] in "?？":
                                word = "？" + word[1:]
                        if word.isdigit():
                                word = str(int(word))
                        if word:
                                processed.append(word)
                return processed

        def build_vocabulary_from_label_dir(self, label_dir: str, min_freq: int = 1) -> Dict[str, int]:
                """
                读取 label/train.csv, label/dev.csv, label/test.csv，统计 Gloss 词频构建词表
                """
                paths = []
                for name in ["train.csv", "dev.csv", "test.csv"]:
                        p = Path(label_dir) / name
                        if p.exists():
                                paths.append(p.as_posix())
                word_counts = defaultdict(int)
                for label_file in paths:
                        with open(label_file, "r", encoding="utf-8") as f:
                                reader = csv.reader(f)
                                for i, row in enumerate(reader):
                                        if i == 0:
                                                continue
                                        if len(row) >= 4:
                                                gloss = row[3]
                                                words = self.preprocess_words(gloss.split("/"))
                                                for w in words:
                                                        word_counts[w] += 1

                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                for w, c in sorted_words:
                        if c >= min_freq and w not in self.word2idx:
                                self.word2idx[w] = len(self.idx2word)
                                self.idx2word.append(w)
                logger.info(f"Gloss词表构建完成: {len(self.idx2word)}")
                return self.word2idx

        def save_vocabulary(self, vocab_path: str):
                data = {"word2idx": self.word2idx, "idx2word": self.idx2word, "vocab_size": len(self.idx2word)}
                with open(vocab_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"词表已保存: {vocab_path}")

        def load_vocabulary(self, vocab_path: str):
                with open(vocab_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                self.word2idx = data["word2idx"]
                self.idx2word = data["idx2word"]
                logger.info(f"词表已加载，大小: {len(self.idx2word)}")

# ---------------------- 分段标注（corpus）词表 ----------------------

def build_corpus_label_vocab(corpus_files: List[str], save_path: Optional[str] = None, use_cleaned: bool = True) -> Dict[str, int]:
        """
        从 *.corpus.csv 提取 label 列，构建简单标签词表（用于片段分类/识别）
        """
        # 如果存在清理后的词汇表，优先使用
        if use_cleaned and save_path:
                cleaned_vocab_path = Path(save_path).parent / "cleaned_vocab.json"
                if cleaned_vocab_path.exists():
                        try:
                                with open(cleaned_vocab_path, 'r', encoding='utf-8') as f:
                                        vocab_data = json.load(f)
                                label2idx = vocab_data.get('word2idx', {})
                                if label2idx:
                                        logger.info(f"使用清理后的词汇表: {cleaned_vocab_path}，共{len(label2idx)}类")
                                        return label2idx
                        except Exception as e:
                                logger.warning(f"加载清理后词汇表失败: {e}")
        
        # 原始方法
        labels = []
        for p in corpus_files:
                if not Path(p).exists():
                        continue
                with open(p, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                                labels.append(row["label"])
        uniq = sorted(list(set(labels)))
        label2idx = {lab: i for i, lab in enumerate(uniq)}
        if save_path:
                with open(save_path, "w", encoding="utf-8") as f:
                        json.dump({"label2idx": label2idx, "idx2label": uniq}, f, indent=2, ensure_ascii=False)
                logger.info(f"corpus标签词表已保存: {save_path}，共{len(uniq)}类")
        return label2idx

# ---------------------- 简单视频变换 ----------------------

class VideoTransform:
        """
        在 Python 中逐帧变换，避免 pipeline 复杂性
        输出 shape: (T, C, H, W)，float32
        """
        def __init__(self, size=(224, 224), flip_prob: float = 0.0, normalize=True, to_chw=True):
                self.size = size
                self.flip_prob = flip_prob
                self.normalize = normalize
                self.to_chw = to_chw
                self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def __call__(self, frames: np.ndarray) -> np.ndarray:
                # frames: (T, H, W, 3), uint8
                out = []
                do_flip = (random.random() < self.flip_prob)
                for img in frames:
                        img = cv2.resize(img, self.size)
                        if do_flip:
                                img = cv2.flip(img, 1)
                        img = img.astype(np.float32) / 255.0
                        if self.normalize:
                                img = (img - self.mean) / self.std
                        if self.to_chw:
                                img = np.transpose(img, (2, 0, 1))  # C,H,W
                        out.append(img)
                return np.stack(out, axis=0)  # T,C,H,W

# ---------------------- 工具：序列采样/填充 ----------------------

def sample_or_pad_sequence(frames: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
        """
        将任意长度序列变为固定长度 target_len
        - 过长: 均匀采样
        - 过短: 末尾补零帧
        返回: (T,H,W,3), 实际原始长度 orig_len
        """
        T = frames.shape[0]
        orig_len = T
        if T == 0:
                return np.zeros((target_len, *frames.shape[1:]), dtype=np.uint8), 0
        if T == target_len:
                return frames, orig_len
        if T > target_len:
                idx = np.linspace(0, T - 1, target_len, dtype=int)
                return frames[idx], orig_len
        # pad
        pad = np.zeros((target_len - T, *frames.shape[1:]), dtype=frames.dtype)
        return np.concatenate([frames, pad], axis=0), orig_len

# ---------------------- 基于 corpus 的数据集 ----------------------

class CECSLSegmentDataset:
        """
        使用 {split}.corpus.csv + processed/{split}/{video_id}_frames.npy
        每条样本为一个片段 [start_frame, end_frame)，对齐/填充为固定长度 clip_len
        """
        def __init__(self, root: str, split: str, clip_len: int, label2idx: Dict[str, int],
                                 transform: Optional[VideoTransform] = None):
                self.root = Path(root)
                self.split = split
                self.clip_len = clip_len
                self.label2idx = label2idx
                self.transform = transform

                self.processed_dir = self.root / "processed" / split
                self.corpus_csv = self.root / f"{split}.corpus.csv"

                if not self.corpus_csv.exists():
                        raise FileNotFoundError(f"缺少 {self.corpus_csv}")

                self.items = []
                with open(self.corpus_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)  # video_id,start_frame,end_frame,label
                        for row in reader:
                                vid = row["video_id"]
                                s = int(row["start_frame"])
                                e = int(row["end_frame"])
                                lab = row["label"]
                                npy = self.processed_dir / f"{vid}_frames.npy"
                                if not npy.exists():
                                        logger.warning(f"样本跳过，未找到: {npy}")
                                        continue
                                if lab not in self.label2idx:
                                        logger.warning(f"样本跳过，未知标签: {lab}")
                                        continue
                                self.items.append({
                                        "video_id": vid,
                                        "start": s,
                                        "end": e,
                                        "label_text": lab,
                                        "label": self.label2idx[lab],
                                        "npy_path": npy.as_posix()
                                })

                logger.info(f"[{split}] 样本数: {len(self.items)} | 来自: {self.corpus_csv}")

        def __len__(self):
                return len(self.items)

        def __getitem__(self, idx):
                it = self.items[idx]
                arr = np.load(it["npy_path"])  # (T,H,W,3), uint8
                # 约定 end_frame 为开区间，frames[s:e]
                s, e = it["start"], it["end"]
                s = max(0, s)
                e = min(arr.shape[0], e)
                clip = arr[s:e] if e > s else arr[0:0]
                # 固定长度
                clip, orig_len = sample_or_pad_sequence(clip, self.clip_len)
                # 变换到 T,C,H,W
                if self.transform:
                        clip = self.transform(clip)
                return {
                        "video": clip.astype(np.float32),           # (T,C,H,W)
                        "label": np.int32(it["label"]),             # 标量
                        "length": np.int32(orig_len),               # 原始长度（未pad前）
                        "video_id": it["video_id"]
                }

# ---------------------- DataLoader 创建（基于 corpus） ----------------------

def create_cecsl_segment_dataloaders(data_config: Dict):
        """
        data_config 参考:
        {
            "root": "data/CE-CSL",
            "splits": ["train", "dev", "test"],
            "batch_size": 4,
            "clip_len": 32,
            "size": [224, 224],
            "train_flip": 0.5
        }
        返回: (train_loader, val_loader, test_loader, num_classes)
        """
        root = data_config["root"]
        splits = data_config.get("splits", ["train", "dev", "test"])
        batch_size = int(data_config.get("batch_size", 4))
        clip_len = int(data_config.get("clip_len", 32))
        size = tuple(data_config.get("size", (224, 224)))
        train_flip = float(data_config.get("train_flip", 0.5))

        # 构建/加载 corpus 标签词表
        corpus_files = [str(Path(root) / f"{sp}.corpus.csv") for sp in splits if (Path(root) / f"{sp}.corpus.csv").exists()]
        vocab_path = Path(root) / "corpus_vocab.json"
        if vocab_path.exists():
                with open(vocab_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                label2idx = data["label2idx"]
                logger.info(f"已加载 corpus 标签词表: {vocab_path} (classes={len(label2idx)})")
        else:
                label2idx = build_corpus_label_vocab(corpus_files, save_path=vocab_path.as_posix())

        # 变换
        train_tf = VideoTransform(size=size, flip_prob=train_flip, normalize=True, to_chw=True)
        test_tf = VideoTransform(size=size, flip_prob=0.0, normalize=True, to_chw=True)

        # 数据集
        train_ds = CECSLSegmentDataset(root, "train", clip_len, label2idx, transform=train_tf) if "train" in splits else None
        val_ds = CECSLSegmentDataset(root, "dev", clip_len, label2idx, transform=test_tf) if "dev" in splits else None
        test_ds = CECSLSegmentDataset(root, "test", clip_len, label2idx, transform=test_tf) if "test" in splits else None

        # 由于已固定长度 (clip_len)，可直接 batch
        def make_loader(dataset, shuffle, bs):
                if dataset is None:
                        return None
                return ds.GeneratorDataset(
                        source=dataset,
                        column_names=["video", "label", "length", "video_id"],
                        shuffle=shuffle
                ).batch(bs, drop_remainder=False)

        train_loader = make_loader(train_ds, shuffle=True, bs=batch_size)
        val_loader = make_loader(val_ds, shuffle=False, bs=1)
        test_loader = make_loader(test_ds, shuffle=False, bs=1)

        num_classes = len(label2idx)
        logger.info(f"DataLoaders: train={len(train_ds) if train_ds else 0}, "
                                f"dev={len(val_ds) if val_ds else 0}, test={len(test_ds) if test_ds else 0}, "
                                f"classes={num_classes}")

        return train_loader, val_loader, test_loader, num_classes

# ---------------------- 可选：原 label.csv 的 Gloss 处理接口 ----------------------

def build_and_save_gloss_vocab_from_label_dir(root: str, min_freq: int = 1, save_name: str = "gloss_vocab.json"):
        """
        从 label/{split}.csv 构建 Gloss 词表并保存
        """
        proc = CECSLLabelProcessor()
        proc.build_vocabulary_from_label_dir(os.path.join(root, "label"), min_freq=min_freq)
        out = Path(root) / save_name
        proc.save_vocabulary(out.as_posix())
        return out.as_posix()

# ---------------------- 主入口（示例） ----------------------

if __name__ == "__main__":
        seed_everything(42)

        # 路径根：data/CE-CSL
        root = "data/CE-CSL"

        # 1) 批量预处理视频 => processed/{split}/*.npy
        video_processor = CECSLVideoProcessor(target_size=(256, 256))
        # 注意：为保持与 {split}.corpus.csv 的 video_id 对齐，遍历顺序为：
        # 译员目录字母序 + 目录内视频文件字典序
        # 如果你的 {split}.corpus.csv 已基于此规则，可直接运行
        # 如果担心截断影响分段范围，请设置 max_frames=None
        # video_processor.batch_process(root=root, splits=["train", "dev", "test"], max_frames=None)

        # 2) （可选）从 label/{split}.csv 构建 Gloss 词表
        # build_and_save_gloss_vocab_from_label_dir(root, min_freq=1, save_name="gloss_vocab.json")

        # 3) 基于 corpus 的数据加载器（片段级任务）
        data_config = {
                "root": root,
                "splits": ["train", "dev", "test"],
                "batch_size": 4,
                "clip_len": 32,
                "size": (224, 224),
                "train_flip": 0.5
        }
        train_loader, val_loader, test_loader, num_classes = create_cecsl_segment_dataloaders(data_config)
        logger.info("CE-CSL 数据处理模块就绪")
