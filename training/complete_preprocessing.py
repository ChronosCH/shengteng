#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的CE-CSL数据预处理脚本（优化版 + 增强补丁）
改进汇总：
- argparse 参数化 data_root / max_frames / size / overwrite
- 标签读取 pick 增加 strip()
- video_to_frames 增加 frame_count==0 兜底采样逻辑（利用 fps 或假设 25fps）
- 断点续跑：已存在且未 --overwrite 时跳过重新解码
- 支持更多视频扩展名大小写 (mp4/avi/mov/mkv)
- 相对路径写入 frames_path（相对于 data_root）方便迁移
- 过滤异常短视频 (<4 帧) 视为坏样本
- 训练/验证集空标签告警
- 覆盖统计信息 & 一致性检查兼容相对路径
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import Dict, Tuple, Optional


def resize_letterbox(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """等比缩放并用黑边填充到目标大小，避免拉伸变形。"""
    th, tw = target_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("非法图像尺寸，h或w为0")
    scale = min(tw / w, th / h)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def video_to_frames(
    video_path: str,
    target_size: Tuple[int, int] = (224, 224),
    max_frames: Optional[int] = 96,
    stride: Optional[int] = None,
) -> np.ndarray:
    """将视频转换为帧数组（RGB，uint8，形状 (T,H,W,3)）。
    - 优先均匀采样到 max_frames
    - 若 CAP_PROP_FRAME_COUNT=0 采用 fps 估计 + stride 兜底
    - BGR->RGB + letterbox
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    take_indices_set = None
    dynamic_stride = stride
    if total > 0:
        if max_frames and total > max_frames:
            take_indices = np.linspace(0, total - 1, max_frames).astype(int).tolist()
            take_indices_set = set(take_indices)
    else:
        # 无法获取总帧数，使用时间步采样兜底（假设 ~2 秒间隔）
        if max_frames and dynamic_stride is None:
            est_fps = fps if fps and fps < 120 else 25
            # 目标：约 uniform 覆盖，取每 ~ (duration/max_frames) 秒；无总帧数难估时使用时间间隔 2 秒或估算 stride
            dynamic_stride = max(1, int(round(est_fps * 2 / max_frames)))  # 简单兜底

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if take_indices_set is not None:
            if idx in take_indices_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(resize_letterbox(rgb, target_size))
        else:
            use_it = False
            if dynamic_stride is not None:
                if idx % dynamic_stride == 0:
                    use_it = True
            else:
                if stride is None or (idx % stride == 0):
                    use_it = True
            if use_it:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(resize_letterbox(rgb, target_size))
        idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"{video_path} 无有效帧")

    return np.stack(frames, axis=0)


def create_label_mapping_from_labelcsv(data_root: Path, use_first_gloss_token: bool = False) -> Dict[str, Dict[str, str]]:
    """
    从 label/*.csv 创建映射: original_name -> {label_text, gloss, chinese, signer, split}
    兼容以下表头（大小写/空格不敏感）：

      - 原始名: id, video_id, name, 编号, 样本名, number
      - 译员: signer, speaker, translator, 说话人, 手语员
      - 中文: chinese, sentence, text, 中文, 中文句子, chinese sentences
      - 手语: gloss, label, 注释, 词序列
    """
    import re
    def canon(s: str) -> str:
        # 小写 + 去掉空白与下划线（中文不变）
        return re.sub(r"[\s_]+", "", s.strip().lower()) if s is not None else ""

    def pick(row_norm: dict, keys_norm):
        for k in keys_norm:
            if k in row_norm and row_norm[k]:
                return row_norm[k].strip()
        return ""

    label_mapping: Dict[str, Dict[str, str]] = {}
    label_dir = data_root / "label"
    if not label_dir.exists():
        print("警告: label 目录不存在，将不生成标签映射")
        return {}

    # 同义词集合（已 canonicalize）
    id_keys = {"id", "videoid", "name", "编号", "样本名", "number"}
    signer_keys = {"signer", "speaker", "translator", "说话人", "手语员"}
    chinese_keys = {"chinese", "sentence", "text", "中文", "中文句子", "chinesesentences"}
    gloss_keys = {"gloss", "label", "注释", "词序列"}

    for split in ["train", "dev", "test"]:
        p = label_dir / f"{split}.csv"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 生成 canonicalized 的键->值映射
                row_norm = {canon(k): (v or "").strip() for k, v in row.items()}

                original_name = pick(row_norm, id_keys)
                original_name = original_name.replace(".mp4", "").replace(".avi", "")
                if not original_name:
                    continue

                signer = pick(row_norm, signer_keys)
                chinese = pick(row_norm, chinese_keys)
                gloss = pick(row_norm, gloss_keys)

                if use_first_gloss_token and gloss:
                    # 取第1个 gloss token（以 / 或空格分割）
                    token = gloss.split("/")[0].strip()
                    token = token.split()[0] if token else token
                    gloss = token

                label_text = gloss if gloss else chinese  # 优先用 gloss；没有就用中文

                label_mapping[original_name] = {
                    "label_text": label_text,
                    "gloss": gloss,
                    "chinese": chinese,
                    "signer": signer,
                    "split": split,
                }

    print(f"从 label/*.csv 加载了 {len(label_mapping)} 条标签")
    return label_mapping


def complete_preprocessing(
    data_root: str = "../data/CE-CSL",
    max_frames: int = 96,
    target_size: Tuple[int, int] = (224, 224),
    overwrite: bool = False,
):
    """完整预处理所有视频，支持断点续跑。"""
    data_path = Path(data_root).resolve()
    video_root = data_path / "video"
    processed_root = data_path / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    label_mapping = create_label_mapping_from_labelcsv(data_path, use_first_gloss_token=False)

    for split in ["train", "dev", "test"]:
        print(f"\n🔄 处理 {split} 数据集...")
        split_video_dir = video_root / split
        split_output_dir = processed_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        if not split_video_dir.exists():
            print(f"跳过 {split}: 视频目录不存在")
            continue

        # 收集视频
        all_videos = []
        translators = sorted([d for d in split_video_dir.iterdir() if d.is_dir()])
        exts = ["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV"]
        for translator_dir in translators:
            video_files = []
            for pat in exts:
                video_files.extend(translator_dir.glob(pat))
            video_files = sorted(video_files)
            for vf in video_files:
                all_videos.append({
                    'path': vf,
                    'translator': translator_dir.name,
                    'filename': vf.name,
                    'original_name': vf.stem
                })
        print(f"找到 {len(all_videos)} 个视频文件")
        if not all_videos:
            print(f"警告: {split} 没有找到视频文件")
            continue

        metadata_list = []
        corpus_records = []
        bad_count = 0

        for idx, video_info in enumerate(tqdm(all_videos, desc=f"处理{split}视频")):
            video_id = f"{split}_video_{idx:06d}"
            frames_file = split_output_dir / f"{video_id}_frames.npz"
            original_name = video_info['original_name']
            lm = label_mapping.get(original_name, {})
            label_text = lm.get('label_text', '')
            signer = lm.get('signer', video_info['translator'])

            # 断点续跑：若存在且不过期
            if frames_file.exists() and not overwrite:
                try:
                    with np.load(frames_file) as old:
                        n_frames = int(old['frames'].shape[0])
                        frame_shape = old['frames'].shape
                except Exception:
                    pass  # 失败则重新解码
                else:
                    frames_rel = frames_file.relative_to(data_path)
                    metadata = {
                        'video_id': video_id,
                        'translator': video_info['translator'],
                        'original_filename': video_info['filename'],
                        'original_name': original_name,
                        'frames_shape': tuple(int(x) for x in frame_shape),
                        'label': label_text,
                        'frames_path': str(frames_rel),
                        'n_frames': n_frames,
                        'split': split,
                        'signer': signer,
                        'cached': True,
                    }
                    metadata_list.append(metadata)
                    corpus_records.append({
                        'video_id': video_id,
                        'original_name': original_name,
                        'signer': signer,
                        'frames_path': str(frames_rel),
                        'n_frames': n_frames,
                        'label': label_text,
                        'split': split,
                    })
                    continue

            try:
                frames = video_to_frames(str(video_info['path']), target_size=target_size, max_frames=max_frames)
            except Exception as e:
                bad_count += 1
                print(f"[跳过] {video_info['path']} 失败: {e}")
                continue
            if frames.shape[0] < 4:
                bad_count += 1
                print(f"[跳过] {video_info['path']} 帧数过少: {frames.shape[0]}")
                continue

            np.savez_compressed(frames_file, frames=frames)
            n_frames = int(frames.shape[0])
            frames_rel = frames_file.relative_to(data_path)

            metadata = {
                'video_id': video_id,
                'translator': video_info['translator'],
                'original_filename': video_info['filename'],
                'original_name': original_name,
                'frames_shape': tuple(int(x) for x in frames.shape),
                'label': label_text,
                'frames_path': str(frames_rel),
                'n_frames': n_frames,
                'split': split,
                'signer': signer,
                'cached': False,
            }
            metadata_list.append(metadata)
            corpus_records.append({
                'video_id': video_id,
                'original_name': original_name,
                'signer': signer,
                'frames_path': str(frames_rel),
                'n_frames': n_frames,
                'label': label_text,
                'split': split,
            })

        metadata_file = split_output_dir / f"{split}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

        frame_counts = [m['n_frames'] for m in metadata_list]
        stats = {
            'split': split,
            'total_videos_found': len(all_videos),
            'processed_videos': len(metadata_list),
            'bad_videos_skipped': bad_count,
            'translators_dirs': len(translators),
            'avg_frames': float(np.mean(frame_counts)) if frame_counts else 0.0,
            'min_frames': int(np.min(frame_counts)) if frame_counts else 0,
            'max_frames': int(np.max(frame_counts)) if frame_counts else 0,
            'target_size': list(target_size),
            'stored_format': 'npz_compressed',
            'sampling': f'uniform_to_max_frames={max_frames}',
            'overwrite': overwrite,
        }
        stats_file = split_output_dir / f"{split}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        corpus_file = data_path / f"{split}.corpus.csv"
        with open(corpus_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['video_id', 'original_name', 'signer', 'frames_path', 'n_frames', 'label', 'split']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(corpus_records)

        empty_labels = sum(1 for r in corpus_records if r['split'] in ('train', 'dev') and not r['label'])
        if empty_labels:
            print(f"⚠️  {split}: 训练/验证集中出现空标签 {empty_labels} 条，请检查 label/*.csv")

        print(f"✅ {split} 处理完成:")
        print(f"   - 处理视频: {len(metadata_list)} 个 (跳过坏样本 {bad_count})")
        print(f"   - 元数据: {metadata_file}")
        print(f"   - 统计: {stats_file}")
        print(f"   - Corpus: {corpus_file}")


def sanity_check_corpus(corpus_csv: Path, data_root: Path):
    """轻量一致性检查：统计行数与缺失帧文件数量（支持相对路径）。"""
    if not corpus_csv.exists():
        print(f"[Check] 缺失: {corpus_csv}")
        return
    rows = 0
    missing = 0
    examples_missing = []
    with open(corpus_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            rel = row.get('frames_path', '')
            if not rel:
                missing += 1
                if len(examples_missing) < 5:
                    examples_missing.append(rel)
                continue
            p_abs = (data_root / rel).resolve()
            if not p_abs.exists():
                missing += 1
                if len(examples_missing) < 5:
                    examples_missing.append(rel)
    print(f"[Check] {corpus_csv.name}: rows={rows}, missing_files={missing}")
    if examples_missing:
        print("  示例缺失：", examples_missing)


def sanity_check_all(data_root: Path):
    for split in ["train", "dev", "test"]:
        c = data_root / f"{split}.corpus.csv"
        sanity_check_corpus(c, data_root)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../data/CE-CSL", help="数据根目录 (含 video/ label/ )")
    ap.add_argument("--max_frames", type=int, default=96, help="统一采样帧数 (>0 均匀采样)" )
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224], help="目标尺寸 H W")
    ap.add_argument("--overwrite", action="store_true", help="重新解码覆盖已存在的 .npz")
    args = ap.parse_args()

    print("🚀 开始完整的CE-CSL数据预处理...")
    complete_preprocessing(
        data_root=args.data_root,
        max_frames=args.max_frames,
        target_size=tuple(args.size),
        overwrite=args.overwrite,
    )
    print("\n🎉 预处理完成!")
    print("\n运行一致性检查...")
    sanity_check_all(Path(args.data_root).resolve())


if __name__ == "__main__":
    main()