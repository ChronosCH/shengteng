#!/usr/bin/env python3
"""
将数据集中所有视频预提取为图像帧，存放到 Config.FRAMES_DIR
目录结构：
  FRAMES_DIR/
    <视频基名>/
      000.jpg, 001.jpg, ...

可选参数：
  --ext jpg|png|webp  输出格式（默认 jpg）
  --quality 0-100     压缩质量（jpg/webp 有效，默认 85）
  --size 160x160      统一尺寸（默认与 Config.INPUT_SIZE 一致）
  --workers N         并行进程数（默认 4）

使用：
  conda run -n mind python preextract_frames.py --ext jpg --quality 85 --workers 8
"""
from __future__ import annotations
import os
import argparse
import concurrent.futures as fut
from typing import Tuple

from config import Config

import cv2
import numpy as np

try:
    import decord
    from decord import VideoReader
    from decord import cpu as decord_cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    import av
    HAS_PYAV = True
except Exception:
    HAS_PYAV = False


def decode_all(video_path: str) -> np.ndarray:
    if HAS_DECORD:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        return vr.get_batch(list(range(len(vr)))).asnumpy()
    if HAS_PYAV:
        container = av.open(video_path)
        frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
        container.close()
        return np.stack(frames, axis=0) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)
    # OpenCV 回退
    cap = cv2.VideoCapture(video_path)
    out = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append(frame)
    cap.release()
    return np.stack(out, axis=0) if out else np.zeros((0, 0, 0, 3), dtype=np.uint8)


def write_frames(frames: np.ndarray, out_dir: str, size: Tuple[int, int], ext: str, quality: int):
    os.makedirs(out_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        img = cv2.resize(fr, size)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fp = os.path.join(out_dir, f"{i:03d}.{ext}")
        if ext == 'webp':
            cv2.imwrite(fp, bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
        elif ext in ('jpg', 'jpeg'):
            cv2.imwrite(fp, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(fp, bgr)


def process_one(video_path: str, frames_root: str, size: Tuple[int, int], ext: str, quality: int):
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(frames_root, base)
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        return f"skip {base}"
    frames = decode_all(video_path)
    if frames.size == 0:
        return f"empty {base}"
    write_frames(frames, out_dir, size, ext, quality)
    return f"ok {base} ({frames.shape[0]} frames)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ext', default='jpg', choices=['jpg', 'jpeg', 'png', 'webp'])
    parser.add_argument('--quality', type=int, default=85)
    parser.add_argument('--size', type=str, default=f"{Config.INPUT_SIZE[1]}x{Config.INPUT_SIZE[0]}")
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    w, h = args.size.split('x')
    size = (int(w), int(h))

    videos_dir = Config.VIDEO_DIR
    frames_root = Config.FRAMES_DIR
    os.makedirs(frames_root, exist_ok=True)

    videos = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if f.lower().endswith('.mp4')]
    videos.sort()

    print(f"Total videos: {len(videos)}; output: {frames_root}")

    with fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(lambda vp: process_one(vp, frames_root, size, args.ext, args.quality), videos):
            print(msg)


if __name__ == '__main__':
    main()
