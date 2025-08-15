"""
离线/批处理 手语视频识别服务
实现流程:
1. 读取视频 -> 抽帧(可下采样) -> MediaPipe 提取 543 关键点 (x,y,z)
2. 关键点序列 -> 分窗口 -> 调用 CSLRService.predict 获取 gloss 片段
3. 合并 gloss 序列 -> 简单规则翻译成自然中文
4. 保存结果 JSON + 任务状态管理
"""
from __future__ import annotations
import cv2
import os
import json
import uuid
import time
import math
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RecognitionSegment:
    gloss_sequence: List[str]
    start_frame: int
    end_frame: int
    confidence: float
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class RecognitionResult:
    task_id: str
    file_path: str
    gloss_sequence: List[str]
    text: str
    segments: List[RecognitionSegment]
    overall_confidence: float
    frame_count: int
    fps: float
    duration: float
    srt_path: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "file_path": self.file_path,
            "gloss_sequence": self.gloss_sequence,
            "text": self.text,
            "segments": [
                {
                    "gloss_sequence": seg.gloss_sequence,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "confidence": seg.confidence,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                } for seg in self.segments
            ],
            "overall_confidence": self.overall_confidence,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration": self.duration,
            "srt_path": self.srt_path,
            "created_at": self.created_at,
        }


class SignRecognitionService:
    def __init__(self, mediapipe_service, cslr_service, result_dir: str = "temp/sign_results"):
        self.mediapipe_service = mediapipe_service
        self.cslr_service = cslr_service
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.target_fps = 25
        self.window_length = getattr(self.cslr_service.config, "max_sequence_length", 64)
        self.window_overlap = 0.4  # 40% 重叠
        self.gloss_dict_path = os.path.join(self.result_dir, "gloss_dictionary.json")
        self.gloss_dict = self._load_or_create_gloss_dict()
        logger.info("SignRecognitionService 初始化完成")

    async def start_video_recognition(self, file_path: str) -> str:
        task_id = str(uuid.uuid4())
        async with self._lock:
            self.tasks[task_id] = {"status": "queued", "progress": 0.0, "file_path": file_path}
        asyncio.create_task(self._process_task(task_id, file_path))
        return task_id

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self.tasks.get(task_id)

    async def _update_task(self, task_id: str, **kwargs):
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)

    async def _process_task(self, task_id: str, file_path: str):
        try:
            await self._update_task(task_id, status="processing", progress=0.01)
            result = await self._run_pipeline(task_id, file_path)
            # 保存结果 JSON
            result_path = os.path.join(self.result_dir, f"{task_id}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            await self._update_task(task_id, status="finished", progress=1.0, result=result.to_dict(), result_path=result_path)
        except Exception as e:
            logger.error(f"任务 {task_id} 处理失败: {e}")
            await self._update_task(task_id, status="error", progress=1.0, error=str(e))

    async def _run_pipeline(self, task_id: str, file_path: str) -> RecognitionResult:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = int(max(1, round(fps / self.target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        landmark_vectors: List[List[float]] = []
        frame_indices: List[int] = []

        frame_id = 0
        processed_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_interval == 0:
                result = self.mediapipe_service.extract_landmarks(frame)
                if result.get("success"):
                    landmarks_array = self.mediapipe_service.landmarks_to_array(result["landmarks"])  # (543,3)
                    # 展平
                    flat_vec = landmarks_array.reshape(-1).tolist()
                    landmark_vectors.append(flat_vec)
                    frame_indices.append(frame_id)
                processed_frames += 1
                if processed_frames % 50 == 0:
                    await self._update_task(task_id, progress=min(0.8, 0.1 + 0.6 * processed_frames / total_frames))
            frame_id += 1

        cap.release()
        if not landmark_vectors:
            raise RuntimeError("未提取到任何关键点")

        # 分窗口推理
        win = self.window_length
        step = int(win * (1 - self.window_overlap)) or 1
        segments: List[RecognitionSegment] = []
        gloss_full: List[str] = []
        confidences: List[float] = []
        T = len(landmark_vectors)
        idx = 0
        window_id = 0
        while idx < T:
            window_seq = landmark_vectors[idx: idx + win]
            # 重置 cslr sequence buffer
            self.cslr_service.sequence_buffer.clear()
            for vec in window_seq:
                self.cslr_service.sequence_buffer.append(vec)
            pred = await self.cslr_service.predict(window_seq)
            if pred.status == "success" and pred.gloss_sequence:
                segments.append(RecognitionSegment(
                    gloss_sequence=pred.gloss_sequence,
                    start_frame=frame_indices[idx] if idx < len(frame_indices) else 0,
                    end_frame=frame_indices[min(idx + len(window_seq)-1, len(frame_indices)-1)] if frame_indices else 0,
                    confidence=pred.confidence,
                ))
                gloss_full.extend(pred.gloss_sequence)
                confidences.append(pred.confidence)
            idx += step
            window_id += 1
            await self._update_task(task_id, progress=min(0.9, 0.8 + 0.1 * idx / T))
        # 计算时间戳
        for seg in segments:
            seg.start_time = seg.start_frame / fps if fps > 0 else 0.0
            seg.end_time = seg.end_frame / fps if fps > 0 else seg.start_time

        # 合并重复 (简单相邻去重)
        merged_gloss: List[str] = []
        for g in gloss_full:
            if not merged_gloss or merged_gloss[-1] != g:
                merged_gloss.append(g)

        text = self._translate_gloss_to_text(merged_gloss)
        overall_conf = float(np.mean(confidences)) if confidences else 0.0
        # 生成 SRT
        srt_path = self._generate_srt(task_id, segments, text)
        return RecognitionResult(
            task_id=task_id,
            file_path=file_path,
            gloss_sequence=merged_gloss,
            text=text,
            segments=segments,
            overall_confidence=overall_conf,
            frame_count=frame_id,
            fps=fps,
            duration=frame_id / fps if fps > 0 else 0.0,
            srt_path=srt_path,
        )

    def _load_or_create_gloss_dict(self) -> Dict[str, str]:
        if os.path.exists(self.gloss_dict_path):
            try:
                with open(self.gloss_dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        sample = {
            "我": "我", "你": "你", "他": "他", "她": "她",
            "学习": "学习", "工作": "工作", "医院": "医院", "学校": "学校",
            "谢谢": "谢谢", "你好": "你好", "再见": "再见", "今天": "今天",
            "昨天": "昨天", "明天": "明天", "想": "想", "去": "去", "吃": "吃",
            "喝": "喝", "家": "家", "是": "是", "不是": "不是"
        }
        try:
            with open(self.gloss_dict_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return sample

    def _translate_gloss_to_text(self, gloss_seq: List[str]) -> str:
        if not gloss_seq:
            return ""
        mapped = [self.gloss_dict.get(g, g) for g in gloss_seq]
        # 去除连续重复
        cleaned = []
        for g in mapped:
            if not cleaned or cleaned[-1] != g:
                cleaned.append(g)
        # 分句规则增强
        pronouns = {"我", "你", "他", "她"}
        verbs = {"学习", "工作", "吃", "喝", "去", "想", "睡觉"}
        time_words = {"今天", "昨天", "明天"}
        question_words = {"吗", "请问", "什么", "怎么", "为什么", "谁", "哪儿", "哪里"}
        exclam_words = {"啊", "呀", "哇", "太棒了", "太好了"}
        logic_words = {"因为", "所以", "但是", "如果", "然后"}
        polite_words = {"谢谢", "再见"}
        result = []
        sentence = []
        for i, word in enumerate(cleaned):
            sentence.append(word)
            # 分句条件：遇到时间词、逻辑词、礼貌词、问句/感叹词、主语后动宾结构
            if word in time_words or word in logic_words or word in polite_words:
                result.append(''.join(sentence))
                sentence = []
            elif word in question_words or (i+1 < len(cleaned) and cleaned[i+1] in question_words):
                result.append(''.join(sentence))
                sentence = []
            elif word in exclam_words or (i+1 < len(cleaned) and cleaned[i+1] in exclam_words):
                result.append(''.join(sentence))
                sentence = []
            elif word in pronouns and i+1 < len(cleaned) and cleaned[i+1] in verbs:
                # 主语+动词后如有宾语或时间词，分句
                if i+2 < len(cleaned) and (cleaned[i+2] not in pronouns and cleaned[i+2] not in verbs):
                    sentence.append(cleaned[i+2])
                    result.append(''.join(sentence))
                    sentence = []
        if sentence:
            result.append(''.join(sentence))
        # 标点插入
        sentences = []
        for s in result:
            if any(q in s for q in question_words):
                sentences.append(s + '？')
            elif any(e in s for e in exclam_words):
                sentences.append(s + '！')
            elif any(p in s for p in polite_words):
                sentences.append(s + '。')
            else:
                sentences.append(s + '。')
        text = ''.join(sentences)
        # 进一步去除多余标点
        text = text.replace('。。', '。').replace('！！', '！').replace('？？', '？')
        return text

    def _generate_srt(self, task_id: str, segments: List[RecognitionSegment], full_text: str) -> Optional[str]:
        if not segments:
            return None
        def format_ts(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start_ts = format_ts(seg.start_time)
            end_ts = format_ts(max(seg.end_time, seg.start_time + 0.04))
            line_text = "".join(seg.gloss_sequence)
            srt_lines.append(str(i))
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.append(line_text)
            srt_lines.append("")
        # 添加一条合并总句
        srt_lines.append(str(len(segments) + 1))
        srt_lines.append("00:00:00,000 --> 00:00:59,999")  # 粗略覆盖前一分钟
        srt_lines.append(full_text)
        srt_lines.append("")
        path = os.path.join(self.result_dir, f"{task_id}.srt")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("\n".join(srt_lines))
            return path
        except Exception as e:
            logger.error(f"写入SRT失败: {e}")
            return None

    async def get_status(self, task_id: str) -> Dict[str, Any]:
        task = await self.get_task(task_id)
        if not task:
            return {"status": "not_found"}
        data = {k: v for k, v in task.items() if k != "result"}
        return data

    async def get_result(self, task_id: str) -> Dict[str, Any]:
        task = await self.get_task(task_id)
        if not task:
            return {"status": "not_found"}
        if task.get("status") != "finished":
            return {"status": task.get("status"), "progress": task.get("progress")}
        return {"status": "finished", "result": task.get("result")}

    async def cleanup(self):
        # 未来可清理过期任务文件
        pass
