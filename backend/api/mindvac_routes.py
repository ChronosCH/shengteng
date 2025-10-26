"""Mind-VAC 实时识别相关 API"""

from __future__ import annotations

import base64
import logging
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mind-vac", tags=["Mind-VAC"])


class MindVacRecognitionRequest(BaseModel):
    """前端发送的 Mind-VAC 实时识别请求"""

    frames: List[str] = Field(..., min_length=8, max_length=512, description="按时间顺序排列的 base64 图片数据")
    fps: float = Field(default=25.0, gt=0.0, le=120.0, description="采集帧率")
    use_llm: bool = Field(default=True, alias="use_llm", description="是否启用 LLM 增强")

    class Config:
        populate_by_name = True


class MindVacRecognitionResponse(BaseModel):
    """Mind-VAC 实时识别响应"""

    text: str
    gloss_sequence: List[str]
    raw_gloss_text: str
    confidence: float
    frame_count: int
    duration: float
    baseline_text: Optional[str] = None
    llm_result: Optional[dict] = None


def _decode_base64_frame(data: str, index: int) -> np.ndarray:
    """解析单帧 base64 图像数据并转换为 RGB 图像"""
    try:
        # 兼容 data URL 和纯 base64 两种形式
        if "," in data:
            data = data.split(",", 1)[1]
        image_bytes = base64.b64decode(data, validate=True)
    except Exception as exc:  # pragma: no cover - base64 验证
        raise HTTPException(status_code=400, detail=f"第 {index} 帧 base64 解码失败: {exc}") from exc

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail=f"第 {index} 帧图像解码失败")

    # Mind-VAC 期望 RGB 帧
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb


@router.post("/recognize-frames", response_model=MindVacRecognitionResponse)
async def mind_vac_recognize_frames(payload: MindVacRecognitionRequest, request: Request) -> MindVacRecognitionResponse:
    """接收前端上传的帧序列并调用 Mind-VAC 模型完成识别"""

    app_state = request.app.state
    sign_recognition_service = getattr(app_state, "sign_recognition_service", None)
    if not sign_recognition_service:
        raise HTTPException(status_code=503, detail="连续手语识别服务不可用")

    mind_vac_engine = getattr(sign_recognition_service, "mind_vac_engine", None)
    if not mind_vac_engine or not getattr(mind_vac_engine, "enabled", False):
        raise HTTPException(status_code=503, detail="Mind-VAC 功能未启用")

    if not getattr(mind_vac_engine, "available", False):
        detail = getattr(mind_vac_engine, "last_error", None) or "Mind-VAC 引擎未初始化"
        raise HTTPException(status_code=503, detail=detail)

    frames_rgb: List[np.ndarray] = []
    for idx, encoded in enumerate(payload.frames):
        frame = _decode_base64_frame(encoded, idx)
        frames_rgb.append(frame)

    if len(frames_rgb) < 8:
        raise HTTPException(status_code=400, detail="有效帧数量不足，至少需要 8 帧")

    fps = float(payload.fps)
    duration = len(frames_rgb) / fps if fps > 0 else 0.0

    try:
        inference = await mind_vac_engine.run_on_frames(frames_rgb, use_llm=payload.use_llm)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.error("Mind-VAC 推理失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"Mind-VAC 推理失败: {exc}") from exc

    gloss_sequence = inference.get("gloss_sequence", []) or []
    raw_gloss_text = inference.get("raw_gloss_text", "") or ""
    confidence = float(inference.get("confidence", 0.0) or 0.0)
    llm_result = inference.get("llm_result")

    # 使用 SignRecognitionService 的后处理逻辑生成文本
    text = raw_gloss_text.replace(" ", "") if raw_gloss_text else ""
    baseline_text = text
    try:
        if gloss_sequence:
            baseline_text = sign_recognition_service._translate_gloss_to_text(gloss_sequence)  # pylint: disable=protected-access
            text = baseline_text
        if llm_result and llm_result.get("success") and llm_result.get("chinese"):
            text = llm_result.get("chinese") or text
    except Exception as exc:  # pragma: no cover - 防御性处理
        logger.warning("Mind-VAC 结果后处理失败: %s", exc)

    return MindVacRecognitionResponse(
        text=text or "",
        gloss_sequence=gloss_sequence,
        raw_gloss_text=raw_gloss_text,
        confidence=confidence,
        frame_count=len(frames_rgb),
        duration=duration,
        baseline_text=baseline_text,
        llm_result=llm_result,
    )
