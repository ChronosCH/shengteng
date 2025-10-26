"""
LLM 对话路由
提供基于识别结果的智能对话功能
"""
import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM对话"])


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: user/assistant/system")
    content: str = Field(..., description="消息内容")
    timestamp: float = Field(..., description="时间戳")


class ChatContext(BaseModel):
    recognitionResult: Optional[str] = Field(None, description="识别结果文本")
    glossSequence: Optional[List[str]] = Field(None, description="Gloss序列")
    baselineText: Optional[str] = Field(None, description="基础翻译文本")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    context: Optional[ChatContext] = Field(None, description="识别上下文")
    history: Optional[List[ChatMessage]] = Field(None, description="历史消息")


class ChatResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None


def build_context_prompt(context: ChatContext) -> str:
    """构建带上下文的提示词"""
    parts = ["以下是手语识别的结果:\n"]

    if context.recognitionResult:
        parts.append(f"识别文本: {context.recognitionResult}")

    if context.glossSequence and len(context.glossSequence) > 0:
        parts.append(f"Gloss序列: {' '.join(context.glossSequence)}")

    if context.baselineText:
        parts.append(f"基础翻译: {context.baselineText}")

    parts.append("\n请基于以上识别结果，用专业、友好的方式回答用户的问题。")
    return "\n".join(parts)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    LLM 对话接口
    
    接收用户消息和上下文，返回AI回复
    """
    try:
        # 检查API密钥
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            return ChatResponse(
                success=False,
                message="",
                error="未配置 DASHSCOPE_API_KEY，无法使用LLM对话功能"
            )

        # 导入通义千问API
        try:
            from mind_vac.qwen_api import QwenAPI
        except ImportError:
            return ChatResponse(
                success=False,
                message="",
                error="通义千问API模块未安装"
            )

        # 创建API客户端
        qwen_client = QwenAPI(api_key=api_key, model="qwen-plus")

        # 构建对话提示词
        messages = []

        # 添加系统提示词（包含识别上下文）
        if request.context:
            system_prompt = build_context_prompt(request.context)
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": "你是一个专业的手语翻译助手，善于解释手语含义、提供翻译建议。"
            })

        # 添加历史对话（最多保留最近5轮）
        if request.history:
            recent_history = request.history[-10:] if len(request.history) > 10 else request.history
            for msg in recent_history:
                if msg.role in ["user", "assistant"]:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

        # 添加当前用户消息
        messages.append({
            "role": "user",
            "content": request.message
        })

        # 调用通义千问API
        import requests
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        payload = {
            "model": "qwen-plus",
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 1500,
                "result_format": "message"
            }
        }

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()

        # 提取回复
        if 'output' in result and 'choices' in result['output']:
            reply = result['output']['choices'][0]['message']['content']
            return ChatResponse(
                success=True,
                message=reply
            )
        else:
            return ChatResponse(
                success=False,
                message="",
                error="API返回格式异常"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API请求失败: {e}")
        return ChatResponse(
            success=False,
            message="",
            error=f"API请求失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"LLM对话处理失败: {e}")
        return ChatResponse(
            success=False,
            message="",
            error=f"服务器错误: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """健康检查"""
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    return {
        "status": "ok",
        "llm_enabled": bool(api_key),
    }
