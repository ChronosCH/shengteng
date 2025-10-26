"""
手语学习训练API路由
提供完整的学习训练功能API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Request, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os

import requests

from ..services.learning_training_service import LearningTrainingService, DifficultyLevel
from ..utils.security import SecurityManager
from ..services import IsolatedSignService

logger = logging.getLogger(__name__)

# 初始化路由和服务（main 中会再挂载前缀）
router = APIRouter(tags=["学习训练"])
learning_service = LearningTrainingService()
security_manager = SecurityManager()
get_current_user = security_manager.get_current_user
get_current_user_optional = security_manager.get_current_user_optional

class PredictRequest(BaseModel):
    file_path: str

async def get_isolated_service(request: Request) -> IsolatedSignService:
    svc = getattr(request.app.state, "isolated_sign_service", None)
    if svc:
        return svc
    raise HTTPException(status_code=503, detail="孤立手语识别服务未初始化")

@router.get("/modules", response_model=List[Dict[str, Any]])
async def get_learning_modules(
    current_user: dict = Depends(get_current_user),
    difficulty: Optional[str] = Query(None, description="难度筛选"),
    category: Optional[str] = Query(None, description="分类筛选"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    """获取学习模块列表"""
    try:
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        modules = await learning_service.get_learning_modules(user_id)
        
        # 应用筛选条件
        if difficulty:
            try:
                difficulty_level = DifficultyLevel(difficulty)
                modules = [m for m in modules if m.get('level') == difficulty_level.value]
            except ValueError:
                pass
        
        if category:
            modules = [m for m in modules if m.get('category') == category]
        
        if search:
            search_lower = search.lower()
            modules = [
                m for m in modules 
                if (search_lower in m.get('title', '').lower() or 
                    search_lower in m.get('description', '').lower() or
                    any(search_lower in skill.lower() for skill in m.get('skills', [])))
            ]
        
        return modules
    except Exception as e:
        logger.error(f"获取学习模块失败: {e}")
        raise HTTPException(status_code=500, detail="获取学习模块失败")

@router.get("/modules/{module_id}/lessons", response_model=List[Dict[str, Any]])
async def get_module_lessons(
    module_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取模块的课程列表"""
    try:
        # user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        lessons = await learning_service.get_module_lessons(module_id)
        return lessons
    except Exception as e:
        logger.error(f"获取模块课程失败: {e}")
        raise HTTPException(status_code=500, detail="获取模块课程失败")

@router.post("/lessons/{lesson_id}/complete")
async def complete_lesson(
    lesson_id: str,
    score: float = 100.0,
    time_spent: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """完成课程"""
    try:
        if not 0 <= score <= 100:
            raise HTTPException(status_code=400, detail="分数必须在0-100之间")
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        result = await learning_service.complete_lesson(user_id, lesson_id, score, time_spent)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "完成课程失败"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"完成课程失败: {e}")
        raise HTTPException(status_code=500, detail="完成课程失败")

@router.get("/progress", response_model=Dict[str, Any])
async def get_user_progress(current_user: dict = Depends(get_current_user)):
    """获取用户学习进度"""
    try:
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        progress = await learning_service.get_user_progress(user_id)
        stats = await learning_service.get_learning_stats(user_id)
        return {
            "progress": progress.__dict__,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"获取用户进度失败: {e}")
        raise HTTPException(status_code=500, detail="获取用户进度失败")

@router.get("/achievements", response_model=List[Dict[str, Any]])
async def get_user_achievements(current_user: dict = Depends(get_current_user)):
    """获取用户成就"""
    try:
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        achievements = await learning_service.get_user_achievements(user_id)
        return achievements
    except Exception as e:
        logger.error(f"获取用户成就失败: {e}")
        raise HTTPException(status_code=500, detail="获取用户成就失败")

@router.get("/daily-tasks", response_model=List[Dict[str, Any]])
async def get_daily_tasks(current_user: dict = Depends(get_current_user)):
    """获取每日任务"""
    try:
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        tasks = await learning_service.get_daily_tasks(user_id)
        return tasks
    except Exception as e:
        logger.error(f"获取每日任务失败: {e}")
        raise HTTPException(status_code=500, detail="获取每日任务失败")

@router.post("/daily-tasks/{task_type}/complete")
async def complete_daily_task(
    task_type: str,
    current_user: dict = Depends(get_current_user)
):
    """完成每日任务"""
    try:
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        result = await learning_service.complete_daily_task(user_id, task_type)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "完成任务失败"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"完成每日任务失败: {e}")
        raise HTTPException(status_code=500, detail="完成每日任务失败")

@router.get("/search", response_model=Dict[str, Any])
async def search_learning_content(
    query: str = Query(..., description="搜索关键词"),
    content_type: str = Query("all", description="内容类型: all, modules, lessons, achievements"),
    current_user: dict = Depends(get_current_user)
):
    """搜索学习内容"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="搜索关键词不能为空")
        
        results = await learning_service.search_content(query, content_type)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索学习内容失败: {e}")
        raise HTTPException(status_code=500, detail="搜索学习内容失败")

@router.get("/learning-paths", response_model=List[Dict[str, Any]])
async def get_learning_paths(current_user: dict = Depends(get_current_user)):
    """获取学习路径"""
    try:
        # 这里返回预定义的学习路径
        paths = [
            {
                "id": "quick-start",
                "title": "快速入门路径",
                "description": "适合零基础学习者的快速入门课程，7天掌握基础手语",
                "duration": "1-2周",
                "modules": ["basic-signs", "numbers-time"],
                "difficulty": "beginner",
                "color": "#B5EAD7",
                "estimatedHours": 6,
                "skills": ["基础词汇", "数字表达", "简单交流"],
                "completionRate": 85,
                "enrolled": 1250,
                "steps": [
                    {"title": "问候语学习", "description": "学习基本问候用语"},
                    {"title": "数字掌握", "description": "掌握数字0-100"},
                    {"title": "自我介绍", "description": "学会用手语自我介绍"},
                    {"title": "日常对话", "description": "进行简单日常对话"},
                ]
            },
            {
                "id": "daily-communication",
                "title": "日常交流路径",
                "description": "学习日常生活中最常用的手语表达，满足基本交流需求",
                "duration": "3-4周",
                "modules": ["basic-signs", "family-relations", "numbers-time", "daily-activities"],
                "difficulty": "intermediate",
                "color": "#FFDAB9",
                "estimatedHours": 12,
                "skills": ["生活用语", "家庭交流", "社交表达"],
                "completionRate": 78,
                "enrolled": 890,
                "steps": [
                    {"title": "基础巩固", "description": "巩固基础手语知识"},
                    {"title": "家庭交流", "description": "学习家庭相关表达"},
                    {"title": "日常活动", "description": "掌握日常活动用语"},
                    {"title": "综合应用", "description": "综合运用所学知识"},
                ]
            },
            {
                "id": "professional-advanced",
                "title": "专业进阶路径",
                "description": "深入学习手语语法和高级表达技巧，达到专业水平",
                "duration": "6-8周",
                "modules": ["basic-signs", "family-relations", "daily-activities", "advanced-grammar", "professional-signs"],
                "difficulty": "advanced",
                "color": "#C7CEDB",
                "estimatedHours": 25,
                "skills": ["高级语法", "专业术语", "流畅表达"],
                "completionRate": 65,
                "enrolled": 456,
                "steps": [
                    {"title": "语法深化", "description": "学习复杂语法结构"},
                    {"title": "专业应用", "description": "掌握职场手语"},
                    {"title": "高级技巧", "description": "学习高级表达技巧"},
                    {"title": "实战演练", "description": "实际场景应用练习"},
                ]
            }
        ]
        return paths
    except Exception as e:
        logger.error(f"获取学习路径失败: {e}")
        raise HTTPException(status_code=500, detail="获取学习路径失败")

@router.get("/statistics", response_model=Dict[str, Any])
async def get_learning_statistics(current_user: dict = Depends(get_current_user)):
    """获取学习统计数据"""
    try:
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        stats = await learning_service.get_learning_stats(user_id)
        
        # 添加额外的统计信息
        extra_stats = {
            "today_stats": {
                "lessons_completed": 2,
                "time_spent": 45,
                "xp_earned": 120,
                "goal": 60,
            },
            "weekly_stats": {
                "goal": 300,
                "progress": 180,
                "lessons": 12,
                "average_score": 87,
            },
            "monthly_stats": {
                "goal": 1200,
                "progress": 650,
                "lessons": 45,
                "achievements_unlocked": 3,
            }
        }
        
        return {**stats, **extra_stats}
    except Exception as e:
        logger.error(f"获取学习统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取学习统计失败")

@router.get("/recommendations", response_model=Dict[str, Any])
async def get_learning_recommendations(current_user: dict = Depends(get_current_user)):
    """获取个性化学习推荐"""
    try:
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        # 获取用户进度
        user_progress = await learning_service.get_user_progress(user_id)
        
        # 基于用户进度生成推荐
        recommendations = {
            "recommended_modules": [],
            "suggested_daily_goal": 30,  # 分钟
            "personalized_tips": [
                "根据您的学习进度，建议重点练习'数字表达'",
                "建议每天学习20-30分钟，保持连续性",
                "多与其他学习者交流，分享学习心得",
                "使用移动端随时随地练习手语"
            ],
            "next_milestone": {
                "title": "基础大师",
                "description": "完成所有基础课程",
                "progress": 67,
                "remaining_lessons": 4
            }
        }
        
        # 根据用户等级推荐模块
        if user_progress.level < 5:
            recommendations["recommended_modules"] = ["basic-signs", "numbers-time"]
        elif user_progress.level < 10:
            recommendations["recommended_modules"] = ["family-relations", "daily-activities"]
        else:
            recommendations["recommended_modules"] = ["advanced-grammar", "professional-signs"]
        
        return recommendations
    except Exception as e:
        logger.error(f"获取学习推荐失败: {e}")
        raise HTTPException(status_code=500, detail="获取学习推荐失败")

@router.post("/feedback")
async def submit_learning_feedback(
    module_id: str,
    rating: float,
    comment: str = "",
    current_user: dict = Depends(get_current_user)
):
    """提交学习反馈"""
    try:
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="评分必须在1-5之间")
        user_id = str(current_user.get("id") or current_user.get("user_id") or "guest")
        # 这里可以保存用户反馈到数据库
        feedback_data = {
            "user_id": user_id,
            "module_id": module_id,
            "rating": rating,
            "comment": comment,
            "submitted_at": datetime.now().isoformat()
        }
        
        # TODO: 保存到数据库
        logger.info(f"收到用户反馈: {feedback_data}")
        
        return {"success": True, "message": "反馈提交成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交学习反馈失败: {e}")
        raise HTTPException(status_code=500, detail="提交学习反馈失败")

@router.get("/leaderboard", response_model=Dict[str, Any])
async def get_learning_leaderboard(
    period: str = Query("weekly", description="排行榜周期: daily, weekly, monthly"),
    limit: int = Query(10, description="返回数量限制"),
    current_user: dict = Depends(get_current_user)
):
    """获取学习排行榜"""
    try:
        # 模拟排行榜数据
        leaderboard = {
            "period": period,
            "user_rank": 15,
            "total_users": 1250,
            "rankings": [
                {"rank": 1, "username": "学习达人", "score": 2850, "avatar": "", "streak": 30},
                {"rank": 2, "username": "手语新星", "score": 2720, "avatar": "", "streak": 25},
                {"rank": 3, "username": "勤奋小蜜蜂", "score": 2680, "avatar": "", "streak": 28},
                {"rank": 4, "username": "语言大师", "score": 2590, "avatar": "", "streak": 22},
                {"rank": 5, "username": "沟通专家", "score": 2450, "avatar": "", "streak": 20},
            ]
        }
        
        return leaderboard
    except Exception as e:
        logger.error(f"获取学习排行榜失败: {e}")
        raise HTTPException(status_code=500, detail="获取学习排行榜失败")

@router.get("/categories", response_model=List[Dict[str, Any]])
async def get_learning_categories():
    """获取学习分类"""
    try:
        categories = [
            {"id": "basic", "name": "基础入门", "icon": "🎯", "color": "#B5EAD7"},
            {"id": "daily", "name": "生活应用", "icon": "🏠", "color": "#FFDAB9"},
            {"id": "professional", "name": "专业应用", "icon": "💼", "color": "#FFB3BA"},
            {"id": "advanced", "name": "高级进阶", "icon": "🎓", "color": "#C7CEDB"},
            {"id": "social", "name": "社交交流", "icon": "👥", "color": "#E8E3F0"},
            {"id": "entertainment", "name": "娱乐休闲", "icon": "🎮", "color": "#B8A9C9"},
        ]
        return categories
    except Exception as e:
        logger.error(f"获取学习分类失败: {e}")
        raise HTTPException(status_code=500, detail="获取学习分类失败")

@router.post("/isolated-sign/upload")
async def upload_isolated_sign(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    try:
        file_manager = getattr(request.app.state, "file_manager", None)
        if file_manager is None:
            raise HTTPException(status_code=503, detail="文件服务不可用")

        info = await file_manager.save_file(file, user_id=current_user.get("id"))
        return {"success": True, "file_path": info["file_path"], "filename": info["filename"]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"孤立手语视频上传失败: {exc}")
        raise HTTPException(status_code=500, detail="上传失败")


@router.post("/isolated-sign/predict")
async def predict_isolated_sign(
    request: Request,
    payload: Optional[PredictRequest] = Body(default=None),
    current_user: dict = Depends(get_current_user),
    service: IsolatedSignService = Depends(get_isolated_service),
):
    try:
        file_path = payload.file_path if payload else None

        if not file_path:
            try:
                raw_body = await request.json()
                if isinstance(raw_body, dict):
                    file_path = raw_body.get("file_path")
            except Exception:
                file_path = None

        if not file_path:
            raise HTTPException(status_code=422, detail="文件路径缺失，请提供 file_path")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="视频文件不存在")

        result = await service.predict(file_path)
        
        # 生成学习反馈
        feedback = _generate_learning_feedback(result.predicted_gloss, result.confidence)
        
        return {
            "success": True,
            "prediction": {
                "gloss": result.predicted_gloss,
                "confidence": result.confidence,
                "top_k_predictions": result.top_k_predictions,  # 新增：返回 Top-K 预测结果
            },
            "feedback": feedback,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"孤立手语识别失败: {exc}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(exc)}")

def _generate_learning_feedback(gloss: str, confidence: float) -> Dict[str, Any]:
    """生成学习反馈"""
    feedback = {
        "recognized_sign": gloss,
        "accuracy_level": "",
        "message": "",
        "tips": [],
        "next_steps": []
    }
    
    if confidence >= 0.9:
        feedback["accuracy_level"] = "excellent"
        feedback["message"] = f"太棒了！识别到标准的'{gloss}'手语，准确率高达{confidence*100:.1f}%！"
        feedback["tips"] = [
            "动作非常标准，继续保持！",
            "可以尝试加快手语速度，模拟真实对话",
        ]
        feedback["next_steps"] = [
            "尝试学习下一个手语动作",
            "挑战更复杂的手语组合",
        ]
    elif confidence >= 0.75:
        feedback["accuracy_level"] = "good"
        feedback["message"] = f"很不错！识别到'{gloss}'手语，准确率{confidence*100:.1f}%。"
        feedback["tips"] = [
            "动作基本标准，稍加练习会更好",
            "注意手型的清晰度和动作幅度",
        ]
        feedback["next_steps"] = [
            "多练几次以提高熟练度",
            "可以尝试结合面部表情",
        ]
    elif confidence >= 0.6:
        feedback["accuracy_level"] = "fair"
        feedback["message"] = f"识别到'{gloss}'手语，但准确率只有{confidence*100:.1f}%，还需要改进。"
        feedback["tips"] = [
            "建议重新观看标准教学视频",
            "注意手指的姿态和动作的连贯性",
            "确保光线充足，背景简洁",
        ]
        feedback["next_steps"] = [
            "观看标准示范视频",
            "对照镜子慢速练习",
            "再次录制视频上传",
        ]
    else:
        feedback["accuracy_level"] = "needs_improvement"
        feedback["message"] = f"识别到'{gloss}'，但准确率较低({confidence*100:.1f}%)，动作可能不够标准。"
        feedback["tips"] = [
            "请仔细观看教学视频，注意每个细节",
            "确保手部完整出现在画面中",
            "动作要清晰、幅度要适中",
            "避免背景干扰和光线不足",
        ]
        feedback["next_steps"] = [
            "从基础手型开始练习",
            "分解动作，逐步掌握",
            "可以向AI助手请教正确做法",
        ]
    
    return feedback


class AITutorRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None


@router.post("/ai-tutor/chat")
async def chat_with_ai_tutor(
    request: AITutorRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    AI手语教学助手对话接口
    支持联网搜索学习资源
    （可选认证 - 未登录用户也可以使用，但功能有限）
    """
    try:
        # 检查API密钥
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="未配置 DASHSCOPE_API_KEY，无法使用AI助手功能"
            )

        # 构建系统提示词 - 让AI扮演手语教学老师
        system_prompt = """你是一位专业、耐心的手语教学老师。你的任务是帮助用户学习手语。

你的能力：
1. 解答手语学习相关的各种问题
2. 推荐适合的学习内容和进度
3. 解释手语动作要领和技巧
4. 提供鼓励和学习建议
5. 可以联网搜索并推荐优质的手语学习视频（B站、YouTube等）

回答要求：
1. 语言要亲切、鼓励性强
2. 解释要清晰易懂
3. 给出具体可操作的建议
4. 如果用户询问某个手语如何做，除了文字描述，也要推荐相关学习视频链接
5. 当推荐视频时，优先推荐B站（bilibili.com）的中文手语教程

当前用户信息：
- 用户ID: """ + str((current_user.get("id") if current_user else None) or (current_user.get("user_id") if current_user else None) or "guest")

        # 如果有识别上下文，加入提示词
        if request.context:
            if request.context.get("recognized_sign"):
                system_prompt += f"\n- 刚刚识别到的手语：{request.context['recognized_sign']}"
            if request.context.get("confidence"):
                system_prompt += f"\n- 识别准确率：{request.context['confidence']*100:.1f}%"
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史消息（保留最近10条）
        if request.history:
            recent_history = request.history[-10:] if len(request.history) > 10 else request.history
            for msg in recent_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # 添加当前用户消息
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # 调用通义千问API（启用联网搜索）
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
                "max_tokens": 2000,
                "result_format": "message",
                "enable_search": True  # 启用联网搜索
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
        
        # 提取AI回复
        if 'output' in result and 'choices' in result['output']:
            ai_reply = result['output']['choices'][0]['message']['content']
            
            return {
                "success": True,
                "message": ai_reply,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="AI响应格式异常")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"AI助手API请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"AI服务暂时不可用: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI助手对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.post("/ai-tutor/suggest-videos")
async def suggest_learning_videos(
    sign_name: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    """
    AI推荐手语学习视频
    """
    try:
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="未配置 DASHSCOPE_API_KEY"
            )

        prompt = f"""请帮我搜索关于"{sign_name}"手语的学习视频，要求：
1. 优先推荐B站（bilibili.com）的中文手语教程
2. 也可以推荐YouTube上的优质教程
3. 确保视频链接真实可用
4. 每个推荐包含：标题、链接、简短说明
5. 最多推荐5个视频

请以JSON格式返回：
{{
  "videos": [
    {{
      "title": "视频标题",
      "url": "视频链接",
      "platform": "B站/YouTube",
      "description": "简短说明"
    }}
  ]
}}"""
        
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        payload = {
            "model": "qwen-plus",
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": 0.3,
                "max_tokens": 1500,
                "result_format": "message",
                "enable_search": True
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if 'output' in result and 'choices' in result['output']:
            ai_reply = result['output']['choices'][0]['message']['content']
            
            # 尝试解析JSON
            import json
            import re
            
            # 清理可能的markdown标记
            ai_reply = ai_reply.strip()
            if '```json' in ai_reply:
                ai_reply = re.sub(r'```json\s*', '', ai_reply)
            if '```' in ai_reply:
                ai_reply = re.sub(r'```\s*', '', ai_reply)
            ai_reply = ai_reply.strip()
            
            try:
                videos_data = json.loads(ai_reply)
                return {
                    "success": True,
                    "videos": videos_data.get("videos", []),
                    "raw_response": ai_reply
                }
            except json.JSONDecodeError:
                # 如果无法解析JSON，返回原始文本
                return {
                    "success": True,
                    "videos": [],
                    "raw_response": ai_reply,
                    "note": "AI返回了文本格式的推荐，请查看raw_response"
                }
        else:
            raise HTTPException(status_code=500, detail="AI响应格式异常")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"视频推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 导出路由
__all__ = ["router"]
