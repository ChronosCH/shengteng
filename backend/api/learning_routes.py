"""
手语学习训练API路由
提供完整的学习训练功能API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..services.learning_training_service import LearningTrainingService, DifficultyLevel, LearningType
from ..utils.auth import get_current_user

logger = logging.getLogger(__name__)

# 初始化路由和服务
router = APIRouter(prefix="/api/learning", tags=["学习训练"])
learning_service = LearningTrainingService()

@router.get("/modules", response_model=List[Dict[str, Any]])
async def get_learning_modules(
    user_id: str = Depends(get_current_user),
    difficulty: Optional[str] = Query(None, description="难度筛选"),
    category: Optional[str] = Query(None, description="分类筛选"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    """获取学习模块列表"""
    try:
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
    user_id: str = Depends(get_current_user)
):
    """获取模块的课程列表"""
    try:
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
    user_id: str = Depends(get_current_user)
):
    """完成课程"""
    try:
        if not 0 <= score <= 100:
            raise HTTPException(status_code=400, detail="分数必须在0-100之间")
        
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
async def get_user_progress(user_id: str = Depends(get_current_user)):
    """获取用户学习进度"""
    try:
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
async def get_user_achievements(user_id: str = Depends(get_current_user)):
    """获取用户成就"""
    try:
        achievements = await learning_service.get_user_achievements(user_id)
        return achievements
    except Exception as e:
        logger.error(f"获取用户成就失败: {e}")
        raise HTTPException(status_code=500, detail="获取用户成就失败")

@router.get("/daily-tasks", response_model=List[Dict[str, Any]])
async def get_daily_tasks(user_id: str = Depends(get_current_user)):
    """获取每日任务"""
    try:
        tasks = await learning_service.get_daily_tasks(user_id)
        return tasks
    except Exception as e:
        logger.error(f"获取每日任务失败: {e}")
        raise HTTPException(status_code=500, detail="获取每日任务失败")

@router.post("/daily-tasks/{task_type}/complete")
async def complete_daily_task(
    task_type: str,
    user_id: str = Depends(get_current_user)
):
    """完成每日任务"""
    try:
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
    user_id: str = Depends(get_current_user)
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
async def get_learning_paths(user_id: str = Depends(get_current_user)):
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
async def get_learning_statistics(user_id: str = Depends(get_current_user)):
    """获取学习统计数据"""
    try:
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
async def get_learning_recommendations(user_id: str = Depends(get_current_user)):
    """获取个性化学习推荐"""
    try:
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
    user_id: str = Depends(get_current_user)
):
    """提交学习反馈"""
    try:
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="评分必须在1-5之间")
        
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
    user_id: str = Depends(get_current_user)
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

# 导出路由
__all__ = ["router"]
