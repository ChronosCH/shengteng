"""
增强的手语学习训练服务
Enhanced Sign Language Learning Training Service
提供更完善的学习功能，包括自适应学习、智能推荐、学习分析等
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import sqlite3
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class LearningStyle(Enum):
    """学习风格"""
    VISUAL = "visual"          # 视觉型
    AUDITORY = "auditory"      # 听觉型
    KINESTHETIC = "kinesthetic" # 动觉型
    MIXED = "mixed"            # 混合型

class SkillLevel(Enum):
    """技能水平"""
    NOVICE = "novice"          # 新手
    BEGINNER = "beginner"      # 初学者
    INTERMEDIATE = "intermediate" # 中级
    ADVANCED = "advanced"      # 高级
    EXPERT = "expert"          # 专家

class LearningGoal(Enum):
    """学习目标"""
    BASIC_COMMUNICATION = "basic_communication"  # 基础交流
    DAILY_CONVERSATION = "daily_conversation"    # 日常对话
    PROFESSIONAL = "professional"                # 专业手语
    INTERPRETATION = "interpretation"            # 手语翻译

@dataclass
class LearningProfile:
    """学习档案"""
    user_id: str
    learning_style: LearningStyle = LearningStyle.MIXED
    skill_level: SkillLevel = SkillLevel.NOVICE
    learning_goals: List[LearningGoal] = None
    preferred_session_duration: int = 30  # 分钟
    daily_goal_minutes: int = 60
    weak_areas: List[str] = None
    strong_areas: List[str] = None
    learning_pace: float = 1.0  # 学习速度倍数
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.learning_goals is None:
            self.learning_goals = [LearningGoal.BASIC_COMMUNICATION]
        if self.weak_areas is None:
            self.weak_areas = []
        if self.strong_areas is None:
            self.strong_areas = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class LearningSession:
    """学习会话"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: int = 0
    lessons_completed: List[str] = None
    exercises_completed: List[str] = None
    accuracy_score: float = 0.0
    engagement_score: float = 0.0
    xp_earned: int = 0
    mistakes: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.lessons_completed is None:
            self.lessons_completed = []
        if self.exercises_completed is None:
            self.exercises_completed = []
        if self.mistakes is None:
            self.mistakes = []

@dataclass
class SmartRecommendation:
    """智能推荐"""
    recommendation_id: str
    user_id: str
    type: str  # lesson, exercise, review, practice
    content_id: str
    title: str
    description: str
    reason: str
    confidence: float
    priority: int
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class EnhancedLearningService:
    """增强的学习训练服务"""
    
    def __init__(self, db_path: str = "data/enhanced_learning.db"):
        self.db_path = db_path
        self.learning_profiles: Dict[str, LearningProfile] = {}
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.recommendations: Dict[str, List[SmartRecommendation]] = {}
        
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 加载学习数据
        self._load_learning_data()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 学习档案表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS learning_profiles (
                        user_id TEXT PRIMARY KEY,
                        learning_style TEXT,
                        skill_level TEXT,
                        learning_goals TEXT,
                        preferred_session_duration INTEGER,
                        daily_goal_minutes INTEGER,
                        weak_areas TEXT,
                        strong_areas TEXT,
                        learning_pace REAL,
                        created_at TEXT,
                        updated_at TEXT
                    )
                ''')
                
                # 学习会话表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        start_time TEXT,
                        end_time TEXT,
                        duration_minutes INTEGER,
                        lessons_completed TEXT,
                        exercises_completed TEXT,
                        accuracy_score REAL,
                        engagement_score REAL,
                        xp_earned INTEGER,
                        mistakes TEXT
                    )
                ''')
                
                # 智能推荐表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS smart_recommendations (
                        recommendation_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        type TEXT,
                        content_id TEXT,
                        title TEXT,
                        description TEXT,
                        reason TEXT,
                        confidence REAL,
                        priority INTEGER,
                        created_at TEXT
                    )
                ''')
                
                # 学习分析表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS learning_analytics (
                        user_id TEXT,
                        date TEXT,
                        total_time INTEGER,
                        lessons_completed INTEGER,
                        accuracy_rate REAL,
                        engagement_rate REAL,
                        xp_earned INTEGER,
                        PRIMARY KEY (user_id, date)
                    )
                ''')
                
                conn.commit()
                logger.info("✅ 增强学习服务数据库初始化完成")
                
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
            raise
    
    def _load_learning_data(self):
        """加载学习数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 加载学习档案
                cursor.execute("SELECT * FROM learning_profiles")
                for row in cursor.fetchall():
                    profile = LearningProfile(
                        user_id=row[0],
                        learning_style=LearningStyle(row[1]),
                        skill_level=SkillLevel(row[2]),
                        learning_goals=[LearningGoal(g) for g in json.loads(row[3])],
                        preferred_session_duration=row[4],
                        daily_goal_minutes=row[5],
                        weak_areas=json.loads(row[6]),
                        strong_areas=json.loads(row[7]),
                        learning_pace=row[8],
                        created_at=datetime.fromisoformat(row[9]),
                        updated_at=datetime.fromisoformat(row[10])
                    )
                    self.learning_profiles[profile.user_id] = profile
                
                logger.info(f"✅ 加载了 {len(self.learning_profiles)} 个学习档案")
                
        except Exception as e:
            logger.error(f"❌ 学习数据加载失败: {e}")
    
    async def create_learning_profile(self, user_id: str, **kwargs) -> LearningProfile:
        """创建学习档案"""
        try:
            profile = LearningProfile(user_id=user_id, **kwargs)
            self.learning_profiles[user_id] = profile
            
            # 保存到数据库
            await self._save_learning_profile(profile)
            
            # 生成初始推荐
            await self._generate_initial_recommendations(user_id)
            
            logger.info(f"✅ 为用户 {user_id} 创建学习档案")
            return profile
            
        except Exception as e:
            logger.error(f"❌ 创建学习档案失败: {e}")
            raise
    
    async def _save_learning_profile(self, profile: LearningProfile):
        """保存学习档案"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO learning_profiles 
                    (user_id, learning_style, skill_level, learning_goals, 
                     preferred_session_duration, daily_goal_minutes, weak_areas, 
                     strong_areas, learning_pace, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile.user_id,
                    profile.learning_style.value,
                    profile.skill_level.value,
                    json.dumps([g.value for g in profile.learning_goals]),
                    profile.preferred_session_duration,
                    profile.daily_goal_minutes,
                    json.dumps(profile.weak_areas),
                    json.dumps(profile.strong_areas),
                    profile.learning_pace,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 保存学习档案失败: {e}")
            raise
    
    async def start_learning_session(self, user_id: str) -> str:
        """开始学习会话"""
        try:
            session_id = f"session_{user_id}_{int(time.time())}"
            session = LearningSession(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now()
            )
            
            self.learning_sessions[session_id] = session
            logger.info(f"✅ 用户 {user_id} 开始学习会话 {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ 开始学习会话失败: {e}")
            raise
    
    async def end_learning_session(self, session_id: str, **kwargs) -> LearningSession:
        """结束学习会话"""
        try:
            if session_id not in self.learning_sessions:
                raise ValueError(f"学习会话 {session_id} 不存在")
            
            session = self.learning_sessions[session_id]
            session.end_time = datetime.now()
            session.duration_minutes = int((session.end_time - session.start_time).total_seconds() / 60)
            
            # 更新会话数据
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            # 保存会话数据
            await self._save_learning_session(session)
            
            # 更新学习档案
            await self._update_learning_profile_from_session(session)
            
            # 生成新的推荐
            await self._generate_recommendations_from_session(session)
            
            logger.info(f"✅ 学习会话 {session_id} 已结束")
            return session
            
        except Exception as e:
            logger.error(f"❌ 结束学习会话失败: {e}")
            raise
    
    async def _generate_initial_recommendations(self, user_id: str):
        """生成初始推荐"""
        try:
            profile = self.learning_profiles.get(user_id)
            if not profile:
                return
            
            recommendations = []
            
            # 基于技能水平推荐
            if profile.skill_level == SkillLevel.NOVICE:
                recommendations.append(SmartRecommendation(
                    recommendation_id=f"rec_{user_id}_basic_1",
                    user_id=user_id,
                    type="lesson",
                    content_id="lesson_basic_greetings",
                    title="基础问候手语",
                    description="学习日常问候的基本手语",
                    reason="适合初学者的基础课程",
                    confidence=0.9,
                    priority=1
                ))
            
            # 基于学习目标推荐
            for goal in profile.learning_goals:
                if goal == LearningGoal.BASIC_COMMUNICATION:
                    recommendations.append(SmartRecommendation(
                        recommendation_id=f"rec_{user_id}_comm_1",
                        user_id=user_id,
                        type="lesson",
                        content_id="lesson_basic_communication",
                        title="基础交流手语",
                        description="学习基本的交流手语词汇",
                        reason="符合您的基础交流学习目标",
                        confidence=0.8,
                        priority=2
                    ))
            
            # 保存推荐
            self.recommendations[user_id] = recommendations
            await self._save_recommendations(user_id, recommendations)
            
        except Exception as e:
            logger.error(f"❌ 生成初始推荐失败: {e}")
    
    async def get_personalized_recommendations(self, user_id: str, limit: int = 5) -> List[SmartRecommendation]:
        """获取个性化推荐"""
        try:
            recommendations = self.recommendations.get(user_id, [])
            
            # 按优先级和置信度排序
            recommendations.sort(key=lambda x: (x.priority, -x.confidence))
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"❌ 获取个性化推荐失败: {e}")
            return []

    async def _save_learning_session(self, session: LearningSession):
        """保存学习会话"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO learning_sessions
                    (session_id, user_id, start_time, end_time, duration_minutes,
                     lessons_completed, exercises_completed, accuracy_score,
                     engagement_score, xp_earned, mistakes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    session.user_id,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.duration_minutes,
                    json.dumps(session.lessons_completed),
                    json.dumps(session.exercises_completed),
                    session.accuracy_score,
                    session.engagement_score,
                    session.xp_earned,
                    json.dumps(session.mistakes)
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"❌ 保存学习会话失败: {e}")
            raise

    async def _save_recommendations(self, user_id: str, recommendations: List[SmartRecommendation]):
        """保存推荐"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 清除旧推荐
                cursor.execute("DELETE FROM smart_recommendations WHERE user_id = ?", (user_id,))

                # 保存新推荐
                for rec in recommendations:
                    cursor.execute('''
                        INSERT INTO smart_recommendations
                        (recommendation_id, user_id, type, content_id, title,
                         description, reason, confidence, priority, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rec.recommendation_id,
                        rec.user_id,
                        rec.type,
                        rec.content_id,
                        rec.title,
                        rec.description,
                        rec.reason,
                        rec.confidence,
                        rec.priority,
                        rec.created_at.isoformat()
                    ))

                conn.commit()

        except Exception as e:
            logger.error(f"❌ 保存推荐失败: {e}")
            raise

    async def _update_learning_profile_from_session(self, session: LearningSession):
        """从学习会话更新学习档案"""
        try:
            profile = self.learning_profiles.get(session.user_id)
            if not profile:
                return

            # 更新弱项和强项
            if session.accuracy_score < 0.7:
                # 识别弱项
                for lesson_id in session.lessons_completed:
                    if lesson_id not in profile.weak_areas:
                        profile.weak_areas.append(lesson_id)
            elif session.accuracy_score > 0.9:
                # 识别强项
                for lesson_id in session.lessons_completed:
                    if lesson_id not in profile.strong_areas:
                        profile.strong_areas.append(lesson_id)
                    # 从弱项中移除
                    if lesson_id in profile.weak_areas:
                        profile.weak_areas.remove(lesson_id)

            # 调整学习速度
            if session.engagement_score > 0.8 and session.accuracy_score > 0.8:
                profile.learning_pace = min(2.0, profile.learning_pace * 1.1)
            elif session.engagement_score < 0.5 or session.accuracy_score < 0.6:
                profile.learning_pace = max(0.5, profile.learning_pace * 0.9)

            profile.updated_at = datetime.now()

            # 保存更新
            await self._save_learning_profile(profile)

        except Exception as e:
            logger.error(f"❌ 更新学习档案失败: {e}")

    async def _generate_recommendations_from_session(self, session: LearningSession):
        """从学习会话生成推荐"""
        try:
            profile = self.learning_profiles.get(session.user_id)
            if not profile:
                return

            recommendations = []

            # 基于错误生成复习推荐
            if session.mistakes:
                mistake_topics = set()
                for mistake in session.mistakes:
                    if 'topic' in mistake:
                        mistake_topics.add(mistake['topic'])

                for topic in mistake_topics:
                    recommendations.append(SmartRecommendation(
                        recommendation_id=f"rec_{session.user_id}_review_{topic}_{int(time.time())}",
                        user_id=session.user_id,
                        type="review",
                        content_id=f"review_{topic}",
                        title=f"复习 {topic}",
                        description=f"针对 {topic} 的专项复习",
                        reason="基于学习中的错误生成的复习建议",
                        confidence=0.8,
                        priority=1
                    ))

            # 基于准确率生成推荐
            if session.accuracy_score < 0.7:
                recommendations.append(SmartRecommendation(
                    recommendation_id=f"rec_{session.user_id}_practice_{int(time.time())}",
                    user_id=session.user_id,
                    type="practice",
                    content_id="practice_basic",
                    title="基础练习",
                    description="加强基础技能的练习",
                    reason="准确率较低，建议进行基础练习",
                    confidence=0.7,
                    priority=2
                ))
            elif session.accuracy_score > 0.9:
                recommendations.append(SmartRecommendation(
                    recommendation_id=f"rec_{session.user_id}_advance_{int(time.time())}",
                    user_id=session.user_id,
                    type="lesson",
                    content_id="lesson_advanced",
                    title="进阶课程",
                    description="挑战更高难度的课程",
                    reason="表现优秀，可以尝试更高难度的内容",
                    confidence=0.8,
                    priority=3
                ))

            # 更新推荐
            if recommendations:
                existing_recs = self.recommendations.get(session.user_id, [])
                existing_recs.extend(recommendations)
                self.recommendations[session.user_id] = existing_recs
                await self._save_recommendations(session.user_id, existing_recs)

        except Exception as e:
            logger.error(f"❌ 生成会话推荐失败: {e}")

    async def get_learning_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """获取学习分析"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 获取指定天数内的学习数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                cursor.execute('''
                    SELECT date, total_time, lessons_completed, accuracy_rate,
                           engagement_rate, xp_earned
                    FROM learning_analytics
                    WHERE user_id = ? AND date >= ? AND date <= ?
                    ORDER BY date
                ''', (user_id, start_date.date().isoformat(), end_date.date().isoformat()))

                daily_data = cursor.fetchall()

                # 计算统计数据
                total_time = sum(row[1] for row in daily_data)
                total_lessons = sum(row[2] for row in daily_data)
                avg_accuracy = sum(row[3] for row in daily_data) / len(daily_data) if daily_data else 0
                avg_engagement = sum(row[4] for row in daily_data) / len(daily_data) if daily_data else 0
                total_xp = sum(row[5] for row in daily_data)

                # 学习趋势
                recent_week = daily_data[-7:] if len(daily_data) >= 7 else daily_data
                prev_week = daily_data[-14:-7] if len(daily_data) >= 14 else []

                trend = "stable"
                if recent_week and prev_week:
                    recent_avg = sum(row[1] for row in recent_week) / len(recent_week)
                    prev_avg = sum(row[1] for row in prev_week) / len(prev_week)
                    if recent_avg > prev_avg * 1.1:
                        trend = "improving"
                    elif recent_avg < prev_avg * 0.9:
                        trend = "declining"

                analytics = {
                    "period_days": days,
                    "total_learning_time": total_time,
                    "total_lessons_completed": total_lessons,
                    "average_accuracy": round(avg_accuracy, 2),
                    "average_engagement": round(avg_engagement, 2),
                    "total_xp_earned": total_xp,
                    "learning_trend": trend,
                    "daily_data": [
                        {
                            "date": row[0],
                            "time": row[1],
                            "lessons": row[2],
                            "accuracy": row[3],
                            "engagement": row[4],
                            "xp": row[5]
                        } for row in daily_data
                    ],
                    "insights": await self._generate_learning_insights(user_id, daily_data)
                }

                return analytics

        except Exception as e:
            logger.error(f"❌ 获取学习分析失败: {e}")
            return {}

    async def _generate_learning_insights(self, user_id: str, daily_data: List) -> List[str]:
        """生成学习洞察"""
        insights = []

        try:
            if not daily_data:
                insights.append("开始您的学习之旅吧！")
                return insights

            # 学习时间分析
            total_time = sum(row[1] for row in daily_data)
            avg_daily_time = total_time / len(daily_data)

            if avg_daily_time < 30:
                insights.append("建议每天至少学习30分钟以获得更好的效果")
            elif avg_daily_time > 120:
                insights.append("您的学习时间很充足，注意劳逸结合")

            # 准确率分析
            accuracies = [row[3] for row in daily_data if row[3] > 0]
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                if avg_accuracy > 0.9:
                    insights.append("您的准确率很高，可以尝试更有挑战性的内容")
                elif avg_accuracy < 0.7:
                    insights.append("建议多进行基础练习以提高准确率")

            # 学习连续性分析
            consecutive_days = 0
            for i in range(len(daily_data) - 1, -1, -1):
                if daily_data[i][1] > 0:  # 有学习时间
                    consecutive_days += 1
                else:
                    break

            if consecutive_days >= 7:
                insights.append(f"太棒了！您已经连续学习了{consecutive_days}天")
            elif consecutive_days == 0:
                insights.append("保持学习的连续性很重要，今天就开始吧！")

            return insights

        except Exception as e:
            logger.error(f"❌ 生成学习洞察失败: {e}")
            return ["继续努力学习！"]
