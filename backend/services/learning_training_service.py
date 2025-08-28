"""
手语学习训练服务
Sign Language Learning Training Service
提供系统化的手语学习功能，包括课程管理、进度跟踪、成就系统等
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

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    """难度等级"""
    BEGINNER = "beginner"      # 初级
    INTERMEDIATE = "intermediate"  # 中级
    ADVANCED = "advanced"      # 高级

class LessonType(Enum):
    """课程类型"""
    VIDEO_DEMO = "video_demo"     # 视频演示
    INTERACTIVE = "interactive"   # 互动练习
    TEST = "test"                 # 测试评估
    GAME = "game"                 # 游戏化学习

class AchievementType(Enum):
    """成就类型"""
    COMPLETION = "completion"     # 完成类
    STREAK = "streak"            # 连续类
    SPEED = "speed"              # 速度类
    MASTERY = "mastery"          # 掌握类

@dataclass
class LearningModule:
    """学习模块"""
    id: str
    title: str
    description: str
    level: DifficultyLevel
    total_lessons: int
    completed_lessons: int = 0
    progress: float = 0.0
    skills: List[str] = None
    estimated_time: str = ""
    prerequisite_modules: List[str] = None
    locked: bool = False
    rating: float = 0.0
    reviews: int = 0
    
    def __post_init__(self):
        if self.skills is None:
            self.skills = []
        if self.prerequisite_modules is None:
            self.prerequisite_modules = []
    
    def update_progress(self):
        """更新进度"""
        if self.total_lessons > 0:
            self.progress = (self.completed_lessons / self.total_lessons) * 100

@dataclass
class Lesson:
    """课程"""
    id: str
    module_id: str
    title: str
    description: str
    type: LessonType
    content: Dict[str, Any]
    duration_minutes: int
    difficulty: DifficultyLevel
    prerequisites: List[str] = None
    learning_objectives: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.learning_objectives is None:
            self.learning_objectives = []

@dataclass
class UserProgress:
    """用户进度"""
    user_id: str
    total_learning_time: int = 0  # 分钟
    completed_lessons: int = 0
    current_streak: int = 0
    level: int = 1
    total_xp: int = 0
    last_activity: datetime = None
    
    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()

@dataclass
class Achievement:
    """成就"""
    id: str
    title: str
    description: str
    type: AchievementType
    requirement: Dict[str, Any]
    xp_reward: int
    unlocked: bool = False
    progress: int = 0
    unlocked_at: Optional[datetime] = None

class LearningTrainingService:
    """手语学习训练服务"""
    
    def __init__(self, db_path: str = "data/learning.db"):
        self.db_path = db_path
        self.modules: Dict[str, LearningModule] = {}
        self.lessons: Dict[str, Lesson] = {}
        self.achievements: Dict[str, Achievement] = {}
        self.user_progress: Dict[str, UserProgress] = {}
        
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 加载默认数据
        self._load_default_data()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 用户进度表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_progress (
                        user_id TEXT PRIMARY KEY,
                        total_learning_time INTEGER DEFAULT 0,
                        completed_lessons INTEGER DEFAULT 0,
                        current_streak INTEGER DEFAULT 0,
                        level INTEGER DEFAULT 1,
                        total_xp INTEGER DEFAULT 0,
                        last_activity TEXT
                    )
                ''')
                
                # 课程完成记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS lesson_completions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        lesson_id TEXT,
                        module_id TEXT,
                        completed_at TEXT,
                        score REAL,
                        time_spent INTEGER,
                        FOREIGN KEY (user_id) REFERENCES user_progress (user_id)
                    )
                ''')
                
                # 成就解锁记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS achievement_unlocks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        achievement_id TEXT,
                        unlocked_at TEXT,
                        FOREIGN KEY (user_id) REFERENCES user_progress (user_id)
                    )
                ''')
                
                # 每日任务记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        task_type TEXT,
                        completed_at TEXT,
                        xp_earned INTEGER,
                        FOREIGN KEY (user_id) REFERENCES user_progress (user_id)
                    )
                ''')
                
                conn.commit()
                logger.info("学习训练数据库初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
    
    def _load_default_data(self):
        """加载默认学习数据"""
        try:
            # 初始化学习模块
            self._init_learning_modules()
            
            # 初始化课程
            self._init_lessons()
            
            # 初始化成就系统
            self._init_achievements()
            
            logger.info("默认学习数据加载完成")
            
        except Exception as e:
            logger.error(f"默认数据加载失败: {e}")
    
    def _init_learning_modules(self):
        """初始化学习模块"""
        modules_data = [
            {
                "id": "basic-signs",
                "title": "基础手语",
                "description": "学习最常用的手语词汇和基本表达",
                "level": DifficultyLevel.BEGINNER,
                "total_lessons": 12,
                "skills": ["基础词汇", "日常用语", "问候语"],
                "estimated_time": "2-3小时",
                "rating": 4.8,
                "reviews": 156
            },
            {
                "id": "numbers-time",
                "title": "数字与时间",
                "description": "掌握数字、时间和日期的手语表达",
                "level": DifficultyLevel.BEGINNER,
                "total_lessons": 8,
                "skills": ["数字0-100", "时间表达", "日期表达"],
                "estimated_time": "1-2小时",
                "rating": 4.6,
                "reviews": 89
            },
            {
                "id": "family-relations",
                "title": "家庭关系",
                "description": "学习家庭成员和人际关系相关手语",
                "level": DifficultyLevel.INTERMEDIATE,
                "total_lessons": 10,
                "skills": ["家庭成员", "关系称谓", "情感表达"],
                "estimated_time": "2-3小时",
                "prerequisite_modules": ["basic-signs"],
                "rating": 4.7,
                "reviews": 124
            },
            {
                "id": "daily-activities",
                "title": "日常活动",
                "description": "学习日常生活中常见活动的手语表达",
                "level": DifficultyLevel.INTERMEDIATE,
                "total_lessons": 15,
                "skills": ["生活用语", "动作表达", "场所名称"],
                "estimated_time": "3-4小时",
                "prerequisite_modules": ["basic-signs", "numbers-time"],
                "rating": 4.5,
                "reviews": 98
            },
            {
                "id": "advanced-grammar",
                "title": "高级语法",
                "description": "掌握复杂的手语语法结构和表达技巧",
                "level": DifficultyLevel.ADVANCED,
                "total_lessons": 15,
                "skills": ["语法结构", "时态表达", "复合句型"],
                "estimated_time": "4-6小时",
                "prerequisite_modules": ["family-relations", "daily-activities"],
                "locked": True,
                "rating": 4.9,
                "reviews": 67
            },
            {
                "id": "professional-signs",
                "title": "职业手语",
                "description": "学习不同职业和工作场景的专业手语",
                "level": DifficultyLevel.ADVANCED,
                "total_lessons": 20,
                "skills": ["职业名称", "工作用语", "商务交流"],
                "estimated_time": "5-7小时",
                "prerequisite_modules": ["daily-activities"],
                "locked": True,
                "rating": 4.6,
                "reviews": 45
            }
        ]
        
        for module_data in modules_data:
            module = LearningModule(**module_data)
            self.modules[module.id] = module
    
    def _init_lessons(self):
        """初始化课程"""
        lessons_data = [
            # 基础手语课程
            {
                "id": "basic-001",
                "module_id": "basic-signs",
                "title": "问候语手语",
                "description": "学习你好、再见、谢谢等基本问候语",
                "type": LessonType.VIDEO_DEMO,
                "duration_minutes": 15,
                "difficulty": DifficultyLevel.BEGINNER,
                "learning_objectives": ["掌握基本问候语", "学会正确手型", "理解手语礼仪"],
                "content": {
                    "signs": ["你好", "再见", "谢谢", "对不起", "请"],
                    "video_urls": [],
                    "practice_exercises": []
                }
            },
            {
                "id": "basic-002",
                "module_id": "basic-signs",
                "title": "自我介绍",
                "description": "学习如何用手语进行简单的自我介绍",
                "type": LessonType.INTERACTIVE,
                "duration_minutes": 20,
                "difficulty": DifficultyLevel.BEGINNER,
                "prerequisites": ["basic-001"],
                "learning_objectives": ["掌握自我介绍手语", "学会表达个人信息", "练习流畅表达"],
                "content": {
                    "signs": ["我", "叫", "名字", "年龄", "来自"],
                    "practice_scenarios": ["介绍姓名", "介绍年龄", "介绍家乡"]
                }
            },
            # 数字与时间课程
            {
                "id": "numbers-001",
                "module_id": "numbers-time",
                "title": "数字0-10",
                "description": "学习基础数字0到10的手语表达",
                "type": LessonType.VIDEO_DEMO,
                "duration_minutes": 10,
                "difficulty": DifficultyLevel.BEGINNER,
                "learning_objectives": ["掌握0-10数字手语", "学会正确手型", "练习数字组合"],
                "content": {
                    "signs": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "practice_sets": ["单个数字", "数字序列", "简单计算"]
                }
            },
            {
                "id": "numbers-002",
                "module_id": "numbers-time",
                "title": "时间表达",
                "description": "学习时间相关的手语表达方式",
                "type": LessonType.INTERACTIVE,
                "duration_minutes": 25,
                "difficulty": DifficultyLevel.BEGINNER,
                "prerequisites": ["numbers-001"],
                "learning_objectives": ["掌握时间手语", "学会询问时间", "练习时间表达"],
                "content": {
                    "signs": ["几点", "小时", "分钟", "上午", "下午", "晚上"],
                    "time_expressions": ["整点表达", "半点表达", "具体时间"]
                }
            },
            # 测试课程
            {
                "id": "test-001",
                "module_id": "basic-signs",
                "title": "基础手语测试",
                "description": "测试基础手语学习成果",
                "type": LessonType.TEST,
                "duration_minutes": 30,
                "difficulty": DifficultyLevel.BEGINNER,
                "prerequisites": ["basic-001", "basic-002"],
                "learning_objectives": ["评估学习成果", "识别薄弱环节", "巩固知识点"],
                "content": {
                    "test_types": ["手语识别", "手语表达", "综合应用"],
                    "question_count": 20,
                    "passing_score": 70
                }
            }
        ]
        
        for lesson_data in lessons_data:
            lesson = Lesson(**lesson_data)
            self.lessons[lesson.id] = lesson
    
    def _init_achievements(self):
        """初始化成就系统"""
        achievements_data = [
            {
                "id": "first-lesson",
                "title": "初学者",
                "description": "完成第一节课程",
                "type": AchievementType.COMPLETION,
                "requirement": {"lessons_completed": 1},
                "xp_reward": 50
            },
            {
                "id": "week-streak",
                "title": "坚持一周",
                "description": "连续学习7天",
                "type": AchievementType.STREAK,
                "requirement": {"consecutive_days": 7},
                "xp_reward": 200
            },
            {
                "id": "speed-learner",
                "title": "学习达人",
                "description": "在一天内完成5节课程",
                "type": AchievementType.SPEED,
                "requirement": {"lessons_per_day": 5},
                "xp_reward": 150
            },
            {
                "id": "master-basic",
                "title": "基础大师",
                "description": "完成所有基础课程",
                "type": AchievementType.MASTERY,
                "requirement": {"module_completed": "basic-signs"},
                "xp_reward": 300
            },
            {
                "id": "numbers-expert",
                "title": "数字专家",
                "description": "完成数字与时间模块",
                "type": AchievementType.MASTERY,
                "requirement": {"module_completed": "numbers-time"},
                "xp_reward": 250
            },
            {
                "id": "perfect-score",
                "title": "满分达人",
                "description": "在测试中获得满分",
                "type": AchievementType.MASTERY,
                "requirement": {"perfect_test_score": True},
                "xp_reward": 400
            },
            {
                "id": "dedicated-learner",
                "title": "专注学习者",
                "description": "累计学习时间达到10小时",
                "type": AchievementType.COMPLETION,
                "requirement": {"total_learning_minutes": 600},
                "xp_reward": 500
            }
        ]
        
        for achievement_data in achievements_data:
            achievement = Achievement(**achievement_data)
            self.achievements[achievement.id] = achievement
    
    async def get_user_progress(self, user_id: str) -> UserProgress:
        """获取用户进度"""
        if user_id not in self.user_progress:
            # 从数据库加载
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT * FROM user_progress WHERE user_id = ?",
                        (user_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        progress = UserProgress(
                            user_id=row[0],
                            total_learning_time=row[1],
                            completed_lessons=row[2],
                            current_streak=row[3],
                            level=row[4],
                            total_xp=row[5],
                            last_activity=datetime.fromisoformat(row[6]) if row[6] else datetime.now()
                        )
                    else:
                        # 创建新用户
                        progress = UserProgress(user_id=user_id)
                        await self._save_user_progress(progress)
                    
                    self.user_progress[user_id] = progress
                    
            except Exception as e:
                logger.error(f"加载用户进度失败: {e}")
                self.user_progress[user_id] = UserProgress(user_id=user_id)
        
        return self.user_progress[user_id]
    
    async def _save_user_progress(self, progress: UserProgress):
        """保存用户进度"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_progress 
                    (user_id, total_learning_time, completed_lessons, current_streak, level, total_xp, last_activity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    progress.user_id,
                    progress.total_learning_time,
                    progress.completed_lessons,
                    progress.current_streak,
                    progress.level,
                    progress.total_xp,
                    progress.last_activity.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存用户进度失败: {e}")
    
    async def get_learning_modules(self, user_id: str) -> List[Dict[str, Any]]:
        """获取学习模块列表"""
        user_progress = await self.get_user_progress(user_id)
        modules_list = []
        
        for module in self.modules.values():
            # 检查模块是否解锁
            module_unlocked = await self._is_module_unlocked(user_id, module.id)
            
            # 计算模块进度
            completed_lessons = await self._get_completed_lessons_count(user_id, module.id)
            module.completed_lessons = completed_lessons
            module.update_progress()
            module.locked = not module_unlocked
            
            modules_list.append(asdict(module))
        
        return modules_list
    
    async def _is_module_unlocked(self, user_id: str, module_id: str) -> bool:
        """检查模块是否解锁"""
        module = self.modules.get(module_id)
        if not module or not module.prerequisite_modules:
            return True
        
        # 检查前置模块是否完成
        for prereq_id in module.prerequisite_modules:
            prereq_module = self.modules.get(prereq_id)
            if prereq_module:
                completed_lessons = await self._get_completed_lessons_count(user_id, prereq_id)
                if completed_lessons < prereq_module.total_lessons:
                    return False
        
        return True
    
    async def _get_completed_lessons_count(self, user_id: str, module_id: str) -> int:
        """获取已完成课程数量"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM lesson_completions WHERE user_id = ? AND module_id = ?",
                    (user_id, module_id)
                )
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"获取完成课程数量失败: {e}")
            return 0
    
    async def get_module_lessons(self, module_id: str) -> List[Dict[str, Any]]:
        """获取模块的课程列表"""
        lessons_list = []
        
        for lesson in self.lessons.values():
            if lesson.module_id == module_id:
                lessons_list.append(asdict(lesson))
        
        return sorted(lessons_list, key=lambda x: x['id'])
    
    async def complete_lesson(self, user_id: str, lesson_id: str, score: float = 100.0, time_spent: int = 0) -> Dict[str, Any]:
        """完成课程"""
        try:
            lesson = self.lessons.get(lesson_id)
            if not lesson:
                return {"success": False, "message": "课程不存在"}
            
            # 检查是否已完成
            if await self._is_lesson_completed(user_id, lesson_id):
                return {"success": False, "message": "课程已完成"}
            
            # 记录完成
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO lesson_completions 
                    (user_id, lesson_id, module_id, completed_at, score, time_spent)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    lesson_id,
                    lesson.module_id,
                    datetime.now().isoformat(),
                    score,
                    time_spent
                ))
                conn.commit()
            
            # 更新用户进度
            user_progress = await self.get_user_progress(user_id)
            user_progress.completed_lessons += 1
            user_progress.total_learning_time += lesson.duration_minutes
            user_progress.last_activity = datetime.now()
            
            # 计算经验值奖励
            xp_reward = self._calculate_lesson_xp(lesson, score)
            user_progress.total_xp += xp_reward
            
            # 更新等级
            new_level = self._calculate_level(user_progress.total_xp)
            level_up = new_level > user_progress.level
            user_progress.level = new_level
            
            # 更新连续天数
            await self._update_streak(user_progress)
            
            await self._save_user_progress(user_progress)
            
            # 检查成就
            unlocked_achievements = await self._check_achievements(user_id)
            
            return {
                "success": True,
                "xp_earned": xp_reward,
                "level_up": level_up,
                "new_level": user_progress.level,
                "unlocked_achievements": unlocked_achievements
            }
            
        except Exception as e:
            logger.error(f"完成课程失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def _is_lesson_completed(self, user_id: str, lesson_id: str) -> bool:
        """检查课程是否已完成"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM lesson_completions WHERE user_id = ? AND lesson_id = ?",
                    (user_id, lesson_id)
                )
                return cursor.fetchone()[0] > 0
                
        except Exception as e:
            logger.error(f"检查课程完成状态失败: {e}")
            return False
    
    def _calculate_lesson_xp(self, lesson: Lesson, score: float) -> int:
        """计算课程经验值"""
        base_xp = {
            DifficultyLevel.BEGINNER: 30,
            DifficultyLevel.INTERMEDIATE: 50,
            DifficultyLevel.ADVANCED: 80
        }
        
        lesson_xp = base_xp.get(lesson.difficulty, 30)
        
        # 根据评分调整
        score_multiplier = score / 100.0
        final_xp = int(lesson_xp * score_multiplier)
        
        # 额外奖励
        if score >= 90:
            final_xp += 10  # 优秀奖励
        
        return final_xp
    
    def _calculate_level(self, total_xp: int) -> int:
        """计算用户等级"""
        # 每级所需经验值递增
        level = 1
        required_xp = 0
        
        while total_xp >= required_xp:
            level += 1
            required_xp += level * 100  # 每级需要更多经验值
        
        return level - 1
    
    async def _update_streak(self, progress: UserProgress):
        """更新连续学习天数"""
        now = datetime.now()
        last_activity = progress.last_activity
        
        if last_activity:
            days_diff = (now.date() - last_activity.date()).days
            
            if days_diff == 1:
                # 连续学习
                progress.current_streak += 1
            elif days_diff > 1:
                # 中断了连续学习
                progress.current_streak = 1
            # days_diff == 0 表示同一天，不更新streak
        else:
            progress.current_streak = 1
    
    async def _check_achievements(self, user_id: str) -> List[Dict[str, Any]]:
        """检查并解锁成就"""
        user_progress = await self.get_user_progress(user_id)
        unlocked_achievements = []
        
        for achievement in self.achievements.values():
            if await self._is_achievement_unlocked(user_id, achievement.id):
                continue
            
            if await self._check_achievement_requirement(user_id, achievement, user_progress):
                # 解锁成就
                await self._unlock_achievement(user_id, achievement.id)
                unlocked_achievements.append({
                    "id": achievement.id,
                    "title": achievement.title,
                    "description": achievement.description,
                    "xp_reward": achievement.xp_reward
                })
                
                # 添加经验值奖励
                user_progress.total_xp += achievement.xp_reward
                await self._save_user_progress(user_progress)
        
        return unlocked_achievements
    
    async def _is_achievement_unlocked(self, user_id: str, achievement_id: str) -> bool:
        """检查成就是否已解锁"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM achievement_unlocks WHERE user_id = ? AND achievement_id = ?",
                    (user_id, achievement_id)
                )
                return cursor.fetchone()[0] > 0
                
        except Exception as e:
            logger.error(f"检查成就解锁状态失败: {e}")
            return False
    
    async def _check_achievement_requirement(self, user_id: str, achievement: Achievement, progress: UserProgress) -> bool:
        """检查成就要求是否满足"""
        req = achievement.requirement
        
        if achievement.type == AchievementType.COMPLETION:
            if "lessons_completed" in req:
                return progress.completed_lessons >= req["lessons_completed"]
            elif "total_learning_minutes" in req:
                return progress.total_learning_time >= req["total_learning_minutes"]
                
        elif achievement.type == AchievementType.STREAK:
            if "consecutive_days" in req:
                return progress.current_streak >= req["consecutive_days"]
                
        elif achievement.type == AchievementType.MASTERY:
            if "module_completed" in req:
                module_id = req["module_completed"]
                module = self.modules.get(module_id)
                if module:
                    completed_lessons = await self._get_completed_lessons_count(user_id, module_id)
                    return completed_lessons >= module.total_lessons
                    
        elif achievement.type == AchievementType.SPEED:
            if "lessons_per_day" in req:
                # 检查今天完成的课程数量
                today_lessons = await self._get_today_completed_lessons(user_id)
                return today_lessons >= req["lessons_per_day"]
        
        return False
    
    async def _get_today_completed_lessons(self, user_id: str) -> int:
        """获取今天完成的课程数量"""
        try:
            today = datetime.now().date()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM lesson_completions 
                    WHERE user_id = ? AND DATE(completed_at) = ?
                ''', (user_id, today.isoformat()))
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"获取今日完成课程数量失败: {e}")
            return 0
    
    async def _unlock_achievement(self, user_id: str, achievement_id: str):
        """解锁成就"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO achievement_unlocks (user_id, achievement_id, unlocked_at)
                    VALUES (?, ?, ?)
                ''', (user_id, achievement_id, datetime.now().isoformat()))
                conn.commit()
                
        except Exception as e:
            logger.error(f"解锁成就失败: {e}")
    
    async def get_user_achievements(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户成就"""
        achievements_list = []
        
        for achievement in self.achievements.values():
            unlocked = await self._is_achievement_unlocked(user_id, achievement.id)
            
            achievement_data = asdict(achievement)
            achievement_data["unlocked"] = unlocked
            
            if not unlocked:
                # 计算进度
                progress = await self._calculate_achievement_progress(user_id, achievement)
                achievement_data["progress"] = progress
            
            achievements_list.append(achievement_data)
        
        return achievements_list
    
    async def _calculate_achievement_progress(self, user_id: str, achievement: Achievement) -> int:
        """计算成就进度"""
        user_progress = await self.get_user_progress(user_id)
        req = achievement.requirement
        
        if achievement.type == AchievementType.COMPLETION:
            if "lessons_completed" in req:
                return min(user_progress.completed_lessons, req["lessons_completed"])
            elif "total_learning_minutes" in req:
                return min(user_progress.total_learning_time, req["total_learning_minutes"])
                
        elif achievement.type == AchievementType.STREAK:
            if "consecutive_days" in req:
                return min(user_progress.current_streak, req["consecutive_days"])
                
        elif achievement.type == AchievementType.MASTERY:
            if "module_completed" in req:
                module_id = req["module_completed"]
                completed_lessons = await self._get_completed_lessons_count(user_id, module_id)
                module = self.modules.get(module_id)
                if module:
                    return min(completed_lessons, module.total_lessons)
                    
        elif achievement.type == AchievementType.SPEED:
            if "lessons_per_day" in req:
                today_lessons = await self._get_today_completed_lessons(user_id)
                return min(today_lessons, req["lessons_per_day"])
        
        return 0
    
    async def get_daily_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        """获取每日任务"""
        tasks = [
            {
                "task": "完成一节基础课程",
                "type": "lesson_completion",
                "target": 1,
                "xp": 50,
                "completed": False
            },
            {
                "task": "练习10个新词汇",
                "type": "vocabulary_practice",
                "target": 10,
                "xp": 30,
                "completed": False
            },
            {
                "task": "通过一次手语测试",
                "type": "test_completion",
                "target": 1,
                "xp": 80,
                "completed": False
            },
            {
                "task": "观看3个演示视频",
                "type": "video_watch",
                "target": 3,
                "xp": 40,
                "completed": False
            }
        ]
        
        # 检查任务完成状态
        for task in tasks:
            task["completed"] = await self._is_daily_task_completed(user_id, task["type"])
        
        return tasks
    
    async def _is_daily_task_completed(self, user_id: str, task_type: str) -> bool:
        """检查每日任务是否完成"""
        try:
            today = datetime.now().date()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM daily_tasks 
                    WHERE user_id = ? AND task_type = ? AND DATE(completed_at) = ?
                ''', (user_id, task_type, today.isoformat()))
                return cursor.fetchone()[0] > 0
                
        except Exception as e:
            logger.error(f"检查每日任务完成状态失败: {e}")
            return False
    
    async def complete_daily_task(self, user_id: str, task_type: str) -> Dict[str, Any]:
        """完成每日任务"""
        try:
            if await self._is_daily_task_completed(user_id, task_type):
                return {"success": False, "message": "任务已完成"}
            
            # 记录任务完成
            task_xp = {
                "lesson_completion": 50,
                "vocabulary_practice": 30,
                "test_completion": 80,
                "video_watch": 40
            }.get(task_type, 20)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO daily_tasks (user_id, task_type, completed_at, xp_earned)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, task_type, datetime.now().isoformat(), task_xp))
                conn.commit()
            
            # 更新用户经验值
            user_progress = await self.get_user_progress(user_id)
            user_progress.total_xp += task_xp
            await self._save_user_progress(user_progress)
            
            return {
                "success": True,
                "xp_earned": task_xp,
                "message": "任务完成！"
            }
            
        except Exception as e:
            logger.error(f"完成每日任务失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def get_learning_stats(self, user_id: str) -> Dict[str, Any]:
        """获取学习统计"""
        user_progress = await self.get_user_progress(user_id)
        
        # 计算下一级经验值
        current_level_xp = (user_progress.level - 1) * user_progress.level * 50
        next_level_xp = user_progress.level * (user_progress.level + 1) * 50
        
        return {
            "user_progress": asdict(user_progress),
            "next_level_xp": next_level_xp,
            "total_modules": len(self.modules),
            "total_lessons": len(self.lessons),
            "total_achievements": len(self.achievements)
        }
    
    async def search_content(self, query: str, content_type: str = "all") -> Dict[str, Any]:
        """搜索学习内容"""
        results = {
            "modules": [],
            "lessons": [],
            "achievements": []
        }
        
        query_lower = query.lower()
        
        if content_type in ["all", "modules"]:
            for module in self.modules.values():
                if (query_lower in module.title.lower() or 
                    query_lower in module.description.lower() or
                    any(query_lower in skill.lower() for skill in module.skills)):
                    results["modules"].append(asdict(module))
        
        if content_type in ["all", "lessons"]:
            for lesson in self.lessons.values():
                if (query_lower in lesson.title.lower() or 
                    query_lower in lesson.description.lower()):
                    results["lessons"].append(asdict(lesson))
        
        if content_type in ["all", "achievements"]:
            for achievement in self.achievements.values():
                if (query_lower in achievement.title.lower() or 
                    query_lower in achievement.description.lower()):
                    results["achievements"].append(asdict(achievement))
        
        return results
