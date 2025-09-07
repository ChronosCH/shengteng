"""
成就系统服务
Achievement System Service
提供完整的成就系统，包括成就定义、进度跟踪、奖励发放等
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import sqlite3
import os

logger = logging.getLogger(__name__)

class AchievementCategory(Enum):
    """成就类别"""
    LEARNING = "learning"        # 学习类
    PRACTICE = "practice"        # 练习类
    SOCIAL = "social"           # 社交类
    MILESTONE = "milestone"      # 里程碑类
    SPECIAL = "special"         # 特殊类

class AchievementRarity(Enum):
    """成就稀有度"""
    COMMON = "common"           # 普通
    UNCOMMON = "uncommon"       # 不常见
    RARE = "rare"              # 稀有
    EPIC = "epic"              # 史诗
    LEGENDARY = "legendary"     # 传说

@dataclass
class Achievement:
    """成就定义"""
    achievement_id: str
    title: str
    description: str
    category: AchievementCategory
    rarity: AchievementRarity
    icon: str
    xp_reward: int
    badge_color: str = "#4CAF50"
    requirements: Dict[str, Any] = None
    hidden: bool = False
    repeatable: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class UserAchievement:
    """用户成就"""
    user_id: str
    achievement_id: str
    progress: float = 0.0
    is_unlocked: bool = False
    unlocked_at: Optional[datetime] = None
    current_count: int = 0
    target_count: int = 1
    
    def update_progress(self):
        """更新进度"""
        if self.target_count > 0:
            self.progress = min(1.0, self.current_count / self.target_count)
            if self.progress >= 1.0 and not self.is_unlocked:
                self.is_unlocked = True
                self.unlocked_at = datetime.now()

class AchievementService:
    """成就系统服务"""
    
    def __init__(self, db_path: str = "data/achievements.db"):
        self.db_path = db_path
        self.achievements: Dict[str, Achievement] = {}
        self.user_achievements: Dict[str, Dict[str, UserAchievement]] = {}  # user_id -> achievement_id -> user_achievement
        self.achievement_handlers: Dict[str, Callable] = {}
        
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 加载数据
        self._load_data()
        
        # 创建默认成就
        self._create_default_achievements()
        
        # 注册成就检查器
        self._register_achievement_handlers()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 成就定义表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS achievements (
                        achievement_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        category TEXT,
                        rarity TEXT,
                        icon TEXT,
                        xp_reward INTEGER,
                        badge_color TEXT,
                        requirements TEXT,
                        hidden BOOLEAN,
                        repeatable BOOLEAN,
                        created_at TEXT
                    )
                ''')
                
                # 用户成就表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_achievements (
                        user_id TEXT,
                        achievement_id TEXT,
                        progress REAL,
                        is_unlocked BOOLEAN,
                        unlocked_at TEXT,
                        current_count INTEGER,
                        target_count INTEGER,
                        PRIMARY KEY (user_id, achievement_id),
                        FOREIGN KEY (achievement_id) REFERENCES achievements (achievement_id)
                    )
                ''')
                
                conn.commit()
                logger.info("✅ 成就系统数据库初始化完成")
                
        except Exception as e:
            logger.error(f"❌ 成就系统数据库初始化失败: {e}")
            raise
    
    def _load_data(self):
        """加载数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 加载成就定义
                cursor.execute("SELECT * FROM achievements")
                for row in cursor.fetchall():
                    achievement = Achievement(
                        achievement_id=row[0],
                        title=row[1],
                        description=row[2],
                        category=AchievementCategory(row[3]),
                        rarity=AchievementRarity(row[4]),
                        icon=row[5],
                        xp_reward=row[6],
                        badge_color=row[7],
                        requirements=json.loads(row[8]) if row[8] else {},
                        hidden=bool(row[9]),
                        repeatable=bool(row[10]),
                        created_at=datetime.fromisoformat(row[11])
                    )
                    self.achievements[achievement.achievement_id] = achievement
                
                # 加载用户成就
                cursor.execute("SELECT * FROM user_achievements")
                for row in cursor.fetchall():
                    user_achievement = UserAchievement(
                        user_id=row[0],
                        achievement_id=row[1],
                        progress=row[2],
                        is_unlocked=bool(row[3]),
                        unlocked_at=datetime.fromisoformat(row[4]) if row[4] else None,
                        current_count=row[5],
                        target_count=row[6]
                    )
                    
                    if user_achievement.user_id not in self.user_achievements:
                        self.user_achievements[user_achievement.user_id] = {}
                    self.user_achievements[user_achievement.user_id][user_achievement.achievement_id] = user_achievement
                
                logger.info(f"✅ 加载了 {len(self.achievements)} 个成就定义")
                
        except Exception as e:
            logger.error(f"❌ 成就数据加载失败: {e}")
    
    def _create_default_achievements(self):
        """创建默认成就"""
        if self.achievements:
            return  # 已有成就，不创建默认成就
        
        default_achievements = [
            # 学习类成就
            Achievement(
                achievement_id="first_lesson",
                title="初学者",
                description="完成第一个课程",
                category=AchievementCategory.LEARNING,
                rarity=AchievementRarity.COMMON,
                icon="🎓",
                xp_reward=50,
                requirements={"lessons_completed": 1}
            ),
            Achievement(
                achievement_id="dedicated_learner",
                title="专注学习者",
                description="连续学习7天",
                category=AchievementCategory.LEARNING,
                rarity=AchievementRarity.UNCOMMON,
                icon="📚",
                xp_reward=200,
                requirements={"consecutive_days": 7}
            ),
            Achievement(
                achievement_id="course_master",
                title="课程大师",
                description="完成10个课程",
                category=AchievementCategory.LEARNING,
                rarity=AchievementRarity.RARE,
                icon="🏆",
                xp_reward=500,
                requirements={"courses_completed": 10}
            ),
            
            # 练习类成就
            Achievement(
                achievement_id="practice_makes_perfect",
                title="熟能生巧",
                description="完成100次练习",
                category=AchievementCategory.PRACTICE,
                rarity=AchievementRarity.UNCOMMON,
                icon="💪",
                xp_reward=300,
                requirements={"practices_completed": 100}
            ),
            Achievement(
                achievement_id="accuracy_expert",
                title="精准专家",
                description="在练习中达到95%以上准确率",
                category=AchievementCategory.PRACTICE,
                rarity=AchievementRarity.RARE,
                icon="🎯",
                xp_reward=400,
                requirements={"accuracy_rate": 0.95}
            ),
            
            # 里程碑类成就
            Achievement(
                achievement_id="time_master",
                title="时间大师",
                description="累计学习100小时",
                category=AchievementCategory.MILESTONE,
                rarity=AchievementRarity.EPIC,
                icon="⏰",
                xp_reward=1000,
                requirements={"total_hours": 100}
            ),
            Achievement(
                achievement_id="sign_language_expert",
                title="手语专家",
                description="掌握1000个手语词汇",
                category=AchievementCategory.MILESTONE,
                rarity=AchievementRarity.LEGENDARY,
                icon="🌟",
                xp_reward=2000,
                requirements={"vocabulary_size": 1000}
            ),
            
            # 特殊成就
            Achievement(
                achievement_id="early_bird",
                title="早起鸟儿",
                description="在早上6点前开始学习",
                category=AchievementCategory.SPECIAL,
                rarity=AchievementRarity.UNCOMMON,
                icon="🐦",
                xp_reward=150,
                requirements={"early_morning_sessions": 1}
            ),
            Achievement(
                achievement_id="night_owl",
                title="夜猫子",
                description="在晚上10点后学习",
                category=AchievementCategory.SPECIAL,
                rarity=AchievementRarity.UNCOMMON,
                icon="🦉",
                xp_reward=150,
                requirements={"late_night_sessions": 1}
            )
        ]
        
        try:
            for achievement in default_achievements:
                self.achievements[achievement.achievement_id] = achievement
                self._save_achievement(achievement)
            
            logger.info("✅ 创建了默认成就系统")
            
        except Exception as e:
            logger.error(f"❌ 创建默认成就失败: {e}")
    
    def _register_achievement_handlers(self):
        """注册成就检查器"""
        self.achievement_handlers = {
            "lesson_completed": self._check_lesson_achievements,
            "course_completed": self._check_course_achievements,
            "practice_completed": self._check_practice_achievements,
            "daily_goal_reached": self._check_daily_achievements,
            "accuracy_achieved": self._check_accuracy_achievements,
            "time_milestone": self._check_time_achievements,
        }
    
    async def trigger_achievement_check(self, user_id: str, event_type: str, event_data: Dict[str, Any]) -> List[str]:
        """触发成就检查"""
        try:
            unlocked_achievements = []
            
            # 确保用户有成就记录
            await self._ensure_user_achievements(user_id)
            
            # 调用对应的检查器
            if event_type in self.achievement_handlers:
                handler = self.achievement_handlers[event_type]
                new_unlocks = await handler(user_id, event_data)
                unlocked_achievements.extend(new_unlocks)
            
            # 检查通用成就
            general_unlocks = await self._check_general_achievements(user_id, event_data)
            unlocked_achievements.extend(general_unlocks)
            
            return unlocked_achievements
            
        except Exception as e:
            logger.error(f"❌ 成就检查失败: {e}")
            return []
    
    async def _ensure_user_achievements(self, user_id: str):
        """确保用户有所有成就的记录"""
        if user_id not in self.user_achievements:
            self.user_achievements[user_id] = {}
        
        for achievement_id in self.achievements:
            if achievement_id not in self.user_achievements[user_id]:
                achievement = self.achievements[achievement_id]
                target_count = achievement.requirements.get("target_count", 1)
                
                user_achievement = UserAchievement(
                    user_id=user_id,
                    achievement_id=achievement_id,
                    target_count=target_count
                )
                
                self.user_achievements[user_id][achievement_id] = user_achievement
                await self._save_user_achievement(user_achievement)
    
    async def _check_lesson_achievements(self, user_id: str, event_data: Dict[str, Any]) -> List[str]:
        """检查课程相关成就"""
        unlocked = []
        
        try:
            lessons_completed = event_data.get("total_lessons_completed", 0)
            
            # 检查第一课成就
            if lessons_completed >= 1:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "first_lesson", lessons_completed
                ))
            
            return unlocked
            
        except Exception as e:
            logger.error(f"❌ 检查课程成就失败: {e}")
            return []
    
    async def _check_course_achievements(self, user_id: str, event_data: Dict[str, Any]) -> List[str]:
        """检查课程完成成就"""
        unlocked = []
        
        try:
            courses_completed = event_data.get("courses_completed", 0)
            
            # 检查课程大师成就
            if courses_completed >= 10:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "course_master", courses_completed
                ))
            
            return unlocked
            
        except Exception as e:
            logger.error(f"❌ 检查课程完成成就失败: {e}")
            return []
    
    async def _check_practice_achievements(self, user_id: str, event_data: Dict[str, Any]) -> List[str]:
        """检查练习成就"""
        unlocked = []
        
        try:
            practices_completed = event_data.get("practices_completed", 0)
            
            # 检查练习成就
            if practices_completed >= 100:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "practice_makes_perfect", practices_completed
                ))
            
            return unlocked
            
        except Exception as e:
            logger.error(f"❌ 检查练习成就失败: {e}")
            return []
    
    async def _check_accuracy_achievements(self, user_id: str, event_data: Dict[str, Any]) -> List[str]:
        """检查准确率成就"""
        unlocked = []
        
        try:
            accuracy_rate = event_data.get("accuracy_rate", 0.0)
            
            # 检查精准专家成就
            if accuracy_rate >= 0.95:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "accuracy_expert", 1
                ))
            
            return unlocked
            
        except Exception as e:
            logger.error(f"❌ 检查准确率成就失败: {e}")
            return []
    
    async def _check_time_achievements(self, user_id: str, event_data: Dict[str, Any]) -> List[str]:
        """检查时间相关成就"""
        unlocked = []
        
        try:
            total_hours = event_data.get("total_hours", 0)
            
            # 检查时间大师成就
            if total_hours >= 100:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "time_master", total_hours
                ))
            
            return unlocked
            
        except Exception as e:
            logger.error(f"❌ 检查时间成就失败: {e}")
            return []
    
    async def _check_daily_achievements(self, user_id: str, event_data: Dict[str, Any]) -> List[str]:
        """检查每日成就"""
        unlocked = []
        
        try:
            consecutive_days = event_data.get("consecutive_days", 0)
            session_time = event_data.get("session_time", datetime.now())
            
            # 检查连续学习成就
            if consecutive_days >= 7:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "dedicated_learner", consecutive_days
                ))
            
            # 检查早起鸟儿成就
            if session_time.hour < 6:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "early_bird", 1
                ))
            
            # 检查夜猫子成就
            if session_time.hour >= 22:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "night_owl", 1
                ))
            
            return unlocked
            
        except Exception as e:
            logger.error(f"❌ 检查每日成就失败: {e}")
            return []
    
    async def _check_general_achievements(self, user_id: str, event_data: Dict[str, Any]) -> List[str]:
        """检查通用成就"""
        unlocked = []
        
        try:
            # 这里可以添加更多通用的成就检查逻辑
            vocabulary_size = event_data.get("vocabulary_size", 0)
            
            # 检查手语专家成就
            if vocabulary_size >= 1000:
                unlocked.extend(await self._update_achievement_progress(
                    user_id, "sign_language_expert", vocabulary_size
                ))
            
            return unlocked
            
        except Exception as e:
            logger.error(f"❌ 检查通用成就失败: {e}")
            return []
    
    async def _update_achievement_progress(self, user_id: str, achievement_id: str, current_value: int) -> List[str]:
        """更新成就进度"""
        try:
            if achievement_id not in self.achievements:
                return []
            
            user_achievement = self.user_achievements[user_id][achievement_id]
            
            if user_achievement.is_unlocked and not self.achievements[achievement_id].repeatable:
                return []  # 已解锁且不可重复
            
            user_achievement.current_count = max(user_achievement.current_count, current_value)
            user_achievement.update_progress()
            
            await self._save_user_achievement(user_achievement)
            
            if user_achievement.is_unlocked:
                logger.info(f"🏆 用户 {user_id} 解锁成就: {self.achievements[achievement_id].title}")
                return [achievement_id]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ 更新成就进度失败: {e}")
            return []
    
    async def get_user_achievements(self, user_id: str, include_locked: bool = True) -> List[Dict[str, Any]]:
        """获取用户成就"""
        try:
            await self._ensure_user_achievements(user_id)
            
            result = []
            user_achievements = self.user_achievements.get(user_id, {})
            
            for achievement_id, user_achievement in user_achievements.items():
                achievement = self.achievements[achievement_id]
                
                # 跳过隐藏的未解锁成就
                if achievement.hidden and not user_achievement.is_unlocked:
                    continue
                
                # 跳过未解锁成就（如果不包含）
                if not include_locked and not user_achievement.is_unlocked:
                    continue
                
                result.append({
                    "achievement_id": achievement_id,
                    "title": achievement.title,
                    "description": achievement.description,
                    "category": achievement.category.value,
                    "rarity": achievement.rarity.value,
                    "icon": achievement.icon,
                    "xp_reward": achievement.xp_reward,
                    "badge_color": achievement.badge_color,
                    "progress": user_achievement.progress,
                    "is_unlocked": user_achievement.is_unlocked,
                    "unlocked_at": user_achievement.unlocked_at.isoformat() if user_achievement.unlocked_at else None,
                    "current_count": user_achievement.current_count,
                    "target_count": user_achievement.target_count
                })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 获取用户成就失败: {e}")
            return []
    
    async def _save_achievement(self, achievement: Achievement):
        """保存成就定义"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO achievements 
                    (achievement_id, title, description, category, rarity, icon,
                     xp_reward, badge_color, requirements, hidden, repeatable, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    achievement.achievement_id, achievement.title, achievement.description,
                    achievement.category.value, achievement.rarity.value, achievement.icon,
                    achievement.xp_reward, achievement.badge_color,
                    json.dumps(achievement.requirements), achievement.hidden,
                    achievement.repeatable, achievement.created_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存成就定义失败: {e}")
            raise
    
    async def _save_user_achievement(self, user_achievement: UserAchievement):
        """保存用户成就"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_achievements 
                    (user_id, achievement_id, progress, is_unlocked, unlocked_at,
                     current_count, target_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_achievement.user_id, user_achievement.achievement_id,
                    user_achievement.progress, user_achievement.is_unlocked,
                    user_achievement.unlocked_at.isoformat() if user_achievement.unlocked_at else None,
                    user_achievement.current_count, user_achievement.target_count
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存用户成就失败: {e}")
            raise
