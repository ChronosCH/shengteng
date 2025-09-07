"""
课程管理服务
Course Management Service
提供完整的课程体系管理，包括课程创建、组织、进度跟踪等
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import sqlite3
import os

logger = logging.getLogger(__name__)

class CourseType(Enum):
    """课程类型"""
    BASIC = "basic"              # 基础课程
    INTERMEDIATE = "intermediate" # 中级课程
    ADVANCED = "advanced"        # 高级课程
    SPECIALIZED = "specialized"   # 专业课程

class LessonType(Enum):
    """课时类型"""
    VIDEO = "video"              # 视频课程
    INTERACTIVE = "interactive"   # 互动练习
    QUIZ = "quiz"               # 测验
    PRACTICE = "practice"        # 实践练习
    ASSESSMENT = "assessment"    # 评估

class ContentFormat(Enum):
    """内容格式"""
    VIDEO_MP4 = "video/mp4"
    IMAGE_PNG = "image/png"
    IMAGE_JPG = "image/jpeg"
    TEXT_HTML = "text/html"
    APPLICATION_JSON = "application/json"

@dataclass
class Course:
    """课程"""
    course_id: str
    title: str
    description: str
    course_type: CourseType
    difficulty_level: int  # 1-10
    estimated_hours: float
    prerequisites: List[str] = None
    learning_objectives: List[str] = None
    tags: List[str] = None
    thumbnail_url: str = ""
    instructor: str = ""
    language: str = "zh-CN"
    is_published: bool = False
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.learning_objectives is None:
            self.learning_objectives = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class Lesson:
    """课时"""
    lesson_id: str
    course_id: str
    title: str
    description: str
    lesson_type: LessonType
    order_index: int
    duration_minutes: int
    content: Dict[str, Any]
    resources: List[Dict[str, Any]] = None
    exercises: List[Dict[str, Any]] = None
    is_mandatory: bool = True
    passing_score: float = 0.7
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = []
        if self.exercises is None:
            self.exercises = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class CourseProgress:
    """课程进度"""
    user_id: str
    course_id: str
    enrollment_date: datetime
    completion_percentage: float = 0.0
    current_lesson_id: str = ""
    completed_lessons: List[str] = None
    quiz_scores: Dict[str, float] = None
    total_time_spent: int = 0  # 分钟
    last_accessed: datetime = None
    is_completed: bool = False
    completion_date: Optional[datetime] = None
    certificate_issued: bool = False
    
    def __post_init__(self):
        if self.completed_lessons is None:
            self.completed_lessons = []
        if self.quiz_scores is None:
            self.quiz_scores = {}
        if self.last_accessed is None:
            self.last_accessed = datetime.now()

class CourseManagementService:
    """课程管理服务"""
    
    def __init__(self, db_path: str = "data/courses.db"):
        self.db_path = db_path
        self.courses: Dict[str, Course] = {}
        self.lessons: Dict[str, Lesson] = {}
        self.course_progress: Dict[str, Dict[str, CourseProgress]] = {}  # user_id -> course_id -> progress
        
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 加载数据
        self._load_data()
        
        # 创建默认课程
        self._create_default_courses()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 课程表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS courses (
                        course_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        course_type TEXT,
                        difficulty_level INTEGER,
                        estimated_hours REAL,
                        prerequisites TEXT,
                        learning_objectives TEXT,
                        tags TEXT,
                        thumbnail_url TEXT,
                        instructor TEXT,
                        language TEXT,
                        is_published BOOLEAN,
                        created_at TEXT,
                        updated_at TEXT
                    )
                ''')
                
                # 课时表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS lessons (
                        lesson_id TEXT PRIMARY KEY,
                        course_id TEXT,
                        title TEXT NOT NULL,
                        description TEXT,
                        lesson_type TEXT,
                        order_index INTEGER,
                        duration_minutes INTEGER,
                        content TEXT,
                        resources TEXT,
                        exercises TEXT,
                        is_mandatory BOOLEAN,
                        passing_score REAL,
                        created_at TEXT,
                        updated_at TEXT,
                        FOREIGN KEY (course_id) REFERENCES courses (course_id)
                    )
                ''')
                
                # 课程进度表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS course_progress (
                        user_id TEXT,
                        course_id TEXT,
                        enrollment_date TEXT,
                        completion_percentage REAL,
                        current_lesson_id TEXT,
                        completed_lessons TEXT,
                        quiz_scores TEXT,
                        total_time_spent INTEGER,
                        last_accessed TEXT,
                        is_completed BOOLEAN,
                        completion_date TEXT,
                        certificate_issued BOOLEAN,
                        PRIMARY KEY (user_id, course_id),
                        FOREIGN KEY (course_id) REFERENCES courses (course_id)
                    )
                ''')
                
                conn.commit()
                logger.info("✅ 课程管理数据库初始化完成")
                
        except Exception as e:
            logger.error(f"❌ 课程管理数据库初始化失败: {e}")
            raise
    
    def _load_data(self):
        """加载数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 加载课程
                cursor.execute("SELECT * FROM courses")
                for row in cursor.fetchall():
                    course = Course(
                        course_id=row[0],
                        title=row[1],
                        description=row[2],
                        course_type=CourseType(row[3]),
                        difficulty_level=row[4],
                        estimated_hours=row[5],
                        prerequisites=json.loads(row[6]) if row[6] else [],
                        learning_objectives=json.loads(row[7]) if row[7] else [],
                        tags=json.loads(row[8]) if row[8] else [],
                        thumbnail_url=row[9] or "",
                        instructor=row[10] or "",
                        language=row[11] or "zh-CN",
                        is_published=bool(row[12]),
                        created_at=datetime.fromisoformat(row[13]),
                        updated_at=datetime.fromisoformat(row[14])
                    )
                    self.courses[course.course_id] = course
                
                # 加载课时
                cursor.execute("SELECT * FROM lessons ORDER BY course_id, order_index")
                for row in cursor.fetchall():
                    lesson = Lesson(
                        lesson_id=row[0],
                        course_id=row[1],
                        title=row[2],
                        description=row[3],
                        lesson_type=LessonType(row[4]),
                        order_index=row[5],
                        duration_minutes=row[6],
                        content=json.loads(row[7]) if row[7] else {},
                        resources=json.loads(row[8]) if row[8] else [],
                        exercises=json.loads(row[9]) if row[9] else [],
                        is_mandatory=bool(row[10]),
                        passing_score=row[11],
                        created_at=datetime.fromisoformat(row[12]),
                        updated_at=datetime.fromisoformat(row[13])
                    )
                    self.lessons[lesson.lesson_id] = lesson
                
                logger.info(f"✅ 加载了 {len(self.courses)} 个课程和 {len(self.lessons)} 个课时")
                
        except Exception as e:
            logger.error(f"❌ 课程数据加载失败: {e}")
    
    def _create_default_courses(self):
        """创建默认课程"""
        if self.courses:
            return  # 已有课程，不创建默认课程
        
        try:
            # 基础手语课程
            basic_course = Course(
                course_id="course_basic_sign_language",
                title="基础手语入门",
                description="从零开始学习手语，掌握基本的手语词汇和语法",
                course_type=CourseType.BASIC,
                difficulty_level=1,
                estimated_hours=20.0,
                learning_objectives=[
                    "掌握基本手语字母",
                    "学会常用问候语",
                    "理解基本手语语法",
                    "能进行简单日常交流"
                ],
                tags=["基础", "入门", "日常交流"],
                instructor="手语教学团队",
                is_published=True
            )
            
            self.courses[basic_course.course_id] = basic_course
            
            # 创建基础课程的课时
            lessons_data = [
                {
                    "lesson_id": "lesson_alphabet",
                    "title": "手语字母学习",
                    "description": "学习26个英文字母的手语表达",
                    "lesson_type": LessonType.VIDEO,
                    "order_index": 1,
                    "duration_minutes": 30,
                    "content": {
                        "video_url": "/videos/alphabet.mp4",
                        "practice_words": ["A", "B", "C", "D", "E"]
                    }
                },
                {
                    "lesson_id": "lesson_greetings",
                    "title": "基础问候语",
                    "description": "学习日常问候的手语表达",
                    "lesson_type": LessonType.INTERACTIVE,
                    "order_index": 2,
                    "duration_minutes": 25,
                    "content": {
                        "phrases": ["你好", "再见", "谢谢", "对不起", "请"]
                    }
                },
                {
                    "lesson_id": "lesson_numbers",
                    "title": "数字手语",
                    "description": "学习1-100的数字手语表达",
                    "lesson_type": LessonType.PRACTICE,
                    "order_index": 3,
                    "duration_minutes": 35,
                    "content": {
                        "number_range": [1, 100],
                        "practice_exercises": 10
                    }
                }
            ]
            
            for lesson_data in lessons_data:
                lesson = Lesson(
                    lesson_id=lesson_data["lesson_id"],
                    course_id=basic_course.course_id,
                    title=lesson_data["title"],
                    description=lesson_data["description"],
                    lesson_type=lesson_data["lesson_type"],
                    order_index=lesson_data["order_index"],
                    duration_minutes=lesson_data["duration_minutes"],
                    content=lesson_data["content"]
                )
                self.lessons[lesson.lesson_id] = lesson
            
            # 保存到数据库
            self._save_course(basic_course)
            for lesson in [l for l in self.lessons.values() if l.course_id == basic_course.course_id]:
                self._save_lesson(lesson)
            
            logger.info("✅ 创建了默认基础手语课程")
            
        except Exception as e:
            logger.error(f"❌ 创建默认课程失败: {e}")
    
    async def enroll_user_in_course(self, user_id: str, course_id: str) -> CourseProgress:
        """用户注册课程"""
        try:
            if course_id not in self.courses:
                raise ValueError(f"课程 {course_id} 不存在")
            
            # 检查是否已注册
            if user_id in self.course_progress and course_id in self.course_progress[user_id]:
                return self.course_progress[user_id][course_id]
            
            # 创建进度记录
            progress = CourseProgress(
                user_id=user_id,
                course_id=course_id,
                enrollment_date=datetime.now()
            )
            
            # 设置第一个课时为当前课时
            course_lessons = [l for l in self.lessons.values() if l.course_id == course_id]
            if course_lessons:
                course_lessons.sort(key=lambda x: x.order_index)
                progress.current_lesson_id = course_lessons[0].lesson_id
            
            # 保存进度
            if user_id not in self.course_progress:
                self.course_progress[user_id] = {}
            self.course_progress[user_id][course_id] = progress
            
            await self._save_course_progress(progress)
            
            logger.info(f"✅ 用户 {user_id} 注册课程 {course_id}")
            return progress
            
        except Exception as e:
            logger.error(f"❌ 用户注册课程失败: {e}")
            raise
    
    async def _save_course(self, course: Course):
        """保存课程"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO courses 
                    (course_id, title, description, course_type, difficulty_level,
                     estimated_hours, prerequisites, learning_objectives, tags,
                     thumbnail_url, instructor, language, is_published, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    course.course_id, course.title, course.description,
                    course.course_type.value, course.difficulty_level, course.estimated_hours,
                    json.dumps(course.prerequisites), json.dumps(course.learning_objectives),
                    json.dumps(course.tags), course.thumbnail_url, course.instructor,
                    course.language, course.is_published,
                    course.created_at.isoformat(), course.updated_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存课程失败: {e}")
            raise
    
    async def _save_lesson(self, lesson: Lesson):
        """保存课时"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO lessons 
                    (lesson_id, course_id, title, description, lesson_type, order_index,
                     duration_minutes, content, resources, exercises, is_mandatory,
                     passing_score, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    lesson.lesson_id, lesson.course_id, lesson.title, lesson.description,
                    lesson.lesson_type.value, lesson.order_index, lesson.duration_minutes,
                    json.dumps(lesson.content), json.dumps(lesson.resources),
                    json.dumps(lesson.exercises), lesson.is_mandatory, lesson.passing_score,
                    lesson.created_at.isoformat(), lesson.updated_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存课时失败: {e}")
            raise
    
    async def _save_course_progress(self, progress: CourseProgress):
        """保存课程进度"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO course_progress 
                    (user_id, course_id, enrollment_date, completion_percentage,
                     current_lesson_id, completed_lessons, quiz_scores, total_time_spent,
                     last_accessed, is_completed, completion_date, certificate_issued)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    progress.user_id, progress.course_id, progress.enrollment_date.isoformat(),
                    progress.completion_percentage, progress.current_lesson_id,
                    json.dumps(progress.completed_lessons), json.dumps(progress.quiz_scores),
                    progress.total_time_spent, progress.last_accessed.isoformat(),
                    progress.is_completed,
                    progress.completion_date.isoformat() if progress.completion_date else None,
                    progress.certificate_issued
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存课程进度失败: {e}")
            raise
