/**
 * 完善优化的手语学习训练页面
 * 提供系统化的手语学习体验，包括课程管理、进度跟踪、成就系统等
 */

import { useState, useCallback } from 'react'
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Tab,
  Tabs,
  Paper,
  Chip,
  Stack,
  LinearProgress,
  Avatar,
  Fade,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Rating,
  TextField,
  InputAdornment,
  Alert,
  Snackbar,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
} from '@mui/material'
import {
  PlayArrow,
  School,
  Quiz,
  EmojiEvents,
  Star,
  CheckCircle,
  Lock,
  Close,
  Timer,
  Psychology,
  Speed,
  LocalFireDepartment,
  MenuBook,
  VideoLibrary,
  Games,
  Search,
  ExpandMore,
  TrendingUp,
  Assessment,
  Bookmark,
  Share,
  PlayCircle,
  Assignment,
  Group,
  Category,
  AccessTime,
  Language,
  VolumeUp,
  Visibility,
  TouchApp,
  AutoAwesome,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import HandSignDemo from '../components/HandSignDemo'
import HandSignTestPanel from '../components/HandSignTestPanel'
import SimpleHandSignTest from '../components/SimpleHandSignTest'
import { useAuth } from '../contexts/AuthContext'
import AuthModal from '../components/auth/AuthModal'
import UserDashboard from '../components/learning/UserDashboard'
import LearningRecommendations from '../components/learning/LearningRecommendations'
import LearningAnalytics from '../components/learning/LearningAnalytics'
import InteractiveTutorial from '../components/learning/InteractiveTutorial'
import PracticeSession from '../components/learning/PracticeSession'
import GamificationSystem from '../components/learning/GamificationSystem'
import ExternalResources from '../components/learning/ExternalResources'
import isolatedSignLearningService from '../services/isolatedSignLearningService'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`learning-tabpanel-${index}`}
      aria-labelledby={`learning-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  )
}

// 学习类型枚举
enum LearningType {
  VIDEO_DEMO = 'video_demo',
  INTERACTIVE = 'interactive',
  TEST = 'test',
  GAME = 'game',
}

// 难度级别枚举
enum DifficultyLevel {
  BEGINNER = 'beginner',
  INTERMEDIATE = 'intermediate',
  ADVANCED = 'advanced',
}

function LearningPage() {
  const [currentTab, setCurrentTab] = useState(0)
  const [selectedModule, setSelectedModule] = useState<any>(null)
  const [selectedLesson, setSelectedLesson] = useState<any>(null)
  const [showAchievements, setShowAchievements] = useState(false)
  const [showLearningPath, setShowLearningPath] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterDifficulty, setFilterDifficulty] = useState('all')
  const [filterType, setFilterType] = useState('all')
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as any })
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const [showTutorial, setShowTutorial] = useState(false)
  const [showPractice, setShowPractice] = useState(false)
  const [selectedTutorial, setSelectedTutorial] = useState<any>(null)
  const [selectedPractice, setSelectedPractice] = useState<any>(null)
  const [isolatedUploadLoading, setIsolatedUploadLoading] = useState(false)
  const [isolatedPrediction, setIsolatedPrediction] = useState<{ gloss: string | null; confidence: number } | null>(null)
  const [isolatedVideoPath, setIsolatedVideoPath] = useState('')

  // 认证状态
  const { isAuthenticated, user, loading } = useAuth()
  
  // 模拟用户数据
  const [userStats, setUserStats] = useState({
    totalLearningTime: 245,
    completedLessons: 28,
    currentStreak: 7,
    level: 15,
    totalXP: 1580,
    nextLevelXP: 1800,
    weeklyGoal: 300, // 分钟
    weeklyProgress: 180,
    monthlyGoal: 1200,
    monthlyProgress: 650,
  })

  // 今日学习统计
  const [todayStats, setTodayStats] = useState({
    lessonsCompleted: 2,
    timeSpent: 45,
    xpEarned: 120,
    goal: 60,
  })

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue)
  }

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'warning' | 'info' = 'info') => {
    setSnackbar({ open: true, message, severity })
  }

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false })
  }

  const handleIsolatedSignUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (!isAuthenticated) {
      setAuthModalOpen(true)
      showSnackbar('请先登录后再进行上传', 'warning')
      return
    }

    try {
      setIsolatedUploadLoading(true)
      setIsolatedPrediction(null)

      const uploadResp = await isolatedSignLearningService.uploadIsolatedVideo(file)
      setIsolatedVideoPath(uploadResp.file_path)

      const predictResp = await isolatedSignLearningService.predictIsolatedVideo(uploadResp.file_path)
      setIsolatedPrediction({
        gloss: predictResp.prediction.gloss,
        confidence: predictResp.prediction.confidence,
      })
      showSnackbar('识别完成，查看结果', 'success')
    } catch (error: any) {
      console.error('isolated sign upload failed', error)
      showSnackbar(error?.message || '孤立手语识别失败', 'error')
    } finally {
      setIsolatedUploadLoading(false)
    }
  }, [isAuthenticated])

  // 开始课程 - 需要认证
  const startLesson = useCallback((lesson: any) => {
    if (!isAuthenticated) {
      setAuthModalOpen(true)
      showSnackbar('请先登录以开始学习', 'warning')
      return
    }
    setSelectedLesson(lesson)
    showSnackbar(`开始学习: ${lesson.title}`, 'info')
  }, [isAuthenticated])

  // 完成课程 - 需要认证
  const completeLesson = useCallback((lessonId: string, score: number = 100) => {
    if (!isAuthenticated) {
      setAuthModalOpen(true)
      showSnackbar('请先登录以保存学习进度', 'warning')
      return
    }

    // 更新统计数据
    setUserStats(prev => ({
      ...prev,
      completedLessons: prev.completedLessons + 1,
      totalXP: prev.totalXP + Math.floor(score * 0.5),
      totalLearningTime: prev.totalLearningTime + 15,
    }))

    setTodayStats(prev => ({
      ...prev,
      lessonsCompleted: prev.lessonsCompleted + 1,
      timeSpent: prev.timeSpent + 15,
      xpEarned: prev.xpEarned + Math.floor(score * 0.5),
    }))
    
    showSnackbar(`课程完成！获得 ${Math.floor(score * 0.5)} XP`, 'success')
  }, [])

  // 收藏课程 - 需要认证
  const bookmarkLesson = useCallback((lessonId: string) => {
    if (!isAuthenticated) {
      setAuthModalOpen(true)
      showSnackbar('请先登录以收藏课程', 'warning')
      return
    }
    showSnackbar('已添加到书签', 'success')
  }, [isAuthenticated])

  // 分享课程
  const shareLesson = useCallback((lesson: any) => {
    if (navigator.share) {
      navigator.share({
        title: lesson.title,
        text: lesson.description,
        url: window.location.href,
      })
    } else {
      navigator.clipboard.writeText(window.location.href)
      showSnackbar('链接已复制到剪贴板', 'success')
    }
  }, [])

  // 完善的学习模块数据
  const learningModules = [
    {
      id: 'basic-signs',
      title: '基础手语',
      description: '学习最常用的手语词汇和基本表达，包括问候语、自我介绍等',
      level: DifficultyLevel.BEGINNER,
      completedLessons: 8,
      totalLessons: 12,
      progress: 67,
      icon: <MenuBook />,
      color: '#B5EAD7',
      locked: false,
      estimatedTime: '2-3小时',
      skills: ['基础词汇', '日常用语', '问候语', '自我介绍'],
      rating: 4.8,
      reviews: 156,
      category: '基础入门',
      lastUpdated: '2024-01-15',
      instructor: '张老师',
      difficulty: '★☆☆',
      prerequisites: [],
      learningOutcomes: [
        '掌握50个基础手语词汇',
        '能够进行简单的自我介绍',
        '理解手语的基本规则和礼仪'
      ]
    },
    {
      id: 'numbers-time',
      title: '数字与时间',
      description: '掌握数字、时间和日期的手语表达，学会表达具体的时间概念',
      level: DifficultyLevel.BEGINNER,
      completedLessons: 5,
      totalLessons: 8,
      progress: 63,
      icon: <Timer />,
      color: '#FFDAB9',
      locked: false,
      estimatedTime: '1-2小时',
      skills: ['数字0-100', '时间表达', '日期表达', '计量单位'],
      rating: 4.6,
      reviews: 89,
      category: '基础入门',
      lastUpdated: '2024-01-10',
      instructor: '李老师',
      difficulty: '★☆☆',
      prerequisites: ['basic-signs'],
      learningOutcomes: [
        '掌握0-100的数字表达',
        '能够询问和表达时间',
        '学会日期和年份的表达方式'
      ]
    },
    {
      id: 'family-relations',
      title: '家庭关系',
      description: '学习家庭成员和人际关系相关手语，表达亲情和友情',
      level: DifficultyLevel.INTERMEDIATE,
      completedLessons: 3,
      totalLessons: 10,
      progress: 30,
      icon: <Group />,
      color: '#FFB3BA',
      locked: false,
      estimatedTime: '2-3小时',
      skills: ['家庭成员', '关系称谓', '情感表达', '人际交往'],
      rating: 4.7,
      reviews: 124,
      category: '生活应用',
      lastUpdated: '2024-01-08',
      instructor: '王老师',
      difficulty: '★★☆',
      prerequisites: ['basic-signs'],
      learningOutcomes: [
        '掌握各种家庭关系称谓',
        '能够表达基本情感',
        '学会描述人际关系'
      ]
    },
    {
      id: 'daily-activities',
      title: '日常活动',
      description: '学习日常生活中常见活动的手语表达，涵盖吃住行等方面',
      level: DifficultyLevel.INTERMEDIATE,
      completedLessons: 6,
      totalLessons: 15,
      progress: 40,
      icon: <Category />,
      color: '#C7CEDB',
      locked: false,
      estimatedTime: '3-4小时',
      skills: ['生活用语', '动作表达', '场所名称', '交通工具'],
      rating: 4.5,
      reviews: 98,
      category: '生活应用',
      lastUpdated: '2024-01-12',
      instructor: '陈老师',
      difficulty: '★★☆',
      prerequisites: ['basic-signs', 'numbers-time'],
      learningOutcomes: [
        '掌握日常活动的表达',
        '能够描述生活场景',
        '学会常用动词和名词'
      ]
    },
    {
      id: 'advanced-grammar',
      title: '高级语法',
      description: '掌握复杂的手语语法结构和表达技巧，提升表达的准确性',
      level: DifficultyLevel.ADVANCED,
      completedLessons: 0,
      totalLessons: 15,
      progress: 0,
      icon: <Psychology />,
      color: '#E8E3F0',
      locked: true,
      estimatedTime: '4-6小时',
      skills: ['语法结构', '时态表达', '复合句型', '修辞技巧'],
      rating: 4.9,
      reviews: 67,
      category: '高级进阶',
      lastUpdated: '2024-01-05',
      instructor: '赵教授',
      difficulty: '★★★',
      prerequisites: ['family-relations', 'daily-activities'],
      learningOutcomes: [
        '掌握复杂语法结构',
        '能够表达抽象概念',
        '提升表达的流畅性和准确性'
      ]
    },
    {
      id: 'professional-signs',
      title: '职业手语',
      description: '学习不同职业和工作场景的专业手语，适用于职场交流',
      level: DifficultyLevel.ADVANCED,
      completedLessons: 0,
      totalLessons: 20,
      progress: 0,
      icon: <Assessment />,
      color: '#B8A9C9',
      locked: true,
      estimatedTime: '5-7小时',
      skills: ['职业名称', '工作用语', '商务交流', '专业术语'],
      rating: 4.6,
      reviews: 45,
      category: '专业应用',
      lastUpdated: '2024-01-03',
      instructor: '孙老师',
      difficulty: '★★★',
      prerequisites: ['daily-activities'],
      learningOutcomes: [
        '掌握各行业专业术语',
        '能够进行职场交流',
        '学会商务手语礼仪'
      ]
    }
  ]

  // 扩展的成就系统
  const achievements = [
    {
      id: 'first-lesson',
      title: '初学者',
      description: '完成第一节课程',
      icon: <School />,
      unlocked: true,
      progress: 1,
      maxProgress: 1,
      color: '#B5EAD7',
      unlockedAt: '2024-01-15',
      category: '入门成就',
      xpReward: 50,
    },
    {
      id: 'week-streak',
      title: '坚持一周',
      description: '连续学习7天',
      icon: <LocalFireDepartment />,
      unlocked: true,
      progress: 7,
      maxProgress: 7,
      color: '#FFB3BA',
      unlockedAt: '2024-01-22',
      category: '坚持成就',
      xpReward: 200,
    },
    {
      id: 'speed-learner',
      title: '学习达人',
      description: '在一天内完成5节课程',
      icon: <Speed />,
      unlocked: false,
      progress: 3,
      maxProgress: 5,
      color: '#FFDAB9',
      category: '效率成就',
      xpReward: 150,
    },
    {
      id: 'master-basic',
      title: '基础大师',
      description: '完成所有基础课程',
      icon: <Star />,
      unlocked: false,
      progress: 8,
      maxProgress: 12,
      color: '#C7CEDB',
      category: '掌握成就',
      xpReward: 300,
    },
    {
      id: 'perfect-score',
      title: '满分达人',
      description: '在测试中获得满分',
      icon: <EmojiEvents />,
      unlocked: false,
      progress: 0,
      maxProgress: 1,
      color: '#FFD700',
      category: '卓越成就',
      xpReward: 400,
    },
    {
      id: 'social-learner',
      title: '社交学习者',
      description: '与其他学习者互动10次',
      icon: <Group />,
      unlocked: false,
      progress: 3,
      maxProgress: 10,
      color: '#98FB98',
      category: '社交成就',
      xpReward: 120,
    },
    {
      id: 'time-master',
      title: '时间大师',
      description: '累计学习时间达到50小时',
      icon: <AccessTime />,
      unlocked: false,
      progress: 245,
      maxProgress: 3000,
      color: '#DDA0DD',
      category: '时间成就',
      xpReward: 500,
    }
  ]

  // 详细的学习路径
  const learningPaths = [
    {
      id: 'quick-start',
      title: '快速入门路径',
      description: '适合零基础学习者的快速入门课程，7天掌握基础手语',
      duration: '1-2周',
      modules: ['basic-signs', 'numbers-time'],
      difficulty: DifficultyLevel.BEGINNER,
      color: '#B5EAD7',
      estimatedHours: 6,
      skills: ['基础词汇', '数字表达', '简单交流'],
      completionRate: 85,
      enrolled: 1250,
      steps: [
        { title: '问候语学习', description: '学习基本问候用语' },
        { title: '数字掌握', description: '掌握数字0-100' },
        { title: '自我介绍', description: '学会用手语自我介绍' },
        { title: '日常对话', description: '进行简单日常对话' },
      ]
    },
    {
      id: 'daily-communication',
      title: '日常交流路径',
      description: '学习日常生活中最常用的手语表达，满足基本交流需求',
      duration: '3-4周',
      modules: ['basic-signs', 'family-relations', 'numbers-time', 'daily-activities'],
      difficulty: DifficultyLevel.INTERMEDIATE,
      color: '#FFDAB9',
      estimatedHours: 12,
      skills: ['生活用语', '家庭交流', '社交表达'],
      completionRate: 78,
      enrolled: 890,
      steps: [
        { title: '基础巩固', description: '巩固基础手语知识' },
        { title: '家庭交流', description: '学习家庭相关表达' },
        { title: '日常活动', description: '掌握日常活动用语' },
        { title: '综合应用', description: '综合运用所学知识' },
      ]
    },
    {
      id: 'professional-advanced',
      title: '专业进阶路径',
      description: '深入学习手语语法和高级表达技巧，达到专业水平',
      duration: '6-8周',
      modules: ['basic-signs', 'family-relations', 'daily-activities', 'advanced-grammar', 'professional-signs'],
      difficulty: DifficultyLevel.ADVANCED,
      color: '#C7CEDB',
      estimatedHours: 25,
      skills: ['高级语法', '专业术语', '流畅表达'],
      completionRate: 65,
      enrolled: 456,
      steps: [
        { title: '语法深化', description: '学习复杂语法结构' },
        { title: '专业应用', description: '掌握职场手语' },
        { title: '高级技巧', description: '学习高级表达技巧' },
        { title: '实战演练', description: '实际场景应用练习' },
      ]
    },
  ]

  // 扩展的每日任务
  const dailyTasks = [
    { 
      id: 'daily-lesson',
      task: '完成一节课程', 
      completed: true, 
      xp: 50,
      type: 'lesson',
      progress: 1,
      target: 1,
      icon: <PlayCircle />
    },
    { 
      id: 'vocabulary-practice',
      task: '练习10个新词汇', 
      completed: true, 
      xp: 30,
      type: 'vocabulary',
      progress: 10,
      target: 10,
      icon: <Language />
    },
    { 
      id: 'test-complete',
      task: '通过一次手语测试', 
      completed: false, 
      xp: 80,
      type: 'test',
      progress: 0,
      target: 1,
      icon: <Quiz />
    },
    { 
      id: 'video-watch',
      task: '观看3个演示视频', 
      completed: false, 
      xp: 40,
      type: 'video',
      progress: 1,
      target: 3,
      icon: <VideoLibrary />
    },
    { 
      id: 'practice-time',
      task: '练习手语30分钟', 
      completed: false, 
      xp: 60,
      type: 'time',
      progress: 15,
      target: 30,
      icon: <Timer />
    }
  ]

  // 学习课程详细数据
  const lessonTypes = [
    {
      type: LearningType.VIDEO_DEMO,
      title: '视频演示',
      description: '观看标准手语动作演示',
      icon: <VideoLibrary />,
      color: '#B5EAD7'
    },
    {
      type: LearningType.INTERACTIVE,
      title: '互动练习',
      description: '实时手语练习和反馈',
      icon: <TouchApp />,
      color: '#FFDAB9'
    },
    {
      type: LearningType.TEST,
      title: '能力测试',
      description: '评估学习成果和能力',
      icon: <Quiz />,
      color: '#FFB3BA'
    },
    {
      type: LearningType.GAME,
      title: '游戏化学习',
      description: '通过游戏轻松学习',
      icon: <Games />,
      color: '#C7CEDB'
    }
  ]

  const getLevelColor = (level: DifficultyLevel) => {
    switch (level) {
      case DifficultyLevel.BEGINNER: return '#B5EAD7'
      case DifficultyLevel.INTERMEDIATE: return '#FFDAB9'
      case DifficultyLevel.ADVANCED: return '#C7CEDB'
      default: return '#B5EAD7'
    }
  }

  const getLevelLabel = (level: DifficultyLevel) => {
    switch (level) {
      case DifficultyLevel.BEGINNER: return '初级'
      case DifficultyLevel.INTERMEDIATE: return '中级'
      case DifficultyLevel.ADVANCED: return '高级'
      default: return '初级'
    }
  }

  // 过滤学习模块
  const filteredModules = learningModules.filter(module => {
    const matchesSearch = module.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         module.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         module.skills.some(skill => skill.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesDifficulty = filterDifficulty === 'all' || module.level === filterDifficulty
    
    return matchesSearch && matchesDifficulty
  })

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 个性化欢迎消息 */}
      {isAuthenticated && user && (
        <Fade in timeout={400}>
          <Alert
            severity="success"
            sx={{
              mb: 3,
              borderRadius: 3,
              background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
              border: 'none',
            }}
          >
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              欢迎回来，{user.full_name || user.username}！ 🎉
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              继续您的手语学习之旅，今天也要加油哦！
            </Typography>
          </Alert>
        </Fade>
      )}

      {/* 未登录提示 */}
      {!isAuthenticated && (
        <Fade in timeout={400}>
          <Alert
            severity="info"
            action={
              <Button
                color="inherit"
                size="small"
                onClick={() => setAuthModalOpen(true)}
                sx={{ fontWeight: 600 }}
              >
                立即登录
              </Button>
            }
            sx={{
              mb: 3,
              borderRadius: 3,
              background: 'linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%)',
              border: 'none',
            }}
          >
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              登录以获得个性化学习体验 📚
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              保存学习进度、获得成就奖励、享受专属推荐内容
            </Typography>
          </Alert>
        </Fade>
      )}

      {/* 页面标题和用户统计增强版 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 6 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} lg={8}>
              <Stack direction="row" spacing={3} alignItems="center">
                <Avatar
                  sx={{
                    width: 80,
                    height: 80,
                    background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                    boxShadow: '0 12px 32px rgba(181, 234, 215, 0.4)',
                  }}
                >
                  <School sx={{ fontSize: 40, color: 'white' }} />
                </Avatar>
                <Box>
                  <Typography 
                    variant="h2" 
                    sx={{ 
                      fontWeight: 700,
                      background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      mb: 1,
                    }}
                  >
                    手语学习训练
                  </Typography>
                  <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                    系统化学习手语，掌握沟通技能
                  </Typography>
                  <Stack direction="row" spacing={2}>
                    <Chip 
                      icon={<TrendingUp />} 
                      label={`等级 ${userStats.level}`} 
                      color="primary" 
                      size="small" 
                    />
                    <Chip 
                      icon={<LocalFireDepartment />} 
                      label={`连续 ${userStats.currentStreak} 天`} 
                      color="warning" 
                      size="small" 
                    />
                  </Stack>
                </Box>
              </Stack>
            </Grid>
            
            <Grid item xs={12} lg={4}>
              <Paper 
                sx={{ 
                  p: 3, 
                  background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                  color: 'white',
                  borderRadius: 4,
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    top: -10,
                    right: -10,
                    width: 40,
                    height: 40,
                    background: 'rgba(255,255,255,0.2)',
                    borderRadius: '50%',
                  }}
                />
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  学习统计
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                      {userStats.level}
                    </Typography>
                    <Typography variant="caption" sx={{ opacity: 0.9 }}>
                      当前等级
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                      {userStats.currentStreak}
                    </Typography>
                    <Typography variant="caption" sx={{ opacity: 0.9 }}>
                      连续天数
                    </Typography>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption">经验值</Typography>
                    <Typography variant="caption">{userStats.totalXP}/{userStats.nextLevelXP}</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={(userStats.totalXP / userStats.nextLevelXP) * 100}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      backgroundColor: 'rgba(255,255,255,0.3)',
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 3,
                        backgroundColor: 'rgba(255,255,255,0.9)',
                      },
                    }}
                  />
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Fade>

      {/* 今日学习概览 */}
      <Fade in timeout={700}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
            📊 今日学习概览
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#B5EAD7', mx: 'auto', mb: 1 }}>
                  <Assignment />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {todayStats.lessonsCompleted}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  已完成课程
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#FFDAB9', mx: 'auto', mb: 1 }}>
                  <AccessTime />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {todayStats.timeSpent}分钟
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  学习时长
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#FFB3BA', mx: 'auto', mb: 1 }}>
                  <AutoAwesome />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {todayStats.xpEarned}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  获得经验
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#C7CEDB', mx: 'auto', mb: 1 }}>
                  <EmojiEvents />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {Math.round((todayStats.timeSpent / todayStats.goal) * 100)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  目标达成
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Fade>

      {/* 搜索和筛选工具栏 */}
      <Fade in timeout={800}>
        <Box sx={{ mb: 4 }}>
          <Paper sx={{ p: 3, borderRadius: 4 }}>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  placeholder="搜索课程、技能或关键词..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Search />
                      </InputAdornment>
                    ),
                  }}
                  sx={{ '& .MuiOutlinedInput-root': { borderRadius: 3 } }}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>难度筛选</InputLabel>
                  <Select
                    value={filterDifficulty}
                    onChange={(e) => setFilterDifficulty(e.target.value)}
                    label="难度筛选"
                    sx={{ borderRadius: 3 }}
                  >
                    <MenuItem value="all">全部难度</MenuItem>
                    <MenuItem value={DifficultyLevel.BEGINNER}>初级</MenuItem>
                    <MenuItem value={DifficultyLevel.INTERMEDIATE}>中级</MenuItem>
                    <MenuItem value={DifficultyLevel.ADVANCED}>高级</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={3}>
                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<Category />}
                  onClick={() => setShowLearningPath(true)}
                  sx={{ borderRadius: 3, height: 56 }}
                >
                  学习路径
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Box>
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧主要内容 */}
        <Grid item xs={12} lg={8}>
          <Stack spacing={4}>
            {/* 学习路径推荐卡片 */}
            <Fade in timeout={900}>
              <Box>
                <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                  📍 推荐学习路径
                </Typography>
                <Grid container spacing={3}>
                  {learningPaths.slice(0, 3).map((path, index) => (
                    <Grid item xs={12} md={4} key={path.id}>
                      <Card
                        sx={{
                          height: '100%',
                          background: `linear-gradient(135deg, ${path.color}20 0%, ${path.color}10 100%)`,
                          border: `1px solid ${path.color}30`,
                          borderRadius: 3,
                          transition: 'all 0.3s ease',
                          cursor: 'pointer',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: `0 8px 25px ${path.color}30`,
                          },
                        }}
                        onClick={() => setShowLearningPath(true)}
                      >
                        <CardContent sx={{ p: 3 }}>
                          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                            {path.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 40 }}>
                            {path.description}
                          </Typography>
                          <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap', gap: 1 }}>
                            <Chip
                              label={getLevelLabel(path.difficulty)}
                              size="small"
                              sx={{ backgroundColor: path.color, color: 'white' }}
                            />
                            <Chip
                              label={path.duration}
                              size="small"
                              variant="outlined"
                            />
                            <Chip
                              label={`${path.estimatedHours}小时`}
                              size="small"
                              variant="outlined"
                            />
                          </Stack>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="caption" color="text.secondary">
                              {path.enrolled} 人已报名
                            </Typography>
                            <Typography variant="caption" color="success.main" sx={{ fontWeight: 600 }}>
                              {path.completionRate}% 完成率
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={path.completionRate}
                            sx={{
                              height: 4,
                              borderRadius: 2,
                              backgroundColor: `${path.color}20`,
                              '& .MuiLinearProgress-bar': {
                                borderRadius: 2,
                                backgroundColor: path.color,
                              },
                            }}
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Fade>

            {/* 学习模块 */}
            <Fade in timeout={1000}>
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    📚 学习模块
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    找到 {filteredModules.length} 个模块
                  </Typography>
                </Box>
                <Grid container spacing={3}>
                  {filteredModules.map((module, index) => (
                    <Grid item xs={12} sm={6} key={module.id}>
                      <Card
                        sx={{
                          height: '100%',
                          background: `linear-gradient(135deg, ${module.color}20 0%, ${module.color}10 100%)`,
                          border: `1px solid ${module.color}30`,
                          borderRadius: 3,
                          opacity: module.locked ? 0.7 : 1,
                          transition: 'all 0.3s ease',
                          position: 'relative',
                          '&:hover': {
                            transform: module.locked ? 'none' : 'translateY(-4px)',
                            boxShadow: module.locked ? 'none' : `0 8px 25px ${module.color}30`,
                          },
                          cursor: module.locked ? 'not-allowed' : 'pointer',
                        }}
                        onClick={() => !module.locked && setSelectedModule(module)}
                      >
                        {module.locked && (
                          <Box
                            sx={{
                              position: 'absolute',
                              top: 12,
                              right: 12,
                              zIndex: 1,
                            }}
                          >
                            <Avatar sx={{ bgcolor: 'text.disabled', width: 32, height: 32 }}>
                              <Lock sx={{ fontSize: 16 }} />
                            </Avatar>
                          </Box>
                        )}
                        
                        <CardContent sx={{ p: 3 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                            <Avatar
                              sx={{
                                bgcolor: module.color,
                                mr: 2,
                                width: 48,
                                height: 48,
                              }}
                            >
                              {module.locked ? <Lock /> : module.icon}
                            </Avatar>
                            <Box sx={{ flexGrow: 1 }}>
                              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                {module.title}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {module.category} • 更新于 {module.lastUpdated}
                              </Typography>
                            </Box>
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 48 }}>
                            {module.description}
                          </Typography>
                          
                          <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap', gap: 1 }}>
                            <Chip
                              label={getLevelLabel(module.level)}
                              size="small"
                              sx={{ 
                                backgroundColor: getLevelColor(module.level), 
                                color: 'white',
                                fontSize: '0.7rem',
                              }}
                            />
                            <Chip
                              label={module.estimatedTime}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem' }}
                            />
                            <Chip
                              label={module.difficulty}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem' }}
                            />
                          </Stack>
                          
                          {!module.locked && (
                            <>
                              <Box sx={{ mb: 2 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                  <Typography variant="body2">学习进度</Typography>
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {module.progress}%
                                  </Typography>
                                </Box>
                                <LinearProgress
                                  variant="determinate"
                                  value={module.progress}
                                  sx={{
                                    height: 6,
                                    borderRadius: 3,
                                    backgroundColor: `${module.color}20`,
                                    '& .MuiLinearProgress-bar': {
                                      borderRadius: 3,
                                      backgroundColor: module.color,
                                    },
                                  }}
                                />
                              </Box>
                              
                              <Grid container spacing={2} sx={{ mb: 2 }}>
                                <Grid item xs={6}>
                                  <Typography variant="caption" color="text.secondary">
                                    课程进度
                                  </Typography>
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {module.completedLessons}/{module.totalLessons}
                                  </Typography>
                                </Grid>
                                <Grid item xs={6}>
                                  <Typography variant="caption" color="text.secondary">
                                    讲师
                                  </Typography>
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {module.instructor}
                                  </Typography>
                                </Grid>
                              </Grid>
                              
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Rating value={module.rating} precision={0.1} size="small" readOnly />
                                  <Typography variant="caption" sx={{ ml: 0.5 }}>
                                    ({module.reviews})
                                  </Typography>
                                </Box>
                                <Stack direction="row" spacing={1}>
                                  <Tooltip title="收藏课程">
                                    <IconButton
                                      size="small"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        bookmarkLesson(module.id)
                                      }}
                                    >
                                      <Bookmark />
                                    </IconButton>
                                  </Tooltip>
                                  <Tooltip title="分享课程">
                                    <IconButton
                                      size="small"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        shareLesson(module)
                                      }}
                                    >
                                      <Share />
                                    </IconButton>
                                  </Tooltip>
                                </Stack>
                              </Box>
                            </>
                          )}
                          
                          {module.locked && (
                            <Alert severity="info" sx={{ mt: 2 }}>
                              完成前置课程后解锁：{module.prerequisites.join(', ')}
                            </Alert>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Fade>

            {/* 学习内容标签页 */}
            <Fade in timeout={1200}>
              <Paper sx={{ borderRadius: 4 }}>
                <Tabs
                  value={currentTab}
                  onChange={handleTabChange}
                  variant="fullWidth"
                  sx={{
                    '& .MuiTab-root': { py: 2.5, fontSize: '1rem', fontWeight: 600 }
                  }}
                >
                  <Tab
                    icon={<Assessment />}
                    label="学习仪表板"
                    iconPosition="start"
                  />
                  <Tab
                    icon={<VideoLibrary />}
                    label="视频演示"
                    iconPosition="start"
                  />
                  <Tab
                    icon={<TouchApp />}
                    label="互动练习"
                    iconPosition="start"
                  />
                  <Tab
                    icon={<Quiz />}
                    label="能力测试"
                    iconPosition="start"
                  />
                  <Tab
                    icon={<Games />}
                    label="游戏学习"
                    iconPosition="start"
                  />
                  <Tab
                    icon={<Language />}
                    label="外部资源"
                    iconPosition="start"
                  />
                </Tabs>

                <TabPanel value={currentTab} index={0}>
                  <Box sx={{ p: 4 }}>
                    {/* 用户仪表板 */}
                    {isAuthenticated ? (
                      <Stack spacing={4}>
                        <UserDashboard
                          userStats={userStats}
                          onStartLesson={startLesson}
                        />

                        <LearningRecommendations
                          userLevel={userStats.level}
                          completedLessons={userStats.completedLessons}
                          currentStreak={userStats.currentStreak}
                          onStartLesson={startLesson}
                          onBookmarkLesson={bookmarkLesson}
                        />

                        <LearningAnalytics userStats={userStats} />
                      </Stack>
                    ) : (
                      <Box textAlign="center" py={8}>
                        <Typography variant="h5" gutterBottom>
                          🔐 登录查看个人仪表板
                        </Typography>
                        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                          登录后可查看详细的学习进度、个性化推荐和学习分析
                        </Typography>
                        <Button
                          variant="contained"
                          size="large"
                          onClick={() => setAuthModalOpen(true)}
                          sx={{
                            borderRadius: 3,
                            px: 4,
                            py: 1.5,
                            background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                          }}
                        >
                          立即登录
                        </Button>
                      </Box>
                    )}
                  </Box>
                </TabPanel>

                <TabPanel value={currentTab} index={1}>
                  <Box sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      📺 手语动作演示
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      观看标准的手语动作演示，学习正确的表达方式和手型技巧
                    </Typography>
                    
                    {/* 视频演示功能区域 */}
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={8}>
                        <ErrorBoundary>
                          <HandSignDemo />
                        </ErrorBoundary>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 3, bgcolor: '#f8f9fa', borderRadius: 3 }}>
                          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                            🎯 学习要点
                          </Typography>
                          <List dense>
                            <ListItem>
                              <ListItemIcon>
                                <Visibility color="primary" />
                              </ListItemIcon>
                              <ListItemText 
                                primary="仔细观察手型"
                                secondary="注意手指的位置和角度"
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemIcon>
                                <Timer color="primary" />
                              </ListItemIcon>
                              <ListItemText 
                                primary="控制动作节奏"
                                secondary="保持适当的动作速度"
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemIcon>
                                <TouchApp color="primary" />
                              </ListItemIcon>
                              <ListItemText 
                                primary="重复练习"
                                secondary="多次练习直到熟练"
                              />
                            </ListItem>
                          </List>
                        </Paper>
                      </Grid>
                    </Grid>
                  </Box>
                </TabPanel>

                <TabPanel value={currentTab} index={2}>
                  <Box sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      🤝 互动练习
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      通过实时交互练习手语动作，获得即时反馈和指导
                    </Typography>

                    <Grid container spacing={3}>
                      {/* 交互式教程卡片 */}
                      <Grid item xs={12} md={6}>
                        <Card sx={{ borderRadius: 3, height: '100%' }}>
                          <CardContent>
                            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                              <Avatar sx={{ bgcolor: '#B5EAD7' }}>
                                <School />
                              </Avatar>
                              <Typography variant="h6" fontWeight="bold">
                                交互式教程
                              </Typography>
                            </Stack>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                              跟随步骤式指导学习手语，获得实时反馈和提示
                            </Typography>
                            <Stack spacing={2}>
                              <Button
                                variant="contained"
                                fullWidth
                                startIcon={<PlayArrow />}
                                onClick={() => {
                                  if (!isAuthenticated) {
                                    setAuthModalOpen(true)
                                    return
                                  }
                                  setSelectedTutorial({
                                    id: 'basic-numbers',
                                    title: '基础数字手语',
                                    description: '学习0-10的数字手语表达',
                                    steps: [
                                      {
                                        id: 'step1',
                                        title: '数字0',
                                        description: '学习数字0的手语表达',
                                        instruction: '将手握成拳头，拇指向上',
                                        tips: ['保持手型稳定', '动作要清晰'],
                                        expectedAction: '握拳拇指向上',
                                        difficulty: 'easy',
                                        estimatedTime: 2,
                                      },
                                      {
                                        id: 'step2',
                                        title: '数字1',
                                        description: '学习数字1的手语表达',
                                        instruction: '伸出食指，其他手指握拳',
                                        tips: ['食指要直立', '其他手指紧握'],
                                        expectedAction: '食指直立',
                                        difficulty: 'easy',
                                        estimatedTime: 2,
                                      },
                                    ]
                                  })
                                  setShowTutorial(true)
                                }}
                                sx={{ borderRadius: 2 }}
                              >
                                开始教程
                              </Button>
                              <Button
                                variant="outlined"
                                fullWidth
                                startIcon={<Assignment />}
                                sx={{ borderRadius: 2 }}
                              >
                                查看所有教程
                              </Button>
                            </Stack>
                          </CardContent>
                        </Card>
                      </Grid>

                      {/* 练习会话卡片 */}
                      <Grid item xs={12} md={6}>
                        <Card sx={{ borderRadius: 3, height: '100%' }}>
                          <CardContent>
                            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                              <Avatar sx={{ bgcolor: '#FFB3BA' }}>
                                <TouchApp />
                              </Avatar>
                              <Typography variant="h6" fontWeight="bold">
                                实时练习
                              </Typography>
                            </Stack>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                              通过摄像头进行实时手语练习，获得准确率反馈
                            </Typography>
                            <Stack spacing={2}>
                              <Button
                                variant="contained"
                                fullWidth
                                startIcon={<PlayArrow />}
                                onClick={() => {
                                  if (!isAuthenticated) {
                                    setAuthModalOpen(true)
                                    return
                                  }
                                  setSelectedPractice({
                                    sessionTitle: '基础手语练习',
                                    exercises: [
                                      {
                                        id: 'ex1',
                                        word: '你好',
                                        description: '学习基本问候语',
                                        difficulty: 'easy',
                                        category: '问候',
                                        expectedGesture: '右手举起，手掌向前',
                                        hints: ['保持手掌平直', '动作要自然'],
                                      },
                                      {
                                        id: 'ex2',
                                        word: '谢谢',
                                        description: '学习感谢表达',
                                        difficulty: 'easy',
                                        category: '礼貌用语',
                                        expectedGesture: '双手合十，微微鞠躬',
                                        hints: ['双手要对齐', '表情要真诚'],
                                      },
                                    ]
                                  })
                                  setShowPractice(true)
                                }}
                                sx={{ borderRadius: 2 }}
                              >
                                开始练习
                              </Button>
                              <Button
                                variant="outlined"
                                fullWidth
                                startIcon={<Speed />}
                                sx={{ borderRadius: 2 }}
                              >
                                快速测试
                              </Button>
                            </Stack>
                          </CardContent>
                        </Card>
                      </Grid>

                      {/* 游戏化系统 */}
                      <Grid item xs={12}>
                        <GamificationSystem
                          userStats={userStats}
                          onClaimReward={(achievementId) => {
                            showSnackbar('奖励已领取！', 'success')
                          }}
                        />
                      </Grid>

                      {/* 原有的简单测试组件 */}
                      <Grid item xs={12}>
                        <Card sx={{ borderRadius: 3 }}>
                          <CardContent>
                            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                              🎯 基础练习
                            </Typography>
                            <ErrorBoundary>
                              <SimpleHandSignTest />
                            </ErrorBoundary>
                          </CardContent>
                        </Card>
                      </Grid>
                    </Grid>
                  </Box>
                </TabPanel>

                <TabPanel value={currentTab} index={3}>
                  <Box sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      📝 能力测试
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      测试您的手语识别和表达能力，获得专业的能力评估报告
                    </Typography>
                    
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={8}>
                        <ErrorBoundary>
                          <HandSignTestPanel />
                        </ErrorBoundary>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Stack spacing={2}>
                          <Paper sx={{ p: 3, bgcolor: '#f3e5f5', borderRadius: 3 }}>
                            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                              🎯 测试类型
                            </Typography>
                            <List dense>
                              <ListItem>
                                <ListItemIcon>
                                  <Assignment />
                                </ListItemIcon>
                                <ListItemText 
                                  primary="基础能力测试"
                                  secondary="评估基础手语掌握程度"
                                />
                              </ListItem>
                              <ListItem>
                                <ListItemIcon>
                                  <Psychology />
                                </ListItemIcon>
                                <ListItemText 
                                  primary="综合能力测试"
                                  secondary="全面评估手语技能"
                                />
                              </ListItem>
                              <ListItem>
                                <ListItemIcon>
                                  <Speed />
                                </ListItemIcon>
                                <ListItemText 
                                  primary="速度测试"
                                  secondary="测试手语表达速度"
                                />
                              </ListItem>
                            </List>
                          </Paper>
                          
                          <Paper sx={{ p: 3, bgcolor: '#e3f2fd', borderRadius: 3 }}>
                            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                              📊 历史成绩
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              最近测试成绩趋势
                            </Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Typography variant="caption">基础测试</Typography>
                              <Typography variant="caption" sx={{ fontWeight: 600 }}>85分</Typography>
                            </Box>
                            <LinearProgress variant="determinate" value={85} sx={{ mb: 1, height: 6, borderRadius: 3 }} />
                            
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Typography variant="caption">综合测试</Typography>
                              <Typography variant="caption" sx={{ fontWeight: 600 }}>78分</Typography>
                            </Box>
                            <LinearProgress variant="determinate" value={78} sx={{ mb: 1, height: 6, borderRadius: 3 }} />
                          </Paper>
                        </Stack>
                      </Grid>
                    </Grid>
                  </Box>
                </TabPanel>

                <TabPanel value={currentTab} index={4}>
                  <Box sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      🎮 游戏化学习
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      通过有趣的游戏方式学习手语，在娱乐中提高技能
                    </Typography>
                    
                    <Grid container spacing={3}>
                      {[
                        {
                          title: '手语猜词',
                          description: '根据手语动作猜测词汇',
                          icon: <Psychology />,
                          color: '#B5EAD7',
                          difficulty: '简单',
                          time: '5-10分钟'
                        },
                        {
                          title: '节奏手语',
                          description: '跟随节拍进行手语表达',
                          icon: <VolumeUp />,
                          color: '#FFDAB9',
                          difficulty: '中等',
                          time: '10-15分钟'
                        },
                        {
                          title: '手语对话',
                          description: '模拟真实对话场景',
                          icon: <Group />,
                          color: '#FFB3BA',
                          difficulty: '困难',
                          time: '15-20分钟'
                        },
                        {
                          title: '速度挑战',
                          description: '在限定时间内完成手语表达',
                          icon: <Speed />,
                          color: '#C7CEDB',
                          difficulty: '专家',
                          time: '5分钟'
                        }
                      ].map((game, index) => (
                        <Grid item xs={12} sm={6} key={index}>
                          <Card 
                            sx={{ 
                              height: '100%',
                              background: `linear-gradient(135deg, ${game.color}20 0%, ${game.color}10 100%)`,
                              border: `1px solid ${game.color}30`,
                              borderRadius: 3,
                              cursor: 'pointer',
                              transition: 'all 0.3s ease',
                              '&:hover': {
                                transform: 'translateY(-4px)',
                                boxShadow: `0 8px 25px ${game.color}30`,
                              },
                            }}
                            onClick={() => showSnackbar(`开始游戏: ${game.title}`, 'info')}
                          >
                            <CardContent sx={{ p: 3 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <Avatar sx={{ bgcolor: game.color, mr: 2 }}>
                                  {game.icon}
                                </Avatar>
                                <Box>
                                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                    {game.title}
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {game.time}
                                  </Typography>
                                </Box>
                              </Box>
                              
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                {game.description}
                              </Typography>
                              
                              <Stack direction="row" spacing={1}>
                                <Chip
                                  label={game.difficulty}
                                  size="small"
                                  sx={{ backgroundColor: game.color, color: 'white' }}
                                />
                                <Chip
                                  label="免费"
                                  size="small"
                                  variant="outlined"
                                />
                              </Stack>
                            </CardContent>
                            
                            <CardActions sx={{ p: 3, pt: 0 }}>
                              <Button
                                variant="contained"
                                startIcon={<PlayArrow />}
                                sx={{
                                  background: `linear-gradient(135deg, ${game.color} 0%, ${game.color}CC 100%)`,
                                  color: 'white',
                                }}
                              >
                                开始游戏
                              </Button>
                            </CardActions>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                </TabPanel>

                <TabPanel value={currentTab} index={5}>
                  <Box sx={{ p: 4 }}>
                    <ExternalResources
                      onBookmark={(resourceId) => {
                        showSnackbar('资源已收藏', 'success')
                      }}
                    />
                  </Box>
                </TabPanel>
              </Paper>
            </Fade>
          </Stack>
        </Grid>

        {/* 右侧边栏 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={4}>
            {/* 今日任务增强版 */}
            <Fade in timeout={1400}>
              <Paper sx={{ p: 3, borderRadius: 4 }}>
                <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                  📋 今日任务
                </Typography>
                <Stack spacing={2}>
                  {dailyTasks.map((task, index) => (
                    <Box
                      key={task.id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        p: 2.5,
                        borderRadius: 3,
                        backgroundColor: task.completed ? 'success.light' : 'background.paper',
                        border: '1px solid',
                        borderColor: task.completed ? 'success.main' : 'divider',
                        opacity: task.completed ? 0.8 : 1,
                        transition: 'all 0.3s ease',
                        cursor: task.completed ? 'default' : 'pointer',
                        '&:hover': {
                          backgroundColor: task.completed ? 'success.light' : 'action.hover',
                        },
                      }}
                      onClick={() => !task.completed && completeLesson(task.id, 100)}
                    >
                      <Avatar 
                        sx={{ 
                          mr: 2,
                          bgcolor: task.completed ? 'success.main' : 'text.disabled',
                          width: 40,
                          height: 40,
                        }}
                      >
                        {task.completed ? <CheckCircle /> : task.icon}
                      </Avatar>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            textDecoration: task.completed ? 'line-through' : 'none',
                            fontWeight: 500,
                            mb: 0.5,
                          }}
                        >
                          {task.task}
                        </Typography>
                        {!task.completed && task.progress > 0 && (
                          <LinearProgress
                            variant="determinate"
                            value={(task.progress / task.target) * 100}
                            sx={{ height: 4, borderRadius: 2, mb: 0.5 }}
                          />
                        )}
                        <Typography variant="caption" color="text.secondary">
                          {task.completed ? '已完成' : `${task.progress}/${task.target}`}
                        </Typography>
                      </Box>
                      <Chip
                        label={`${task.xp} XP`}
                        size="small"
                        color={task.completed ? 'success' : 'default'}
                        sx={{ fontSize: '0.7rem', fontWeight: 600 }}
                      />
                    </Box>
                  ))}
                </Stack>
                
                <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                  <Typography variant="caption" color="text.secondary">
                    完成所有任务可获得额外奖励
                  </Typography>
                </Box>
              </Paper>
            </Fade>

            {/* 成就系统增强版 */}
            <Fade in timeout={1600}>
              <Paper sx={{ p: 3, borderRadius: 4 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    🏆 成就系统
                  </Typography>
                  <Button
                    variant="text"
                    size="small"
                    onClick={() => setShowAchievements(true)}
                  >
                    查看全部
                  </Button>
                </Box>
                <Stack spacing={2}>
                  {achievements.slice(0, 4).map((achievement) => (
                    <Box
                      key={achievement.id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        p: 2,
                        borderRadius: 3,
                        backgroundColor: achievement.unlocked ? `${achievement.color}20` : 'background.paper',
                        border: '1px solid',
                        borderColor: achievement.unlocked ? `${achievement.color}40` : 'divider',
                        position: 'relative',
                        overflow: 'hidden',
                      }}
                    >
                      {achievement.unlocked && (
                        <Box
                          sx={{
                            position: 'absolute',
                            top: 0,
                            right: 0,
                            background: 'linear-gradient(45deg, #FFD700, #FFA500)',
                            color: 'white',
                            px: 1,
                            py: 0.5,
                            fontSize: '0.6rem',
                            fontWeight: 600,
                            clipPath: 'polygon(0 0, 100% 0, 100% 70%, 85% 100%, 0 100%)',
                          }}
                        >
                          已解锁
                        </Box>
                      )}
                      
                      <Avatar
                        sx={{
                          bgcolor: achievement.unlocked ? achievement.color : 'text.disabled',
                          mr: 2,
                          width: 36,
                          height: 36,
                        }}
                      >
                        {achievement.icon}
                      </Avatar>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {achievement.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {achievement.description}
                        </Typography>
                        {!achievement.unlocked && (
                          <Box sx={{ mt: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={(achievement.progress / achievement.maxProgress) * 100}
                              sx={{ height: 4, borderRadius: 2 }}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {achievement.progress}/{achievement.maxProgress}
                            </Typography>
                          </Box>
                        )}
                        {achievement.unlocked && achievement.unlockedAt && (
                          <Typography variant="caption" color="success.main">
                            获得于 {achievement.unlockedAt}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  ))}
                </Stack>
              </Paper>
            </Fade>

            {/* 学习进度周报 */}
            <Fade in timeout={1800}>
              <Paper sx={{ p: 3, borderRadius: 4 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  📈 本周学习报告
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">周学习目标</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {userStats.weeklyProgress}/{userStats.weeklyGoal} 分钟
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={(userStats.weeklyProgress / userStats.weeklyGoal) * 100}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      backgroundColor: '#B5EAD720',
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 4,
                        background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                      },
                    }}
                  />
                </Box>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">本周课程</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>12节</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">平均分数</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>87分</Typography>
                  </Grid>
                </Grid>
              </Paper>
            </Fade>

            {/* 学习建议增强版 */}
            <Fade in timeout={2000}>
              <Paper 
                sx={{ 
                  p: 3, 
                  background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                  color: 'white',
                  borderRadius: 4,
                }}
              >
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  💡 智能学习建议
                </Typography>
                <Stack spacing={2}>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    🎯 根据您的学习进度，建议重点练习"数字表达"
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    ⏰ 建议每天学习20-30分钟，保持连续性
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    🤝 多与其他学习者交流，分享学习心得
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    📱 使用移动端随时随地练习手语
                  </Typography>
                </Stack>
                
                <Button
                  variant="contained"
                  size="small"
                  sx={{ 
                    mt: 2,
                    bgcolor: 'rgba(255,255,255,0.2)',
                    color: 'white',
                    '&:hover': { bgcolor: 'rgba(255,255,255,0.3)' }
                  }}
                  onClick={() => showSnackbar('已为您定制个性化学习计划', 'success')}
                >
                  获取定制计划
                </Button>
              </Paper>
            </Fade>
          </Stack>
        </Grid>
      </Grid>

      {/* 学习模块详情对话框 */}
      <Dialog
        open={!!selectedModule}
        onClose={() => setSelectedModule(null)}
        maxWidth="lg"
        fullWidth
        PaperProps={{ sx: { borderRadius: 4 } }}
      >
        {selectedModule && (
          <>
            <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', pb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Avatar
                  sx={{
                    bgcolor: selectedModule.color,
                    mr: 2,
                    width: 56,
                    height: 56,
                  }}
                >
                  {selectedModule.icon}
                </Avatar>
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 600, mb: 0.5 }}>
                    {selectedModule.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {selectedModule.category} • {selectedModule.instructor}
                  </Typography>
                </Box>
              </Box>
              <IconButton onClick={() => setSelectedModule(null)}>
                <Close />
              </IconButton>
            </DialogTitle>
            
            <DialogContent sx={{ pb: 2 }}>
              <Grid container spacing={4}>
                <Grid item xs={12} md={8}>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                    课程介绍
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {selectedModule.description}
                  </Typography>
                  
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, mt: 3 }}>
                    学习成果
                  </Typography>
                  <List dense>
                    {selectedModule.learningOutcomes?.map((outcome: string, index: number) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <CheckCircle color="success" />
                        </ListItemIcon>
                        <ListItemText primary={outcome} />
                      </ListItem>
                    ))}
                  </List>
                  
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, mt: 3 }}>
                    前置要求
                  </Typography>
                  {selectedModule.prerequisites?.length > 0 ? (
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                      {selectedModule.prerequisites.map((prereq: string, index: number) => (
                        <Chip
                          key={index}
                          label={prereq}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      无前置要求，适合初学者
                    </Typography>
                  )}
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Paper sx={{ p: 3, bgcolor: 'background.default', borderRadius: 3 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                      课程信息
                    </Typography>
                    <Stack spacing={2}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">难度等级:</Typography>
                        <Chip
                          label={getLevelLabel(selectedModule.level)}
                          size="small"
                          sx={{ backgroundColor: getLevelColor(selectedModule.level), color: 'white' }}
                        />
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">预计时间:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {selectedModule.estimatedTime}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">课程数量:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {selectedModule.totalLessons} 节
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">课程进度:</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {selectedModule.completedLessons}/{selectedModule.totalLessons}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">用户评分:</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Rating value={selectedModule.rating} precision={0.1} size="small" readOnly />
                          <Typography variant="caption" sx={{ ml: 0.5 }}>
                            ({selectedModule.reviews})
                          </Typography>
                        </Box>
                      </Box>
                    </Stack>
                  </Paper>
                  
                  <Paper sx={{ p: 3, bgcolor: 'background.default', borderRadius: 3, mt: 2 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                      学习技能
                    </Typography>
                    <Stack spacing={1}>
                      {selectedModule.skills?.map((skill: string, index: number) => (
                        <Chip
                          key={index}
                          label={skill}
                          size="small"
                          variant="outlined"
                          sx={{ alignSelf: 'flex-start' }}
                        />
                      ))}
                    </Stack>
                  </Paper>
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 4 }}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                  学习进度
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">完成度</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {selectedModule.progress}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={selectedModule.progress}
                  sx={{
                    height: 12,
                    borderRadius: 6,
                    backgroundColor: `${selectedModule.color}20`,
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 6,
                      backgroundColor: selectedModule.color,
                    },
                  }}
                />
              </Box>
            </DialogContent>
            
            <DialogActions sx={{ p: 3, gap: 2 }}>
              <Button 
                onClick={() => setSelectedModule(null)} 
                variant="outlined"
                sx={{ borderRadius: 3 }}
              >
                稍后学习
              </Button>
              <Button 
                onClick={() => bookmarkLesson(selectedModule.id)}
                variant="outlined"
                startIcon={<Bookmark />}
                sx={{ borderRadius: 3 }}
              >
                收藏课程
              </Button>
              <Button 
                variant="contained"
                startIcon={<PlayArrow />}
                sx={{
                  background: `linear-gradient(135deg, ${selectedModule.color} 0%, ${selectedModule.color}CC 100%)`,
                  borderRadius: 3,
                }}
                onClick={() => {
                  setSelectedModule(null)
                  showSnackbar(`开始学习: ${selectedModule.title}`, 'success')
                }}
              >
                开始学习
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* 学习路径详情对话框 */}
      <Dialog
        open={showLearningPath}
        onClose={() => setShowLearningPath(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{ sx: { borderRadius: 4 } }}
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            🛣️ 学习路径详情
          </Typography>
          <IconButton onClick={() => setShowLearningPath(false)}>
            <Close />
          </IconButton>
        </DialogTitle>
        
        <DialogContent>
          <Grid container spacing={3}>
            {learningPaths.map((path) => (
              <Grid item xs={12} key={path.id}>
                <Accordion sx={{ borderRadius: 3, border: `1px solid ${path.color}30` }}>
                  <AccordionSummary 
                    expandIcon={<ExpandMore />}
                    sx={{ 
                      bgcolor: `${path.color}10`,
                      borderRadius: '12px 12px 0 0',
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                      <Avatar sx={{ bgcolor: path.color, mr: 2 }}>
                        <MenuBook />
                      </Avatar>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {path.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {path.duration} • {path.estimatedHours}小时 • {path.enrolled}人已报名
                        </Typography>
                      </Box>
                      <Chip
                        label={`${path.completionRate}%完成率`}
                        size="small"
                        color="success"
                      />
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {path.description}
                    </Typography>
                    
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, mt: 2 }}>
                      学习步骤：
                    </Typography>
                    <Stepper orientation="vertical">
                      {path.steps.map((step, index) => (
                        <Step key={index} active>
                          <StepLabel>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {step.title}
                            </Typography>
                          </StepLabel>
                          <StepContent>
                            <Typography variant="body2" color="text.secondary">
                              {step.description}
                            </Typography>
                          </StepContent>
                        </Step>
                      ))}
                    </Stepper>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 3 }}>
                      <Stack direction="row" spacing={1}>
                        {path.skills.map((skill, index) => (
                          <Chip
                            key={index}
                            label={skill}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                      </Stack>
                      <Button
                        variant="contained"
                        startIcon={<PlayArrow />}
                        sx={{
                          background: `linear-gradient(135deg, ${path.color} 0%, ${path.color}CC 100%)`,
                        }}
                        onClick={() => {
                          setShowLearningPath(false)
                          showSnackbar(`开始学习路径: ${path.title}`, 'success')
                        }}
                      >
                        开始学习
                      </Button>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
      </Dialog>

      {/* 成就详情对话框 */}
      <Dialog
        open={showAchievements}
        onClose={() => setShowAchievements(false)}
        maxWidth="lg"
        fullWidth
        PaperProps={{ sx: { borderRadius: 4 } }}
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            🏆 成就收藏馆
          </Typography>
          <IconButton onClick={() => setShowAchievements(false)}>
            <Close />
          </IconButton>
        </DialogTitle>
        
        <DialogContent>
          {/* 成就分类 */}
          {['入门成就', '坚持成就', '效率成就', '掌握成就', '卓越成就', '社交成就', '时间成就'].map((category) => (
            <Box key={category} sx={{ mb: 4 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'primary.main' }}>
                {category}
              </Typography>
              <Grid container spacing={2}>
                {achievements
                  .filter(achievement => achievement.category === category)
                  .map((achievement) => (
                    <Grid item xs={12} sm={6} md={4} key={achievement.id}>
                      <Card
                        sx={{
                          background: achievement.unlocked ? 
                            `linear-gradient(135deg, ${achievement.color}20 0%, ${achievement.color}10 100%)` : 
                            'background.paper',
                          border: '1px solid',
                          borderColor: achievement.unlocked ? `${achievement.color}40` : 'divider',
                          borderRadius: 3,
                          position: 'relative',
                          overflow: 'hidden',
                        }}
                      >
                        {achievement.unlocked && (
                          <Box
                            sx={{
                              position: 'absolute',
                              top: 0,
                              right: 0,
                              background: 'linear-gradient(45deg, #FFD700, #FFA500)',
                              color: 'white',
                              px: 2,
                              py: 0.5,
                              fontSize: '0.7rem',
                              fontWeight: 600,
                              clipPath: 'polygon(0 0, 100% 0, 100% 70%, 85% 100%, 0 100%)',
                            }}
                          >
                            +{achievement.xpReward}XP
                          </Box>
                        )}
                        
                        <CardContent sx={{ p: 3, textAlign: 'center' }}>
                          <Avatar
                            sx={{
                              bgcolor: achievement.unlocked ? achievement.color : 'text.disabled',
                              mx: 'auto',
                              mb: 2,
                              width: 64,
                              height: 64,
                            }}
                          >
                            {achievement.icon}
                          </Avatar>
                          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                            {achievement.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {achievement.description}
                          </Typography>
                          
                          {achievement.unlocked ? (
                            <Chip
                              label={`已获得 • ${achievement.unlockedAt}`}
                              color="success"
                              size="small"
                              icon={<CheckCircle />}
                            />
                          ) : (
                            <Box>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                <Typography variant="caption">进度</Typography>
                                <Typography variant="caption">
                                  {achievement.progress}/{achievement.maxProgress}
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={(achievement.progress / achievement.maxProgress) * 100}
                                sx={{ height: 6, borderRadius: 3 }}
                              />
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
              </Grid>
            </Box>
          ))}
        </DialogContent>
      </Dialog>

      {/* 全局提示 */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity}
          sx={{ borderRadius: 3 }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* 认证模态框 */}
      <AuthModal
        open={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        initialMode="login"
      />

      {/* 交互式教程模态框 */}
      {selectedTutorial && (
        <InteractiveTutorial
          tutorialId={selectedTutorial.id}
          title={selectedTutorial.title}
          description={selectedTutorial.description}
          steps={selectedTutorial.steps}
          onComplete={(score) => {
            setShowTutorial(false)
            setSelectedTutorial(null)
            showSnackbar(`教程完成！得分: ${score}%`, 'success')
            completeLesson(selectedTutorial.id, score)
          }}
          onClose={() => {
            setShowTutorial(false)
            setSelectedTutorial(null)
          }}
        />
      )}

      {/* 练习会话模态框 */}
      {selectedPractice && (
        <PracticeSession
          exercises={selectedPractice.exercises}
          sessionTitle={selectedPractice.sessionTitle}
          onComplete={(results) => {
            setShowPractice(false)
            setSelectedPractice(null)
            const averageScore = results.reduce((sum, result) => sum + result.accuracy, 0) / results.length
            showSnackbar(`练习完成！平均准确率: ${Math.round(averageScore)}%`, 'success')
            completeLesson('practice-session', averageScore)
          }}
          onClose={() => {
            setShowPractice(false)
            setSelectedPractice(null)
          }}
        />
      )}
    </Container>
  )
}

export default LearningPage