/**
 * 用户学习仪表板组件
 * 显示个性化学习进度、统计和推荐
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Avatar,
  Stack,
  Chip,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Button,
  IconButton,
  Tooltip,
  CircularProgress,
  Badge,
} from '@mui/material'
import {
  TrendingUp,
  EmojiEvents,
  LocalFireDepartment,
  School,
  Timer,
  Star,
  Assignment,
  BookmarkBorder,
  PlayArrow,
  CheckCircle,
  Schedule,
  Psychology,
  Speed,
  Lightbulb,
  AutoAwesome,
  CalendarToday,
} from '@mui/icons-material'

import { useAuth } from '../../contexts/AuthContext'

interface UserStats {
  totalLearningTime: number
  completedLessons: number
  currentStreak: number
  level: number
  totalXP: number
  nextLevelXP: number
  weeklyGoal: number
  weeklyProgress: number
  monthlyGoal: number
  monthlyProgress: number
}

interface Achievement {
  id: string
  title: string
  description: string
  icon: React.ReactNode
  unlocked: boolean
  unlockedAt?: string
  progress?: number
  maxProgress?: number
}

interface LearningRecommendation {
  id: string
  title: string
  description: string
  type: 'lesson' | 'practice' | 'review'
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  estimatedTime: number
  icon: React.ReactNode
}

interface UserDashboardProps {
  userStats: UserStats
  onStartLesson?: (lessonId: string) => void
}

const UserDashboard: React.FC<UserDashboardProps> = ({
  userStats,
  onStartLesson
}) => {
  const { user } = useAuth()
  const [recentAchievements, setRecentAchievements] = useState<Achievement[]>([])
  const [recommendations, setRecommendations] = useState<LearningRecommendation[]>([])
  const [weeklyActivity, setWeeklyActivity] = useState<number[]>([])

  useEffect(() => {
    // 模拟加载最近成就
    setRecentAchievements([
      {
        id: '1',
        title: '连续学习者',
        description: '连续学习7天',
        icon: <LocalFireDepartment />,
        unlocked: true,
        unlockedAt: '2024-01-15',
      },
      {
        id: '2',
        title: '快速学习者',
        description: '单日完成5个课程',
        icon: <Speed />,
        unlocked: true,
        unlockedAt: '2024-01-14',
      },
      {
        id: '3',
        title: '知识探索者',
        description: '完成30个课程',
        icon: <Psychology />,
        unlocked: false,
        progress: 28,
        maxProgress: 30,
      },
    ])

    // 模拟加载个性化推荐
    setRecommendations([
      {
        id: '1',
        title: '数字手语进阶',
        description: '基于您的学习进度推荐',
        type: 'lesson',
        difficulty: 'intermediate',
        estimatedTime: 15,
        icon: <School />,
      },
      {
        id: '2',
        title: '日常对话练习',
        description: '巩固已学内容',
        type: 'practice',
        difficulty: 'beginner',
        estimatedTime: 10,
        icon: <Assignment />,
      },
      {
        id: '3',
        title: '复习基础手语',
        description: '温故而知新',
        type: 'review',
        difficulty: 'beginner',
        estimatedTime: 8,
        icon: <AutoAwesome />,
      },
    ])

    // 模拟一周学习活动数据
    setWeeklyActivity([45, 30, 60, 25, 40, 55, 35])
  }, [])

  const levelProgress = ((userStats.totalXP % 100) / 100) * 100
  const weeklyProgressPercent = (userStats.weeklyProgress / userStats.weeklyGoal) * 100
  const monthlyProgressPercent = (userStats.monthlyProgress / userStats.monthlyGoal) * 100

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '#4CAF50'
      case 'intermediate': return '#FF9800'
      case 'advanced': return '#F44336'
      default: return '#2196F3'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'lesson': return <School />
      case 'practice': return <Assignment />
      case 'review': return <AutoAwesome />
      default: return <Lightbulb />
    }
  }

  return (
    <Box>
      <Grid container spacing={3}>
        {/* 用户信息卡片 */}
        <Grid item xs={12} md={4}>
          <Card sx={{ borderRadius: 3, background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)' }}>
            <CardContent sx={{ textAlign: 'center', color: 'white' }}>
              <Avatar
                sx={{
                  width: 80,
                  height: 80,
                  mx: 'auto',
                  mb: 2,
                  bgcolor: 'rgba(255,255,255,0.2)',
                  fontSize: '2rem',
                }}
              >
                {user?.username?.charAt(0).toUpperCase() || 'U'}
              </Avatar>
              <Typography variant="h5" fontWeight="bold" gutterBottom>
                {user?.full_name || user?.username || '学习者'}
              </Typography>
              <Stack direction="row" spacing={1} justifyContent="center" sx={{ mb: 2 }}>
                <Chip
                  icon={<TrendingUp />}
                  label={`等级 ${userStats.level}`}
                  size="small"
                  sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                />
                <Chip
                  icon={<LocalFireDepartment />}
                  label={`${userStats.currentStreak}天`}
                  size="small"
                  sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                />
              </Stack>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" sx={{ mb: 1, opacity: 0.9 }}>
                  距离下一级还需 {userStats.nextLevelXP - userStats.totalXP} XP
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={levelProgress}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    bgcolor: 'rgba(255,255,255,0.2)',
                    '& .MuiLinearProgress-bar': {
                      bgcolor: 'white',
                    }
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 学习统计 */}
        <Grid item xs={12} md={8}>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#B5EAD7', mx: 'auto', mb: 1 }}>
                  <Timer />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {userStats.totalLearningTime}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  总学习时长(分钟)
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#FFB3BA', mx: 'auto', mb: 1 }}>
                  <CheckCircle />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {userStats.completedLessons}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  完成课程
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#FFDAB9', mx: 'auto', mb: 1 }}>
                  <Star />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {userStats.totalXP}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  总经验值
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#C7CEDB', mx: 'auto', mb: 1 }}>
                  <EmojiEvents />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {recentAchievements.filter(a => a.unlocked).length}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  获得成就
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Grid>

        {/* 学习目标进度 */}
        <Grid item xs={12} md={6}>
          <Card sx={{ borderRadius: 3, height: '100%' }}>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                📊 学习目标
              </Typography>
              <Stack spacing={3}>
                <Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="body2">本周目标</Typography>
                    <Typography variant="body2" color="primary">
                      {userStats.weeklyProgress}/{userStats.weeklyGoal} 分钟
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(weeklyProgressPercent, 100)}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
                <Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="body2">本月目标</Typography>
                    <Typography variant="body2" color="primary">
                      {userStats.monthlyProgress}/{userStats.monthlyGoal} 分钟
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(monthlyProgressPercent, 100)}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* 最近成就 */}
        <Grid item xs={12} md={6}>
          <Card sx={{ borderRadius: 3, height: '100%' }}>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                🏆 最近成就
              </Typography>
              <List dense>
                {recentAchievements.slice(0, 3).map((achievement) => (
                  <ListItem key={achievement.id} sx={{ px: 0 }}>
                    <ListItemIcon>
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          bgcolor: achievement.unlocked ? '#FFD700' : '#E0E0E0',
                          color: achievement.unlocked ? 'white' : 'text.secondary',
                        }}
                      >
                        {achievement.icon}
                      </Avatar>
                    </ListItemIcon>
                    <ListItemText
                      primary={achievement.title}
                      secondary={
                        achievement.unlocked
                          ? `已解锁 • ${achievement.unlockedAt}`
                          : `进度: ${achievement.progress}/${achievement.maxProgress}`
                      }
                      primaryTypographyProps={{
                        fontWeight: achievement.unlocked ? 600 : 400,
                        color: achievement.unlocked ? 'text.primary' : 'text.secondary',
                      }}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default UserDashboard
