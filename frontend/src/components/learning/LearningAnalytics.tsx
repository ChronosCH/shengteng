/**
 * 学习分析组件
 * 显示详细的学习统计和进度分析
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Stack,
  Chip,
  Avatar,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tab,
  Tabs,
  CircularProgress,
} from '@mui/material'
import {
  TrendingUp,
  TrendingDown,
  Schedule,
  EmojiEvents,
  Assignment,
  Speed,
  Psychology,
  CalendarToday,
  Timer,
  Star,
  CheckCircle,
  LocalFireDepartment,
  School,
  AutoAwesome,
} from '@mui/icons-material'

interface LearningSession {
  date: string
  duration: number
  lessonsCompleted: number
  xpEarned: number
  accuracy: number
}

interface SkillProgress {
  skill: string
  level: number
  maxLevel: number
  progress: number
  color: string
  icon: React.ReactNode
}

interface LearningAnalyticsProps {
  userStats: {
    totalLearningTime: number
    completedLessons: number
    currentStreak: number
    level: number
    totalXP: number
  }
}

const LearningAnalytics: React.FC<LearningAnalyticsProps> = ({ userStats }) => {
  const [currentTab, setCurrentTab] = useState(0)
  const [weeklyData, setWeeklyData] = useState<LearningSession[]>([])
  const [skillProgress, setSkillProgress] = useState<SkillProgress[]>([])
  const [learningTrends, setLearningTrends] = useState({
    timeSpentTrend: 15, // 百分比变化
    accuracyTrend: 8,
    speedTrend: -3,
    consistencyScore: 85,
  })

  useEffect(() => {
    // 模拟一周学习数据
    setWeeklyData([
      { date: '2024-01-15', duration: 45, lessonsCompleted: 3, xpEarned: 120, accuracy: 92 },
      { date: '2024-01-14', duration: 30, lessonsCompleted: 2, xpEarned: 80, accuracy: 88 },
      { date: '2024-01-13', duration: 60, lessonsCompleted: 4, xpEarned: 160, accuracy: 95 },
      { date: '2024-01-12', duration: 25, lessonsCompleted: 2, xpEarned: 70, accuracy: 85 },
      { date: '2024-01-11', duration: 40, lessonsCompleted: 3, xpEarned: 110, accuracy: 90 },
      { date: '2024-01-10', duration: 55, lessonsCompleted: 4, xpEarned: 150, accuracy: 93 },
      { date: '2024-01-09', duration: 35, lessonsCompleted: 2, xpEarned: 90, accuracy: 87 },
    ])

    // 模拟技能进度数据
    setSkillProgress([
      {
        skill: '基础手语',
        level: 8,
        maxLevel: 10,
        progress: 80,
        color: '#4CAF50',
        icon: <School />,
      },
      {
        skill: '数字表达',
        level: 6,
        maxLevel: 10,
        progress: 60,
        color: '#2196F3',
        icon: <Assignment />,
      },
      {
        skill: '日常对话',
        level: 5,
        maxLevel: 10,
        progress: 50,
        color: '#FF9800',
        icon: <Psychology />,
      },
      {
        skill: '情感表达',
        level: 3,
        maxLevel: 10,
        progress: 30,
        color: '#E91E63',
        icon: <EmojiEvents />,
      },
      {
        skill: '语法规则',
        level: 4,
        maxLevel: 10,
        progress: 40,
        color: '#9C27B0',
        icon: <AutoAwesome />,
      },
    ])
  }, [])

  const averageAccuracy = weeklyData.reduce((sum, session) => sum + session.accuracy, 0) / weeklyData.length
  const totalWeeklyTime = weeklyData.reduce((sum, session) => sum + session.duration, 0)
  const totalWeeklyXP = weeklyData.reduce((sum, session) => sum + session.xpEarned, 0)
  const averageDailyTime = totalWeeklyTime / 7

  const getTrendIcon = (trend: number) => {
    if (trend > 0) return <TrendingUp color="success" />
    if (trend < 0) return <TrendingDown color="error" />
    return <TrendingUp color="disabled" />
  }

  const getTrendColor = (trend: number) => {
    if (trend > 0) return 'success.main'
    if (trend < 0) return 'error.main'
    return 'text.secondary'
  }

  const TabPanel = ({ children, value, index }: any) => (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  )

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
        📊 学习分析
      </Typography>

      <Tabs value={currentTab} onChange={(_, newValue) => setCurrentTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="总览" />
        <Tab label="技能进度" />
        <Tab label="学习趋势" />
      </Tabs>

      <TabPanel value={currentTab} index={0}>
        <Grid container spacing={3}>
          {/* 本周统计 */}
          <Grid item xs={12} md={8}>
            <Card sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  📅 本周学习概览
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 2 }}>
                      <Avatar sx={{ bgcolor: '#B5EAD7', mx: 'auto', mb: 1 }}>
                        <Timer />
                      </Avatar>
                      <Typography variant="h6" fontWeight="bold">
                        {totalWeeklyTime}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        总时长(分钟)
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 2 }}>
                      <Avatar sx={{ bgcolor: '#FFB3BA', mx: 'auto', mb: 1 }}>
                        <CheckCircle />
                      </Avatar>
                      <Typography variant="h6" fontWeight="bold">
                        {weeklyData.reduce((sum, session) => sum + session.lessonsCompleted, 0)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        完成课程
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 2 }}>
                      <Avatar sx={{ bgcolor: '#FFDAB9', mx: 'auto', mb: 1 }}>
                        <Star />
                      </Avatar>
                      <Typography variant="h6" fontWeight="bold">
                        {totalWeeklyXP}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        获得经验
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 2 }}>
                      <Avatar sx={{ bgcolor: '#C7CEDB', mx: 'auto', mb: 1 }}>
                        <Speed />
                      </Avatar>
                      <Typography variant="h6" fontWeight="bold">
                        {Math.round(averageAccuracy)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        平均准确率
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>

                <Divider sx={{ my: 3 }} />

                {/* 每日学习记录 */}
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  每日学习记录
                </Typography>
                <List dense>
                  {weeklyData.slice(0, 5).map((session, index) => (
                    <ListItem key={session.date} sx={{ px: 0 }}>
                      <ListItemIcon>
                        <Avatar sx={{ width: 32, height: 32, bgcolor: '#E3F2FD' }}>
                          <CalendarToday fontSize="small" />
                        </Avatar>
                      </ListItemIcon>
                      <ListItemText
                        primary={`${session.date} - ${session.duration}分钟`}
                        secondary={`${session.lessonsCompleted}个课程 • ${session.xpEarned} XP • ${session.accuracy}% 准确率`}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* 学习趋势 */}
          <Grid item xs={12} md={4}>
            <Card sx={{ borderRadius: 3, height: '100%' }}>
              <CardContent>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  📈 学习趋势
                </Typography>
                <Stack spacing={3}>
                  <Box>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2">学习时长</Typography>
                      <Stack direction="row" alignItems="center" spacing={0.5}>
                        {getTrendIcon(learningTrends.timeSpentTrend)}
                        <Typography
                          variant="body2"
                          color={getTrendColor(learningTrends.timeSpentTrend)}
                          fontWeight="bold"
                        >
                          {learningTrends.timeSpentTrend > 0 ? '+' : ''}{learningTrends.timeSpentTrend}%
                        </Typography>
                      </Stack>
                    </Stack>
                  </Box>

                  <Box>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2">准确率</Typography>
                      <Stack direction="row" alignItems="center" spacing={0.5}>
                        {getTrendIcon(learningTrends.accuracyTrend)}
                        <Typography
                          variant="body2"
                          color={getTrendColor(learningTrends.accuracyTrend)}
                          fontWeight="bold"
                        >
                          {learningTrends.accuracyTrend > 0 ? '+' : ''}{learningTrends.accuracyTrend}%
                        </Typography>
                      </Stack>
                    </Stack>
                  </Box>

                  <Box>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2">学习速度</Typography>
                      <Stack direction="row" alignItems="center" spacing={0.5}>
                        {getTrendIcon(learningTrends.speedTrend)}
                        <Typography
                          variant="body2"
                          color={getTrendColor(learningTrends.speedTrend)}
                          fontWeight="bold"
                        >
                          {learningTrends.speedTrend > 0 ? '+' : ''}{learningTrends.speedTrend}%
                        </Typography>
                      </Stack>
                    </Stack>
                  </Box>

                  <Divider />

                  <Box>
                    <Typography variant="body2" gutterBottom>
                      学习一致性
                    </Typography>
                    <Stack direction="row" alignItems="center" spacing={2}>
                      <CircularProgress
                        variant="determinate"
                        value={learningTrends.consistencyScore}
                        size={40}
                        thickness={4}
                        sx={{
                          color: learningTrends.consistencyScore > 80 ? 'success.main' : 
                                 learningTrends.consistencyScore > 60 ? 'warning.main' : 'error.main'
                        }}
                      />
                      <Typography variant="h6" fontWeight="bold">
                        {learningTrends.consistencyScore}%
                      </Typography>
                    </Stack>
                    <Typography variant="caption" color="text.secondary">
                      基于学习频率和时长计算
                    </Typography>
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={currentTab} index={1}>
        <Grid container spacing={3}>
          {skillProgress.map((skill) => (
            <Grid item xs={12} sm={6} md={4} key={skill.skill}>
              <Card sx={{ borderRadius: 3, border: `2px solid ${skill.color}30` }}>
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                    <Avatar sx={{ bgcolor: skill.color }}>
                      {skill.icon}
                    </Avatar>
                    <Box flex={1}>
                      <Typography variant="h6" fontWeight="bold">
                        {skill.skill}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        等级 {skill.level}/{skill.maxLevel}
                      </Typography>
                    </Box>
                  </Stack>
                  <LinearProgress
                    variant="determinate"
                    value={skill.progress}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      bgcolor: skill.color + '20',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: skill.color,
                      }
                    }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    {skill.progress}% 完成
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>

      <TabPanel value={currentTab} index={2}>
        <Card sx={{ borderRadius: 3 }}>
          <CardContent>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              📊 详细趋势分析
            </Typography>
            <Typography variant="body2" color="text.secondary">
              详细的趋势分析图表将在后续版本中实现，包括：
            </Typography>
            <List dense sx={{ mt: 2 }}>
              <ListItem>
                <ListItemIcon><TrendingUp /></ListItemIcon>
                <ListItemText primary="学习时长变化趋势" />
              </ListItem>
              <ListItem>
                <ListItemIcon><Speed /></ListItemIcon>
                <ListItemText primary="准确率提升曲线" />
              </ListItem>
              <ListItem>
                <ListItemIcon><EmojiEvents /></ListItemIcon>
                <ListItemText primary="技能掌握进度" />
              </ListItem>
              <ListItem>
                <ListItemIcon><LocalFireDepartment /></ListItemIcon>
                <ListItemText primary="学习连续性分析" />
              </ListItem>
            </List>
          </CardContent>
        </Card>
      </TabPanel>
    </Box>
  )
}

export default LearningAnalytics
