/**
 * 优化的学习训练页面 - 增强学习体验和进度跟踪
 */

import { useState, useEffect } from 'react'
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
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import HandSignDemo from '../components/HandSignDemo'
import HandSignTestPanel from '../components/HandSignTestPanel'
import SimpleHandSignTest from '../components/SimpleHandSignTest'

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

function LearningPage() {
  const [currentTab, setCurrentTab] = useState(0)
  const [selectedModule, setSelectedModule] = useState<any>(null)
  const [showAchievements, setShowAchievements] = useState(false)
  const [userStats] = useState({
    totalLearningTime: 245,
    completedLessons: 28,
    currentStreak: 7,
    level: 15,
    totalXP: 1580,
    nextLevelXP: 1800,
  })

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue)
  }

  // 学习模块数据
  const learningModules = [
    {
      id: 'basic-signs',
      title: '基础手语',
      description: '学习最常用的手语词汇和基本表达',
      level: 'beginner',
      completedLessons: 8,
      totalLessons: 12,
      progress: 67,
      icon: <MenuBook />,
      color: '#B5EAD7',
      locked: false,
      estimatedTime: '2-3小时',
      skills: ['基础词汇', '日常用语', '问候语'],
      rating: 4.8,
      reviews: 156,
    },
    {
      id: 'numbers-time',
      title: '数字与时间',
      description: '掌握数字、时间和日期的手语表达',
      level: 'beginner',
      completedLessons: 5,
      totalLessons: 8,
      progress: 63,
      icon: <Timer />,
      color: '#FFDAB9',
      locked: false,
      estimatedTime: '1-2小时',
      skills: ['数字0-100', '时间表达', '日期表达'],
      rating: 4.6,
      reviews: 89,
    },
    {
      id: 'family-relations',
      title: '家庭关系',
      description: '学习家庭成员和人际关系相关手语',
      level: 'intermediate',
      completedLessons: 3,
      totalLessons: 10,
      progress: 30,
      icon: <EmojiEvents />,
      color: '#FFB3BA',
      locked: false,
      estimatedTime: '2-3小时',
      skills: ['家庭成员', '关系称谓', '情感表达'],
      rating: 4.7,
      reviews: 124,
    },
    {
      id: 'advanced-grammar',
      title: '高级语法',
      description: '掌握复杂的手语语法结构和表达技巧',
      level: 'advanced',
      completedLessons: 0,
      totalLessons: 15,
      progress: 0,
      icon: <Psychology />,
      color: '#C7CEDB',
      locked: true,
      estimatedTime: '4-6小时',
      skills: ['语法结构', '时态表达', '复合句型'],
      rating: 4.9,
      reviews: 67,
    },
  ]

  // 成就系统
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
    },
  ]

  const learningPaths = [
    {
      title: '快速入门路径',
      description: '适合零基础学习者的快速入门课程',
      duration: '1-2周',
      modules: ['basic-signs', 'numbers-time'],
      difficulty: 'beginner',
      color: '#B5EAD7',
    },
    {
      title: '日常交流路径',
      description: '学习日常生活中最常用的手语表达',
      duration: '3-4周',
      modules: ['basic-signs', 'family-relations', 'numbers-time'],
      difficulty: 'intermediate',
      color: '#FFDAB9',
    },
    {
      title: '专业进阶路径',
      description: '深入学习手语语法和高级表达技巧',
      duration: '6-8周',
      modules: ['basic-signs', 'family-relations', 'advanced-grammar'],
      difficulty: 'advanced',
      color: '#C7CEDB',
    },
  ]

  const dailyTasks = [
    { task: '完成一节基础课程', completed: true, xp: 50 },
    { task: '练习10个新词汇', completed: true, xp: 30 },
    { task: '通过一次手语测试', completed: false, xp: 80 },
    { task: '观看3个演示视频', completed: false, xp: 40 },
  ]

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'beginner': return '#B5EAD7'
      case 'intermediate': return '#FFDAB9'
      case 'advanced': return '#C7CEDB'
      default: return '#B5EAD7'
    }
  }

  const getLevelLabel = (level: string) => {
    switch (level) {
      case 'beginner': return '初级'
      case 'intermediate': return '中级'
      case 'advanced': return '高级'
      default: return '初级'
    }
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题和用户统计 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 6 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8}>
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
                    学习训练
                  </Typography>
                  <Typography variant="h6" color="text.secondary">
                    系统化学习手语，提升沟通技能
                  </Typography>
                </Box>
              </Stack>
            </Grid>
            
            <Grid item xs={12} md={4}>
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

      <Grid container spacing={4}>
        {/* 左侧主要内容 */}
        <Grid item xs={12} lg={8}>
          <Stack spacing={4}>
            {/* 学习路径推荐 */}
            <Fade in timeout={800}>
              <Box>
                <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                  推荐学习路径
                </Typography>
                <Grid container spacing={3}>
                  {learningPaths.map((path, index) => (
                    <Grid item xs={12} md={4} key={path.title}>
                      <Card
                        sx={{
                          height: '100%',
                          background: `linear-gradient(135deg, ${path.color}20 0%, ${path.color}10 100%)`,
                          border: `1px solid ${path.color}30`,
                          borderRadius: 3,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: `0 8px 25px ${path.color}30`,
                          },
                        }}
                      >
                        <CardContent sx={{ p: 3 }}>
                          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                            {path.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {path.description}
                          </Typography>
                          <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
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
                          </Stack>
                          <Typography variant="caption" color="text.secondary">
                            包含 {path.modules.length} 个模块
                          </Typography>
                        </CardContent>
                        <CardActions sx={{ p: 3, pt: 0 }}>
                          <Button
                            variant="outlined"
                            size="small"
                            sx={{
                              borderColor: path.color,
                              color: path.color,
                              '&:hover': {
                                backgroundColor: path.color,
                                color: 'white',
                              },
                            }}
                          >
                            开始学习
                          </Button>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Fade>

            {/* 学习模块 */}
            <Fade in timeout={1000}>
              <Box>
                <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                  学习模块
                </Typography>
                <Grid container spacing={3}>
                  {learningModules.map((module, index) => (
                    <Grid item xs={12} sm={6} key={module.id}>
                      <Card
                        sx={{
                          height: '100%',
                          background: `linear-gradient(135deg, ${module.color}20 0%, ${module.color}10 100%)`,
                          border: `1px solid ${module.color}30`,
                          borderRadius: 3,
                          opacity: module.locked ? 0.7 : 1,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: module.locked ? 'none' : 'translateY(-4px)',
                            boxShadow: module.locked ? 'none' : `0 8px 25px ${module.color}30`,
                          },
                          cursor: module.locked ? 'not-allowed' : 'pointer',
                        }}
                        onClick={() => !module.locked && setSelectedModule(module)}
                      >
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
                              <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
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
                              </Stack>
                            </Box>
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {module.description}
                          </Typography>
                          
                          {!module.locked && (
                            <>
                              <Box sx={{ mb: 2 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                  <Typography variant="body2">进度</Typography>
                                  <Typography variant="body2">{module.progress}%</Typography>
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
                              
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" color="text.secondary">
                                  {module.completedLessons}/{module.totalLessons} 课程
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Rating value={module.rating} precision={0.1} size="small" readOnly />
                                  <Typography variant="caption" sx={{ ml: 0.5 }}>
                                    ({module.reviews})
                                  </Typography>
                                </Box>
                              </Box>
                            </>
                          )}
                          
                          {module.locked && (
                            <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                              完成前置课程后解锁
                            </Typography>
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
                    icon={<VideoLibrary />} 
                    label="演示学习" 
                    iconPosition="start"
                  />
                  <Tab 
                    icon={<Quiz />} 
                    label="能力测试" 
                    iconPosition="start"
                  />
                  <Tab 
                    icon={<Games />} 
                    label="互动练习" 
                    iconPosition="start"
                  />
                </Tabs>

                <TabPanel value={currentTab} index={0}>
                  <Box sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      手语动作演示
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      观看标准的手语动作演示，学习正确的表达方式
                    </Typography>
                    <ErrorBoundary>
                      <HandSignDemo />
                    </ErrorBoundary>
                  </Box>
                </TabPanel>

                <TabPanel value={currentTab} index={1}>
                  <Box sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      手语能力测试
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      测试您的手语识别能力，获得专业的能力评估
                    </Typography>
                    <ErrorBoundary>
                      <HandSignTestPanel />
                    </ErrorBoundary>
                  </Box>
                </TabPanel>

                <TabPanel value={currentTab} index={2}>
                  <Box sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      简单手语练习
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      从简单的手语动作开始，逐步提高你的手语技能
                    </Typography>
                    <ErrorBoundary>
                      <SimpleHandSignTest />
                    </ErrorBoundary>
                  </Box>
                </TabPanel>
              </Paper>
            </Fade>
          </Stack>
        </Grid>

        {/* 右侧边栏 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={4}>
            {/* 今日任务 */}
            <Fade in timeout={1400}>
              <Paper sx={{ p: 3, borderRadius: 4 }}>
                <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                  📋 今日任务
                </Typography>
                <Stack spacing={2}>
                  {dailyTasks.map((task, index) => (
                    <Box
                      key={index}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        p: 2,
                        borderRadius: 2,
                        backgroundColor: task.completed ? 'success.light' : 'background.paper',
                        border: '1px solid',
                        borderColor: task.completed ? 'success.main' : 'divider',
                        opacity: task.completed ? 0.8 : 1,
                      }}
                    >
                      <CheckCircle 
                        sx={{ 
                          mr: 2, 
                          color: task.completed ? 'success.main' : 'text.disabled',
                        }} 
                      />
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            textDecoration: task.completed ? 'line-through' : 'none',
                            fontWeight: 500,
                          }}
                        >
                          {task.task}
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
              </Paper>
            </Fade>

            {/* 成就系统 */}
            <Fade in timeout={1600}>
              <Paper sx={{ p: 3, borderRadius: 4 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    🏆 成就
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
                  {achievements.slice(0, 3).map((achievement) => (
                    <Box
                      key={achievement.id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        p: 2,
                        borderRadius: 2,
                        backgroundColor: achievement.unlocked ? `${achievement.color}20` : 'background.paper',
                        border: '1px solid',
                        borderColor: achievement.unlocked ? `${achievement.color}40` : 'divider',
                      }}
                    >
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
                          <LinearProgress
                            variant="determinate"
                            value={(achievement.progress / achievement.maxProgress) * 100}
                            size="small"
                            sx={{ mt: 1, height: 4, borderRadius: 2 }}
                          />
                        )}
                      </Box>
                    </Box>
                  ))}
                </Stack>
              </Paper>
            </Fade>

            {/* 学习建议 */}
            <Fade in timeout={1800}>
              <Paper 
                sx={{ 
                  p: 3, 
                  background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                  color: 'white',
                  borderRadius: 4,
                }}
              >
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  💡 学习建议
                </Typography>
                <Stack spacing={2}>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    • 每天坚持学习15-30分钟，保持连续性
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    • 多练习手语动作，注意手型和动作规范
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    • 与其他学习者交流，分享学习心得
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.95 }}>
                    • 观看手语视频，模仿标准动作
                  </Typography>
                </Stack>
              </Paper>
            </Fade>
          </Stack>
        </Grid>
      </Grid>

      {/* 学习模块详情对话框 */}
      <Dialog
        open={!!selectedModule}
        onClose={() => setSelectedModule(null)}
        maxWidth="md"
        fullWidth
        PaperProps={{ sx: { borderRadius: 4 } }}
      >
        {selectedModule && (
          <>
            <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Avatar
                  sx={{
                    bgcolor: selectedModule.color,
                    mr: 2,
                    width: 48,
                    height: 48,
                  }}
                >
                  {selectedModule.icon}
                </Avatar>
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {selectedModule.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {selectedModule.description}
                  </Typography>
                </Box>
              </Box>
              <IconButton onClick={() => setSelectedModule(null)}>
                <Close />
              </IconButton>
            </DialogTitle>
            
            <DialogContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
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
                      <Typography variant="body2">{selectedModule.estimatedTime}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">课程进度:</Typography>
                      <Typography variant="body2">{selectedModule.completedLessons}/{selectedModule.totalLessons}</Typography>
                    </Box>
                  </Stack>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                    学习技能
                  </Typography>
                  <Stack spacing={1}>
                    {selectedModule.skills.map((skill: string, index: number) => (
                      <Chip
                        key={index}
                        label={skill}
                        size="small"
                        variant="outlined"
                        sx={{ alignSelf: 'flex-start' }}
                      />
                    ))}
                  </Stack>
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                  学习进度
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">完成度</Typography>
                  <Typography variant="body2">{selectedModule.progress}%</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={selectedModule.progress}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: `${selectedModule.color}20`,
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 4,
                      backgroundColor: selectedModule.color,
                    },
                  }}
                />
              </Box>
            </DialogContent>
            
            <DialogActions sx={{ p: 3, pt: 0 }}>
              <Button onClick={() => setSelectedModule(null)} variant="outlined">
                稍后学习
              </Button>
              <Button 
                variant="contained"
                startIcon={<PlayArrow />}
                sx={{
                  background: `linear-gradient(135deg, ${selectedModule.color} 0%, ${selectedModule.color}CC 100%)`,
                }}
              >
                开始学习
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* 成就详情对话框 */}
      <Dialog
        open={showAchievements}
        onClose={() => setShowAchievements(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{ sx: { borderRadius: 4 } }}
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            🏆 我的成就
          </Typography>
          <IconButton onClick={() => setShowAchievements(false)}>
            <Close />
          </IconButton>
        </DialogTitle>
        
        <DialogContent>
          <Grid container spacing={2}>
            {achievements.map((achievement) => (
              <Grid item xs={12} sm={6} key={achievement.id}>
                <Card
                  sx={{
                    background: achievement.unlocked ? `${achievement.color}20` : 'background.paper',
                    border: '1px solid',
                    borderColor: achievement.unlocked ? `${achievement.color}40` : 'divider',
                    borderRadius: 3,
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Avatar
                        sx={{
                          bgcolor: achievement.unlocked ? achievement.color : 'text.disabled',
                          mr: 2,
                          width: 48,
                          height: 48,
                        }}
                      >
                        {achievement.icon}
                      </Avatar>
                      <Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {achievement.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {achievement.description}
                        </Typography>
                      </Box>
                    </Box>
                    
                    {achievement.unlocked ? (
                      <Chip
                        label={`已获得 · ${achievement.unlockedAt}`}
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
        </DialogContent>
      </Dialog>
    </Container>
  )
}

export default LearningPage