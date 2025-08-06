/**
 * 智能手语学习助手 - AI驱动的个性化学习体验
 */

import React, { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Grid,
  Chip,
  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  Tooltip,
  IconButton,
  Fade,
  Zoom,
} from '@mui/material'
import {
  School,
  Psychology,
  TrendingUp,
  Star,
  CheckCircle,
  PlayArrow,
  Pause,
  Replay,
  Lightbulb,
  Assessment,
  EmojiEvents,
  Close,
  Visibility,
  Speed,
  Favorite,
} from '@mui/icons-material'

interface LearningSession {
  id: string
  word: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  description: string
  keyPoints: string[]
  commonMistakes: string[]
  practiceCount: number
  accuracy: number
  lastPracticed: Date
  mastered: boolean
}

interface AIFeedback {
  score: number
  strengths: string[]
  improvements: string[]
  nextSteps: string[]
  encouragement: string
}

interface LearningStats {
  totalWordsLearned: number
  averageAccuracy: number
  streakDays: number
  weeklyGoalProgress: number
  skillLevel: string
  nextMilestone: string
}

interface IntelligentLearningAssistantProps {
  onWordSelect?: (word: string) => void
  currentAccuracy?: number
  onFeedbackRequest?: () => void
}

const IntelligentLearningAssistant: React.FC<IntelligentLearningAssistantProps> = ({
  onWordSelect,
  currentAccuracy = 0,
  onFeedbackRequest
}) => {
  const [currentSession, setCurrentSession] = useState<LearningSession | null>(null)
  const [aiFeedback, setAiFeedback] = useState<AIFeedback | null>(null)
  const [learningStats, setLearningStats] = useState<LearningStats>({
    totalWordsLearned: 0,
    averageAccuracy: 0,
    streakDays: 0,
    weeklyGoalProgress: 0,
    skillLevel: 'beginner',
    nextMilestone: '学会10个基础词汇'
  })
  
  const [activeStep, setActiveStep] = useState(0)
  const [isPracticing, setIsPracticing] = useState(false)
  const [showAIDialog, setShowAIDialog] = useState(false)
  const [practiceProgress, setPracticeProgress] = useState(0)
  const [achievements, setAchievements] = useState<string[]>([])

  // 预设学习内容
  const learningWords: LearningSession[] = [
    {
      id: '1',
      word: '你好',
      difficulty: 'beginner',
      description: '最基础的问候语，是手语学习的第一步',
      keyPoints: [
        '右手食指指向对方',
        '然后指向自己',
        '动作要自然流畅',
        '保持微笑表情'
      ],
      commonMistakes: [
        '手势过于僵硬',
        '缺乏眼神交流',
        '动作过快或过慢'
      ],
      practiceCount: 0,
      accuracy: 0,
      lastPracticed: new Date(),
      mastered: false
    },
    {
      id: '2',
      word: '谢谢',
      difficulty: 'beginner',
      description: '表达感谢的基本手语，日常交流中经常使用',
      keyPoints: [
        '右手掌心向上',
        '从胸前向前推出',
        '同时轻微点头',
        '表情要真诚'
      ],
      commonMistakes: [
        '手掌方向错误',
        '动作幅度过大',
        '忘记配合表情'
      ],
      practiceCount: 0,
      accuracy: 0,
      lastPracticed: new Date(),
      mastered: false
    },
    {
      id: '3',
      word: '我爱你',
      difficulty: 'intermediate',
      description: '表达爱意的手语，包含复杂的手部动作',
      keyPoints: [
        '食指指向自己(我)',
        '双手交叉放在胸前(爱)',
        '指向对方(你)',
        '动作要有感情'
      ],
      commonMistakes: [
        '动作顺序错乱',
        '手势不标准',
        '缺乏情感表达'
      ],
      practiceCount: 0,
      accuracy: 0,
      lastPracticed: new Date(),
      mastered: false
    }
  ]

  // AI智能推荐下一个学习内容
  const getAIRecommendation = useCallback(() => {
    const unmastered = learningWords.filter(word => !word.mastered)
    const beginnerWords = unmastered.filter(word => word.difficulty === 'beginner')
    
    // 优先推荐初级词汇
    if (beginnerWords.length > 0) {
      return beginnerWords[0]
    }
    
    // 根据准确率推荐适合难度
    if (currentAccuracy >= 80) {
      return unmastered.find(word => word.difficulty === 'advanced') || unmastered[0]
    } else if (currentAccuracy >= 60) {
      return unmastered.find(word => word.difficulty === 'intermediate') || unmastered[0]
    }
    
    return unmastered[0] || learningWords[0]
  }, [currentAccuracy])

  // 生成AI反馈
  const generateAIFeedback = useCallback((accuracy: number, word: string) => {
    let feedback: AIFeedback

    if (accuracy >= 90) {
      feedback = {
        score: accuracy,
        strengths: ['手势标准', '动作流畅', '表情自然'],
        improvements: ['可以尝试更复杂的词汇'],
        nextSteps: ['学习词汇组合', '练习对话场景'],
        encouragement: '太棒了！你的手语表达非常标准！'
      }
    } else if (accuracy >= 70) {
      feedback = {
        score: accuracy,
        strengths: ['基本动作正确', '学习态度认真'],
        improvements: ['注意手势细节', '增加练习频率'],
        nextSteps: ['重复练习当前词汇', '注意常见错误'],
        encouragement: '很好！继续保持这样的学习状态！'
      }
    } else if (accuracy >= 50) {
      feedback = {
        score: accuracy,
        strengths: ['有学习潜力', '勇于尝试'],
        improvements: ['放慢动作速度', '注意手势准确性'],
        nextSteps: ['多看示范视频', '分步骤练习'],
        encouragement: '不要气馁，每个人都有学习过程！'
      }
    } else {
      feedback = {
        score: accuracy,
        strengths: ['坚持学习的精神'],
        improvements: ['重新学习基本手势', '多练习基础动作'],
        nextSteps: ['回到基础教程', '寻求人工指导'],
        encouragement: '学习需要时间，相信自己一定能做到！'
      }
    }

    return feedback
  }, [])

  // 开始练习
  const startPractice = useCallback((word: LearningSession) => {
    setCurrentSession(word)
    setIsPracticing(true)
    setPracticeProgress(0)
    setActiveStep(0)
    onWordSelect?.(word.word)

    // 模拟练习进度
    const progressInterval = setInterval(() => {
      setPracticeProgress(prev => {
        const newProgress = prev + 10
        if (newProgress >= 100) {
          clearInterval(progressInterval)
          setIsPracticing(false)
          
          // 生成模拟准确率
          const mockAccuracy = Math.random() * 40 + 60 // 60-100%
          const feedback = generateAIFeedback(mockAccuracy, word.word)
          setAiFeedback(feedback)
          setShowAIDialog(true)
          
          // 更新学习统计
          setLearningStats(prev => ({
            ...prev,
            totalWordsLearned: prev.totalWordsLearned + (mockAccuracy > 80 ? 1 : 0),
            averageAccuracy: (prev.averageAccuracy + mockAccuracy) / 2,
            weeklyGoalProgress: Math.min(prev.weeklyGoalProgress + 10, 100)
          }))

          // 检查成就
          if (mockAccuracy > 95) {
            setAchievements(prev => [...prev, '完美表达'])
          }
          
          return 100
        }
        return newProgress
      })
    }, 500)
  }, [onWordSelect, generateAIFeedback])

  // 获取技能等级颜色
  const getSkillLevelColor = (level: string) => {
    switch (level) {
      case 'beginner': return 'info'
      case 'intermediate': return 'warning'
      case 'advanced': return 'success'
      default: return 'primary'
    }
  }

  // 练习步骤
  const practiceSteps = currentSession ? [
    {
      label: '观看示范',
      content: `观看"${currentSession.word}"的标准手语示范`
    },
    {
      label: '学习要点',
      content: '掌握关键动作要点和注意事项'
    },
    {
      label: '跟随练习',
      content: '跟随视频进行练习，注意手势准确性'
    },
    {
      label: '独立表演',
      content: '独立完成手语表达，系统进行识别评分'
    }
  ] : []

  useEffect(() => {
    // 初始化推荐内容
    const recommended = getAIRecommendation()
    setCurrentSession(recommended)
  }, [getAIRecommendation])

  return (
    <Box>
      {/* 学习统计卡片 */}
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Psychology color="primary" />
            智能学习助手
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="primary.main">
                  {learningStats.totalWordsLearned}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  已掌握词汇
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={6} sm={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="success.main">
                  {Math.round(learningStats.averageAccuracy)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  平均准确率
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={6} sm={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="warning.main">
                  {learningStats.streakDays}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  连续学习天数
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={6} sm={3}>
              <Box textAlign="center">
                <Chip
                  label={learningStats.skillLevel}
                  color={getSkillLevelColor(learningStats.skillLevel) as any}
                  icon={<Star />}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  当前等级
                </Typography>
              </Box>
            </Grid>
          </Grid>

          {/* 周目标进度 */}
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              本周学习目标: {learningStats.nextMilestone}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={learningStats.weeklyGoalProgress}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="caption" color="text.secondary">
              {learningStats.weeklyGoalProgress}% 完成
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* 当前学习会话 */}
      {currentSession && (
        <Card elevation={2} sx={{ mb: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <School color="primary" />
                正在学习: {currentSession.word}
              </Typography>
              
              <Chip
                label={currentSession.difficulty}
                color={getSkillLevelColor(currentSession.difficulty) as any}
                size="small"
              />
            </Box>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {currentSession.description}
            </Typography>

            {isPracticing ? (
              <Box>
                <Stepper activeStep={activeStep} orientation="vertical">
                  {practiceSteps.map((step, index) => (
                    <Step key={step.label}>
                      <StepLabel>{step.label}</StepLabel>
                      <StepContent>
                        <Typography variant="body2" sx={{ mb: 2 }}>
                          {step.content}
                        </Typography>
                        {index < practiceSteps.length - 1 && (
                          <Button
                            variant="contained"
                            onClick={() => setActiveStep(index + 1)}
                            size="small"
                          >
                            下一步
                          </Button>
                        )}
                      </StepContent>
                    </Step>
                  ))}
                </Stepper>

                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    练习进度: {practiceProgress}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={practiceProgress}
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </Box>
              </Box>
            ) : (
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  学习要点:
                </Typography>
                <List dense>
                  {currentSession.keyPoints.map((point, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Lightbulb color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={point} />
                    </ListItem>
                  ))}
                </List>

                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={() => startPractice(currentSession)}
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  开始练习
                </Button>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* 推荐学习内容 */}
      <Card elevation={2}>
        <CardContent>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <TrendingUp color="primary" />
            AI 推荐学习
          </Typography>

          <Grid container spacing={2}>
            {learningWords.slice(0, 3).map((word) => (
              <Grid item xs={12} sm={4} key={word.id}>
                <Card
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    '&:hover': {
                      boxShadow: 2,
                      transform: 'translateY(-2px)',
                    },
                  }}
                  onClick={() => setCurrentSession(word)}
                >
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Avatar
                      sx={{
                        width: 48,
                        height: 48,
                        mx: 'auto',
                        mb: 1,
                        bgcolor: word.mastered ? 'success.main' : 'primary.main',
                      }}
                    >
                      {word.mastered ? <CheckCircle /> : <School />}
                    </Avatar>
                    
                    <Typography variant="h6" gutterBottom>
                      {word.word}
                    </Typography>
                    
                    <Chip
                      label={word.difficulty}
                      size="small"
                      color={getSkillLevelColor(word.difficulty) as any}
                      sx={{ mb: 1 }}
                    />
                    
                    <Typography variant="body2" color="text.secondary">
                      练习 {word.practiceCount} 次
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* AI反馈对话框 */}
      <Dialog
        open={showAIDialog}
        onClose={() => setShowAIDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Psychology color="primary" />
            AI 学习反馈
          </Box>
          <IconButton onClick={() => setShowAIDialog(false)}>
            <Close />
          </IconButton>
        </DialogTitle>

        {aiFeedback && (
          <DialogContent>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Typography variant="h2" color="primary.main">
                {aiFeedback.score}
              </Typography>
              <Typography variant="h6" color="text.secondary">
                本次得分
              </Typography>
            </Box>

            <Alert severity="success" sx={{ mb: 2 }}>
              {aiFeedback.encouragement}
            </Alert>

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Favorite color="success" />
                  做得好的地方
                </Typography>
                <List>
                  {aiFeedback.strengths.map((strength, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <CheckCircle color="success" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={strength} />
                    </ListItem>
                  ))}
                </List>
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TrendingUp color="warning" />
                  可以改进的地方
                </Typography>
                <List>
                  {aiFeedback.improvements.map((improvement, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Lightbulb color="warning" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={improvement} />
                    </ListItem>
                  ))}
                </List>
              </Grid>
            </Grid>

            <Typography variant="h6" gutterBottom sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <Assessment color="primary" />
              下一步建议
            </Typography>
            <List>
              {aiFeedback.nextSteps.map((step, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <PlayArrow color="primary" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary={step} />
                </ListItem>
              ))}
            </List>
          </DialogContent>
        )}

        <DialogActions>
          <Button onClick={() => setShowAIDialog(false)}>
            关闭
          </Button>
          <Button variant="contained" onClick={() => {
            setShowAIDialog(false)
            const nextWord = getAIRecommendation()
            if (nextWord) {
              setCurrentSession(nextWord)
            }
          }}>
            继续学习
          </Button>
        </DialogActions>
      </Dialog>

      {/* 成就提示 */}
      {achievements.length > 0 && (
        <Zoom in={true}>
          <Box
            sx={{
              position: 'fixed',
              top: 20,
              right: 20,
              zIndex: 1000,
            }}
          >
            <Alert
              severity="success"
              icon={<EmojiEvents />}
              action={
                <IconButton
                  size="small"
                  onClick={() => setAchievements([])}
                >
                  <Close />
                </IconButton>
              }
            >
              获得成就: {achievements[achievements.length - 1]}
            </Alert>
          </Box>
        </Zoom>
      )}
    </Box>
  )
}

export default IntelligentLearningAssistant