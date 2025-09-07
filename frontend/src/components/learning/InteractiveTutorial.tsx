/**
 * 交互式手语教程组件
 * 提供步骤式指导和实时反馈的学习体验
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  Stack,
  Avatar,
  Chip,
  LinearProgress,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Divider,
  Grid,
} from '@mui/material'
import {
  PlayArrow,
  Pause,
  Replay,
  CheckCircle,
  School,
  Visibility,
  TouchApp,
  Timer,
  Star,
  Close,
  NavigateNext,
  NavigateBefore,
  EmojiEvents,
  Lightbulb,
  Assignment,
} from '@mui/icons-material'

interface TutorialStep {
  id: string
  title: string
  description: string
  instruction: string
  videoUrl?: string
  imageUrl?: string
  tips: string[]
  expectedAction: string
  difficulty: 'easy' | 'medium' | 'hard'
  estimatedTime: number
}

interface InteractiveTutorialProps {
  tutorialId: string
  title: string
  description: string
  steps: TutorialStep[]
  onComplete?: (score: number) => void
  onClose?: () => void
}

const InteractiveTutorial: React.FC<InteractiveTutorialProps> = ({
  tutorialId,
  title,
  description,
  steps,
  onComplete,
  onClose
}) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set())
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [showHints, setShowHints] = useState(false)
  const [stepScores, setStepScores] = useState<number[]>([])
  const [tutorialStartTime, setTutorialStartTime] = useState<Date | null>(null)

  useEffect(() => {
    setTutorialStartTime(new Date())
  }, [])

  useEffect(() => {
    const newProgress = (completedSteps.size / steps.length) * 100
    setProgress(newProgress)
  }, [completedSteps, steps.length])

  const handleStepComplete = (stepIndex: number, score: number = 100) => {
    const newCompletedSteps = new Set(completedSteps)
    newCompletedSteps.add(stepIndex)
    setCompletedSteps(newCompletedSteps)
    
    const newStepScores = [...stepScores]
    newStepScores[stepIndex] = score
    setStepScores(newStepScores)

    // 自动进入下一步
    if (stepIndex < steps.length - 1) {
      setTimeout(() => {
        setCurrentStep(stepIndex + 1)
      }, 1000)
    } else {
      // 教程完成
      handleTutorialComplete()
    }
  }

  const handleTutorialComplete = () => {
    const averageScore = stepScores.reduce((sum, score) => sum + score, 0) / stepScores.length
    const timeSpent = tutorialStartTime ? (new Date().getTime() - tutorialStartTime.getTime()) / 1000 / 60 : 0
    
    onComplete?.(averageScore)
  }

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return '#4CAF50'
      case 'medium': return '#FF9800'
      case 'hard': return '#F44336'
      default: return '#2196F3'
    }
  }

  const getDifficultyLabel = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return '简单'
      case 'medium': return '中等'
      case 'hard': return '困难'
      default: return '未知'
    }
  }

  const currentStepData = steps[currentStep]

  return (
    <Dialog
      open={true}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { borderRadius: 3, minHeight: '80vh' }
      }}
    >
      <DialogTitle sx={{ pb: 1 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h5" fontWeight="bold">
              {title}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {description}
            </Typography>
          </Box>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Stack>
        
        {/* 总体进度 */}
        <Box sx={{ mt: 2 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              总体进度
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {completedSteps.size}/{steps.length} 步骤完成
            </Typography>
          </Stack>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: 8,
              borderRadius: 4,
              bgcolor: 'grey.200',
              '& .MuiLinearProgress-bar': {
                borderRadius: 4,
                bgcolor: 'success.main',
              }
            }}
          />
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 0 }}>
        <Grid container sx={{ height: '100%' }}>
          {/* 左侧步骤导航 */}
          <Grid item xs={12} md={4} sx={{ borderRight: 1, borderColor: 'divider', p: 2 }}>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              学习步骤
            </Typography>
            <Stepper activeStep={currentStep} orientation="vertical">
              {steps.map((step, index) => (
                <Step key={step.id} completed={completedSteps.has(index)}>
                  <StepLabel
                    onClick={() => setCurrentStep(index)}
                    sx={{ cursor: 'pointer' }}
                    StepIconComponent={() => (
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          bgcolor: completedSteps.has(index) 
                            ? 'success.main' 
                            : index === currentStep 
                              ? 'primary.main' 
                              : 'grey.300',
                          fontSize: '0.8rem',
                        }}
                      >
                        {completedSteps.has(index) ? (
                          <CheckCircle fontSize="small" />
                        ) : (
                          index + 1
                        )}
                      </Avatar>
                    )}
                  >
                    <Typography variant="body2" fontWeight={index === currentStep ? 600 : 400}>
                      {step.title}
                    </Typography>
                  </StepLabel>
                  <StepContent>
                    <Typography variant="caption" color="text.secondary">
                      {step.description}
                    </Typography>
                    <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                      <Chip
                        label={getDifficultyLabel(step.difficulty)}
                        size="small"
                        sx={{
                          bgcolor: getDifficultyColor(step.difficulty),
                          color: 'white',
                          fontSize: '0.7rem',
                        }}
                      />
                      <Chip
                        label={`${step.estimatedTime}分钟`}
                        size="small"
                        variant="outlined"
                        sx={{ fontSize: '0.7rem' }}
                      />
                    </Stack>
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </Grid>

          {/* 右侧内容区域 */}
          <Grid item xs={12} md={8} sx={{ p: 3 }}>
            {currentStepData && (
              <Stack spacing={3}>
                {/* 步骤标题和信息 */}
                <Box>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                    <Typography variant="h5" fontWeight="bold">
                      {currentStepData.title}
                    </Typography>
                    <Stack direction="row" spacing={1}>
                      <Chip
                        icon={<Timer />}
                        label={`${currentStepData.estimatedTime}分钟`}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={getDifficultyLabel(currentStepData.difficulty)}
                        size="small"
                        sx={{
                          bgcolor: getDifficultyColor(currentStepData.difficulty),
                          color: 'white',
                        }}
                      />
                    </Stack>
                  </Stack>
                  <Typography variant="body1" color="text.secondary">
                    {currentStepData.description}
                  </Typography>
                </Box>

                {/* 学习内容区域 */}
                <Card sx={{ borderRadius: 3, border: '2px solid #E3F2FD' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      📋 学习指导
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      {currentStepData.instruction}
                    </Typography>

                    {/* 模拟视频/图片区域 */}
                    <Paper
                      sx={{
                        height: 200,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: 'grey.100',
                        borderRadius: 2,
                        mb: 2,
                      }}
                    >
                      <Stack alignItems="center" spacing={2}>
                        <Avatar sx={{ width: 60, height: 60, bgcolor: 'primary.main' }}>
                          <School fontSize="large" />
                        </Avatar>
                        <Typography variant="body2" color="text.secondary">
                          手语演示视频区域
                        </Typography>
                        <Stack direction="row" spacing={1}>
                          <IconButton
                            onClick={() => setIsPlaying(!isPlaying)}
                            sx={{ bgcolor: 'primary.main', color: 'white' }}
                          >
                            {isPlaying ? <Pause /> : <PlayArrow />}
                          </IconButton>
                          <IconButton sx={{ bgcolor: 'grey.300' }}>
                            <Replay />
                          </IconButton>
                        </Stack>
                      </Stack>
                    </Paper>

                    {/* 预期动作 */}
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        <strong>预期动作：</strong> {currentStepData.expectedAction}
                      </Typography>
                    </Alert>
                  </CardContent>
                </Card>

                {/* 提示和技巧 */}
                <Card sx={{ borderRadius: 3 }}>
                  <CardContent>
                    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                      <Typography variant="h6">
                        💡 学习提示
                      </Typography>
                      <Button
                        size="small"
                        onClick={() => setShowHints(!showHints)}
                        startIcon={<Lightbulb />}
                      >
                        {showHints ? '隐藏提示' : '显示提示'}
                      </Button>
                    </Stack>
                    
                    {showHints && (
                      <Stack spacing={1}>
                        {currentStepData.tips.map((tip, index) => (
                          <Typography key={index} variant="body2" sx={{ pl: 2 }}>
                            • {tip}
                          </Typography>
                        ))}
                      </Stack>
                    )}
                  </CardContent>
                </Card>

                {/* 练习区域 */}
                <Card sx={{ borderRadius: 3, bgcolor: 'success.50' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      🤝 开始练习
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      请按照指导进行手语练习，系统将实时检测您的动作
                    </Typography>
                    
                    <Stack direction="row" spacing={2} justifyContent="center">
                      <Button
                        variant="contained"
                        size="large"
                        startIcon={<TouchApp />}
                        onClick={() => handleStepComplete(currentStep, 95)}
                        sx={{
                          borderRadius: 3,
                          bgcolor: 'success.main',
                          '&:hover': { bgcolor: 'success.dark' }
                        }}
                      >
                        开始练习
                      </Button>
                      <Button
                        variant="outlined"
                        size="large"
                        startIcon={<Assignment />}
                        sx={{ borderRadius: 3 }}
                      >
                        查看详情
                      </Button>
                    </Stack>
                  </CardContent>
                </Card>
              </Stack>
            )}
          </Grid>
        </Grid>
      </DialogContent>

      <DialogActions sx={{ p: 3, borderTop: 1, borderColor: 'divider' }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ width: '100%' }}>
          <Button
            startIcon={<NavigateBefore />}
            onClick={handleBack}
            disabled={currentStep === 0}
          >
            上一步
          </Button>
          
          <Typography variant="body2" color="text.secondary">
            第 {currentStep + 1} 步，共 {steps.length} 步
          </Typography>
          
          <Button
            endIcon={<NavigateNext />}
            onClick={handleNext}
            disabled={currentStep === steps.length - 1}
            variant={completedSteps.has(currentStep) ? "contained" : "outlined"}
          >
            下一步
          </Button>
        </Stack>
      </DialogActions>
    </Dialog>
  )
}

export default InteractiveTutorial
