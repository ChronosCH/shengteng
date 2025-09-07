/**
 * 练习会话组件
 * 提供实时反馈的手语练习体验
 */

import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Avatar,
  Chip,
  LinearProgress,
  Paper,
  Grid,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Divider,
} from '@mui/material'
import {
  PlayArrow,
  Pause,
  Stop,
  Replay,
  CheckCircle,
  Error,
  Warning,
  Timer,
  Speed,
  TrendingUp,
  EmojiEvents,
  Star,
  TouchApp,
  Visibility,
  Close,
} from '@mui/icons-material'

interface PracticeExercise {
  id: string
  word: string
  description: string
  difficulty: 'easy' | 'medium' | 'hard'
  category: string
  expectedGesture: string
  hints: string[]
}

interface PracticeResult {
  exerciseId: string
  accuracy: number
  speed: number
  attempts: number
  timeSpent: number
  feedback: string
}

interface PracticeSessionProps {
  exercises: PracticeExercise[]
  sessionTitle: string
  onComplete?: (results: PracticeResult[]) => void
  onClose?: () => void
}

const PracticeSession: React.FC<PracticeSessionProps> = ({
  exercises,
  sessionTitle,
  onComplete,
  onClose
}) => {
  const [currentExercise, setCurrentExercise] = useState(0)
  const [isRecording, setIsRecording] = useState(false)
  const [sessionResults, setSessionResults] = useState<PracticeResult[]>([])
  const [currentResult, setCurrentResult] = useState<Partial<PracticeResult>>({})
  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null)
  const [exerciseStartTime, setExerciseStartTime] = useState<Date | null>(null)
  const [feedback, setFeedback] = useState<{type: 'success' | 'warning' | 'error', message: string} | null>(null)
  const [showResults, setShowResults] = useState(false)
  const [attempts, setAttempts] = useState(0)
  const [realTimeFeedback, setRealTimeFeedback] = useState<string>('')

  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    setSessionStartTime(new Date())
    startExercise()
  }, [])

  const startExercise = () => {
    setExerciseStartTime(new Date())
    setAttempts(0)
    setCurrentResult({
      exerciseId: exercises[currentExercise].id,
      attempts: 0,
      timeSpent: 0,
    })
    setFeedback(null)
    setRealTimeFeedback('')
  }

  const startRecording = async () => {
    try {
      setIsRecording(true)
      setAttempts(prev => prev + 1)
      
      // 模拟实时手语识别反馈
      const feedbackMessages = [
        '检测到手部动作...',
        '正在分析手势...',
        '手势形状良好',
        '注意手指位置',
        '动作幅度可以更大一些',
        '很好，继续保持',
      ]
      
      let messageIndex = 0
      const feedbackInterval = setInterval(() => {
        if (messageIndex < feedbackMessages.length) {
          setRealTimeFeedback(feedbackMessages[messageIndex])
          messageIndex++
        }
      }, 1000)

      // 模拟识别过程（3-5秒）
      setTimeout(() => {
        clearInterval(feedbackInterval)
        stopRecording()
      }, Math.random() * 2000 + 3000)

    } catch (error) {
      console.error('录制启动失败:', error)
      setIsRecording(false)
    }
  }

  const stopRecording = () => {
    setIsRecording(false)
    setRealTimeFeedback('')
    
    // 模拟识别结果
    const accuracy = Math.random() * 40 + 60 // 60-100%
    const speed = Math.random() * 30 + 70 // 70-100%
    const timeSpent = exerciseStartTime ? (new Date().getTime() - exerciseStartTime.getTime()) / 1000 : 0

    const result: PracticeResult = {
      exerciseId: exercises[currentExercise].id,
      accuracy: Math.round(accuracy),
      speed: Math.round(speed),
      attempts: attempts,
      timeSpent: Math.round(timeSpent),
      feedback: generateFeedback(accuracy, speed, attempts)
    }

    setCurrentResult(result)
    
    // 设置反馈
    if (accuracy >= 80) {
      setFeedback({
        type: 'success',
        message: `太棒了！准确率 ${Math.round(accuracy)}%，继续保持！`
      })
    } else if (accuracy >= 60) {
      setFeedback({
        type: 'warning',
        message: `不错！准确率 ${Math.round(accuracy)}%，还可以更好！`
      })
    } else {
      setFeedback({
        type: 'error',
        message: `需要练习！准确率 ${Math.round(accuracy)}%，再试一次吧！`
      })
    }
  }

  const generateFeedback = (accuracy: number, speed: number, attempts: number): string => {
    const feedbacks = []
    
    if (accuracy >= 90) {
      feedbacks.push('手势非常准确')
    } else if (accuracy >= 70) {
      feedbacks.push('手势基本正确，注意细节')
    } else {
      feedbacks.push('手势需要改进，请参考示范')
    }

    if (speed >= 85) {
      feedbacks.push('动作流畅自然')
    } else if (speed >= 70) {
      feedbacks.push('动作稍显生硬，多练习会更好')
    } else {
      feedbacks.push('动作需要更加流畅')
    }

    if (attempts === 1) {
      feedbacks.push('一次成功，很棒！')
    } else if (attempts <= 3) {
      feedbacks.push('经过几次尝试达到了要求')
    } else {
      feedbacks.push('多次练习后有所改善')
    }

    return feedbacks.join('，')
  }

  const nextExercise = () => {
    if (currentResult.accuracy && currentResult.accuracy >= 60) {
      const newResults = [...sessionResults, currentResult as PracticeResult]
      setSessionResults(newResults)

      if (currentExercise < exercises.length - 1) {
        setCurrentExercise(prev => prev + 1)
        startExercise()
      } else {
        // 练习完成
        setShowResults(true)
        onComplete?.(newResults)
      }
    }
  }

  const retryExercise = () => {
    startExercise()
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

  const currentExerciseData = exercises[currentExercise]
  const progress = ((currentExercise + 1) / exercises.length) * 100

  if (showResults) {
    const averageAccuracy = sessionResults.reduce((sum, result) => sum + result.accuracy, 0) / sessionResults.length
    const totalTime = sessionResults.reduce((sum, result) => sum + result.timeSpent, 0)
    const totalAttempts = sessionResults.reduce((sum, result) => sum + result.attempts, 0)

    return (
      <Dialog open={true} onClose={onClose} maxWidth="md" fullWidth>
        <DialogTitle>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h5" fontWeight="bold">
              🎉 练习完成！
            </Typography>
            <Button onClick={onClose} startIcon={<Close />}>
              关闭
            </Button>
          </Stack>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <Paper sx={{ p: 3, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#4CAF50', mx: 'auto', mb: 2, width: 60, height: 60 }}>
                  <TrendingUp fontSize="large" />
                </Avatar>
                <Typography variant="h4" fontWeight="bold">
                  {Math.round(averageAccuracy)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  平均准确率
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Paper sx={{ p: 3, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#2196F3', mx: 'auto', mb: 2, width: 60, height: 60 }}>
                  <Timer fontSize="large" />
                </Avatar>
                <Typography variant="h4" fontWeight="bold">
                  {Math.round(totalTime)}s
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  总用时
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Paper sx={{ p: 3, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#FF9800', mx: 'auto', mb: 2, width: 60, height: 60 }}>
                  <EmojiEvents fontSize="large" />
                </Avatar>
                <Typography variant="h4" fontWeight="bold">
                  {sessionResults.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  完成练习
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" fontWeight="bold" gutterBottom>
            详细结果
          </Typography>
          <Stack spacing={2}>
            {sessionResults.map((result, index) => (
              <Card key={result.exerciseId} sx={{ borderRadius: 2 }}>
                <CardContent sx={{ py: 2 }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="subtitle1" fontWeight="bold">
                        {exercises[index].word}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {result.feedback}
                      </Typography>
                    </Box>
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Chip
                        label={`${result.accuracy}%`}
                        color={result.accuracy >= 80 ? 'success' : result.accuracy >= 60 ? 'warning' : 'error'}
                        size="small"
                      />
                      <Chip
                        label={`${result.attempts}次尝试`}
                        variant="outlined"
                        size="small"
                      />
                    </Stack>
                  </Stack>
                </CardContent>
              </Card>
            ))}
          </Stack>
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <Dialog open={true} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="h5" fontWeight="bold">
            {sessionTitle}
          </Typography>
          <Button onClick={onClose} startIcon={<Close />}>
            退出练习
          </Button>
        </Stack>
        
        <Box sx={{ mt: 2 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              练习进度
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {currentExercise + 1}/{exercises.length}
            </Typography>
          </Stack>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
      </DialogTitle>

      <DialogContent>
        <Grid container spacing={3}>
          {/* 左侧练习内容 */}
          <Grid item xs={12} md={8}>
            <Stack spacing={3}>
              {/* 当前练习信息 */}
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                    <Typography variant="h4" fontWeight="bold">
                      {currentExerciseData.word}
                    </Typography>
                    <Stack direction="row" spacing={1}>
                      <Chip
                        label={currentExerciseData.category}
                        variant="outlined"
                        size="small"
                      />
                      <Chip
                        label={getDifficultyLabel(currentExerciseData.difficulty)}
                        size="small"
                        sx={{
                          bgcolor: getDifficultyColor(currentExerciseData.difficulty),
                          color: 'white',
                        }}
                      />
                    </Stack>
                  </Stack>
                  <Typography variant="body1" color="text.secondary">
                    {currentExerciseData.description}
                  </Typography>
                </CardContent>
              </Card>

              {/* 摄像头区域 */}
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📹 练习区域
                  </Typography>
                  <Paper
                    sx={{
                      height: 300,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      bgcolor: 'grey.100',
                      borderRadius: 2,
                      position: 'relative',
                    }}
                  >
                    {isRecording ? (
                      <Stack alignItems="center" spacing={2}>
                        <CircularProgress size={60} color="error" />
                        <Typography variant="h6" color="error">
                          正在录制...
                        </Typography>
                        {realTimeFeedback && (
                          <Typography variant="body2" color="text.secondary">
                            {realTimeFeedback}
                          </Typography>
                        )}
                      </Stack>
                    ) : (
                      <Stack alignItems="center" spacing={2}>
                        <Avatar sx={{ width: 80, height: 80, bgcolor: 'primary.main' }}>
                          <TouchApp fontSize="large" />
                        </Avatar>
                        <Typography variant="h6">
                          准备开始练习
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          点击开始按钮进行手语练习
                        </Typography>
                      </Stack>
                    )}
                  </Paper>

                  {/* 控制按钮 */}
                  <Stack direction="row" spacing={2} justifyContent="center" sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      size="large"
                      startIcon={isRecording ? <Stop /> : <PlayArrow />}
                      onClick={isRecording ? stopRecording : startRecording}
                      color={isRecording ? "error" : "primary"}
                      sx={{ borderRadius: 3 }}
                    >
                      {isRecording ? '停止录制' : '开始练习'}
                    </Button>
                    <Button
                      variant="outlined"
                      size="large"
                      startIcon={<Replay />}
                      onClick={retryExercise}
                      disabled={isRecording}
                      sx={{ borderRadius: 3 }}
                    >
                      重新开始
                    </Button>
                  </Stack>
                </CardContent>
              </Card>

              {/* 反馈区域 */}
              {feedback && (
                <Alert severity={feedback.type} sx={{ borderRadius: 3 }}>
                  <Typography variant="body1">
                    {feedback.message}
                  </Typography>
                  {currentResult.accuracy && (
                    <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                      <Typography variant="body2">
                        准确率: {currentResult.accuracy}%
                      </Typography>
                      <Typography variant="body2">
                        尝试次数: {attempts}
                      </Typography>
                    </Stack>
                  )}
                </Alert>
              )}
            </Stack>
          </Grid>

          {/* 右侧提示和指导 */}
          <Grid item xs={12} md={4}>
            <Stack spacing={3}>
              {/* 预期手势 */}
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🎯 预期手势
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {currentExerciseData.expectedGesture}
                  </Typography>
                </CardContent>
              </Card>

              {/* 学习提示 */}
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    💡 学习提示
                  </Typography>
                  <Stack spacing={1}>
                    {currentExerciseData.hints.map((hint, index) => (
                      <Typography key={index} variant="body2" sx={{ pl: 1 }}>
                        • {hint}
                      </Typography>
                    ))}
                  </Stack>
                </CardContent>
              </Card>

              {/* 当前统计 */}
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📊 当前统计
                  </Typography>
                  <Stack spacing={2}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        尝试次数
                      </Typography>
                      <Typography variant="h6">
                        {attempts}
                      </Typography>
                    </Box>
                    {currentResult.accuracy && (
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          当前准确率
                        </Typography>
                        <Typography variant="h6" color="primary">
                          {currentResult.accuracy}%
                        </Typography>
                      </Box>
                    )}
                  </Stack>
                </CardContent>
              </Card>
            </Stack>
          </Grid>
        </Grid>
      </DialogContent>

      <DialogActions sx={{ p: 3 }}>
        <Button
          onClick={retryExercise}
          disabled={isRecording}
        >
          重新练习
        </Button>
        <Button
          variant="contained"
          onClick={nextExercise}
          disabled={!currentResult.accuracy || currentResult.accuracy < 60}
          sx={{ borderRadius: 3 }}
        >
          {currentExercise === exercises.length - 1 ? '完成练习' : '下一个'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default PracticeSession
