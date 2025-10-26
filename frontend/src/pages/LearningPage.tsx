/**
 * 手语学习训练页面 - 简化版
 * 核心功能：孤立手语识别、AI教学助手、能力测试
 */

import { useState, useCallback } from 'react'
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Paper,
  Chip,
  Stack,
  Avatar,
  Alert,
  Snackbar,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Divider,
  CircularProgress,
} from '@mui/material'
import {
  School,
  CheckCircle,
  Psychology,
  TouchApp,
} from '@mui/icons-material'

import { useAuth } from '../contexts/AuthContext'
import AuthModal from '../components/auth/AuthModal'
import AITutorChat from '../components/learning/AITutorChat'
import CameraRecorder from '../components/CameraRecorder'
import isolatedSignLearningService from '../services/isolatedSignLearningService'

function LearningPage() {
  // 核心状态
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as any })
  const [authModalOpen, setAuthModalOpen] = useState(false)
  
  // 孤立手语识别相关
  const [isolatedUploadLoading, setIsolatedUploadLoading] = useState(false)
  const [isolatedPrediction, setIsolatedPrediction] = useState<any>(null)
  const [isolatedVideoPath, setIsolatedVideoPath] = useState('')
  
  // AI助手相关
  const [aiTutorOpen, setAiTutorOpen] = useState(false)
  const [recognitionContext, setRecognitionContext] = useState<any>(null)

  // 认证状态
  const { isAuthenticated, user } = useAuth()

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'warning' | 'info' = 'info') => {
    setSnackbar({ open: true, message, severity })
  }

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false })
  }

  // 孤立手语上传识别
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
      
      const predictionData = {
        gloss: predictResp.prediction.gloss,
        confidence: predictResp.prediction.confidence,
        feedback: predictResp.feedback || null,
      }
      
      setIsolatedPrediction(predictionData)
      
      setRecognitionContext({
        recognized_sign: predictResp.prediction.gloss,
        confidence: predictResp.prediction.confidence,
      })
      
      showSnackbar('识别完成，查看结果', 'success')
    } catch (error: any) {
      console.error('手语识别失败:', error)
      showSnackbar(error?.message || '孤立手语识别失败', 'error')
    } finally {
      setIsolatedUploadLoading(false)
    }
  }, [isAuthenticated])

  // 摄像头录制完成后的处理
  const handleCameraRecordComplete = useCallback(async (videoBlob: Blob) => {
    if (!isAuthenticated) {
      setAuthModalOpen(true)
      showSnackbar('请先登录后再进行识别', 'warning')
      return
    }

    try {
      setIsolatedUploadLoading(true)
      setIsolatedPrediction(null)

      // 将 Blob 转换为 File 对象
      const videoFile = new File([videoBlob], `recorded-${Date.now()}.webm`, { type: 'video/webm' })

      const uploadResp = await isolatedSignLearningService.uploadIsolatedVideo(videoFile)
      setIsolatedVideoPath(uploadResp.file_path)

      const predictResp = await isolatedSignLearningService.predictIsolatedVideo(uploadResp.file_path)
      
      const predictionData = {
        gloss: predictResp.prediction.gloss,
        confidence: predictResp.prediction.confidence,
        feedback: predictResp.feedback || null,
      }
      
      setIsolatedPrediction(predictionData)
      
      setRecognitionContext({
        recognized_sign: predictResp.prediction.gloss,
        confidence: predictResp.prediction.confidence,
      })
      
      showSnackbar('识别完成，查看结果', 'success')
    } catch (error: any) {
      console.error('手语识别失败:', error)
      showSnackbar(error?.message || '孤立手语识别失败', 'error')
    } finally {
      setIsolatedUploadLoading(false)
    }
  }, [isAuthenticated])

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Box sx={{ mb: 4 }}>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <Avatar
            sx={{
              width: 64,
              height: 64,
              background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
            }}
          >
            <School sx={{ fontSize: 32, color: 'white' }} />
          </Avatar>
          <Box>
            <Typography 
              variant="h3" 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              手语学习训练
            </Typography>
            <Typography variant="h6" color="text.secondary">
              通过AI辅助学习手语，提升沟通能力
            </Typography>
          </Box>
        </Stack>
      </Box>

      {/* 欢迎提示 */}
      {isAuthenticated && user && (
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
            欢迎回来，{user.full_name || user.username}！ 👋
          </Typography>
          <Typography variant="body2" sx={{ mt: 0.5 }}>
            开始你的手语学习之旅吧！有任何问题都可以向AI助手提问。
          </Typography>
        </Alert>
      )}

      {!isAuthenticated && (
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
          }}
        >
          <Typography variant="body1" sx={{ fontWeight: 600 }}>
            登录以保存学习进度 📚
          </Typography>
        </Alert>
      )}

      {/* 主要内容区 */}
      <Paper sx={{ borderRadius: 4, overflow: 'hidden' }}>
        {/* 手语练习区域 */}
        <Box sx={{ p: 4 }}>
          {/* AI助手引导卡片 */}
          <Alert 
            severity="info" 
            icon={<Psychology />}
            action={
              <Button 
                color="inherit" 
                size="small" 
                onClick={() => setAiTutorOpen(true)}
                sx={{ fontWeight: 600 }}
              >
                立即咨询
              </Button>
            }
            sx={{ mb: 3, borderRadius: 3 }}
          >
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              🤖 AI助手随时待命！点击右下角按钮或这里，向我提问任何手语学习问题。
            </Typography>
          </Alert>

          <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
            🎯 孤立手语识别练习
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            可以通过摄像头实时录制手语动作，或上传视频文件，系统将自动识别并提供详细反馈。
          </Typography>

            <Grid container spacing={3}>
              {/* 摄像头录制区域 */}
              <Grid item xs={12} md={6}>
                <CameraRecorder
                  onRecordComplete={handleCameraRecordComplete}
                  maxDuration={10}
                  countdown={3}
                />
              </Grid>

              {/* 文件上传区域 */}
              <Grid item xs={12} md={6}>
                <Card sx={{ borderRadius: 3, background: '#f0f7ff', height: '100%' }}>
                  <CardContent sx={{ p: 4 }}>
                    <Stack spacing={3} sx={{ height: '100%' }}>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        📤 上传视频文件
                      </Typography>
                      
                      {/* 上传按钮 */}
                      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                        <Button
                          component="label"
                          variant="contained"
                          size="large"
                          disabled={isolatedUploadLoading}
                          sx={{ 
                            borderRadius: 3,
                            px: 4,
                            py: 2,
                            background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
                          }}
                          startIcon={isolatedUploadLoading ? <CircularProgress size={20} /> : <TouchApp />}
                        >
                          {isolatedUploadLoading ? '识别中...' : '选择手语视频'}
                          <input
                            type="file"
                            hidden
                            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
                            onChange={handleIsolatedSignUpload}
                          />
                        </Button>
                        {isolatedVideoPath && (
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                            已上传: {isolatedVideoPath.split('/').pop()}
                          </Typography>
                        )}
                      </Box>

                      <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center' }}>
                        支持格式: MP4, MOV, AVI, MKV, WebM
                      </Typography>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>

              {/* 识别结果区域 */}
              <Grid item xs={12}>
                {isolatedPrediction && (
                  <Card sx={{ borderRadius: 3, background: '#f0f7ff' }}>
                    <CardContent sx={{ p: 4 }}>
                      <Stack spacing={3}>
                        {/* 识别结果和反馈 */}
                        <Paper elevation={2} sx={{ p: 3, borderRadius: 3, backgroundColor: '#fff' }}>
                          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                            识别结果
                          </Typography>
                          
                          <Stack spacing={2}>
                            {/* 识别信息 */}
                            <Stack direction="row" spacing={2} alignItems="center">
                              <Chip
                                color="primary"
                                label={isolatedPrediction.gloss || '未识别'}
                                sx={{ fontSize: '1.2rem', px: 3, py: 2, height: 'auto' }}
                              />
                              <Typography variant="body1" color="text.secondary">
                                准确率：{(isolatedPrediction.confidence * 100).toFixed(1)}%
                              </Typography>
                            </Stack>
                            
                            {/* 学习反馈 */}
                            {isolatedPrediction.feedback && (
                              <Box>
                                <Divider sx={{ my: 2 }} />
                                
                                {/* 反馈消息 */}
                                <Alert 
                                  severity={
                                    isolatedPrediction.feedback.accuracy_level === 'excellent' ? 'success' :
                                    isolatedPrediction.feedback.accuracy_level === 'good' ? 'info' :
                                    isolatedPrediction.feedback.accuracy_level === 'fair' ? 'warning' : 'error'
                                  }
                                  sx={{ mb: 2 }}
                                >
                                  {isolatedPrediction.feedback.message}
                                </Alert>
                                
                                {/* 改进建议 */}
                                {isolatedPrediction.feedback.tips?.length > 0 && (
                                  <Box sx={{ mb: 2 }}>
                                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                      💡 改进建议
                                    </Typography>
                                    <List dense>
                                      {isolatedPrediction.feedback.tips.map((tip: string, index: number) => (
                                        <ListItem key={index}>
                                          <ListItemIcon sx={{ minWidth: 32 }}>
                                            <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />
                                          </ListItemIcon>
                                          <ListItemText primary={tip} />
                                        </ListItem>
                                      ))}
                                    </List>
                                  </Box>
                                )}
                                
                                {/* 下一步行动 */}
                                {isolatedPrediction.feedback.next_steps?.length > 0 && (
                                  <Box>
                                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                      🎯 下一步
                                    </Typography>
                                    <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
                                      {isolatedPrediction.feedback.next_steps.map((step: string, index: number) => (
                                        <Chip
                                          key={index}
                                          label={step}
                                          size="small"
                                          variant="outlined"
                                          color="primary"
                                        />
                                      ))}
                                    </Stack>
                                  </Box>
                                )}
                                
                                {/* AI助手按钮 */}
                                <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                                  <Button
                                    variant="contained"
                                    startIcon={<Psychology />}
                                    onClick={() => setAiTutorOpen(true)}
                                    sx={{
                                      background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                                      color: 'white',
                                      borderRadius: 3,
                                    }}
                                  >
                                    向AI助手请教
                                  </Button>
                                  <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>
                                    有问题？AI助手会根据你的练习情况给出建议！
                                  </Typography>
                                </Box>
                              </Box>
                            )}
                          </Stack>
                        </Paper>
                      </Stack>
                    </CardContent>
                  </Card>
                )}
              </Grid>
            </Grid>
        </Box>
      </Paper>

      {/* AI教学助手对话框 */}
      <AITutorChat
        open={aiTutorOpen}
        onClose={() => setAiTutorOpen(false)}
        recognitionContext={recognitionContext}
      />

      {/* AI助手浮动按钮 */}
      <Box
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 9999,
        }}
      >
        <Tooltip title="🤖 AI手语助手 - 点击向我提问" placement="left">
          <Button
            variant="contained"
            onClick={() => setAiTutorOpen(true)}
            sx={{
              width: 72,
              height: 72,
              borderRadius: '50%',
              minWidth: 'unset',
              background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
              boxShadow: '0 8px 24px rgba(181, 234, 215, 0.5)',
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%, 100%': {
                  boxShadow: '0 8px 24px rgba(181, 234, 215, 0.5)',
                },
                '50%': {
                  boxShadow: '0 12px 32px rgba(181, 234, 215, 0.8)',
                },
              },
              '&:hover': {
                transform: 'scale(1.15)',
                boxShadow: '0 12px 32px rgba(181, 234, 215, 0.6)',
              },
              transition: 'all 0.3s ease',
            }}
          >
            <Psychology sx={{ fontSize: 36, color: 'white' }} />
          </Button>
        </Tooltip>
      </Box>

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
    </Container>
  )
}

export default LearningPage
