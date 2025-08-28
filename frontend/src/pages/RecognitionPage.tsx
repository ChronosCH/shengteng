import { useState, useEffect } from 'react'
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  Chip,
  Stack,
  Card,
  CardContent,
  Alert,
  Snackbar,
  CircularProgress,
  Avatar,
  LinearProgress,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Favorite,
  Warning,
  CheckCircle,
  VideoCall,
  Visibility,
  Settings,
  TipsAndUpdates,
  Security,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import SafeFade from '../components/SafeFade'
import VideoCapture from '../components/VideoCapture'
import SubtitleDisplay from '../components/SubtitleDisplay'
import AvatarViewer from '../components/AvatarViewer'
import ThreeDErrorBoundary from '../components/ThreeDErrorBoundary'
import { useSignLanguageRecognition } from '../hooks/useSignLanguageRecognition'
import VideoFileRecognition from '../components/VideoFileRecognition'
import EnhancedVideoRecognition from '../components/EnhancedVideoRecognition'

function RecognitionPage() {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)
  const [isMounted, setIsMounted] = useState(false)

  const {
    isRecognizing,
    currentText,
    confidence,
    startRecognition,
    stopRecognition,
    websocketService,
  } = useSignLanguageRecognition()

  // 确保组件完全挂载后再显示动画
  useEffect(() => {
    const timer = setTimeout(() => setIsMounted(true), 100)
    return () => clearTimeout(timer)
  }, [])

  // WebSocket连接状态监听
  useEffect(() => {
    if (websocketService) {
      const handleConnect = () => {
        setIsConnected(true)
        setIsConnecting(false)
      }
      const handleDisconnect = () => {
        setIsConnected(false)
        setIsConnecting(false)
      }
      const handleError = (error: string) => {
        setError(error)
        setIsConnecting(false)
      }

      websocketService.on('connect', handleConnect)
      websocketService.on('disconnect', handleDisconnect)
      websocketService.on('error', handleError)

      return () => {
        websocketService.off('connect', handleConnect)
        websocketService.off('disconnect', handleDisconnect)
        websocketService.off('error', handleError)
      }
    }
  }, [websocketService])

  const handleConnect = async () => {
    try {
      setIsConnecting(true)
      setError(null)
      await websocketService.connect()
    } catch (err) {
      setError('连接服务器失败，请检查网络连接或服务器状态')
      setIsConnecting(false)
    }
  }

  const handleStartStop = async () => {
    if (isRecognizing) {
      stopRecognition()
    } else {
      try {
        setError(null)
        await startRecognition()
      } catch (err) {
        setError('启动识别失败，请确保摄像头权限已授予')
      }
    }
  }

  const handleCloseError = () => {
    setError(null)
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题和状态 */}
      <SafeFade in={isMounted} timeout={600}>
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Avatar
            sx={{
              width: 80,
              height: 80,
              mx: 'auto',
              mb: 3,
              background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
              boxShadow: '0 12px 32px rgba(181, 234, 215, 0.4)',
            }}
          >
            <Visibility sx={{ fontSize: 40, color: 'white' }} />
          </Avatar>
          
          <Typography 
            variant="h2" 
            gutterBottom 
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              mb: 2,
            }}
          >
            实时手语识别
          </Typography>
          
          <Typography variant="h6" color="text.secondary" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
            使用深度学习技术实时识别手语动作并转换为文字，支持多种手语类型
          </Typography>
          
          <Stack 
            direction="row" 
            spacing={2} 
            justifyContent="center" 
            flexWrap="wrap"
            sx={{ gap: 2 }}
          >
            <Chip
              icon={isConnected ? <CheckCircle /> : <Warning />}
              label={isConnected ? '服务器已连接' : '服务器未连接'}
              color={isConnected ? 'success' : 'warning'}
              sx={{ 
                px: 2,
                py: 1,
                height: 'auto',
                '& .MuiChip-label': { fontSize: '0.95rem', py: 0.5 }
              }}
            />
            {isRecognizing && (
              <Chip
                icon={<VideoCall />}
                label="识别进行中"
                color="info"
                sx={{ 
                  px: 2,
                  py: 1,
                  height: 'auto',
                  animation: 'pulse 2s infinite',
                  '& .MuiChip-label': { fontSize: '0.95rem', py: 0.5 }
                }}
              />
            )}
            {confidence !== null && (
              <Chip
                label={`置信度: ${(confidence * 100).toFixed(1)}%`}
                color={confidence > 0.8 ? "success" : confidence > 0.6 ? "warning" : "error"}
                sx={{ 
                  px: 2,
                  py: 1,
                  height: 'auto',
                  fontWeight: 600,
                  '& .MuiChip-label': { fontSize: '0.95rem', py: 0.5 }
                }}
              />
            )}
          </Stack>
        </Box>
      </SafeFade>

      {/* 优化的响应式布局 */}
      <Box sx={{ mb: 4 }}>
        <Grid container spacing={3}>
          {/* 顶部控制面板 - 水平排列减少拥挤感 */}
          <Grid item xs={12}>
            <Grid container spacing={3}>
              {/* 连接状态 */}
              <Grid item xs={12} md={4}>
                <SafeFade in={isMounted} timeout={600} key="connection-status">
                  <Card
                    elevation={0}
                    sx={{
                      background: 'linear-gradient(135deg, #FFB3BA15 0%, #FFD6CC08 100%)',
                      border: '1px solid #FFB3BA20',
                      borderRadius: 3,
                      height: '100%',
                    }}
                  >
                    <CardContent sx={{ p: 3, textAlign: 'center' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
                        <Avatar
                          sx={{
                            width: 40,
                            height: 40,
                            mr: 2,
                            backgroundColor: isConnected ? '#B5EAD7' : '#FFB3BA',
                            boxShadow: isConnected 
                              ? '0 4px 12px rgba(181, 234, 215, 0.3)'
                              : '0 4px 12px rgba(255, 179, 186, 0.3)',
                          }}
                        >
                          {isConnected ? <CheckCircle sx={{ fontSize: 20 }} /> : <Settings sx={{ fontSize: 20 }} />}
                        </Avatar>
                        <Box sx={{ textAlign: 'left' }}>
                          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
                            服务器连接
                          </Typography>
                          <Chip
                            label={isConnected ? '已连接' : '未连接'}
                            color={isConnected ? 'success' : 'warning'}
                            size="small"
                            sx={{ fontSize: '0.75rem' }}
                          />
                        </Box>
                      </Box>
                      
                      {!isConnected && (
                        <Button
                          variant="contained"
                          onClick={handleConnect}
                          disabled={isConnecting}
                          size="small"
                          startIcon={isConnecting ? <CircularProgress size={16} /> : null}
                          sx={{
                            borderRadius: 2,
                            background: 'linear-gradient(135deg, #FFB3BA 0%, #FFD6CC 100%)',
                            fontSize: '0.8rem',
                          }}
                        >
                          {isConnecting ? '连接中...' : '连接'}
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                </SafeFade>
              </Grid>

              {/* 识别控制 */}
              <Grid item xs={12} md={4}>
                <SafeFade in={isMounted} timeout={800} key="recognition-control">
                  <Card
                    elevation={0}
                    sx={{
                      background: 'linear-gradient(135deg, #B5EAD715 0%, #C7F0DB08 100%)',
                      border: '1px solid #B5EAD720',
                      borderRadius: 3,
                      height: '100%',
                    }}
                  >
                    <CardContent sx={{ p: 3, textAlign: 'center' }}>
                      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                        识别控制
                      </Typography>
                      
                      <Button
                        variant="contained"
                        color={isRecognizing ? "error" : "primary"}
                        startIcon={isRecognizing ? <Stop /> : <PlayArrow />}
                        onClick={handleStartStop}
                        disabled={!isConnected}
                        size="large"
                        sx={{ 
                          borderRadius: 3,
                          fontWeight: 600,
                          px: 4,
                          background: isRecognizing 
                            ? 'linear-gradient(135deg, #FFB3BA 0%, #FF9AA2 100%)'
                            : 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                        }}
                      >
                        {isRecognizing ? '停止' : '开始'}
                      </Button>

                      {confidence !== null && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="caption" color="text.secondary">
                            置信度: {(confidence * 100).toFixed(1)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={confidence * 100}
                            sx={{
                              mt: 1,
                              height: 4,
                              borderRadius: 2,
                              backgroundColor: 'rgba(181, 234, 215, 0.2)',
                              '& .MuiLinearProgress-bar': {
                                borderRadius: 2,
                                background: confidence > 0.8 
                                  ? 'linear-gradient(90deg, #B5EAD7 0%, #9BC1BC 100%)'
                                  : confidence > 0.6
                                  ? 'linear-gradient(90deg, #FFDAB9 0%, #FFCC99 100%)'
                                  : 'linear-gradient(90deg, #FFB3BA 0%, #FF9AA2 100%)',
                              },
                            }}
                          />
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </SafeFade>
              </Grid>

              {/* 识别结果预览 */}
              <Grid item xs={12} md={4}>
                <SafeFade in={isMounted} timeout={1000} key="recognition-result">
                  <Card
                    elevation={0}
                    sx={{
                      background: 'linear-gradient(135deg, #C7CEDB15 0%, #D6DCE508 100%)',
                      border: '1px solid #C7CEDB20',
                      borderRadius: 3,
                      height: '100%',
                    }}
                  >
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, textAlign: 'center', mb: 2 }}>
                        识别结果
                      </Typography>
                      <Box sx={{ 
                        minHeight: 60, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        background: 'rgba(255, 255, 255, 0.5)',
                        borderRadius: 2,
                        p: 2,
                      }}>
                        {currentText ? (
                          <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main', textAlign: 'center' }}>
                            {currentText}
                          </Typography>
                        ) : (
                          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', fontStyle: 'italic' }}>
                            {isRecognizing ? '正在识别...' : '等待手语输入'}
                          </Typography>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </SafeFade>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Box>

      {/* 主要内容区域 */}
      <Grid container spacing={4}>
        {/* 左侧：摄像头预览 */}
        <Grid item xs={12} lg={4}>
          <SafeFade in={isMounted} timeout={1200} key="video-capture">
            <Card
              elevation={0}
              sx={{
                background: 'linear-gradient(135deg, #C7CEDB20 0%, #D6DCE510 100%)',
                border: '2px solid #C7CEDB30',
                borderRadius: 4,
                height: { xs: 'auto', lg: '600px' },
              }}
            >
              <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, textAlign: 'center', mb: 3 }}>
                  摄像头预览
                </Typography>
                <Box sx={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                  <ErrorBoundary>
                    <Box sx={{ 
                      width: '100%',
                      borderRadius: 3, 
                      overflow: 'hidden',
                      background: 'linear-gradient(135deg, #F0F8FF 0%, #E6F7FF 100%)',
                    }}>
                      <VideoCapture isActive={isRecognizing} />
                    </Box>
                  </ErrorBoundary>
                </Box>
              </CardContent>
            </Card>
          </SafeFade>
        </Grid>

        {/* 中间：3D Avatar - 更大更突出 */}
        <Grid item xs={12} lg={8}>
          <SafeFade in={isMounted} timeout={600} key="avatar-viewer">
            <Paper 
              elevation={0}
              sx={{ 
                p: 4, 
                height: { xs: '500px', lg: '600px' },
                display: 'flex', 
                flexDirection: 'column',
                background: 'linear-gradient(135deg, #FFDAB920 0%, #FFE7CC10 100%)',
                border: '2px solid #FFDAB930',
                borderRadius: 6,
                position: 'relative',
                overflow: 'hidden',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: -20,
                  right: -20,
                  width: 60,
                  height: 60,
                  background: 'radial-gradient(circle, #FFDAB940 0%, transparent 70%)',
                  borderRadius: '50%',
                },
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Avatar
                    sx={{
                      width: 48,
                      height: 48,
                      mr: 2,
                      background: 'linear-gradient(135deg, #FFDAB9 0%, #FFE7CC 100%)',
                    }}
                  >
                    <Visibility />
                  </Avatar>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary' }}>
                    3D手语Avatar
                  </Typography>
                </Box>
                
                <Stack direction="row" spacing={1}>
                  {currentText && (
                    <Chip 
                      label="实时演示"
                      color="success"
                      size="small"
                      sx={{ fontWeight: 600 }}
                    />
                  )}
                  {isRecognizing && (
                    <Chip 
                      label="识别中"
                      color="info"
                      size="small"
                      sx={{ fontWeight: 600, animation: 'pulse 2s infinite' }}
                    />
                  )}
                </Stack>
              </Box>
              
              <Box 
                sx={{ 
                  flex: 1, 
                  minHeight: 0,
                  borderRadius: 4,
                  overflow: 'hidden',
                  background: 'linear-gradient(135deg, #F8FDFF 0%, #E8F5FF 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid rgba(255, 218, 185, 0.3)',
                  position: 'relative',
                }}
              >
                <ThreeDErrorBoundary>
                  <ErrorBoundary>
                    {isConnected ? (
                      <AvatarViewer
                        text={currentText}
                        isActive={isRecognizing}
                      />
                    ) : (
                      <Box sx={{ textAlign: 'center', color: 'text.secondary', p: 4 }}>
                        <Avatar
                          sx={{
                            width: 80,
                            height: 80,
                            mx: 'auto',
                            mb: 3,
                            backgroundColor: 'rgba(255, 218, 185, 0.3)',
                            color: 'text.secondary',
                        }}
                      >
                        <Settings sx={{ fontSize: 40 }} />
                      </Avatar>
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        等待连接服务器
                      </Typography>
                      <Typography variant="body2">
                        连接后即可查看3D Avatar实时演示手语动作
                      </Typography>
                    </Box>
                  )}
                  </ErrorBoundary>
                </ThreeDErrorBoundary>
              </Box>
            </Paper>
          </SafeFade>
        </Grid>
      </Grid>

      {/* 底部辅助信息 */}
      <Box sx={{ mt: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <SafeFade in={isMounted} timeout={1600} key="usage-tips">
              <Paper 
                elevation={0}
                sx={{ 
                  p: 3, 
                  background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                  color: 'white',
                  borderRadius: 4,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TipsAndUpdates sx={{ mr: 1, fontSize: 20 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    使用提示
                  </Typography>
                </Box>
                <Typography variant="body2" sx={{ fontSize: '0.9rem', lineHeight: 1.7, opacity: 0.95 }}>
                  • 确保手部完全在摄像头范围内<br/>
                  • 保持充足稳定的光线条件<br/>
                  • 手语动作要清晰标准规范<br/>
                  • 适当调整与摄像头的距离
                </Typography>
              </Paper>
            </SafeFade>
          </Grid>

          <Grid item xs={12} md={6}>
            <SafeFade in={isMounted} timeout={1800} key="privacy-info">
              <Paper 
                elevation={0}
                sx={{ 
                  p: 3, 
                  background: 'linear-gradient(135deg, #FFB3BA20 0%, #FFD6CC10 100%)',
                  border: '2px solid #FFB3BA30',
                  borderRadius: 4,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Security sx={{ mr: 1, fontSize: 20, color: 'success.main' }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'success.main' }}>
                    隐私保护
                  </Typography>
                </Box>
                <Typography variant="caption" sx={{ fontSize: '0.85rem', lineHeight: 1.6, color: 'text.secondary' }}>
                  我们仅上传手部关键点数据用于识别，原始视频完全在本地处理，
                  确保您的隐私和数据安全。所有传输均采用加密协议保护。
                </Typography>
              </Paper>
            </SafeFade>
          </Grid>

          <Grid item xs={12} md={12}>
            <SafeFade in={isMounted} timeout={2200} key="enhanced-video-recognition">
              <EnhancedVideoRecognition onResult={(r)=>console.log('enhanced cecsl result', r)} />
            </SafeFade>
          </Grid>
        </Grid>
      </Box>

      {/* 错误提示 */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseError} 
          severity="error" 
          sx={{ 
            width: '100%',
            borderRadius: 3,
            '& .MuiAlert-icon': {
              fontSize: 24,
            }
          }}
        >
          {error}
        </Alert>
      </Snackbar>

      {/* CSS动画 */}
      <style>
        {`
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
          }
        `}
      </style>
    </Container>
  )
}

export default RecognitionPage