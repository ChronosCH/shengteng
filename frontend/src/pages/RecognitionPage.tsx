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
  Fade,
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
import VideoCapture from '../components/VideoCapture'
import SubtitleDisplay from '../components/SubtitleDisplay'
import AvatarViewer from '../components/AvatarViewer'
import { useSignLanguageRecognition } from '../hooks/useSignLanguageRecognition'

function RecognitionPage() {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)

  const {
    isRecognizing,
    currentText,
    confidence,
    startRecognition,
    stopRecognition,
    websocketService,
  } = useSignLanguageRecognition()

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
      <Fade in timeout={600}>
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
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧控制面板 */}
        <Grid item xs={12} lg={3}>
          <Stack spacing={3}>
            {/* 连接控制 */}
            <Fade in timeout={800}>
              <Card
                elevation={0}
                sx={{
                  background: 'linear-gradient(135deg, #FFB3BA20 0%, #FFD6CC10 100%)',
                  border: '2px solid #FFB3BA30',
                  borderRadius: 4,
                }}
              >
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Avatar
                    sx={{
                      width: 56,
                      height: 56,
                      mx: 'auto',
                      mb: 3,
                      backgroundColor: isConnected ? '#B5EAD7' : '#FFB3BA',
                      boxShadow: isConnected 
                        ? '0 8px 20px rgba(181, 234, 215, 0.3)'
                        : '0 8px 20px rgba(255, 179, 186, 0.3)',
                    }}
                  >
                    {isConnected ? <CheckCircle /> : <Settings />}
                  </Avatar>
                  
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                    服务器连接
                  </Typography>
                  
                  {!isConnected ? (
                    <Box>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.6 }}>
                        需要先连接到手语识别服务器才能开始识别
                      </Typography>
                      <Button
                        variant="contained"
                        onClick={handleConnect}
                        disabled={isConnecting}
                        fullWidth
                        size="large"
                        startIcon={isConnecting ? <CircularProgress size={20} /> : null}
                        sx={{
                          borderRadius: 3,
                          py: 1.5,
                          background: 'linear-gradient(135deg, #FFB3BA 0%, #FFD6CC 100%)',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: '0 8px 25px rgba(255, 179, 186, 0.4)',
                          },
                          transition: 'all 0.3s ease',
                        }}
                      >
                        {isConnecting ? '连接中...' : '连接服务器'}
                      </Button>
                    </Box>
                  ) : (
                    <Box>
                      <Chip 
                        label="连接正常" 
                        color="success" 
                        icon={<Favorite />}
                        sx={{ mb: 2, fontWeight: 600 }}
                      />
                      <Typography variant="body2" color="success.main" sx={{ fontWeight: 500 }}>
                        服务器连接稳定，可以开始识别
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Fade>

            {/* 识别控制 */}
            <Fade in timeout={1000}>
              <Card
                elevation={0}
                sx={{
                  background: 'linear-gradient(135deg, #B5EAD720 0%, #C7F0DB10 100%)',
                  border: '2px solid #B5EAD730',
                  borderRadius: 4,
                }}
              >
                <CardContent sx={{ py: 4 }}>
                  <Typography variant="h6" gutterBottom sx={{ textAlign: 'center', fontWeight: 600, mb: 3 }}>
                    识别控制
                  </Typography>
                  
                  <Button
                    variant="contained"
                    color={isRecognizing ? "error" : "primary"}
                    startIcon={isRecognizing ? <Stop /> : <PlayArrow />}
                    onClick={handleStartStop}
                    disabled={!isConnected}
                    fullWidth
                    size="large"
                    sx={{ 
                      py: 2,
                      fontSize: '1.1rem',
                      borderRadius: 3,
                      fontWeight: 600,
                      background: isRecognizing 
                        ? 'linear-gradient(135deg, #FFB3BA 0%, #FF9AA2 100%)'
                        : 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: isRecognizing
                          ? '0 8px 25px rgba(255, 179, 186, 0.4)'
                          : '0 8px 25px rgba(181, 234, 215, 0.4)',
                      },
                      transition: 'all 0.3s ease',
                    }}
                  >
                    {isRecognizing ? '停止识别' : '开始识别'}
                  </Button>

                  {!isConnected && (
                    <Typography 
                      variant="caption" 
                      display="block" 
                      sx={{ mt: 2, textAlign: 'center', color: 'text.secondary', fontStyle: 'italic' }}
                    >
                      请先连接服务器
                    </Typography>
                  )}

                  {isRecognizing && (
                    <Box sx={{ mt: 3, textAlign: 'center' }}>
                      <Typography variant="body2" color="info.main" gutterBottom sx={{ fontWeight: 500 }}>
                        正在分析手语动作...
                      </Typography>
                      <LinearProgress 
                        sx={{ 
                          borderRadius: 2,
                          height: 6,
                          background: 'rgba(181, 234, 215, 0.2)',
                          '& .MuiLinearProgress-bar': {
                            background: 'linear-gradient(90deg, #B5EAD7 0%, #9BC1BC 100%)',
                            borderRadius: 2,
                          }
                        }} 
                      />
                    </Box>
                  )}

                  {confidence !== null && (
                    <Box sx={{ mt: 3 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          识别置信度
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {(confidence * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={confidence * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(181, 234, 215, 0.2)',
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 4,
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
            </Fade>

            {/* 摄像头预览 */}
            <Fade in timeout={1200}>
              <Card
                elevation={0}
                sx={{
                  background: 'linear-gradient(135deg, #C7CEDB20 0%, #D6DCE510 100%)',
                  border: '2px solid #C7CEDB30',
                  borderRadius: 4,
                }}
              >
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, textAlign: 'center', mb: 3 }}>
                    摄像头预览
                  </Typography>
                  <ErrorBoundary>
                    <Box sx={{ 
                      borderRadius: 3, 
                      overflow: 'hidden',
                      background: 'linear-gradient(135deg, #F0F8FF 0%, #E6F7FF 100%)',
                    }}>
                      <VideoCapture isActive={isRecognizing} />
                    </Box>
                  </ErrorBoundary>
                </CardContent>
              </Card>
            </Fade>
          </Stack>
        </Grid>

        {/* 中间3D Avatar区域 */}
        <Grid item xs={12} lg={6}>
          <Fade in timeout={600}>
            <Paper 
              elevation={0}
              sx={{ 
                p: 4, 
                height: '750px', 
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
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 4 }}>
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
                      label="有识别内容"
                      color="success"
                      size="small"
                      sx={{ fontWeight: 600 }}
                    />
                  )}
                  {isRecognizing && (
                    <Chip 
                      label="实时演示"
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
              </Box>
            </Paper>
          </Fade>
        </Grid>

        {/* 右侧结果显示 */}
        <Grid item xs={12} lg={3}>
          <Stack spacing={3}>
            <Fade in timeout={1400}>
              <Card
                elevation={0}
                sx={{
                  background: 'linear-gradient(135deg, #C7CEDB20 0%, #D6DCE510 100%)',
                  border: '2px solid #C7CEDB30',
                  borderRadius: 4,
                }}
              >
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, textAlign: 'center', mb: 3 }}>
                    识别结果
                  </Typography>
                  <ErrorBoundary>
                    <SubtitleDisplay
                      text={currentText}
                      confidence={confidence}
                      isRecognizing={isRecognizing}
                    />
                  </ErrorBoundary>
                </CardContent>
              </Card>
            </Fade>

            {/* 使用提示 */}
            <Fade in timeout={1600}>
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
                  • 适当调整与摄像头的距离<br/>
                  • 避免背景干扰和遮挡
                </Typography>
              </Paper>
            </Fade>

            {/* 隐私说明 */}
            <Fade in timeout={1800}>
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
            </Fade>
          </Stack>
        </Grid>
      </Grid>

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