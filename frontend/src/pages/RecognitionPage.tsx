import { useState, useEffect, useRef, useCallback } from 'react'
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
  Warning,
  CheckCircle,
  Speed,
  Visibility,
  TipsAndUpdates,
  Security,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import SafeFade from '../components/SafeFade'
import VideoCapture from '../components/VideoCapture'
import ContinuousVideoRecognition from '../components/ContinuousVideoRecognition'
import mindVacRealtimeService, { MindVacRealtimeResult } from '../services/mindVacRealtimeService'

const CAPTURE_DURATION_MS = 4000
const TARGET_FPS = 24
const MIN_FRAME_COUNT = 32
const BACKEND_MIN_FRAME_COUNT = 8
const CAPTURE_WIDTH = 256
const CAPTURE_HEIGHT = 256

function RecognitionPage() {
  const [isMounted, setIsMounted] = useState(false)
  const [engineAvailable, setEngineAvailable] = useState(true)
  const [isCollecting, setIsCollecting] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [collectionProgress, setCollectionProgress] = useState(0)
  const [fpsEstimate, setFpsEstimate] = useState(0)

  const [currentText, setCurrentText] = useState('')
  const [confidence, setConfidence] = useState<number | null>(null)
  const [glossSequence, setGlossSequence] = useState<string[]>([])
  const [statusMessage, setStatusMessage] = useState('等待识别')
  const [error, setError] = useState<string | null>(null)
  const [realtimeResult, setRealtimeResult] = useState<MindVacRealtimeResult | null>(null)

  const framesRef = useRef<string[]>([])
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const captureStartRef = useRef(0)
  const lastFrameTsRef = useRef(0)

  useEffect(() => {
    const timer = setTimeout(() => setIsMounted(true), 100)
    return () => clearTimeout(timer)
  }, [])

  const resetStatesForNewCapture = useCallback(() => {
    framesRef.current = []
    captureStartRef.current = performance.now()
    lastFrameTsRef.current = captureStartRef.current
  setCollectionProgress(0)
  setFpsEstimate(0)
  setStatusMessage('正在采集手势，请持续保持动作约 10 秒')
    setError(null)
    setRealtimeResult(null)
    setCurrentText('')
    setConfidence(null)
    setGlossSequence([])
  }, [])

  const finalizeCapture = useCallback(
    (endTime?: number) => {
      const stopTime = endTime ?? performance.now()
      setIsCollecting(false)

      let frames = [...framesRef.current]
      const elapsedMs = Math.max(1, stopTime - captureStartRef.current)
      const originalFrameCount = frames.length

      if (frames.length === 0) {
        framesRef.current = []
        setCollectionProgress(0)
        setStatusMessage('未采集到有效画面，请检查摄像头后重试')
        setError('未捕捉到任何帧，请保持手势在取景框内并稍后再试')
        return
      }

      let autoPadded = false

      if (frames.length < MIN_FRAME_COUNT) {
        // 若采集帧过少，先补齐到后端要求的最小帧数，再补齐到页面设定的窗口长度
        if (frames.length < BACKEND_MIN_FRAME_COUNT) {
          const lastFrame = frames[frames.length - 1]
          while (frames.length < BACKEND_MIN_FRAME_COUNT) {
            frames.push(lastFrame)
          }
        }

        const paddedFrames = [...frames]
        while (paddedFrames.length < MIN_FRAME_COUNT) {
          const remaining = MIN_FRAME_COUNT - paddedFrames.length
          const batch = frames.slice(0, remaining)
          paddedFrames.push(...batch)
        }

        frames = paddedFrames
        autoPadded = true
      }

      if (autoPadded) {
        setStatusMessage('采集帧稍少，系统已自动补足，正在调用 Mind-VAC 模型...')
      } else {
        setStatusMessage('正在调用 Mind-VAC 模型进行识别...')
      }

      setIsProcessing(true)

      const fps = Math.max(1, originalFrameCount / (elapsedMs / 1000))
      const normalizedFps = Math.max(12, Math.min(30, Math.round(fps)))

      mindVacRealtimeService
        .recognizeFrames(frames, normalizedFps, true)
        .then((result) => {
          setRealtimeResult(result)
          setCurrentText(result.text || '')
          setConfidence(typeof result.confidence === 'number' ? result.confidence : null)
          setGlossSequence(result.gloss_sequence || [])
          setStatusMessage('识别完成')
          setEngineAvailable(true)
        })
        .catch((err) => {
          const message = err instanceof Error ? err.message : String(err)
          setError(message)
          setStatusMessage('识别失败')
          if (message.toLowerCase().includes('mind-vac')) {
            setEngineAvailable(false)
          }
        })
        .finally(() => {
          framesRef.current = []
          setCollectionProgress(0)
          setIsProcessing(false)
        })
    },
    [],
  )

  const handleStartCapture = useCallback(() => {
    if (isCollecting || isProcessing) {
      return
    }
    resetStatesForNewCapture()
    setIsCollecting(true)
  }, [isCollecting, isProcessing, resetStatesForNewCapture])

  const handleStopCapture = useCallback(() => {
    if (!isCollecting) {
      return
    }
    finalizeCapture(performance.now())
  }, [finalizeCapture, isCollecting])

  const handleVideoFrame = useCallback(
    (video: HTMLVideoElement) => {
      if (!isCollecting) {
        return
      }

      const now = performance.now()
      const minInterval = 1000 / TARGET_FPS
      if (now - lastFrameTsRef.current < minInterval) {
        return
      }

      let canvas = captureCanvasRef.current
      if (!canvas) {
        canvas = document.createElement('canvas')
        canvas.width = CAPTURE_WIDTH
        canvas.height = CAPTURE_HEIGHT
        captureCanvasRef.current = canvas
      }

      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      if (!ctx) {
        return
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85)
      framesRef.current.push(dataUrl)
      lastFrameTsRef.current = now

      const elapsedMs = now - captureStartRef.current
      setCollectionProgress(Math.min(1, elapsedMs / CAPTURE_DURATION_MS))
      const fps = framesRef.current.length / Math.max(0.001, elapsedMs / 1000)
      setFpsEstimate(fps)

      const maxFrameCount = TARGET_FPS * Math.ceil(CAPTURE_DURATION_MS / 1000)
      if (elapsedMs >= CAPTURE_DURATION_MS || framesRef.current.length >= maxFrameCount) {
        finalizeCapture(now)
      }
    },
    [finalizeCapture, isCollecting],
  )

  const handleCloseError = () => setError(null)

  const renderGlossSequence = () => {
    if (!glossSequence.length) {
      return null
    }
    return (
      <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1, mt: 2 }}>
        {glossSequence.map((gloss, index) => (
          <Chip key={`${gloss}-${index}`} label={`${index + 1}. ${gloss}`} size="small" color="secondary" variant="outlined" />
        ))}
      </Stack>
    )
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
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
            Mind-VAC 实时手语识别
          </Typography>

          <Typography variant="h6" color="text.secondary" sx={{ mb: 4, maxWidth: 620, mx: 'auto' }}>
            直接调用本地 Mind-VAC 模型，采集摄像头画面后在后端完成推理，并返回连续手语识别结果
          </Typography>

          <Stack direction="row" spacing={2} justifyContent="center" flexWrap="wrap" sx={{ gap: 2 }}>
            <Chip
              icon={engineAvailable ? <CheckCircle /> : <Warning />}
              label={engineAvailable ? 'Mind-VAC 引擎可用' : 'Mind-VAC 引擎不可用'}
              color={engineAvailable ? 'success' : 'warning'}
              sx={{ px: 2, py: 1, height: 'auto', '& .MuiChip-label': { fontSize: '0.95rem', py: 0.5 } }}
            />
            {isCollecting && (
              <Chip
                icon={<Speed />}
                label={`采集中 · ${(collectionProgress * 100).toFixed(0)}% · ${fpsEstimate.toFixed(1)} FPS`}
                color="info"
                sx={{ px: 2, py: 1, height: 'auto', animation: 'pulse 2s infinite', '& .MuiChip-label': { fontSize: '0.95rem', py: 0.5 } }}
              />
            )}
            {isProcessing && (
              <Chip
                icon={<CircularProgress size={16} color="inherit" />}
                label="Mind-VAC 推理中"
                color="primary"
                sx={{ px: 2, py: 1, height: 'auto', '& .MuiChip-label': { fontSize: '0.95rem', py: 0.5 } }}
              />
            )}
            {confidence !== null && !isProcessing && (
              <Chip
                label={`置信度 ${(confidence * 100).toFixed(1)}%`}
                color={confidence > 0.8 ? 'success' : confidence > 0.6 ? 'warning' : 'error'}
                sx={{ px: 2, py: 1, height: 'auto', fontWeight: 600, '& .MuiChip-label': { fontSize: '0.95rem', py: 0.5 } }}
              />
            )}
          </Stack>
        </Box>
      </SafeFade>

      <Box sx={{ mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <SafeFade in={isMounted} timeout={600} key="engine-status">
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
                        backgroundColor: engineAvailable ? '#B5EAD7' : '#FFB3BA',
                        boxShadow: engineAvailable
                          ? '0 4px 12px rgba(181, 234, 215, 0.3)'
                          : '0 4px 12px rgba(255, 179, 186, 0.3)',
                      }}
                    >
                      {engineAvailable ? <CheckCircle sx={{ fontSize: 20 }} /> : <Warning sx={{ fontSize: 20 }} />}
                    </Avatar>
                    <Box sx={{ textAlign: 'left' }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
                        Mind-VAC 状态
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {engineAvailable ? '模型已准备就绪，可直接识别。' : '模型不可用，请检查后端服务或权重文件。'}
                      </Typography>
                    </Box>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    点击“开始识别”后，将采集约 3 秒的视频帧并在服务端运行 Mind-VAC 模型。
                  </Typography>
                </CardContent>
              </Card>
            </SafeFade>
          </Grid>

          <Grid item xs={12} md={4}>
            <SafeFade in={isMounted} timeout={800} key="control-panel">
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
                    color={isCollecting ? 'error' : 'primary'}
                    startIcon={isCollecting ? <Stop /> : <PlayArrow />}
                    onClick={isCollecting ? handleStopCapture : handleStartCapture}
                    size="large"
                    disabled={isProcessing}
                    sx={{
                      borderRadius: 3,
                      fontWeight: 600,
                      px: 4,
                      background: isCollecting
                        ? 'linear-gradient(135deg, #FFB3BA 0%, #FF9AA2 100%)'
                        : 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                    }}
                  >
                    {isCollecting ? '停止采集' : '开始识别'}
                  </Button>

                  {isCollecting && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="caption" color="text.secondary">
                        正在采集手势... 请保持稳定
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={collectionProgress * 100}
                        sx={{
                          mt: 1,
                          height: 4,
                          borderRadius: 2,
                          backgroundColor: 'rgba(181, 234, 215, 0.2)',
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 2,
                            background: 'linear-gradient(90deg, #B5EAD7 0%, #9BC1BC 100%)',
                          },
                        }}
                      />
                    </Box>
                  )}

                  {isProcessing && (
                    <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={32} color="primary" />
                      <Typography variant="caption" color="text.secondary">
                        Mind-VAC 推理中...
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </SafeFade>
          </Grid>

          <Grid item xs={12} md={4}>
            <SafeFade in={isMounted} timeout={1000} key="result-preview">
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
                  <Box
                    sx={{
                      minHeight: 70,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'rgba(255, 255, 255, 0.5)',
                      borderRadius: 2,
                      p: 2,
                    }}
                  >
                    {currentText ? (
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main', textAlign: 'center' }}>
                        {currentText}
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', fontStyle: 'italic' }}>
                        {isProcessing ? 'Mind-VAC 推理中...' : statusMessage}
                      </Typography>
                    )}
                  </Box>
                  {renderGlossSequence()}
                </CardContent>
              </Card>
            </SafeFade>
          </Grid>
        </Grid>
      </Box>

      <Grid container spacing={4}>
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
                    <Box
                      sx={{
                        width: '100%',
                        borderRadius: 3,
                        overflow: 'hidden',
                        background: 'linear-gradient(135deg, #F0F8FF 0%, #E6F7FF 100%)',
                      }}
                    >
                      <VideoCapture isActive={isCollecting} onFrame={handleVideoFrame} />
                    </Box>
                  </ErrorBoundary>
                </Box>
              </CardContent>
            </Card>
          </SafeFade>
        </Grid>

        <Grid item xs={12} lg={8}>
          <SafeFade in={isMounted} timeout={600} key="result-details">
            <Paper
              elevation={0}
              sx={{
                p: 4,
                height: { xs: 'auto', lg: '600px' },
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
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary', mb: 3 }}>
                Mind-VAC 推理详情
              </Typography>

              {realtimeResult ? (
                <Stack spacing={2} sx={{ color: 'text.primary' }}>
                  <Box>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                      模型输出
                    </Typography>
                    <Typography variant="body1" sx={{ lineHeight: 1.7 }}>
                      {realtimeResult.text || '—'}
                    </Typography>
                  </Box>

                  {realtimeResult.baseline_text && realtimeResult.baseline_text !== realtimeResult.text && (
                    <Box>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        基础翻译
                      </Typography>
                      <Typography variant="body2">{realtimeResult.baseline_text}</Typography>
                    </Box>
                  )}

                  {realtimeResult.raw_gloss_text && (
                    <Box>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Gloss 序列
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-word' }}>
                        {realtimeResult.raw_gloss_text}
                      </Typography>
                    </Box>
                  )}

                  <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
                    <Chip label={`帧数 ${realtimeResult.frame_count}`} size="small" />
                    <Chip label={`时长 ${(realtimeResult.duration || 0).toFixed(2)}s`} size="small" />
                    <Chip label={`置信度 ${(realtimeResult.confidence * 100).toFixed(1)}%`} size="small" />
                  </Stack>
                </Stack>
              ) : (
                <Box
                  sx={{
                    flex: 1,
                    borderRadius: 4,
                    border: '1px dashed rgba(255, 218, 185, 0.6)',
                    background: 'linear-gradient(135deg, rgba(248, 253, 255, 0.6) 0%, rgba(232, 245, 255, 0.8) 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    p: 4,
                    textAlign: 'center',
                    color: 'text.secondary',
                  }}
                >
                  <Typography variant="body2">等待采集并完成 Mind-VAC 推理后将在此显示详细结果</Typography>
                </Box>
              )}
            </Paper>
          </SafeFade>
        </Grid>
      </Grid>

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
                  • 确保手部完整进入画面，避免遮挡<br />• 保持稳定光线与背景对比<br />• 采集中保持动作连续，直到进度条完成<br />• 如果结果为空，尝试延长动作时间或增大动作幅度
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
                  视频帧仅用于即时推理，并不会在服务器上长期保存；如果 Mind-VAC LLM 增强不可用，系统会自动返回基础识别结果。
                </Typography>
              </Paper>
            </SafeFade>
          </Grid>

          <Grid item xs={12} md={12}>
            <SafeFade in={isMounted} timeout={2400} key="continuous-video-recognition">
              <ContinuousVideoRecognition onResult={(r) => console.log('continuous recognition result', r)} />
            </SafeFade>
          </Grid>
        </Grid>
      </Box>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseError}
          severity="error"
          sx={{ width: '100%', borderRadius: 3, '& .MuiAlert-icon': { fontSize: 24 } }}
        >
          {error}
        </Alert>
      </Snackbar>

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