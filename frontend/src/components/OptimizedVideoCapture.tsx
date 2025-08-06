/**
 * 优化的视频捕获组件 - 支持多种分辨率、帧率优化、错误处理
 */

import React, { useRef, useEffect, useState, useCallback } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Slider,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
  Chip,
  Grid,
} from '@mui/material'
import {
  Videocam,
  VideocamOff,
  Settings,
  Fullscreen,
  FullscreenExit,
  Refresh,
  CameraAlt,
  HighQuality,
  Speed,
} from '@mui/icons-material'

interface VideoConstraints {
  width: number
  height: number
  frameRate: number
  facingMode: 'user' | 'environment'
}

interface VideoDevice {
  deviceId: string
  label: string
  kind: string
}

interface OptimizedVideoCaptureProps {
  onFrame?: (imageData: ImageData, canvas: HTMLCanvasElement) => void
  onError?: (error: string) => void
  onStreamStart?: () => void
  onStreamStop?: () => void
  enableLandmarkOverlay?: boolean
  className?: string
}

const OptimizedVideoCapture: React.FC<OptimizedVideoCaptureProps> = ({
  onFrame,
  onError,
  onStreamStart,
  onStreamStop,
  enableLandmarkOverlay = false,
  className
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  const [isStreaming, setIsStreaming] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [devices, setDevices] = useState<VideoDevice[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string>('')
  const [showSettings, setShowSettings] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [currentFPS, setCurrentFPS] = useState(0)

  // 视频配置
  const [videoConstraints, setVideoConstraints] = useState<VideoConstraints>({
    width: 640,
    height: 480,
    frameRate: 30,
    facingMode: 'user'
  })

  // 性能设置
  const [performanceSettings, setPerformanceSettings] = useState({
    enableOptimization: true,
    skipFrames: 1, // 跳帧数量，用于降低处理频率
    enableResize: true,
    targetFPS: 30,
    qualityLevel: 'medium' as 'low' | 'medium' | 'high'
  })

  // 预设分辨率
  const resolutionPresets = [
    { width: 320, height: 240, label: '320x240 (低质量)' },
    { width: 640, height: 480, label: '640x480 (标准)' },
    { width: 1280, height: 720, label: '1280x720 (高清)' },
    { width: 1920, height: 1080, label: '1920x1080 (全高清)' }
  ]

  // FPS计算
  const fpsCounterRef = useRef({
    frameCount: 0,
    lastTime: Date.now(),
    fps: 0
  })

  // 获取可用的视频设备
  const getVideoDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const videoDevices = devices
        .filter(device => device.kind === 'videoinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `摄像头 ${device.deviceId.slice(0, 8)}`,
          kind: device.kind
        }))
      
      setDevices(videoDevices)
      
      if (videoDevices.length > 0 && !selectedDevice) {
        setSelectedDevice(videoDevices[0].deviceId)
      }
    } catch (err) {
      console.error('获取视频设备失败:', err)
      setError('无法获取摄像头设备列表')
    }
  }, [selectedDevice])

  // 启动视频流
  const startStream = useCallback(async () => {
    setIsLoading(true)
    setError('')

    try {
      // 停止现有流
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }

      // 构建媒体约束
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          width: { ideal: videoConstraints.width },
          height: { ideal: videoConstraints.height },
          frameRate: { ideal: videoConstraints.frameRate },
          facingMode: videoConstraints.facingMode
        },
        audio: false
      }

      // 获取媒体流
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream

      // 设置视频元素
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          setIsStreaming(true)
          setIsLoading(false)
          onStreamStart?.()
          startFrameCapture()
        }
      }

    } catch (err: any) {
      setIsLoading(false)
      const errorMessage = err.name === 'NotAllowedError' 
        ? '摄像头权限被拒绝，请允许访问摄像头'
        : err.name === 'NotFoundError'
        ? '未找到摄像头设备'
        : `启动摄像头失败: ${err.message}`
      
      setError(errorMessage)
      onError?.(errorMessage)
    }
  }, [selectedDevice, videoConstraints, onStreamStart, onError])

  // 停止视频流
  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    setIsStreaming(false)
    onStreamStop?.()
  }, [onStreamStop])

  // 帧捕获和处理
  const startFrameCapture = useCallback(() => {
    const captureFrame = () => {
      if (!videoRef.current || !canvasRef.current || !isStreaming) {
        return
      }

      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      if (!ctx) return

      // 更新画布尺寸
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // 性能优化：跳帧处理
      const shouldProcess = fpsCounterRef.current.frameCount % (performanceSettings.skipFrames + 1) === 0

      if (shouldProcess) {
        // 绘制视频帧到画布
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        // 如果启用了性能优化，可能需要缩放
        let processCanvas = canvas
        if (performanceSettings.enableResize && performanceSettings.qualityLevel !== 'high') {
          const scale = performanceSettings.qualityLevel === 'low' ? 0.5 : 0.75
          processCanvas = createScaledCanvas(canvas, scale)
        }

        // 获取图像数据并传递给父组件
        try {
          const imageData = processCanvas.getContext('2d')!.getImageData(
            0, 0, processCanvas.width, processCanvas.height
          )
          onFrame?.(imageData, processCanvas)
        } catch (err) {
          console.error('帧处理失败:', err)
        }
      }

      // FPS计算
      updateFPSCounter()

      // 继续下一帧
      animationFrameRef.current = requestAnimationFrame(captureFrame)
    }

    animationFrameRef.current = requestAnimationFrame(captureFrame)
  }, [isStreaming, onFrame, performanceSettings])

  // 创建缩放画布（性能优化）
  const createScaledCanvas = (sourceCanvas: HTMLCanvasElement, scale: number): HTMLCanvasElement => {
    const scaledCanvas = document.createElement('canvas')
    const scaledCtx = scaledCanvas.getContext('2d')!
    
    scaledCanvas.width = sourceCanvas.width * scale
    scaledCanvas.height = sourceCanvas.height * scale
    
    scaledCtx.drawImage(sourceCanvas, 0, 0, scaledCanvas.width, scaledCanvas.height)
    
    return scaledCanvas
  }

  // 更新FPS计数器
  const updateFPSCounter = () => {
    const now = Date.now()
    fpsCounterRef.current.frameCount++

    if (now - fpsCounterRef.current.lastTime >= 1000) {
      fpsCounterRef.current.fps = fpsCounterRef.current.frameCount
      fpsCounterRef.current.frameCount = 0
      fpsCounterRef.current.lastTime = now
      setCurrentFPS(fpsCounterRef.current.fps)
    }
  }

  // 切换全屏
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      videoRef.current?.parentElement?.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }, [])

  // 监听全屏状态变化
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  // 组件挂载时获取设备列表
  useEffect(() => {
    getVideoDevices()
  }, [getVideoDevices])

  // 组件卸载时清理资源
  useEffect(() => {
    return () => {
      stopStream()
    }
  }, [stopStream])

  return (
    <Card className={className} elevation={2}>
      <CardContent>
        {/* 控制栏 */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CameraAlt color="primary" />
            视频捕获
          </Typography>

          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            {/* FPS显示 */}
            <Chip
              icon={<Speed />}
              label={`${currentFPS} FPS`}
              size="small"
              color={currentFPS >= 25 ? 'success' : currentFPS >= 15 ? 'warning' : 'error'}
            />

            {/* 分辨率显示 */}
            <Chip
              icon={<HighQuality />}
              label={`${videoConstraints.width}x${videoConstraints.height}`}
              size="small"
              variant="outlined"
            />

            {/* 控制按钮 */}
            <Tooltip title="设置">
              <IconButton
                size="small"
                onClick={() => setShowSettings(!showSettings)}
                color={showSettings ? 'primary' : 'default'}
              >
                <Settings />
              </IconButton>
            </Tooltip>

            <Tooltip title={isFullscreen ? '退出全屏' : '全屏'}>
              <IconButton size="small" onClick={toggleFullscreen}>
                {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
              </IconButton>
            </Tooltip>

            <Tooltip title="刷新设备">
              <IconButton size="small" onClick={getVideoDevices}>
                <Refresh />
              </IconButton>
            </Tooltip>

            {/* 主控制按钮 */}
            {!isStreaming ? (
              <Button
                variant="contained"
                startIcon={isLoading ? <CircularProgress size={20} /> : <Videocam />}
                onClick={startStream}
                disabled={isLoading}
              >
                {isLoading ? '启动中...' : '开启摄像头'}
              </Button>
            ) : (
              <Button
                variant="outlined"
                startIcon={<VideocamOff />}
                onClick={stopStream}
                color="error"
              >
                关闭摄像头
              </Button>
            )}
          </Box>
        </Box>

        {/* 设置面板 */}
        {showSettings && (
          <Box sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
            <Grid container spacing={2}>
              {/* 设备选择 */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>摄像头设备</InputLabel>
                  <Select
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    label="摄像头设备"
                  >
                    {devices.map((device) => (
                      <MenuItem key={device.deviceId} value={device.deviceId}>
                        {device.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              {/* 分辨率选择 */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>分辨率</InputLabel>
                  <Select
                    value={`${videoConstraints.width}x${videoConstraints.height}`}
                    onChange={(e) => {
                      const [width, height] = e.target.value.split('x').map(Number)
                      setVideoConstraints(prev => ({ ...prev, width, height }))
                    }}
                    label="分辨率"
                  >
                    {resolutionPresets.map((preset) => (
                      <MenuItem key={`${preset.width}x${preset.height}`} value={`${preset.width}x${preset.height}`}>
                        {preset.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              {/* 帧率设置 */}
              <Grid item xs={12} sm={4}>
                <Typography gutterBottom>目标帧率: {videoConstraints.frameRate} FPS</Typography>
                <Slider
                  value={videoConstraints.frameRate}
                  onChange={(_, value) => setVideoConstraints(prev => ({ ...prev, frameRate: value as number }))}
                  min={15}
                  max={60}
                  step={5}
                  marks
                  size="small"
                />
              </Grid>

              {/* 性能优化选项 */}
              <Grid item xs={12} sm={8}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={performanceSettings.enableOptimization}
                      onChange={(e) => setPerformanceSettings(prev => ({ 
                        ...prev, 
                        enableOptimization: e.target.checked 
                      }))}
                    />
                  }
                  label="启用性能优化"
                />

                <FormControl size="small" sx={{ ml: 2, minWidth: 120 }}>
                  <InputLabel>质量等级</InputLabel>
                  <Select
                    value={performanceSettings.qualityLevel}
                    onChange={(e) => setPerformanceSettings(prev => ({ 
                      ...prev, 
                      qualityLevel: e.target.value as any 
                    }))}
                    label="质量等级"
                  >
                    <MenuItem value="low">低质量</MenuItem>
                    <MenuItem value="medium">中等质量</MenuItem>
                    <MenuItem value="high">高质量</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* 错误提示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        {/* 视频显示区域 */}
        <Box
          sx={{
            position: 'relative',
            width: '100%',
            bgcolor: 'black',
            borderRadius: 2,
            overflow: 'hidden',
            aspectRatio: `${videoConstraints.width}/${videoConstraints.height}`,
          }}
        >
          {/* 视频元素 */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              display: isStreaming ? 'block' : 'none',
            }}
          />

          {/* 关键点叠加画布 */}
          {enableLandmarkOverlay && (
            <canvas
              ref={overlayCanvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
            />
          )}

          {/* 处理用的隐藏画布 */}
          <canvas
            ref={canvasRef}
            style={{ display: 'none' }}
          />

          {/* 加载状态 */}
          {isLoading && (
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                color: 'white',
              }}
            >
              <CircularProgress color="inherit" />
              <Typography variant="body2">正在启动摄像头...</Typography>
            </Box>
          )}

          {/* 未启动状态 */}
          {!isStreaming && !isLoading && (
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                color: 'white',
                textAlign: 'center',
              }}
            >
              <VideocamOff sx={{ fontSize: 64, opacity: 0.5 }} />
              <Typography variant="body1">摄像头未启动</Typography>
              <Typography variant="body2" sx={{ opacity: 0.7 }}>
                点击"开启摄像头"按钮开始
              </Typography>
            </Box>
          )}
        </Box>

        {/* 性能信息 */}
        {isStreaming && (
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              实际分辨率: {videoRef.current?.videoWidth || 0} x {videoRef.current?.videoHeight || 0}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              处理质量: {performanceSettings.qualityLevel} | 跳帧: {performanceSettings.skipFrames}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  )
}

export default OptimizedVideoCapture