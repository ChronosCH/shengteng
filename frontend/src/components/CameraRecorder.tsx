/**
 * 摄像头录制组件
 * 用于孤立手语识别的实时视频采集
 */

import React, { useRef, useState, useEffect } from 'react'
import {
  Box,
  Button,
  Stack,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  IconButton,
} from '@mui/material'
import {
  Videocam,
  VideocamOff,
  FiberManualRecord,
  Stop,
  Replay,
  CloudUpload,
} from '@mui/icons-material'

interface CameraRecorderProps {
  onRecordComplete: (videoBlob: Blob) => void
  maxDuration?: number // 最大录制时长(秒)
  countdown?: number // 开始录制前的倒计时(秒)
}

const CameraRecorder: React.FC<CameraRecorderProps> = ({
  onRecordComplete,
  maxDuration = 10,
  countdown = 3,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  const [cameraActive, setCameraActive] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [recordedUrl, setRecordedUrl] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [countdownValue, setCountdownValue] = useState<number>(0)
  const [recordingTime, setRecordingTime] = useState<number>(0)

  // 当摄像头状态变化时，将流绑定到 video 元素，避免组件初次渲染时出现黑屏
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      const playPromise = videoRef.current.play()
      if (playPromise && typeof playPromise.then === 'function') {
        playPromise.catch(() => {
          // 某些浏览器需要用户交互才能播放，忽略错误即可
        })
      }
    }
  }, [cameraActive, recordedUrl])

  // 启动摄像头
  const startCamera = async () => {
    try {
      setError('')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      })

      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setCameraActive(true)
    } catch (err) {
      console.error('摄像头启动失败:', err)
      setError('无法访问摄像头，请确保已授权摄像头权限')
    }
  }

  // 停止摄像头
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
  }

  // 开始录制(带倒计时)
  const startRecordingWithCountdown = () => {
    setCountdownValue(countdown)
    const timer = setInterval(() => {
      setCountdownValue(prev => {
        if (prev <= 1) {
          clearInterval(timer)
          startRecording()
          return 0
        }
        return prev - 1
      })
    }, 1000)
  }

  // 开始录制
  const startRecording = () => {
    if (!streamRef.current) return

    try {
      chunksRef.current = []
      const options = { mimeType: 'video/webm;codecs=vp8,opus' }
      
      // 尝试不同的编码格式
      let mediaRecorder: MediaRecorder
      if (MediaRecorder.isTypeSupported(options.mimeType)) {
        mediaRecorder = new MediaRecorder(streamRef.current, options)
      } else {
        mediaRecorder = new MediaRecorder(streamRef.current)
      }

      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' })
        setRecordedBlob(blob)
        const url = URL.createObjectURL(blob)
        setRecordedUrl(url)
        setIsRecording(false)
        setRecordingTime(0)
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      // 录制计时器
      const startTime = Date.now()
      const timer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000)
        setRecordingTime(elapsed)
        
        if (elapsed >= maxDuration) {
          stopRecording()
          clearInterval(timer)
        }
      }, 100)

    } catch (err) {
      console.error('录制启动失败:', err)
      setError('录制启动失败，请重试')
    }
  }

  // 停止录制
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
    }
  }

  // 重新录制
  const resetRecording = () => {
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl)
    }
    setRecordedBlob(null)
    setRecordedUrl('')
    setRecordingTime(0)
    chunksRef.current = []
  }

  // 上传录制的视频
  const handleUpload = () => {
    if (recordedBlob) {
      onRecordComplete(recordedBlob)
      resetRecording()
      stopCamera()
    }
  }

  // 清理资源
  useEffect(() => {
    return () => {
      stopCamera()
      if (recordedUrl) {
        URL.revokeObjectURL(recordedUrl)
      }
    }
  }, [recordedUrl])

  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        borderRadius: 3,
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
      }}
    >
      <Stack spacing={3}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          📹 摄像头录制
        </Typography>

        {error && (
          <Alert severity="error" sx={{ borderRadius: 2 }}>
            {error}
          </Alert>
        )}

        {/* 视频预览区域 */}
        <Box
          sx={{
            position: 'relative',
            width: '100%',
            aspectRatio: '16/9',
            backgroundColor: '#000',
            borderRadius: 2,
            overflow: 'hidden',
          }}
        >
          {/* 实时视频流 - 在摄像头激活且没有录制完成的视频时显示 */}
          {cameraActive && !recordedUrl && (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                position: 'absolute',
                top: 0,
                left: 0,
              }}
            />
          )}

          {/* 录制的视频回放 */}
          {recordedUrl && (
            <video
              src={recordedUrl}
              controls
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                position: 'absolute',
                top: 0,
                left: 0,
              }}
            />
          )}

          {/* 占位符 */}
          {!cameraActive && !recordedUrl && (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: '#666',
              }}
            >
              <VideocamOff sx={{ fontSize: 64 }} />
            </Box>
          )}

          {/* 倒计时显示 */}
          {countdownValue > 0 && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'rgba(0,0,0,0.7)',
                zIndex: 10,
              }}
            >
              <Typography
                variant="h1"
                sx={{
                  fontWeight: 700,
                  fontSize: '120px',
                  color: '#fff',
                  animation: 'pulse 1s ease-in-out',
                  '@keyframes pulse': {
                    '0%, 100%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.2)' },
                  },
                }}
              >
                {countdownValue}
              </Typography>
            </Box>
          )}

          {/* 录制中的指示器 */}
          {isRecording && (
            <Box
              sx={{
                position: 'absolute',
                top: 16,
                left: 16,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                backgroundColor: 'rgba(244, 67, 54, 0.9)',
                px: 2,
                py: 1,
                borderRadius: 2,
              }}
            >
              <FiberManualRecord
                sx={{
                  fontSize: 16,
                  animation: 'blink 1s infinite',
                  '@keyframes blink': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.3 },
                  },
                }}
              />
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                录制中 {recordingTime}s / {maxDuration}s
              </Typography>
            </Box>
          )}
        </Box>

        {/* 控制按钮 */}
        <Stack direction="row" spacing={2} justifyContent="center">
          {!cameraActive && !recordedBlob && (
            <Button
              variant="contained"
              size="large"
              startIcon={<Videocam />}
              onClick={startCamera}
              sx={{
                backgroundColor: '#fff',
                color: '#667eea',
                '&:hover': { backgroundColor: '#f5f5f5' },
                borderRadius: 2,
                px: 4,
              }}
            >
              启动摄像头
            </Button>
          )}

          {cameraActive && !isRecording && !recordedBlob && (
            <>
              <Button
                variant="contained"
                size="large"
                startIcon={<FiberManualRecord />}
                onClick={startRecordingWithCountdown}
                disabled={countdownValue > 0}
                sx={{
                  backgroundColor: '#f44336',
                  color: '#fff',
                  '&:hover': { backgroundColor: '#d32f2f' },
                  borderRadius: 2,
                  px: 4,
                }}
              >
                {countdownValue > 0 ? `${countdownValue}秒后开始` : '开始录制'}
              </Button>
              <Button
                variant="outlined"
                size="large"
                startIcon={<VideocamOff />}
                onClick={stopCamera}
                sx={{
                  borderColor: '#fff',
                  color: '#fff',
                  '&:hover': { borderColor: '#fff', backgroundColor: 'rgba(255,255,255,0.1)' },
                  borderRadius: 2,
                }}
              >
                关闭摄像头
              </Button>
            </>
          )}

          {isRecording && (
            <Button
              variant="contained"
              size="large"
              startIcon={<Stop />}
              onClick={stopRecording}
              sx={{
                backgroundColor: '#ff9800',
                color: '#fff',
                '&:hover': { backgroundColor: '#f57c00' },
                borderRadius: 2,
                px: 4,
              }}
            >
              停止录制
            </Button>
          )}

          {recordedBlob && (
            <>
              <Button
                variant="contained"
                size="large"
                startIcon={<CloudUpload />}
                onClick={handleUpload}
                sx={{
                  backgroundColor: '#4caf50',
                  color: '#fff',
                  '&:hover': { backgroundColor: '#388e3c' },
                  borderRadius: 2,
                  px: 4,
                }}
              >
                识别手语
              </Button>
              <Button
                variant="outlined"
                size="large"
                startIcon={<Replay />}
                onClick={resetRecording}
                sx={{
                  borderColor: '#fff',
                  color: '#fff',
                  '&:hover': { borderColor: '#fff', backgroundColor: 'rgba(255,255,255,0.1)' },
                  borderRadius: 2,
                }}
              >
                重新录制
              </Button>
            </>
          )}
        </Stack>

        <Typography variant="caption" sx={{ textAlign: 'center', opacity: 0.9 }}>
          💡 提示：录制时请确保手势清晰可见，建议录制 {maxDuration} 秒左右的视频
        </Typography>
      </Stack>
    </Paper>
  )
}

export default CameraRecorder
