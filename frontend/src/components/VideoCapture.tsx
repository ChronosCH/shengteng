/**
 * 视频捕获组件 - 处理摄像头视频流
 */

import React, { useRef, useEffect, useState } from 'react'
import {
  Box,
  Button,
  Typography,
  Alert,
  Paper,
  Card,
  CardContent,
  Chip,
  Stack,
  IconButton,
  LinearProgress,
} from '@mui/material'
import {
  Videocam,
  VideocamOff,
  CameraAlt,
  PhotoCamera,
  Settings,
} from '@mui/icons-material'

interface VideoCaptureProps {
  onFrame?: (videoElement: HTMLVideoElement) => void
  isActive?: boolean
}

const VideoCapture: React.FC<VideoCaptureProps> = ({
  onFrame,
  isActive = false,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // 启动摄像头
  const startCamera = async () => {
    try {
      setError(null)
      setIsLoading(true)
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      })

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        videoRef.current.play()
      }

      setStream(mediaStream)
      setIsStreaming(true)
    } catch (err) {
      console.error('摄像头启动失败:', err)
      setError('无法访问摄像头，请检查权限设置')
    } finally {
      setIsLoading(false)
    }
  }

  // 停止摄像头
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setIsStreaming(false)
  }

  // 处理视频帧
  useEffect(() => {
    if (!isStreaming || !isActive || !onFrame || !videoRef.current) return

    const processFrame = () => {
      if (videoRef.current && onFrame) {
        onFrame(videoRef.current)
      }
      requestAnimationFrame(processFrame)
    }

    const frameId = requestAnimationFrame(processFrame)
    return () => cancelAnimationFrame(frameId)
  }, [isStreaming, isActive, onFrame])

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <Card 
      sx={{ 
        overflow: 'hidden',
        background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(152,251,152,0.1) 100%)',
        backdropFilter: 'blur(10px)',
      }}
    >
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <PhotoCamera sx={{ mr: 1.5, color: 'primary.main', fontSize: 24 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              视频输入
            </Typography>
          </Box>
          
          <Stack direction="row" spacing={1}>
            {isStreaming && (
              <Chip 
                label="运行中" 
                color="success" 
                size="small"
                sx={{ fontSize: '0.75rem' }}
              />
            )}
            {isActive && (
              <Chip 
                label="识别中" 
                color="info" 
                size="small"
                sx={{ fontSize: '0.75rem' }}
              />
            )}
          </Stack>
        </Box>

        {error && (
          <Alert 
            severity="error" 
            sx={{ 
              mb: 2,
              borderRadius: 2,
              backgroundColor: 'rgba(255, 179, 186, 0.1)',
              border: '1px solid rgba(255, 179, 186, 0.3)',
            }}
          >
            {error}
          </Alert>
        )}

        {isLoading && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              正在启动摄像头...
            </Typography>
            <LinearProgress 
              sx={{ 
                borderRadius: 1,
                height: 6,
                backgroundColor: 'rgba(255, 182, 193, 0.2)',
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, #FFB6C1 0%, #98FB98 100%)',
                }
              }}
            />
          </Box>
        )}

        <Box 
          sx={{ 
            position: 'relative', 
            mb: 2,
            borderRadius: 3,
            overflow: 'hidden',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
          }}
        >
          <video
            ref={videoRef}
            style={{
              width: '100%',
              maxWidth: 640,
              height: 'auto',
              backgroundColor: '#F0F8FF',
              display: 'block',
            }}
            playsInline
            muted
          />
          
          {!isStreaming && !isLoading && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, rgba(240,248,255,0.9) 0%, rgba(230,247,255,0.9) 100%)',
                color: 'text.secondary',
                minHeight: 200,
              }}
            >
              <CameraAlt sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
              <Typography variant="body1" sx={{ fontWeight: 500 }}>
                摄像头未启动
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.7, mt: 0.5 }}>
                点击下方按钮开始
              </Typography>
            </Box>
          )}
          
          {isStreaming && (
            <Box
              sx={{
                position: 'absolute',
                top: 12,
                right: 12,
                display: 'flex',
                gap: 1,
              }}
            >
              <Chip 
                label="LIVE" 
                color="error" 
                size="small"
                sx={{ 
                  fontWeight: 600,
                  fontSize: '0.7rem',
                  animation: 'pulse 2s infinite',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                    '100%': { opacity: 1 },
                  }
                }}
              />
            </Box>
          )}
        </Box>

        <Stack direction="row" spacing={2}>
          {!isStreaming ? (
            <Button
              variant="contained"
              startIcon={<Videocam />}
              onClick={startCamera}
              disabled={isLoading}
              fullWidth
              size="large"
              sx={{ 
                py: 1.5,
                fontWeight: 600,
              }}
            >
              {isLoading ? '启动中...' : '启动摄像头'}
            </Button>
          ) : (
            <Button
              variant="outlined"
              startIcon={<VideocamOff />}
              onClick={stopCamera}
              fullWidth
              size="large"
              sx={{ 
                py: 1.5,
                fontWeight: 600,
              }}
            >
              停止摄像头
            </Button>
          )}
        </Stack>

        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
          width={640}
          height={480}
        />
      </CardContent>
    </Card>
  )
}

export default VideoCapture
