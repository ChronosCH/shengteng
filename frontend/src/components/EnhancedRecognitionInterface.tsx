/**
 * 增强的手语识别界面组件
 * 提供更好的用户体验和交互设计
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Stack,
  LinearProgress,
  Alert,
  Fade,
  Zoom,
  Tooltip,
  IconButton,
  Divider,
  Badge,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Videocam,
  VideocamOff,
  Mic,
  MicOff,
  Settings,
  Info,
  TrendingUp,
  Speed,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material'

interface EnhancedRecognitionInterfaceProps {
  isConnected: boolean
  isRecognizing: boolean
  currentText: string
  confidence: number | null
  onStartStop: () => void
  onConnect: () => void
  isConnecting: boolean
  error?: string | null
}

const EnhancedRecognitionInterface: React.FC<EnhancedRecognitionInterfaceProps> = ({
  isConnected,
  isRecognizing,
  currentText,
  confidence,
  onStartStop,
  onConnect,
  isConnecting,
  error,
}) => {
  const [showDetails, setShowDetails] = useState(false)
  const [recognitionTime, setRecognitionTime] = useState(0)
  const [wordCount, setWordCount] = useState(0)

  // 计算识别时间
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isRecognizing) {
      interval = setInterval(() => {
        setRecognitionTime(prev => prev + 1)
      }, 1000)
    } else {
      setRecognitionTime(0)
    }
    return () => clearInterval(interval)
  }, [isRecognizing])

  // 计算词汇数量
  useEffect(() => {
    const words = currentText.trim().split(/\s+/).filter(word => word.length > 0)
    setWordCount(words.length)
  }, [currentText])

  // 格式化时间显示
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  // 获取置信度颜色
  const getConfidenceColor = (conf: number | null) => {
    if (conf === null) return 'default'
    if (conf > 0.8) return 'success'
    if (conf > 0.6) return 'warning'
    return 'error'
  }

  // 获取置信度描述
  const getConfidenceDescription = (conf: number | null) => {
    if (conf === null) return '未知'
    if (conf > 0.9) return '非常高'
    if (conf > 0.8) return '高'
    if (conf > 0.6) return '中等'
    if (conf > 0.4) return '较低'
    return '低'
  }

  return (
    <Card 
      elevation={8}
      sx={{
        background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
        borderRadius: 3,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* 状态指示条 */}
      {isRecognizing && (
        <LinearProgress 
          sx={{
            height: 4,
            background: 'rgba(59, 130, 246, 0.1)',
            '& .MuiLinearProgress-bar': {
              background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
            }
          }}
        />
      )}

      <CardContent sx={{ p: 4 }}>
        {/* 连接状态和控制区域 */}
        <Box sx={{ mb: 4 }}>
          <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
            <Box>
              <Typography variant="h5" fontWeight={700} gutterBottom>
                手语识别控制台
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center">
                <Chip
                  icon={isConnected ? <Visibility /> : <VisibilityOff />}
                  label={isConnected ? '已连接' : '未连接'}
                  color={isConnected ? 'success' : 'default'}
                  size="small"
                />
                {isRecognizing && (
                  <Chip
                    icon={<Videocam />}
                    label="识别中"
                    color="primary"
                    size="small"
                    sx={{ animation: 'pulse 2s infinite' }}
                  />
                )}
              </Stack>
            </Box>

            <Stack direction="row" spacing={1}>
              <Tooltip title="显示详细信息">
                <IconButton 
                  onClick={() => setShowDetails(!showDetails)}
                  color={showDetails ? 'primary' : 'default'}
                >
                  <Info />
                </IconButton>
              </Tooltip>
              <Tooltip title="设置">
                <IconButton>
                  <Settings />
                </IconButton>
              </Tooltip>
            </Stack>
          </Stack>
        </Box>

        {/* 错误提示 */}
        {error && (
          <Fade in={Boolean(error)}>
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          </Fade>
        )}

        {/* 主控制按钮 */}
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          {!isConnected ? (
            <Button
              variant="contained"
              size="large"
              onClick={onConnect}
              disabled={isConnecting}
              startIcon={isConnecting ? <LinearProgress size={20} /> : <PlayArrow />}
              sx={{
                px: 4,
                py: 1.5,
                fontSize: '1.1rem',
                borderRadius: 2,
                background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                }
              }}
            >
              {isConnecting ? '连接中...' : '连接服务器'}
            </Button>
          ) : (
            <Button
              variant="contained"
              size="large"
              onClick={onStartStop}
              startIcon={isRecognizing ? <Stop /> : <PlayArrow />}
              color={isRecognizing ? 'error' : 'success'}
              sx={{
                px: 4,
                py: 1.5,
                fontSize: '1.1rem',
                borderRadius: 2,
                minWidth: 160,
              }}
            >
              {isRecognizing ? '停止识别' : '开始识别'}
            </Button>
          )}
        </Box>

        {/* 识别结果显示 */}
        {currentText && (
          <Zoom in={Boolean(currentText)}>
            <Card 
              variant="outlined" 
              sx={{ 
                mb: 3,
                background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                border: '2px solid #0ea5e9',
              }}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary" fontWeight={600}>
                  识别结果
                </Typography>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    fontSize: '1.2rem',
                    lineHeight: 1.6,
                    color: '#0f172a',
                    fontWeight: 500,
                  }}
                >
                  {currentText}
                </Typography>
              </CardContent>
            </Card>
          </Zoom>
        )}

        {/* 详细信息面板 */}
        {showDetails && (
          <Fade in={showDetails}>
            <Box>
              <Divider sx={{ mb: 3 }} />
              <Typography variant="h6" gutterBottom fontWeight={600}>
                识别统计
              </Typography>
              <Stack direction="row" spacing={3} flexWrap="wrap">
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" fontWeight={700} color="primary">
                    {formatTime(recognitionTime)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    识别时长
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" fontWeight={700} color="success.main">
                    {wordCount}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    识别词汇
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" fontWeight={700} color={getConfidenceColor(confidence) + '.main'}>
                    {confidence ? `${(confidence * 100).toFixed(0)}%` : '--'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    置信度 ({getConfidenceDescription(confidence)})
                  </Typography>
                </Box>
              </Stack>
            </Box>
          </Fade>
        )}
      </CardContent>

      {/* CSS动画 */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
          }
        `}
      </style>
    </Card>
  )
}

export default EnhancedRecognitionInterface
