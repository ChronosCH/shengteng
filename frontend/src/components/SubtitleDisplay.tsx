/**
 * 字幕显示组件 - 显示识别结果和置信度
 */

import React, { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Chip,
  Card,
  CardContent,
  IconButton,
  Stack,
  Fade,
  Tooltip,
} from '@mui/material'
import {
  Subtitles,
  VolumeUp,
  ContentCopy,
  Favorite,
  Psychology,
  TrendingUp,
} from '@mui/icons-material'

interface SubtitleDisplayProps {
  text: string
  confidence: number
  isRecognizing: boolean
}

const SubtitleDisplay: React.FC<SubtitleDisplayProps> = ({
  text,
  confidence,
  isRecognizing,
}) => {
  const [copySuccess, setCopySuccess] = useState(false)

  // 获取置信度颜色
  const getConfidenceColor = (): 'success' | 'warning' | 'error' | 'primary' => {
    if (confidence >= 0.8) return 'success'
    if (confidence >= 0.6) return 'warning'
    if (confidence > 0) return 'error'
    return 'primary'
  }

  // 语音播报功能
  const speakText = () => {
    if (text && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = 'zh-CN'
      utterance.rate = 1.0
      window.speechSynthesis.speak(utterance)
    }
  }

  // 复制文本功能
  const copyText = async () => {
    if (text) {
      try {
        await navigator.clipboard.writeText(text)
        setCopySuccess(true)
        setTimeout(() => setCopySuccess(false), 2000)
      } catch (err) {
        console.error('复制失败:', err)
      }
    }
  }

  return (
    <Stack spacing={3}>
      {/* 状态卡片 */}
      <Card 
        sx={{ 
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(191,239,255,0.1) 100%)',
          backdropFilter: 'blur(10px)',
        }}
      >
        <CardContent sx={{ py: 2.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Psychology sx={{ mr: 1.5, color: 'info.main', fontSize: 24 }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              识别状态
            </Typography>
          </Box>
          
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Chip 
              label={isRecognizing ? "识别中" : "待机"} 
              color={isRecognizing ? "info" : "default"}
              icon={isRecognizing ? <TrendingUp /> : undefined}
              sx={{ fontWeight: 500 }}
            />
            
            {confidence > 0 && (
              <Chip 
                label={`${(confidence * 100).toFixed(1)}%`}
                color={getConfidenceColor()}
                icon={<Favorite sx={{ fontSize: 16 }} />}
                sx={{ fontWeight: 500 }}
              />
            )}
          </Stack>

          {/* 置信度进度条 */}
          {isRecognizing && confidence > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                识别置信度
              </Typography>
              <LinearProgress
                variant="determinate"
                value={confidence * 100}
                sx={{ 
                  height: 8, 
                  borderRadius: 4,
                  backgroundColor: 'rgba(255, 182, 193, 0.2)',
                  '& .MuiLinearProgress-bar': {
                    background: confidence >= 0.8 
                      ? 'linear-gradient(90deg, #98FB98 0%, #90EE90 100%)'
                      : confidence >= 0.6
                      ? 'linear-gradient(90deg, #FFDFBA 0%, #FFB347 100%)'
                      : 'linear-gradient(90deg, #FFB3BA 0%, #FF6B6B 100%)',
                    borderRadius: 4,
                  }
                }}
              />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* 文本显示卡片 */}
      <Card 
        sx={{ 
          flex: 1,
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,182,193,0.05) 100%)',
          backdropFilter: 'blur(10px)',
        }}
      >
        <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Subtitles sx={{ mr: 1.5, color: 'primary.main', fontSize: 24 }} />
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                识别结果
              </Typography>
            </Box>
            
            {text && (
              <Stack direction="row" spacing={1}>
                <Tooltip title={copySuccess ? "已复制!" : "复制文本"}>
                  <IconButton 
                    onClick={copyText}
                    size="small"
                    sx={{ 
                      bgcolor: copySuccess ? 'success.light' : 'background.paper',
                      '&:hover': { 
                        bgcolor: copySuccess ? 'success.main' : 'primary.light',
                        transform: 'scale(1.1)',
                      }
                    }}
                  >
                    <ContentCopy sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="语音播报">
                  <IconButton 
                    onClick={speakText}
                    size="small"
                    sx={{ 
                      bgcolor: 'background.paper',
                      '&:hover': { 
                        bgcolor: 'secondary.light',
                        transform: 'scale(1.1)',
                      }
                    }}
                  >
                    <VolumeUp sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
              </Stack>
            )}
          </Box>

          {/* 文本显示区域 */}
          <Box 
            sx={{ 
              flex: 1,
              minHeight: 300,
              background: 'linear-gradient(135deg, #FFFEF7 0%, #F0F8FF 100%)',
              p: 3,
              borderRadius: 3,
              border: '2px solid',
              borderColor: text ? 'primary.light' : 'divider',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            {text ? (
              <Fade in timeout={500}>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    color: 'text.primary',
                    fontWeight: 600,
                    lineHeight: 1.6,
                    textAlign: 'center',
                    textShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  }}
                >
                  {text}
                </Typography>
              </Fade>
            ) : (
              <Box sx={{ textAlign: 'center' }}>
                <Subtitles 
                  sx={{ 
                    fontSize: 64, 
                    color: 'text.disabled', 
                    mb: 2,
                    opacity: 0.3,
                  }} 
                />
                <Typography 
                  variant="h6" 
                  color="text.secondary"
                  sx={{ fontStyle: 'italic', fontWeight: 500 }}
                >
                  {isRecognizing ? "等待手语识别结果..." : "请开始手语识别"}
                </Typography>
                <Typography 
                  variant="body2" 
                  color="text.disabled"
                  sx={{ mt: 1 }}
                >
                  识别结果将在这里显示
                </Typography>
              </Box>
            )}
            
            {/* 装饰性背景图案 */}
            <Box
              sx={{
                position: 'absolute',
                top: -20,
                right: -20,
                width: 100,
                height: 100,
                background: 'linear-gradient(135deg, rgba(255,182,193,0.1) 0%, rgba(152,251,152,0.1) 100%)',
                borderRadius: '50%',
                zIndex: -1,
              }}
            />
            <Box
              sx={{
                position: 'absolute',
                bottom: -30,
                left: -30,
                width: 120,
                height: 120,
                background: 'linear-gradient(135deg, rgba(191,239,255,0.1) 0%, rgba(255,223,186,0.1) 100%)',
                borderRadius: '50%',
                zIndex: -1,
              }}
            />
          </Box>
        </CardContent>
      </Card>

      {/* 使用提示卡片 */}
      <Card 
        sx={{ 
          background: 'linear-gradient(135deg, rgba(152,251,152,0.1) 0%, rgba(255,255,255,0.9) 100%)',
          border: '1px solid rgba(152,251,152,0.3)',
        }}
      >
        <CardContent sx={{ py: 2 }}>
          <Typography 
            variant="body2" 
            sx={{ 
              color: 'success.main',
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            💡 <Box component="span" sx={{ ml: 1 }}>
              保持标准的手语动作，确保手部在摄像头视野内，获得更好的识别效果
            </Box>
          </Typography>
        </CardContent>
      </Card>
    </Stack>
  )
}

export default SubtitleDisplay
