/**
 * 手语识别结果展示组件
 * 提供丰富的结果展示和历史记录功能
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Stack,
  IconButton,
  Tooltip,
  Fade,
  Zoom,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material'
import {
  ContentCopy,
  Share,
  Save,
  History,
  TrendingUp,
  AccessTime,
  Translate,
  VolumeUp,
  Download,
  Clear,
} from '@mui/icons-material'

interface RecognitionResult {
  id: string
  text: string
  confidence: number
  timestamp: Date
  duration: number
  wordCount: number
}

interface RecognitionResultDisplayProps {
  currentText: string
  confidence: number | null
  isRecognizing: boolean
  onSave?: (text: string, title: string) => void
  onShare?: (text: string) => void
}

const RecognitionResultDisplay: React.FC<RecognitionResultDisplayProps> = ({
  currentText,
  confidence,
  isRecognizing,
  onSave,
  onShare,
}) => {
  const [history, setHistory] = useState<RecognitionResult[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [saveTitle, setSaveTitle] = useState('')
  const [startTime, setStartTime] = useState<Date | null>(null)

  // 记录识别开始时间
  useEffect(() => {
    if (isRecognizing && !startTime) {
      setStartTime(new Date())
    } else if (!isRecognizing && startTime) {
      // 识别结束，保存结果到历史
      if (currentText.trim()) {
        const duration = (new Date().getTime() - startTime.getTime()) / 1000
        const wordCount = currentText.trim().split(/\s+/).length
        
        const result: RecognitionResult = {
          id: Date.now().toString(),
          text: currentText,
          confidence: confidence || 0,
          timestamp: new Date(),
          duration,
          wordCount,
        }
        
        setHistory(prev => [result, ...prev.slice(0, 9)]) // 保留最近10条记录
      }
      setStartTime(null)
    }
  }, [isRecognizing, currentText, confidence, startTime])

  // 复制到剪贴板
  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      // 可以添加成功提示
    } catch (err) {
      console.error('复制失败:', err)
    }
  }

  // 语音播报
  const handleSpeak = (text: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = 'zh-CN'
      speechSynthesis.speak(utterance)
    }
  }

  // 保存结果
  const handleSave = () => {
    if (onSave && currentText.trim()) {
      onSave(currentText, saveTitle || `识别结果_${new Date().toLocaleString()}`)
      setSaveDialogOpen(false)
      setSaveTitle('')
    }
  }

  // 分享结果
  const handleShare = () => {
    if (onShare && currentText.trim()) {
      onShare(currentText)
    }
  }

  // 获取置信度颜色和描述
  const getConfidenceInfo = (conf: number | null) => {
    if (conf === null) return { color: 'default', label: '未知', description: '置信度未知' }
    if (conf > 0.9) return { color: 'success', label: '优秀', description: '识别准确度很高' }
    if (conf > 0.8) return { color: 'success', label: '良好', description: '识别准确度较高' }
    if (conf > 0.6) return { color: 'warning', label: '一般', description: '识别准确度中等' }
    if (conf > 0.4) return { color: 'warning', label: '较低', description: '识别准确度较低' }
    return { color: 'error', label: '很低', description: '识别准确度很低' }
  }

  const confidenceInfo = getConfidenceInfo(confidence)

  return (
    <Box>
      {/* 当前识别结果 */}
      {currentText && (
        <Zoom in={Boolean(currentText)}>
          <Card 
            elevation={4}
            sx={{
              mb: 3,
              background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
              border: '2px solid #0ea5e9',
              borderRadius: 3,
            }}
          >
            <CardContent sx={{ p: 3 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 2 }}>
                <Typography variant="h6" fontWeight={600} color="primary">
                  识别结果
                </Typography>
                <Stack direction="row" spacing={1}>
                  <Tooltip title="复制文本">
                    <IconButton size="small" onClick={() => handleCopy(currentText)}>
                      <ContentCopy fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="语音播报">
                    <IconButton size="small" onClick={() => handleSpeak(currentText)}>
                      <VolumeUp fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="保存结果">
                    <IconButton size="small" onClick={() => setSaveDialogOpen(true)}>
                      <Save fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="分享结果">
                    <IconButton size="small" onClick={handleShare}>
                      <Share fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Stack>
              </Stack>

              <Typography 
                variant="body1" 
                sx={{ 
                  fontSize: '1.3rem',
                  lineHeight: 1.6,
                  color: '#0f172a',
                  fontWeight: 500,
                  mb: 2,
                  minHeight: '2em',
                }}
              >
                {currentText}
              </Typography>

              {/* 置信度和统计信息 */}
              <Stack direction="row" spacing={2} flexWrap="wrap">
                {confidence !== null && (
                  <Tooltip title={confidenceInfo.description}>
                    <Chip
                      label={`置信度: ${(confidence * 100).toFixed(1)}% (${confidenceInfo.label})`}
                      color={confidenceInfo.color as any}
                      size="small"
                      icon={<TrendingUp />}
                    />
                  </Tooltip>
                )}
                <Chip
                  label={`词汇数: ${currentText.trim().split(/\s+/).filter(w => w).length}`}
                  size="small"
                  variant="outlined"
                />
                {isRecognizing && startTime && (
                  <Chip
                    label={`识别时长: ${Math.floor((new Date().getTime() - startTime.getTime()) / 1000)}s`}
                    size="small"
                    variant="outlined"
                    icon={<AccessTime />}
                  />
                )}
              </Stack>
            </CardContent>
          </Card>
        </Zoom>
      )}

      {/* 历史记录 */}
      {history.length > 0 && (
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent sx={{ p: 2 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
              <Typography variant="h6" fontWeight={600}>
                识别历史
              </Typography>
              <Stack direction="row" spacing={1}>
                <Button
                  size="small"
                  startIcon={<History />}
                  onClick={() => setShowHistory(!showHistory)}
                >
                  {showHistory ? '隐藏' : '显示'}历史
                </Button>
                <Button
                  size="small"
                  startIcon={<Clear />}
                  onClick={() => setHistory([])}
                  color="error"
                >
                  清空
                </Button>
              </Stack>
            </Stack>

            <Fade in={showHistory}>
              <Box>
                {showHistory && (
                  <List dense>
                    {history.map((result, index) => (
                      <React.Fragment key={result.id}>
                        <ListItem
                          sx={{
                            borderRadius: 1,
                            mb: 1,
                            bgcolor: 'rgba(0,0,0,0.02)',
                            '&:hover': { bgcolor: 'rgba(0,0,0,0.04)' }
                          }}
                        >
                          <ListItemIcon>
                            <Typography variant="caption" color="text.secondary">
                              {index + 1}
                            </Typography>
                          </ListItemIcon>
                          <ListItemText
                            primary={result.text}
                            secondary={
                              <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                                <Chip
                                  label={`${(result.confidence * 100).toFixed(0)}%`}
                                  size="small"
                                  color={getConfidenceInfo(result.confidence).color as any}
                                />
                                <Chip
                                  label={`${result.duration.toFixed(1)}s`}
                                  size="small"
                                  variant="outlined"
                                />
                                <Chip
                                  label={result.timestamp.toLocaleTimeString()}
                                  size="small"
                                  variant="outlined"
                                />
                              </Stack>
                            }
                          />
                          <Stack direction="row" spacing={0.5}>
                            <IconButton size="small" onClick={() => handleCopy(result.text)}>
                              <ContentCopy fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleSpeak(result.text)}>
                              <VolumeUp fontSize="small" />
                            </IconButton>
                          </Stack>
                        </ListItem>
                        {index < history.length - 1 && <Divider />}
                      </React.Fragment>
                    ))}
                  </List>
                )}
              </Box>
            </Fade>
          </CardContent>
        </Card>
      )}

      {/* 保存对话框 */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>保存识别结果</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="标题"
            fullWidth
            variant="outlined"
            value={saveTitle}
            onChange={(e) => setSaveTitle(e.target.value)}
            placeholder={`识别结果_${new Date().toLocaleString()}`}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="内容"
            fullWidth
            multiline
            rows={4}
            variant="outlined"
            value={currentText}
            InputProps={{ readOnly: true }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>取消</Button>
          <Button onClick={handleSave} variant="contained">保存</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default RecognitionResultDisplay
