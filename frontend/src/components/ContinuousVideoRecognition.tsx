import React, { useState, useCallback, useRef } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Alert,
  Stack,
  Chip,
  Grid,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from '@mui/material'
import {
  CloudUpload,
  PlayArrow,
  Stop,
  Refresh,
  VideoFile,
  ExpandMore,
} from '@mui/icons-material'

import continuousSignRecognitionService, {
  ContinuousRecognitionResult
} from '../services/continuousSignRecognitionService'

interface Props {
  onResult?: (result: ContinuousRecognitionResult) => void
}

const ContinuousVideoRecognition: React.FC<Props> = ({ onResult }) => {
  const [file, setFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<'idle' | 'processing' | 'completed' | 'error' | 'stopped'>('idle')
  const [statusMessage, setStatusMessage] = useState('')
  const [result, setResult] = useState<ContinuousRecognitionResult | null>(null)
  const [error, setError] = useState('')

  const abortRef = useRef<AbortController | null>(null)

  // 文件选择处理
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      
      // 检查文件大小 (限制100MB)
      if (selectedFile.size > 100 * 1024 * 1024) {
        setError('视频文件过大，请选择小于100MB的文件')
        return
      }

      // 检查文件类型
      if (!selectedFile.type.startsWith('video/')) {
        setError('请选择有效的视频文件')
        return
      }

      setFile(selectedFile)
      setError('')
      setResult(null)
    }
  }

  // 开始识别
  const startRecognition = useCallback(async () => {
    if (!file) return

    setIsProcessing(true)
    setError('')
    setStatus('processing')
    setProgress(0)
    setResult(null)

    abortRef.current = new AbortController()

    try {
      const result = await continuousSignRecognitionService.recognizeVideo(
        file,
        (progress, status) => {
          setProgress(progress)
          setStatusMessage(status)
        }
      )

      setResult(result)
      setStatus('completed')
      onResult?.(result)

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '识别过程中发生未知错误'
      setError(errorMessage)
      setStatus('error')
    } finally {
      setIsProcessing(false)
    }
  }, [file, onResult])

  // 停止处理
  const stopProcessing = useCallback(() => {
    abortRef.current?.abort()
    setIsProcessing(false)
    setStatus('stopped')
    setStatusMessage('已停止')
  }, [])

  // 重置
  const resetRecognition = useCallback(() => {
    setFile(null)
    setIsProcessing(false)
    setProgress(0)
    setStatus('idle')
    setStatusMessage('')
    setResult(null)
    setError('')
  }, [])

  // 格式化文件大小
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 格式化时长
  const formatDuration = (duration?: number): string => {
    if (!duration) return '—'
    const minutes = Math.floor(duration / 60)
    const seconds = Math.floor(duration % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  return (
    <Card sx={{ p: 3 }}>
      <CardContent>
        <Stack spacing={3}>
          {/* 标题 */}
          <Box>
            <Typography variant="h5" gutterBottom fontWeight={600}>
              连续手语识别
            </Typography>
            <Typography variant="body2" color="text.secondary">
              使用真正的CSLR模型进行完整句子识别
            </Typography>
          </Box>

          {/* 错误提示 */}
          {error && (
            <Alert severity="error" onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          {/* 文件选择 */}
          <Card variant="outlined">
            <CardContent>
              <Stack spacing={2}>
                <Stack direction="row" spacing={2} alignItems="center">
                  <Button 
                    variant="contained" 
                    component="label" 
                    startIcon={<CloudUpload />}
                    disabled={isProcessing}
                  >
                    选择视频文件
                    <input 
                      hidden 
                      type="file" 
                      accept="video/*" 
                      onChange={handleFileChange} 
                    />
                  </Button>
                  
                  {file && (
                    <Box>
                      <Chip 
                        icon={<VideoFile />}
                        label={file.name}
                        color="info"
                        variant="outlined"
                      />
                      <Typography variant="caption" display="block" color="text.secondary">
                        {formatFileSize(file.size)}
                      </Typography>
                    </Box>
                  )}
                </Stack>
              </Stack>
            </CardContent>
          </Card>

          {/* 进度显示 */}
          {isProcessing && (
            <Box>
              <LinearProgress 
                variant="determinate" 
                value={progress * 100} 
                sx={{ height: 8, borderRadius: 2 }}
              />
              <Stack direction="row" justifyContent="space-between" sx={{ mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  {statusMessage}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {(progress * 100).toFixed(1)}%
                </Typography>
              </Stack>
            </Box>
          )}

          {/* 控制按钮 */}
          <Stack direction="row" spacing={2}>
            <Button
              disabled={!file || isProcessing}
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={startRecognition}
            >
              开始识别
            </Button>
            
            <Button
              disabled={!isProcessing}
              color="warning"
              variant="outlined"
              startIcon={<Stop />}
              onClick={stopProcessing}
            >
              停止
            </Button>
            
            <Button
              disabled={isProcessing}
              variant="outlined"
              startIcon={<Refresh />}
              onClick={resetRecognition}
            >
              重置
            </Button>
          </Stack>

          {/* 识别结果 */}
          {result && (
            <Box>
              <Typography variant="h6" gutterBottom fontWeight={600}>
                识别结果
              </Typography>

              <Grid container spacing={3}>
                {/* 主要结果 */}
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom fontWeight={600}>
                      识别文本
                    </Typography>
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        minHeight: 60,
                        p: 2,
                        bgcolor: 'background.default',
                        borderRadius: 1,
                        wordBreak: 'break-word',
                        mb: 1
                      }}
                    >
                      {result.text || '无识别结果'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      总体置信度: {((result.overall_confidence || 0) * 100).toFixed(1)}%
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom fontWeight={600}>
                      处理信息
                    </Typography>
                    <Stack spacing={1}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">视频时长:</Typography>
                        <Typography variant="body2">{formatDuration(result.duration)}</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">帧数:</Typography>
                        <Typography variant="body2">{result.frame_count ?? '—'}</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">帧率:</Typography>
                        <Typography variant="body2">{(result.fps ?? 0).toFixed(1)} fps</Typography>
                      </Box>
                    </Stack>
                  </Paper>
                </Grid>
              </Grid>

              {/* Gloss序列详情 */}
              {result.gloss_sequence && result.gloss_sequence.length > 0 && (
                <Accordion sx={{ mt: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1">
                      Gloss序列 ({result.gloss_sequence.length} 个词汇)
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {result.gloss_sequence.map((gloss, index) => (
                        <Chip 
                          key={index} 
                          label={gloss} 
                          variant="outlined" 
                          size="small"
                        />
                      ))}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* 分段详情 */}
              {result.segments && result.segments.length > 0 && (
                <Accordion sx={{ mt: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1">
                      分段详情 ({result.segments.length} 个分段)
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>分段</TableCell>
                          <TableCell>Gloss序列</TableCell>
                          <TableCell>时间</TableCell>
                          <TableCell>置信度</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.segments.map((segment, index) => (
                          <TableRow key={index}>
                            <TableCell>{index + 1}</TableCell>
                            <TableCell>
                              {segment.gloss_sequence.join(' ')}
                            </TableCell>
                            <TableCell>
                              {segment.start_time.toFixed(1)}s - {segment.end_time.toFixed(1)}s
                            </TableCell>
                            <TableCell>
                              {(segment.confidence * 100).toFixed(1)}%
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </AccordionDetails>
                </Accordion>
              )}
            </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

export default ContinuousVideoRecognition
