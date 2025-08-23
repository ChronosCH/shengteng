import React, { useState, useRef, useCallback } from 'react'
import {
  Box,
  Button,
  LinearProgress,
  Typography,
  Paper,
  Stack,
  Chip,
  Card,
  CardContent,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Divider,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid
} from '@mui/material'
import {
  CloudUpload,
  PlayArrow,
  Stop,
  Refresh,
  Psychology,
  VideoFile,
  Timeline,
  TrendingUp,
  ExpandMore,
  CheckCircle,
  Error as ErrorIcon,
  Speed
} from '@mui/icons-material'

import enhancedCECSLService, { VideoProcessResult, PredictionResult } from '../services/enhancedCECSLService'

interface Props {
  onResult?: (result: VideoProcessResult) => void
}

const EnhancedVideoRecognition: React.FC<Props> = ({ onResult }) => {
  const [file, setFile] = useState<File | null>(null)
  const [taskId, setTaskId] = useState<string>('')
  const [progress, setProgress] = useState<number>(0)
  const [status, setStatus] = useState<string>('idle')
  const [statusMessage, setStatusMessage] = useState<string>('')
  const [result, setResult] = useState<VideoProcessResult | null>(null)
  const [error, setError] = useState<string>('')
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [serviceStats, setServiceStats] = useState<any>(null)
  
  const abortRef = useRef<AbortController | null>(null)

  // 重置状态
  const reset = useCallback(() => {
    abortRef.current?.abort()
    setFile(null)
    setTaskId('')
    setProgress(0)
    setStatus('idle')
    setStatusMessage('')
    setResult(null)
    setError('')
    setIsProcessing(false)
  }, [])

  // 处理文件选择
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
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

  // 加载服务统计信息
  const loadServiceStats = useCallback(async () => {
    try {
      const stats = await enhancedCECSLService.getStats()
      setServiceStats(stats)
    } catch (err) {
      console.warn('Failed to load service stats:', err)
    }
  }, [])

  // 开始识别
  const startRecognition = useCallback(async () => {
    if (!file) return

    setIsProcessing(true)
    setError('')
    setProgress(0)

    try {
      const result = await enhancedCECSLService.recognizeVideo(
        file,
        (progress, status) => {
          setProgress(progress)
          setStatusMessage(status)
        }
      )

      setResult(result)
      setStatus('completed')
      onResult?.(result)
      
      // 刷新服务统计
      await loadServiceStats()

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '识别过程中发生未知错误'
      setError(errorMessage)
      setStatus('error')
    } finally {
      setIsProcessing(false)
    }
  }, [file, onResult, loadServiceStats])

  // 停止处理
  const stopProcessing = useCallback(() => {
    abortRef.current?.abort()
    setIsProcessing(false)
    setStatus('stopped')
    setStatusMessage('已停止')
  }, [])

  // 格式化文件大小
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 格式化时间
  const formatDuration = (seconds: number | undefined): string => {
    const s = typeof seconds === 'number' && isFinite(seconds) ? seconds : 0
    const mins = Math.floor(s / 60)
    const secs = Math.floor(s % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // 组件挂载时加载统计信息
  React.useEffect(() => {
    loadServiceStats()
  }, [loadServiceStats])

  return (
    <Paper variant="outlined" sx={{ p: 3, borderRadius: 3 }}>
      <Stack spacing={3}>
        {/* 标题和服务状态 */}
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
            <Psychology sx={{ mr: 1, verticalAlign: 'middle' }} />
            增强版CE-CSL手语识别
          </Typography>
          <Typography variant="body2" color="text.secondary">
            基于训练好的enhanced_cecsl_final_model.ckpt模型
          </Typography>
          
          {serviceStats && (
            <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
              <Chip 
                size="small" 
                label={`词汇: ${serviceStats.model_info?.vocab_size || 0}`}
                color="info"
              />
              <Chip 
                size="small" 
                label={`预测: ${serviceStats.stats?.predictions || 0}次`}
                color="primary"
              />
              <Chip 
                size="small" 
                label={serviceStats.model_info?.is_loaded ? '模型已加载' : '模型未加载'}
                color={serviceStats.model_info?.is_loaded ? 'success' : 'warning'}
              />
            </Stack>
          )}
        </Box>

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

              {/* 状态指示器 */}
              {status !== 'idle' && (
                <Stack direction="row" spacing={1}>
                  {status === 'completed' && (
                    <Chip icon={<CheckCircle />} label="识别完成" color="success" />
                  )}
                  {status === 'error' && (
                    <Chip icon={<ErrorIcon />} label="识别失败" color="error" />
                  )}
                  {status === 'stopped' && (
                    <Chip icon={<Stop />} label="已停止" color="warning" />
                  )}
                  {isProcessing && (
                    <Chip 
                      icon={<CircularProgress size={16} />} 
                      label={statusMessage} 
                      color="primary" 
                    />
                  )}
                </Stack>
              )}
            </Stack>
          </CardContent>
        </Card>

        {/* 进度条 */}
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
            onClick={reset}
          >
            重置
          </Button>
        </Stack>

        {/* 错误信息 */}
        {error && (
          <Alert severity="error" sx={{ borderRadius: 2 }}>
            {error}
          </Alert>
        )}

        {/* 识别结果 */}
        {result && (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                识别结果
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom fontWeight={600}>
                      预测文本
                    </Typography>
                    <Typography 
                      variant="h5" 
                      color="primary" 
                      sx={{ 
                        fontWeight: 'bold',
                        wordBreak: 'break-word',
                        mb: 1
                      }}
                    >
                      {result.recognition_result?.text || '无识别结果'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      置信度: {((result.recognition_result?.confidence || 0) * 100).toFixed(1)}%
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
                        <Typography variant="body2">{formatDuration(result?.duration)}</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">帧数:</Typography>
                        <Typography variant="body2">{result?.frame_count ?? '—'}</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">帧率:</Typography>
                        <Typography variant="body2">{(result?.fps ?? 0).toFixed(1)} fps</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2">处理时间:</Typography>
                        <Typography variant="body2">{(result?.processing_time ?? 0).toFixed(2)}s</Typography>
                      </Box>
                    </Stack>
                  </Paper>
                </Grid>
              </Grid>

              {/* 手势序列详情 */}
              {result.recognition_result?.gloss_sequence && result.recognition_result.gloss_sequence.length > 0 && (
                <Accordion sx={{ mt: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1">
                      <Timeline sx={{ mr: 1, verticalAlign: 'middle' }} />
                      手势序列详情
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      {result.recognition_result.gloss_sequence.map((gloss, index) => (
                        <Chip
                          key={index}
                          label={gloss}
                          variant="outlined"
                          size="small"
                          color="primary"
                        />
                      ))}
                    </Stack>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* 性能指标 */}
              <Accordion sx={{ mt: 1 }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle1">
                    <Speed sx={{ mr: 1, verticalAlign: 'middle' }} />
                    性能指标
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>指标</TableCell>
                        <TableCell align="right">值</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>推理时间</TableCell>
                        <TableCell align="right">
                          {((result.recognition_result?.inference_time || 0) * 1000).toFixed(1)}ms
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>总处理时间</TableCell>
                        <TableCell align="right">{(result?.processing_time ?? 0).toFixed(2)}s</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>关键点提取</TableCell>
                        <TableCell align="right">
                          {result.landmarks_extracted ? '成功' : '失败'}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>状态</TableCell>
                        <TableCell align="right">
                          <Chip 
                            size="small"
                            label={result.status}
                            color={result.status === 'completed' ? 'success' : 'warning'}
                          />
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        )}
      </Stack>
    </Paper>
  )
}

export default EnhancedVideoRecognition
