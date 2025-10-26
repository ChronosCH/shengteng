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
    <Card 
      sx={{ 
        p: 3,
        borderRadius: 3,
        boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
      }}
    >
      <CardContent>
        <Stack spacing={3}>
          {/* 标题 */}
          <Box
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: 2,
              p: 3,
              color: 'white',
            }}
          >
            <Stack direction="row" alignItems="center" spacing={2}>
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: '50%',
                  bgcolor: 'rgba(255,255,255,0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px',
                }}
              >
                🎬
              </Box>
              <Box>
                <Typography variant="h5" gutterBottom fontWeight={700} sx={{ mb: 0.5 }}>
                  连续手语识别
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  使用 Mind-VAC CSLR 模型 + 通义千问大语言模型进行完整句子识别与翻译
                </Typography>
              </Box>
            </Stack>
          </Box>

          {/* 错误提示 */}
          {error && (
            <Alert severity="error" onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          {/* 文件选择 */}
          <Card 
            variant="outlined" 
            sx={{
              background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
              border: '2px dashed',
              borderColor: file ? 'success.main' : 'primary.main',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: file ? 'success.dark' : 'primary.dark',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                transform: 'translateY(-2px)',
              }
            }}
          >
            <CardContent>
              <Stack spacing={2}>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ xs: 'stretch', sm: 'center' }}>
                  <Button 
                    variant="contained" 
                    component="label" 
                    startIcon={<CloudUpload />}
                    disabled={isProcessing}
                    size="large"
                    sx={{
                      py: 1.5,
                      px: 3,
                      fontWeight: 600,
                      boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)',
                      '&:hover': {
                        boxShadow: '0 6px 16px rgba(102, 126, 234, 0.4)',
                      }
                    }}
                  >
                    📁 选择视频文件
                    <input 
                      hidden 
                      type="file" 
                      accept="video/*" 
                      onChange={handleFileChange} 
                    />
                  </Button>
                  
                  {file && (
                    <Box 
                      sx={{ 
                        flex: 1,
                        p: 2,
                        bgcolor: 'rgba(255,255,255,0.9)',
                        borderRadius: 2,
                        border: '1px solid',
                        borderColor: 'success.light',
                      }}
                    >
                      <Stack direction="row" spacing={2} alignItems="center">
                        <VideoFile color="success" sx={{ fontSize: 32 }} />
                        <Box flex={1}>
                          <Typography variant="body1" fontWeight={600} color="success.dark">
                            {file.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            大小: {formatFileSize(file.size)}
                          </Typography>
                        </Box>
                      </Stack>
                    </Box>
                  )}
                </Stack>

                {!file && (
                  <Box 
                    sx={{ 
                      textAlign: 'center',
                      py: 2,
                      color: 'text.secondary',
                    }}
                  >
                    <Typography variant="body2">
                      💡 支持 MP4, AVI, MOV 等常见视频格式，文件大小限制 100MB
                    </Typography>
                  </Box>
                )}
              </Stack>
            </CardContent>
          </Card>

          {/* 进度显示 */}
          {isProcessing && (
            <Card 
              sx={{ 
                p: 3,
                background: 'linear-gradient(135deg, #667eea22 0%, #764ba222 100%)',
                border: '2px solid',
                borderColor: 'primary.main',
              }}
            >
              <Stack spacing={2}>
                <Box display="flex" alignItems="center" gap={2}>
                  <Box 
                    sx={{ 
                      width: 40, 
                      height: 40, 
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      animation: 'pulse 2s ease-in-out infinite',
                      '@keyframes pulse': {
                        '0%, 100%': { transform: 'scale(1)', opacity: 1 },
                        '50%': { transform: 'scale(1.1)', opacity: 0.8 },
                      }
                    }}
                  >
                    <Typography variant="body2" color="white" fontWeight={700}>
                      {(progress * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                  <Box flex={1}>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                      正在处理中...
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {statusMessage}
                    </Typography>
                  </Box>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={progress * 100} 
                  sx={{ 
                    height: 12, 
                    borderRadius: 6,
                    bgcolor: 'rgba(0,0,0,0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 6,
                      background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                    }
                  }}
                />
              </Stack>
            </Card>
          )}

          {/* 控制按钮 */}
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <Button
              disabled={!file || isProcessing}
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={startRecognition}
              size="large"
              sx={{
                flex: 1,
                py: 1.5,
                fontWeight: 600,
                background: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #0d8071 0%, #2dd464 100%)',
                  boxShadow: '0 4px 12px rgba(17, 153, 142, 0.4)',
                },
                '&:disabled': {
                  background: 'grey.300',
                }
              }}
            >
              🚀 开始识别
            </Button>
            
            <Button
              disabled={!isProcessing}
              color="warning"
              variant="contained"
              startIcon={<Stop />}
              onClick={stopProcessing}
              size="large"
              sx={{
                py: 1.5,
                fontWeight: 600,
              }}
            >
              ⏸ 停止
            </Button>
            
            <Button
              disabled={isProcessing}
              variant="outlined"
              startIcon={<Refresh />}
              onClick={resetRecognition}
              size="large"
              sx={{
                py: 1.5,
                fontWeight: 600,
              }}
            >
              🔄 重置
            </Button>
          </Stack>

          {/* 识别结果 */}
          {result && (
            <Box>
              <Typography variant="h6" gutterBottom fontWeight={600} sx={{ mb: 3 }}>
                🎯 识别结果
              </Typography>

              <Grid container spacing={3}>
                {/* LLM增强翻译结果 - 主要展示 */}
                {result.llm_result?.success && (result.llm_result?.chinese || result.llm_result?.english) && (
                  <Grid item xs={12}>
                    <Paper 
                      elevation={3}
                      sx={{ 
                        p: 3,
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        color: 'white',
                        borderRadius: 2,
                      }}
                    >
                      <Stack spacing={2}>
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="h6" fontWeight={700}>
                            ✨ 通义千问大语言模型增强翻译
                          </Typography>
                          {result.llm_result?.confidence && (
                            <Chip 
                              label={`置信度: ${result.llm_result.confidence}`}
                              size="small"
                              sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                            />
                          )}
                        </Box>
                        
                        {result.llm_result?.chinese && (
                          <Box
                            sx={{
                              p: 2.5,
                              bgcolor: 'rgba(255, 255, 255, 0.95)',
                              borderRadius: 1.5,
                              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                            }}
                          >
                            <Typography variant="caption" sx={{ color: '#666', fontWeight: 600, mb: 0.5, display: 'block' }}>
                              🇨🇳 中文翻译
                            </Typography>
                            <Typography 
                              variant="h6"
                              sx={{
                                color: '#333',
                                fontWeight: 600,
                                lineHeight: 1.6,
                                wordBreak: 'break-word',
                              }}
                            >
                              {result.llm_result.chinese}
                            </Typography>
                          </Box>
                        )}

                        {result.llm_result?.english && (
                          <Box
                            sx={{
                              p: 2.5,
                              bgcolor: 'rgba(255, 255, 255, 0.95)',
                              borderRadius: 1.5,
                              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                            }}
                          >
                            <Typography variant="caption" sx={{ color: '#666', fontWeight: 600, mb: 0.5, display: 'block' }}>
                              🇺🇸 English Translation
                            </Typography>
                            <Typography 
                              variant="body1"
                              sx={{
                                color: '#333',
                                fontWeight: 500,
                                lineHeight: 1.6,
                                wordBreak: 'break-word',
                                fontStyle: 'italic',
                              }}
                            >
                              {result.llm_result.english}
                            </Typography>
                          </Box>
                        )}

                        {result.llm_result?.explanation && result.llm_result.explanation.trim() && (
                          <Box
                            sx={{
                              p: 2,
                              bgcolor: 'rgba(255, 255, 255, 0.15)',
                              borderRadius: 1,
                              border: '1px solid rgba(255,255,255,0.3)',
                            }}
                          >
                            <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.95)' }}>
                              💡 <strong>说明:</strong> {result.llm_result.explanation}
                            </Typography>
                          </Box>
                        )}
                      </Stack>
                    </Paper>
                  </Grid>
                )}

                {/* Mind-VAC原始识别结果 */}
                <Grid item xs={12} md={6}>
                  <Paper 
                    elevation={2}
                    sx={{ 
                      p: 2.5,
                      height: '100%',
                      borderRadius: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Typography variant="subtitle1" gutterBottom fontWeight={700} sx={{ color: 'primary.main', mb: 2 }}>
                      🤖 MIND-VAC 原始识别
                    </Typography>
                    <Stack spacing={2}>
                      {result.baseline_text && (
                        <Box
                          sx={{
                            p: 2,
                            bgcolor: 'background.default',
                            borderRadius: 1.5,
                            border: '2px solid',
                            borderColor: 'primary.light',
                          }}
                        >
                          <Typography variant="caption" color="text.secondary" fontWeight={600} display="block" sx={{ mb: 0.5 }}>
                            基础翻译
                          </Typography>
                          <Typography 
                            variant="body1" 
                            sx={{ 
                              wordBreak: 'break-word',
                              fontWeight: 500,
                              color: 'text.primary',
                            }}
                          >
                            {result.baseline_text}
                          </Typography>
                        </Box>
                      )}

                      {result.raw_gloss_text && (
                        <Box
                          sx={{
                            p: 2,
                            borderRadius: 1.5,
                            bgcolor: 'rgba(103, 58, 183, 0.08)',
                            border: '1px dashed',
                            borderColor: 'secondary.main',
                          }}
                        >
                          <Typography variant="caption" color="secondary.main" fontWeight={600} display="block" sx={{ mb: 0.5 }}>
                            📝 Gloss 序列
                          </Typography>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              wordBreak: 'break-word', 
                              fontFamily: 'monospace',
                              color: 'text.secondary',
                              fontSize: '0.85rem',
                            }}
                          >
                            {result.raw_gloss_text}
                          </Typography>
                        </Box>
                      )}

                      <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
                        <Chip 
                          label={`置信度 ${((result.overall_confidence ?? 0) * 100).toFixed(1)}%`}
                          color="primary"
                          size="small"
                          sx={{ fontWeight: 600 }}
                        />
                        <Chip 
                          label={result.pipeline?.toUpperCase() || 'UNKNOWN'}
                          variant="outlined"
                          size="small"
                        />
                      </Stack>
                    </Stack>
                  </Paper>
                </Grid>
                
                {/* 处理信息 */}
                <Grid item xs={12} md={6}>
                  <Paper 
                    elevation={2}
                    sx={{ 
                      p: 2.5,
                      height: '100%',
                      borderRadius: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Typography variant="subtitle1" gutterBottom fontWeight={700} sx={{ color: 'success.main', mb: 2 }}>
                      📊 处理信息
                    </Typography>
                    <Stack spacing={1.5}>
                      <Box 
                        display="flex" 
                        justifyContent="space-between" 
                        alignItems="center"
                        sx={{
                          p: 1.5,
                          bgcolor: 'background.default',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="body2" color="text.secondary">识别管线</Typography>
                        <Chip 
                          label={result.pipeline?.toUpperCase() || '未知'}
                          size="small"
                          color="info"
                          sx={{ fontWeight: 600 }}
                        />
                      </Box>
                      <Box 
                        display="flex" 
                        justifyContent="space-between"
                        sx={{
                          p: 1.5,
                          bgcolor: 'background.default',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="body2" color="text.secondary">视频时长</Typography>
                        <Typography variant="body2" fontWeight={600}>{formatDuration(result.duration)}</Typography>
                      </Box>
                      <Box 
                        display="flex" 
                        justifyContent="space-between"
                        sx={{
                          p: 1.5,
                          bgcolor: 'background.default',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="body2" color="text.secondary">总帧数</Typography>
                        <Typography variant="body2" fontWeight={600}>{result.frame_count ?? '—'}</Typography>
                      </Box>
                      <Box 
                        display="flex" 
                        justifyContent="space-between"
                        sx={{
                          p: 1.5,
                          bgcolor: 'background.default',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="body2" color="text.secondary">视频帧率</Typography>
                        <Typography variant="body2" fontWeight={600}>{(result.fps ?? 0).toFixed(1)} fps</Typography>
                      </Box>
                    </Stack>
                  </Paper>
                </Grid>
              </Grid>

              {/* Gloss序列详情 */}
              {result.gloss_sequence && result.gloss_sequence.length > 0 && (
                <Accordion 
                  sx={{ 
                    mt: 3,
                    borderRadius: 2,
                    '&:before': { display: 'none' },
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  }}
                >
                  <AccordionSummary 
                    expandIcon={<ExpandMore />}
                    sx={{
                      bgcolor: 'background.default',
                      borderRadius: '8px 8px 0 0',
                    }}
                  >
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="subtitle1" fontWeight={600}>
                        📋 Gloss 词汇序列
                      </Typography>
                      <Chip 
                        label={`${result.gloss_sequence.length} 个词汇`}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails sx={{ p: 2.5 }}>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
                      {result.gloss_sequence.map((gloss, index) => (
                        <Chip 
                          key={index} 
                          label={`${index + 1}. ${gloss}`}
                          color="secondary"
                          variant="outlined" 
                          size="medium"
                          sx={{
                            fontWeight: 500,
                            fontSize: '0.9rem',
                            '&:hover': {
                              bgcolor: 'secondary.light',
                              color: 'white',
                            }
                          }}
                        />
                      ))}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* 分段详情 */}
              {result.segments && result.segments.length > 0 && (
                <Accordion 
                  sx={{ 
                    mt: 2,
                    borderRadius: 2,
                    '&:before': { display: 'none' },
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  }}
                >
                  <AccordionSummary 
                    expandIcon={<ExpandMore />}
                    sx={{
                      bgcolor: 'background.default',
                      borderRadius: '8px 8px 0 0',
                    }}
                  >
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="subtitle1" fontWeight={600}>
                        🎬 时序分段详情
                      </Typography>
                      <Chip 
                        label={`${result.segments.length} 个分段`}
                        size="small"
                        color="info"
                        variant="outlined"
                      />
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails sx={{ p: 0 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow sx={{ bgcolor: 'background.default' }}>
                          <TableCell sx={{ fontWeight: 700 }}>分段</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>Gloss序列</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>时间范围</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>置信度</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.segments.map((segment, index) => (
                          <TableRow 
                            key={index}
                            sx={{
                              '&:nth-of-type(odd)': {
                                bgcolor: 'background.default',
                              },
                              '&:hover': {
                                bgcolor: 'action.hover',
                              }
                            }}
                          >
                            <TableCell>
                              <Chip 
                                label={`#${index + 1}`}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                            </TableCell>
                            <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                              {segment.gloss_sequence.join(' ')}
                            </TableCell>
                            <TableCell>
                              {segment.start_time.toFixed(1)}s - {segment.end_time.toFixed(1)}s
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={`${(segment.confidence * 100).toFixed(1)}%`}
                                size="small"
                                color={segment.confidence > 0.8 ? "success" : segment.confidence > 0.6 ? "warning" : "error"}
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* LLM原始响应 - 可选展开 */}
              {result.llm_result?.raw_response && (
                <Accordion 
                  sx={{ 
                    mt: 2,
                    borderRadius: 2,
                    '&:before': { display: 'none' },
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  }}
                >
                  <AccordionSummary 
                    expandIcon={<ExpandMore />}
                    sx={{
                      bgcolor: 'background.default',
                      borderRadius: '8px 8px 0 0',
                    }}
                  >
                    <Typography variant="subtitle2" fontWeight={600} color="text.secondary">
                      🔍 LLM 原始响应（调试信息）
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ p: 2.5 }}>
                    <Box
                      sx={{
                        p: 2,
                        bgcolor: 'grey.100',
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'grey.300',
                        maxHeight: '300px',
                        overflow: 'auto',
                      }}
                    >
                      <Typography 
                        variant="caption" 
                        component="pre"
                        sx={{ 
                          whiteSpace: 'pre-wrap', 
                          wordBreak: 'break-word', 
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          lineHeight: 1.6,
                          margin: 0,
                        }}
                      >
                        {result.llm_result.raw_response}
                      </Typography>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* 错误信息显示 */}
              {result.llm_result?.error && !result.llm_result?.success && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    <strong>LLM 增强失败:</strong> {result.llm_result.error}
                  </Typography>
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    已回退到基础识别结果
                  </Typography>
                </Alert>
              )}
            </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

export default ContinuousVideoRecognition
