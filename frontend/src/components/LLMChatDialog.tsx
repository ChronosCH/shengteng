import React, { useState, useRef, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Stack,
  Paper,
  IconButton,
  Fade,
  CircularProgress,
  Chip,
  Divider,
} from '@mui/material'
import {
  Send,
  Close,
  SmartToy,
  Person,
  Clear,
  ContentCopy,
} from '@mui/icons-material'
import llmChatService, { ChatMessage, ChatContext } from '../services/llmChatService'

interface Props {
  open: boolean
  onClose: () => void
  context?: ChatContext
}

const LLMChatDialog: React.FC<Props> = ({ open, onClose, context }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // 初始化欢迎消息
  useEffect(() => {
    if (open && messages.length === 0) {
      const welcomeMessage: ChatMessage = {
        role: 'assistant',
        content: '你好！我是通义千问AI助手。我已经了解了刚才的手语识别结果，你可以问我关于识别内容的任何问题，比如：\n\n• 这段手语表达的完整意思是什么？\n• 能否详细解释某个词汇的含义？\n• 能否用更正式/口语化的方式重新表达？\n• 有没有更好的翻译方式？',
        timestamp: Date.now(),
      }
      setMessages([welcomeMessage])
    }
  }, [open])

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 发送消息
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: Date.now(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const response = await llmChatService.sendMessage({
        message: inputMessage,
        context,
        history: messages,
      })

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.success ? response.message : `抱歉，我遇到了一些问题：${response.error}`,
        timestamp: Date.now(),
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: '抱歉，发送消息时出现错误，请稍后重试。',
        timestamp: Date.now(),
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  // 清空对话
  const handleClearChat = () => {
    setMessages([])
    setInputMessage('')
  }

  // 复制消息
  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content)
  }

  // 快捷问题
  const quickQuestions = [
    '请详细解释这段手语的含义',
    '能否用更正式的语言重新表达？',
    '这个句子的语法结构是怎样的？',
    '有没有其他可能的翻译方式？',
  ]

  if (!open) return null

  return (
    <Fade in={open}>
      <Box
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          width: { xs: 'calc(100% - 48px)', sm: 400, md: 500 },
          maxHeight: '70vh',
          zIndex: 1300,
        }}
      >
        <Card
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 12px 40px rgba(0,0,0,0.3)',
            borderRadius: 3,
            overflow: 'hidden',
          }}
        >
          {/* 头部 */}
          <Box
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              p: 2,
            }}
          >
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Stack direction="row" alignItems="center" spacing={1.5}>
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: '50%',
                    bgcolor: 'rgba(255,255,255,0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <SmartToy />
                </Box>
                <Box>
                  <Typography variant="subtitle1" fontWeight={700}>
                    通义千问 AI 助手
                  </Typography>
                  <Typography variant="caption" sx={{ opacity: 0.9 }}>
                    在线 · 已读取识别结果
                  </Typography>
                </Box>
              </Stack>
              <Stack direction="row" spacing={0.5}>
                <IconButton size="small" onClick={handleClearChat} sx={{ color: 'white' }}>
                  <Clear />
                </IconButton>
                <IconButton size="small" onClick={onClose} sx={{ color: 'white' }}>
                  <Close />
                </IconButton>
              </Stack>
            </Stack>
          </Box>

          {/* 上下文信息 */}
          {context && (
            <Box
              sx={{
                bgcolor: 'info.lighter',
                p: 1.5,
                borderBottom: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Stack spacing={0.5}>
                <Typography variant="caption" color="text.secondary" fontWeight={600}>
                  📋 识别上下文:
                </Typography>
                {context.recognitionResult && (
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    {context.recognitionResult.length > 60
                      ? context.recognitionResult.slice(0, 60) + '...'
                      : context.recognitionResult}
                  </Typography>
                )}
              </Stack>
            </Box>
          )}

          {/* 消息列表 */}
          <CardContent
            sx={{
              flex: 1,
              overflowY: 'auto',
              p: 2,
              bgcolor: 'background.default',
            }}
          >
            <Stack spacing={2}>
              {messages.map((msg, index) => (
                <Box
                  key={index}
                  sx={{
                    display: 'flex',
                    justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  }}
                >
                  <Paper
                    elevation={1}
                    sx={{
                      maxWidth: '80%',
                      p: 1.5,
                      bgcolor: msg.role === 'user' ? 'primary.main' : 'background.paper',
                      color: msg.role === 'user' ? 'white' : 'text.primary',
                      borderRadius: 2,
                      position: 'relative',
                      '&:hover .copy-btn': {
                        opacity: 1,
                      },
                    }}
                  >
                    <Stack direction="row" spacing={1} alignItems="flex-start">
                      <Box
                        sx={{
                          width: 24,
                          height: 24,
                          borderRadius: '50%',
                          bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'primary.light',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                        }}
                      >
                        {msg.role === 'user' ? (
                          <Person sx={{ fontSize: 16, color: 'white' }} />
                        ) : (
                          <SmartToy sx={{ fontSize: 16, color: 'white' }} />
                        )}
                      </Box>
                      <Box flex={1}>
                        <Typography
                          variant="body2"
                          sx={{
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            lineHeight: 1.6,
                          }}
                        >
                          {msg.content}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            opacity: 0.7,
                            display: 'block',
                            mt: 0.5,
                          }}
                        >
                          {new Date(msg.timestamp).toLocaleTimeString('zh-CN', {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </Typography>
                      </Box>
                      <IconButton
                        size="small"
                        className="copy-btn"
                        onClick={() => handleCopyMessage(msg.content)}
                        sx={{
                          opacity: 0,
                          transition: 'opacity 0.2s',
                          color: msg.role === 'user' ? 'white' : 'text.secondary',
                        }}
                      >
                        <ContentCopy fontSize="small" />
                      </IconButton>
                    </Stack>
                  </Paper>
                </Box>
              ))}

              {isLoading && (
                <Box display="flex" justifyContent="flex-start">
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      bgcolor: 'background.paper',
                      borderRadius: 2,
                    }}
                  >
                    <Stack direction="row" spacing={1} alignItems="center">
                      <CircularProgress size={20} />
                      <Typography variant="body2" color="text.secondary">
                        AI 正在思考中...
                      </Typography>
                    </Stack>
                  </Paper>
                </Box>
              )}

              <div ref={messagesEndRef} />
            </Stack>
          </CardContent>

          {/* 快捷问题 */}
          {messages.length <= 1 && (
            <Box sx={{ px: 2, pb: 1 }}>
              <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                💡 快捷问题:
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
                {quickQuestions.map((question, index) => (
                  <Chip
                    key={index}
                    label={question}
                    size="small"
                    onClick={() => setInputMessage(question)}
                    sx={{
                      cursor: 'pointer',
                      '&:hover': {
                        bgcolor: 'primary.light',
                        color: 'white',
                      },
                    }}
                  />
                ))}
              </Stack>
            </Box>
          )}

          <Divider />

          {/* 输入区域 */}
          <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
            <Stack direction="row" spacing={1}>
              <TextField
                fullWidth
                multiline
                maxRows={3}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
                placeholder="输入你的问题... (Shift+Enter 换行)"
                disabled={isLoading}
                size="small"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 2,
                  },
                }}
              />
              <Button
                variant="contained"
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                sx={{
                  minWidth: 48,
                  height: 48,
                  borderRadius: 2,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                }}
              >
                <Send />
              </Button>
            </Stack>
          </Box>
        </Card>
      </Box>
    </Fade>
  )
}

export default LLMChatDialog
