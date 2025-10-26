/**
 * AI手语教学助手对话组件
 */
import React, { useState, useRef, useEffect } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  Avatar,
  CircularProgress,
  IconButton,
  Chip,
  Divider,
  Alert,
} from '@mui/material'
import {
  Send,
  Close,
  SmartToy,
  Person,
  VideoLibrary,
  Refresh,
  ContentCopy,
  Check,
} from '@mui/icons-material'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

interface AITutorChatProps {
  open: boolean
  onClose: () => void
  recognitionContext?: {
    recognized_sign?: string
    confidence?: number
  }
}

const AITutorChat: React.FC<AITutorChatProps> = ({ open, onClose, recognitionContext }) => {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // 初始化欢迎消息
  useEffect(() => {
    if (open && messages.length === 0) {
      const welcomeMessage: Message = {
        role: 'assistant',
        content: recognitionContext?.recognized_sign
          ? `你好！我注意到你刚刚练习了"${recognitionContext.recognized_sign}"手语。有什么问题我可以帮你吗？`
          : '你好！我是你的AI手语教学助手。我可以帮你：\n\n1. 解答手语学习问题\n2. 推荐学习视频和资源\n3. 解释手语动作要领\n4. 制定学习计划\n\n有什么想问的吗？',
        timestamp: new Date().toISOString(),
      }
      setMessages([welcomeMessage])
    }
  }, [open, recognitionContext])

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || loading) return

    const userMessage: Message = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setLoading(true)
    setError('')

    try {
      const token = localStorage.getItem('access_token')
      const response = await fetch('/api/learning/ai-tutor/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: inputMessage,
          context: recognitionContext,
          history: messages,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'AI响应失败')
      }

      const data = await response.json()
      
      if (data.success) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: data.message,
          timestamp: data.timestamp,
        }
        setMessages(prev => [...prev, assistantMessage])
      } else {
        throw new Error('AI响应失败')
      }
    } catch (err: any) {
      console.error('AI对话失败:', err)
      setError(err.message || '抱歉，AI助手暂时不可用')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleQuickQuestion = (question: string) => {
    setInputMessage(question)
  }

  const quickQuestions = [
    '我该怎么开始学手语？',
    '推荐一些学习视频',
    '如何提高手语表达的准确性？',
    '有什么好的练习方法？',
  ]

  // 复制消息内容
  const handleCopyMessage = (content: string, index: number) => {
    navigator.clipboard.writeText(content)
    setCopiedIndex(index)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  // 重新生成回答
  const handleRegenerate = () => {
    if (messages.length > 0) {
      const lastUserMessage = [...messages].reverse().find(m => m.role === 'user')
      if (lastUserMessage) {
        setInputMessage(lastUserMessage.content)
        handleSendMessage()
      }
    }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          height: '80vh',
          display: 'flex',
          flexDirection: 'column',
        },
      }}
    >
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          py: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Avatar 
            sx={{ 
              bgcolor: 'rgba(255,255,255,0.2)', 
              width: 44, 
              height: 44,
              backdropFilter: 'blur(10px)',
            }}
          >
            <SmartToy sx={{ fontSize: 24 }} />
          </Avatar>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600, lineHeight: 1.2 }}>
              AI手语教学助手
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.9 }}>
              24/7 在线 · 专业耐心 · 即时响应
            </Typography>
          </Box>
        </Box>
        <IconButton 
          onClick={onClose} 
          sx={{ 
            color: 'white',
            '&:hover': {
              bgcolor: 'rgba(255, 255, 255, 0.1)',
            },
          }}
        >
          <Close />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 0 }}>
        {/* 识别上下文显示 */}
        {recognitionContext?.recognized_sign && (
          <Alert
            severity="info"
            sx={{ 
              m: 2, 
              mb: 1,
              borderRadius: 2,
              background: 'linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%)',
              border: '1px solid #90CAF9',
              '& .MuiAlert-icon': {
                color: '#1976D2',
              },
            }}
            icon={<VideoLibrary />}
          >
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              刚刚识别：<strong>{recognitionContext.recognized_sign}</strong> 
              <Chip 
                label={`准确率 ${(recognitionContext.confidence! * 100).toFixed(1)}%`}
                size="small"
                sx={{ 
                  ml: 1,
                  height: 20,
                  bgcolor: 'rgba(25, 118, 210, 0.1)',
                  color: '#1976D2',
                  fontWeight: 600,
                }}
              />
            </Typography>
          </Alert>
        )}

        {/* 快捷问题 */}
        {messages.length <= 1 && (
          <Box sx={{ p: 2, pt: 1, bgcolor: 'white' }}>
            <Typography 
              variant="caption" 
              color="text.secondary" 
              sx={{ 
                mb: 1.5, 
                display: 'block',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: 0.5,
              }}
            >
              💬 快速开始
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {quickQuestions.map((question, index) => (
                <Chip
                  key={index}
                  label={question}
                  onClick={() => handleQuickQuestion(question)}
                  size="medium"
                  sx={{ 
                    cursor: 'pointer',
                    borderRadius: 2,
                    border: '1px solid #E0E0E0',
                    bgcolor: 'white',
                    '&:hover': {
                      bgcolor: '#F5F5F5',
                      borderColor: '#7fcdbb',
                      color: '#5fb89c',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 4px 8px rgba(127, 205, 187, 0.15)',
                    },
                    transition: 'all 0.2s ease',
                  }}
                />
              ))}
            </Box>
            <Divider sx={{ mt: 2 }} />
          </Box>
        )}

        {/* 消息列表 */}
        <Box
          sx={{
            flex: 1,
            overflowY: 'auto',
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            bgcolor: '#F8F9FA',
          }}
        >
          {messages.map((message, index) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                gap: 1,
                animation: 'slideIn 0.3s ease-out',
                '@keyframes slideIn': {
                  from: {
                    opacity: 0,
                    transform: 'translateY(10px)',
                  },
                  to: {
                    opacity: 1,
                    transform: 'translateY(0)',
                  },
                },
              }}
            >
              {message.role === 'assistant' && (
                <Avatar 
                  sx={{ 
                    bgcolor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    width: 36, 
                    height: 36,
                    boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)',
                  }}
                >
                  <SmartToy sx={{ fontSize: 20 }} />
                </Avatar>
              )}

              <Box sx={{ maxWidth: '75%', display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                <Paper
                  sx={{
                    p: 2.5,
                    background: message.role === 'user' 
                      ? 'linear-gradient(135deg, #a8e6cf 0%, #7fcdbb 100%)' 
                      : 'white',
                    color: message.role === 'user' ? 'white' : 'text.primary',
                    borderRadius: message.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                    boxShadow: message.role === 'user'
                      ? '0 4px 12px rgba(127, 205, 187, 0.3)'
                      : '0 2px 8px rgba(0, 0, 0, 0.08)',
                    position: 'relative',
                    '&:hover .message-actions': {
                      opacity: 1,
                    },
                  }}
                  elevation={0}
                >
                  {message.role === 'assistant' ? (
                    <Box
                      sx={{
                        '& p': { margin: '0.5em 0', lineHeight: 1.6 },
                        '& p:first-of-type': { marginTop: 0 },
                        '& p:last-of-type': { marginBottom: 0 },
                        '& ul, & ol': { 
                          margin: '0.5em 0', 
                          paddingLeft: '1.5em',
                        },
                        '& li': { margin: '0.3em 0' },
                        '& code': {
                          bgcolor: '#F5F5F5',
                          color: '#E91E63',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontSize: '0.9em',
                          fontFamily: 'monospace',
                        },
                        '& pre': {
                          bgcolor: '#2D2D2D',
                          color: '#F8F8F2',
                          padding: '12px',
                          borderRadius: '8px',
                          overflow: 'auto',
                          margin: '0.5em 0',
                        },
                        '& pre code': {
                          bgcolor: 'transparent',
                          color: 'inherit',
                          padding: 0,
                        },
                        '& a': {
                          color: '#667eea',
                          textDecoration: 'none',
                          fontWeight: 500,
                          '&:hover': {
                            textDecoration: 'underline',
                          },
                        },
                        '& blockquote': {
                          borderLeft: '4px solid #667eea',
                          paddingLeft: '12px',
                          margin: '0.5em 0',
                          color: 'text.secondary',
                          fontStyle: 'italic',
                        },
                        '& h1, & h2, & h3, & h4, & h5, & h6': {
                          margin: '0.8em 0 0.4em 0',
                          fontWeight: 600,
                        },
                        '& table': {
                          borderCollapse: 'collapse',
                          width: '100%',
                          margin: '0.5em 0',
                        },
                        '& th, & td': {
                          border: '1px solid #ddd',
                          padding: '8px',
                          textAlign: 'left',
                        },
                        '& th': {
                          bgcolor: '#F5F5F5',
                          fontWeight: 600,
                        },
                        '& img': {
                          maxWidth: '100%',
                          borderRadius: '8px',
                          margin: '0.5em 0',
                        },
                      }}
                    >
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </Box>
                  ) : (
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                      {message.content}
                    </Typography>
                  )}
                  
                  {/* 消息操作按钮 */}
                  <Box
                    className="message-actions"
                    sx={{
                      position: 'absolute',
                      top: 8,
                      right: 8,
                      opacity: 0,
                      transition: 'opacity 0.2s',
                      display: 'flex',
                      gap: 0.5,
                    }}
                  >
                    <IconButton
                      size="small"
                      onClick={() => handleCopyMessage(message.content, index)}
                      sx={{
                        bgcolor: 'rgba(255, 255, 255, 0.9)',
                        '&:hover': { bgcolor: 'rgba(255, 255, 255, 1)' },
                        width: 28,
                        height: 28,
                      }}
                    >
                      {copiedIndex === index ? (
                        <Check sx={{ fontSize: 16, color: 'success.main' }} />
                      ) : (
                        <ContentCopy sx={{ fontSize: 16 }} />
                      )}
                    </IconButton>
                  </Box>
                </Paper>

                {/* 时间戳 */}
                <Typography 
                  variant="caption" 
                  color="text.secondary"
                  sx={{ 
                    px: 1,
                    alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                  }}
                >
                  {new Date(message.timestamp).toLocaleTimeString('zh-CN', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </Typography>
              </Box>

              {message.role === 'user' && (
                <Avatar 
                  sx={{ 
                    background: 'linear-gradient(135deg, #a8e6cf 0%, #7fcdbb 100%)',
                    width: 36, 
                    height: 36,
                    boxShadow: '0 2px 8px rgba(127, 205, 187, 0.3)',
                  }}
                >
                  <Person sx={{ fontSize: 20 }} />
                </Avatar>
              )}
            </Box>
          ))}

          {loading && (
            <Box 
              sx={{ 
                display: 'flex', 
                alignItems: 'flex-start', 
                gap: 1,
                animation: 'slideIn 0.3s ease-out',
              }}
            >
              <Avatar sx={{ 
                bgcolor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                width: 36, 
                height: 36 
              }}>
                <SmartToy sx={{ fontSize: 20 }} />
              </Avatar>
              <Paper sx={{ 
                p: 2, 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1.5,
                borderRadius: '18px 18px 18px 4px',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
              }}>
                <CircularProgress size={18} thickness={4} />
                <Typography variant="body2" color="text.secondary">
                  AI正在思考回答...
                </Typography>
              </Paper>
            </Box>
          )}

          {error && (
            <Alert 
              severity="error" 
              onClose={() => setError('')}
              sx={{ borderRadius: 2 }}
            >
              {error}
            </Alert>
          )}

          <div ref={messagesEndRef} />
        </Box>

        {/* 输入框 */}
        <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider', bgcolor: 'white' }}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              placeholder="输入你的问题... (Shift + Enter 换行，Enter 发送)"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 3,
                  bgcolor: '#F8F9FA',
                  '&:hover': {
                    bgcolor: '#F0F1F3',
                  },
                  '&.Mui-focused': {
                    bgcolor: 'white',
                  },
                },
              }}
            />
            <Button
              variant="contained"
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || loading}
              sx={{
                minWidth: 56,
                height: 56,
                borderRadius: 3,
                background: loading 
                  ? 'linear-gradient(135deg, #ccc 0%, #aaa 100%)'
                  : 'linear-gradient(135deg, #a8e6cf 0%, #7fcdbb 100%)',
                boxShadow: '0 4px 12px rgba(127, 205, 187, 0.3)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #8dd9bb 0%, #6bbaa5 100%)',
                  boxShadow: '0 6px 16px rgba(127, 205, 187, 0.4)',
                  transform: 'translateY(-2px)',
                },
                '&:disabled': {
                  background: 'linear-gradient(135deg, #ccc 0%, #aaa 100%)',
                },
                transition: 'all 0.3s ease',
              }}
            >
              <Send />
            </Button>
          </Box>
          
          {/* 提示信息 */}
          <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              💡 支持 Markdown 格式回复
            </Typography>
            {messages.length > 1 && (
              <Button
                size="small"
                startIcon={<Refresh />}
                onClick={handleRegenerate}
                disabled={loading}
                sx={{ 
                  textTransform: 'none',
                  color: 'text.secondary',
                  '&:hover': {
                    bgcolor: 'rgba(127, 205, 187, 0.08)',
                  },
                }}
              >
                重新生成
              </Button>
            )}
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  )
}

export default AITutorChat
