/**
 * 3D组件错误边界
 * 专门处理Three.js和React Three Fiber相关的错误
 */

import React, { Component, ReactNode } from 'react'
import {
  Box,
  Typography,
  Button,
  Alert,
  AlertTitle,
  Paper,
} from '@mui/material'
import {
  Error as ErrorIcon,
  Refresh,
  Warning,
} from '@mui/icons-material'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: any
}

class ThreeDErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('3D组件错误:', error, errorInfo)
    this.setState({ error, errorInfo })
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <Paper
          sx={{
            p: 3,
            textAlign: 'center',
            bgcolor: 'grey.50',
            border: '1px solid',
            borderColor: 'warning.light',
            borderRadius: 2,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
            <AlertTitle>3D模型加载失败</AlertTitle>
            3D虚拟人组件遇到技术问题，已启用简化模式
          </Alert>

          <Box sx={{ mb: 3 }}>
            <Warning sx={{ fontSize: 80, color: 'warning.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              虚拟人暂时不可用
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              可能的原因：
            </Typography>
            <Typography variant="body2" color="text.secondary" component="ul" sx={{ textAlign: 'left', maxWidth: 400 }}>
              <li>3D模型文件缺失</li>
              <li>WebGL支持问题</li>
              <li>浏览器兼容性</li>
              <li>网络连接异常</li>
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', gap: 2, flexDirection: 'column', alignItems: 'center' }}>
            <Button
              variant="contained"
              startIcon={<Refresh />}
              onClick={this.handleRetry}
              size="large"
            >
              重新尝试
            </Button>
            
            <Typography variant="caption" color="text.secondary">
              或者继续使用其他学习功能
            </Typography>
          </Box>

          {/* 开发环境下显示详细错误信息 */}
          {import.meta.env.DEV && this.state.error && (
            <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.100', borderRadius: 1, width: '100%', overflow: 'auto' }}>
              <Typography variant="caption" component="pre" sx={{ fontSize: '0.7rem', wordBreak: 'break-word' }}>
                {this.state.error.message}
                {this.state.error.stack && (
                  <>
                    {'\n\n'}
                    {this.state.error.stack}
                  </>
                )}
              </Typography>
            </Box>
          )}
        </Paper>
      )
    }

    return this.props.children
  }
}

// 简化的替代Avatar组件
export const SimpleAvatarFallback: React.FC<{
  text?: string
  isActive?: boolean
}> = ({ text, isActive }) => {
  return (
    <Paper
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        bgcolor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* 背景动画 */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          opacity: isActive ? 0.9 : 0.7,
          transition: 'opacity 0.3s ease',
        }}
      />
      
      {/* 虚拟人图标 */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          transform: isActive ? 'scale(1.1)' : 'scale(1)',
          transition: 'transform 0.3s ease',
        }}
      >
        <Box
          sx={{
            width: 120,
            height: 120,
            borderRadius: '50%',
            bgcolor: 'rgba(255, 255, 255, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
            fontSize: '4rem',
            animation: isActive ? 'pulse 2s infinite' : 'none',
          }}
        >
          👤
        </Box>
        
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
          手语虚拟人
        </Typography>
        
        {text && (
          <Typography variant="body1" sx={{ 
            bgcolor: 'rgba(255, 255, 255, 0.2)', 
            px: 2, 
            py: 1, 
            borderRadius: 1,
            maxWidth: 200,
            textAlign: 'center',
          }}>
            {text}
          </Typography>
        )}
        
        <Typography variant="body2" sx={{ mt: 2, opacity: 0.8 }}>
          简化模式
        </Typography>
      </Box>

      {/* CSS动画 */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
          }
        `}
      </style>
    </Paper>
  )
}

export default ThreeDErrorBoundary
