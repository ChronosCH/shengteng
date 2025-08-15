/**
 * 错误边界组件 - 捕获和处理React组件错误
 */

import React, { Component, ErrorInfo, ReactNode } from 'react'
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
} from '@mui/material'
import {
  ErrorOutline,
  Refresh,
} from '@mui/icons-material'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // 忽略MUI Fade组件的scrollTop错误
    if (error.message?.includes('scrollTop') || 
        error.message?.includes('Cannot read properties of null')) {
      console.warn('忽略MUI Fade组件的scrollTop错误:', error.message)
      this.setState({ hasError: false })
      return
    }

    console.error('ErrorBoundary caught an error:', error, errorInfo)
    this.setState({
      error,
      errorInfo,
    })
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '50vh',
            p: 2,
          }}
        >
          <Paper sx={{ p: 4, maxWidth: 600, textAlign: 'center' }}>
            <ErrorOutline sx={{ fontSize: 60, color: 'error.main', mb: 2 }} />
            
            <Typography variant="h5" gutterBottom>
              组件渲染出错
            </Typography>
            
            <Typography variant="body1" color="text.secondary" paragraph>
              抱歉，这个组件遇到了一个错误。请尝试刷新页面或重新加载组件。
            </Typography>

            <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
              <Typography variant="body2" component="div">
                <strong>错误信息：</strong> {this.state.error?.message}
              </Typography>
            </Alert>

            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="contained"
                startIcon={<Refresh />}
                onClick={this.handleRetry}
              >
                重试
              </Button>
              
              <Button
                variant="outlined"
                onClick={() => window.location.reload()}
              >
                刷新页面
              </Button>
            </Box>

            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <Box sx={{ mt: 3, textAlign: 'left' }}>
                <Typography variant="h6" gutterBottom>
                  开发信息：
                </Typography>
                <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                    {this.state.errorInfo.componentStack}
                  </Typography>
                </Paper>
              </Box>
            )}
          </Paper>
        </Box>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary