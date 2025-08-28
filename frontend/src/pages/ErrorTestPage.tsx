/**
 * 3D组件错误测试页面
 * 用于验证错误边界是否正常工作
 */

import React, { useState } from 'react'
import { Box, Button, Paper, Typography, Alert } from '@mui/material'
import ThreeDErrorBoundary from '../components/ThreeDErrorBoundary'
import AvatarViewer from '../components/AvatarViewer'

const ErrorTestPage: React.FC = () => {
  const [shouldError, setShouldError] = useState(false)
  const [testCase, setTestCase] = useState<'avatar' | 'webgl' | 'null'>('avatar')

  const renderTestComponent = () => {
    if (shouldError) {
      throw new Error('Intentional test error for error boundary')
    }

    switch (testCase) {
      case 'avatar':
        return <AvatarViewer text="测试手语" isActive={true} />
      case 'webgl':
        // 这个可能会触发WebGL错误
        return (
          <div style={{ width: 400, height: 300 }}>
            <canvas width={400} height={300} />
          </div>
        )
      case 'null':
        return null
      default:
        return <div>默认组件</div>
    }
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        3D组件错误边界测试
      </Typography>

      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          测试控制面板
        </Typography>
        
        <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button
            variant={testCase === 'avatar' ? 'contained' : 'outlined'}
            onClick={() => setTestCase('avatar')}
          >
            Avatar组件
          </Button>
          <Button
            variant={testCase === 'webgl' ? 'contained' : 'outlined'}
            onClick={() => setTestCase('webgl')}
          >
            WebGL Canvas
          </Button>
          <Button
            variant={testCase === 'null' ? 'contained' : 'outlined'}
            onClick={() => setTestCase('null')}
          >
            空组件
          </Button>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Button
            variant="contained"
            color="error"
            onClick={() => setShouldError(!shouldError)}
          >
            {shouldError ? '停止错误' : '触发错误'}
          </Button>
        </Box>

        {shouldError && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            组件将在下次渲染时抛出错误，用于测试错误边界
          </Alert>
        )}
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          3D组件显示区域
        </Typography>
        
        <ThreeDErrorBoundary>
          {renderTestComponent()}
        </ThreeDErrorBoundary>
      </Paper>
    </Box>
  )
}

export default ErrorTestPage
