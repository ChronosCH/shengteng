import React from 'react'
import { Box, Paper, Typography, Avatar } from '@mui/material'
import PersonIcon from '@mui/icons-material/Person'

interface SimpleAvatarFallbackProps {
  currentGesture?: string
  isRecording?: boolean
  width?: number
  height?: number
}

const SimpleAvatarFallback: React.FC<SimpleAvatarFallbackProps> = ({
  currentGesture = '准备中',
  isRecording = false,
  width = 400,
  height = 300
}) => {
  return (
    <Paper
      elevation={3}
      sx={{
        width,
        height,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#f5f5f5',
        border: '2px solid #e0e0e0',
        borderRadius: 2,
        position: 'relative'
      }}
    >
      {/* 状态指示器 */}
      <Box
        sx={{
          position: 'absolute',
          top: 8,
          right: 8,
          width: 12,
          height: 12,
          borderRadius: '50%',
          backgroundColor: isRecording ? '#4caf50' : '#ff9800',
          animation: isRecording ? 'pulse 1s infinite' : 'none',
          '@keyframes pulse': {
            '0%': { opacity: 1 },
            '50%': { opacity: 0.5 },
            '100%': { opacity: 1 }
          }
        }}
      />

      {/* 头像图标 */}
      <Avatar
        sx={{
          width: 80,
          height: 80,
          backgroundColor: '#2196f3',
          mb: 2
        }}
      >
        <PersonIcon sx={{ fontSize: 48 }} />
      </Avatar>

      {/* 当前手势文本 */}
      <Typography variant="h6" color="text.primary" gutterBottom>
        手语演示
      </Typography>

      <Typography 
        variant="body1" 
        color="text.secondary"
        sx={{ 
          textAlign: 'center',
          px: 2,
          mb: 1
        }}
      >
        当前手势: {currentGesture}
      </Typography>

      {/* 状态说明 */}
      <Typography 
        variant="caption" 
        color="text.secondary"
        sx={{ textAlign: 'center', px: 2 }}
      >
        {isRecording ? '正在录制...' : '等待中'}
      </Typography>

      {/* 装饰性手势图标 */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: 1,
          opacity: 0.3
        }}
      >
        <Box sx={{ fontSize: 24 }}>👋</Box>
        <Box sx={{ fontSize: 24 }}>🤝</Box>
        <Box sx={{ fontSize: 24 }}>👍</Box>
      </Box>
    </Paper>
  )
}

export default SimpleAvatarFallback
