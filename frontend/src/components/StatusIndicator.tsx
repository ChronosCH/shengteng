/**
 * 状态指示器组件 - 显示连接状态和识别状态
 */

import React from 'react'
import {
  Box,
  Chip,
  Tooltip,
  Typography,
} from '@mui/material'
import {
  Wifi,
  WifiOff,
  RadioButtonChecked,
  RadioButtonUnchecked,
  SignalCellularAlt,
} from '@mui/icons-material'

interface StatusIndicatorProps {
  isConnected: boolean
  isRecognizing: boolean
  confidence: number
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  isConnected,
  isRecognizing,
  confidence,
}) => {
  // 获取连接状态颜色和图标
  const getConnectionStatus = () => {
    if (isConnected) {
      return {
        color: 'success' as const,
        icon: <Wifi fontSize="small" />,
        label: '已连接',
      }
    } else {
      return {
        color: 'error' as const,
        icon: <WifiOff fontSize="small" />,
        label: '未连接',
      }
    }
  }

  // 获取识别状态颜色和图标
  const getRecognitionStatus = () => {
    if (isRecognizing) {
      return {
        color: 'primary' as const,
        icon: <RadioButtonChecked fontSize="small" />,
        label: '识别中',
      }
    } else {
      return {
        color: 'default' as const,
        icon: <RadioButtonUnchecked fontSize="small" />,
        label: '待机',
      }
    }
  }

  // 获取置信度状态 - 简化颜色处理
  const getConfidenceStatus = () => {
    if (confidence >= 0.8) {
      return { color: '#4caf50', level: '高' } // 直接使用颜色值
    } else if (confidence >= 0.6) {
      return { color: '#ff9800', level: '中' }
    } else if (confidence > 0) {
      return { color: '#f44336', level: '低' }
    } else {
      return { color: '#9e9e9e', level: '-' }
    }
  }

  const connectionStatus = getConnectionStatus()
  const recognitionStatus = getRecognitionStatus()
  const confidenceStatus = getConfidenceStatus()

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      {/* 连接状态 */}
      <Tooltip title={`WebSocket连接: ${connectionStatus.label}`}>
        <Chip
          icon={connectionStatus.icon}
          label={connectionStatus.label}
          color={connectionStatus.color}
          size="small"
          variant="outlined"
        />
      </Tooltip>

      {/* 识别状态 */}
      <Tooltip title={`识别状态: ${recognitionStatus.label}`}>
        <Chip
          icon={recognitionStatus.icon}
          label={recognitionStatus.label}
          color={recognitionStatus.color}
          size="small"
          variant={isRecognizing ? "filled" : "outlined"}
        />
      </Tooltip>

      {/* 置信度指示器 - 简化实现 */}
      {isRecognizing && confidence > 0 && (
        <Tooltip 
          title={`识别置信度: ${confidenceStatus.level} (${(confidence * 100).toFixed(1)}%)`}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <SignalCellularAlt
              fontSize="small"
              sx={{ color: confidenceStatus.color }}
            />
            <Typography 
              variant="caption" 
              sx={{ color: confidenceStatus.color, minWidth: 20 }}
            >
              {confidenceStatus.level}
            </Typography>
          </Box>
        </Tooltip>
      )}
    </Box>
  )
}

export default StatusIndicator
