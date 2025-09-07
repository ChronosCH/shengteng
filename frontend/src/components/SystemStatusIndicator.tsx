/**
 * 系统状态指示器组件
 * 显示系统各个模块的运行状态
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Stack,
  IconButton,
  Collapse,
  Tooltip,
  LinearProgress,
  Alert,
} from '@mui/material'
import {
  ExpandMore,
  ExpandLess,
  CheckCircle,
  Error,
  Warning,
  Info,
  Wifi,
  WifiOff,
  Videocam,
  VideocamOff,
  Memory,
  Speed,
  Storage,
  Computer,
} from '@mui/icons-material'

interface SystemStatus {
  websocket: 'connected' | 'disconnected' | 'connecting' | 'error'
  camera: 'available' | 'unavailable' | 'permission_denied' | 'in_use'
  backend: 'online' | 'offline' | 'error'
  performance: {
    fps: number
    latency: number
    memory: number
  }
}

interface SystemStatusIndicatorProps {
  status: SystemStatus
  onRefresh?: () => void
}

const SystemStatusIndicator: React.FC<SystemStatusIndicatorProps> = ({
  status,
  onRefresh,
}) => {
  const [expanded, setExpanded] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  useEffect(() => {
    setLastUpdate(new Date())
  }, [status])

  // 获取状态图标和颜色
  const getStatusIcon = (statusType: string, value: string) => {
    switch (statusType) {
      case 'websocket':
        switch (value) {
          case 'connected':
            return { icon: <Wifi />, color: 'success' as const }
          case 'connecting':
            return { icon: <Wifi />, color: 'warning' as const }
          case 'disconnected':
            return { icon: <WifiOff />, color: 'default' as const }
          case 'error':
            return { icon: <WifiOff />, color: 'error' as const }
          default:
            return { icon: <WifiOff />, color: 'default' as const }
        }
      case 'camera':
        switch (value) {
          case 'available':
            return { icon: <Videocam />, color: 'success' as const }
          case 'in_use':
            return { icon: <Videocam />, color: 'primary' as const }
          case 'unavailable':
            return { icon: <VideocamOff />, color: 'default' as const }
          case 'permission_denied':
            return { icon: <VideocamOff />, color: 'error' as const }
          default:
            return { icon: <VideocamOff />, color: 'default' as const }
        }
      case 'backend':
        switch (value) {
          case 'online':
            return { icon: <CheckCircle />, color: 'success' as const }
          case 'offline':
            return { icon: <Error />, color: 'default' as const }
          case 'error':
            return { icon: <Error />, color: 'error' as const }
          default:
            return { icon: <Error />, color: 'default' as const }
        }
      default:
        return { icon: <Info />, color: 'default' as const }
    }
  }

  // 获取状态描述
  const getStatusDescription = (statusType: string, value: string) => {
    const descriptions: Record<string, Record<string, string>> = {
      websocket: {
        connected: 'WebSocket已连接',
        connecting: 'WebSocket连接中',
        disconnected: 'WebSocket未连接',
        error: 'WebSocket连接错误',
      },
      camera: {
        available: '摄像头可用',
        in_use: '摄像头使用中',
        unavailable: '摄像头不可用',
        permission_denied: '摄像头权限被拒绝',
      },
      backend: {
        online: '后端服务在线',
        offline: '后端服务离线',
        error: '后端服务错误',
      },
    }
    return descriptions[statusType]?.[value] || '未知状态'
  }

  // 获取性能等级
  const getPerformanceLevel = (type: 'fps' | 'latency' | 'memory', value: number) => {
    switch (type) {
      case 'fps':
        if (value >= 25) return 'success'
        if (value >= 15) return 'warning'
        return 'error'
      case 'latency':
        if (value <= 100) return 'success'
        if (value <= 300) return 'warning'
        return 'error'
      case 'memory':
        if (value <= 70) return 'success'
        if (value <= 85) return 'warning'
        return 'error'
      default:
        return 'default'
    }
  }

  // 计算整体系统健康度
  const getSystemHealth = () => {
    let score = 0
    let total = 0

    // WebSocket状态评分
    if (status.websocket === 'connected') score += 25
    else if (status.websocket === 'connecting') score += 10
    total += 25

    // 摄像头状态评分
    if (status.camera === 'available' || status.camera === 'in_use') score += 25
    else if (status.camera === 'unavailable') score += 10
    total += 25

    // 后端状态评分
    if (status.backend === 'online') score += 25
    total += 25

    // 性能评分
    const fpsScore = Math.min(25, (status.performance.fps / 30) * 25)
    const latencyScore = Math.max(0, 25 - (status.performance.latency / 20))
    const memoryScore = Math.max(0, 25 - (status.performance.memory / 4))
    score += (fpsScore + latencyScore + memoryScore) / 3
    total += 25

    return Math.round((score / total) * 100)
  }

  const systemHealth = getSystemHealth()

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent sx={{ pb: 2 }}>
        {/* 系统健康度总览 */}
        <Box sx={{ mb: 2 }}>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="h6" fontWeight={600}>
              系统状态
            </Typography>
            <Stack direction="row" alignItems="center" spacing={1}>
              <Typography variant="body2" color="text.secondary">
                健康度: {systemHealth}%
              </Typography>
              <IconButton
                size="small"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Stack>
          </Stack>
          
          <LinearProgress
            variant="determinate"
            value={systemHealth}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: 'rgba(0,0,0,0.1)',
              '& .MuiLinearProgress-bar': {
                borderRadius: 4,
                backgroundColor: systemHealth > 80 ? '#4caf50' : systemHealth > 60 ? '#ff9800' : '#f44336',
              },
            }}
          />
        </Box>

        {/* 主要状态指示器 */}
        <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
          {Object.entries({
            websocket: status.websocket,
            camera: status.camera,
            backend: status.backend,
          }).map(([key, value]) => {
            const { icon, color } = getStatusIcon(key, value)
            return (
              <Tooltip key={key} title={getStatusDescription(key, value)}>
                <Chip
                  icon={icon}
                  label={key.charAt(0).toUpperCase() + key.slice(1)}
                  color={color}
                  size="small"
                  variant={color === 'default' ? 'outlined' : 'filled'}
                />
              </Tooltip>
            )
          })}
        </Stack>

        {/* 详细信息 */}
        <Collapse in={expanded}>
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom fontWeight={600}>
              性能指标
            </Typography>
            <Stack spacing={2}>
              {/* FPS */}
              <Box>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">帧率 (FPS)</Typography>
                  <Chip
                    label={`${status.performance.fps} fps`}
                    color={getPerformanceLevel('fps', status.performance.fps)}
                    size="small"
                  />
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, (status.performance.fps / 30) * 100)}
                  sx={{ mt: 0.5, height: 4 }}
                />
              </Box>

              {/* 延迟 */}
              <Box>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">延迟</Typography>
                  <Chip
                    label={`${status.performance.latency}ms`}
                    color={getPerformanceLevel('latency', status.performance.latency)}
                    size="small"
                  />
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={Math.max(0, 100 - (status.performance.latency / 5))}
                  sx={{ mt: 0.5, height: 4 }}
                />
              </Box>

              {/* 内存使用 */}
              <Box>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">内存使用</Typography>
                  <Chip
                    label={`${status.performance.memory}%`}
                    color={getPerformanceLevel('memory', status.performance.memory)}
                    size="small"
                  />
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={status.performance.memory}
                  sx={{ mt: 0.5, height: 4 }}
                />
              </Box>
            </Stack>

            <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
              最后更新: {lastUpdate.toLocaleTimeString()}
            </Typography>
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  )
}

export default SystemStatusIndicator
