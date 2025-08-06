/**
 * 性能监控组件 - 实时显示系统性能指标
 */

import React, { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Grid,
  Chip,
  IconButton,
  Collapse,
  Alert,
  Tooltip,
} from '@mui/material'
import {
  ExpandMore,
  ExpandLess,
  Speed,
  Memory,
  NetworkCheck,
  Error as ErrorIcon,
  CheckCircle,
  Warning,
} from '@mui/icons-material'

interface PerformanceMetrics {
  fps: number
  latency: number
  cpuUsage: number
  memoryUsage: number
  networkStatus: 'good' | 'poor' | 'disconnected'
  recognitionAccuracy: number
  frameDropRate: number
  bufferHealth: number
}

interface PerformanceMonitorProps {
  websocketService?: any
  onPerformanceIssue?: (issue: string) => void
  compact?: boolean
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  websocketService,
  onPerformanceIssue,
  compact = false
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    latency: 0,
    cpuUsage: 0,
    memoryUsage: 0,
    networkStatus: 'disconnected',
    recognitionAccuracy: 0,
    frameDropRate: 0,
    bufferHealth: 100,
  })

  const [expanded, setExpanded] = useState(!compact)
  const [performanceHistory, setPerformanceHistory] = useState<number[]>([])
  const [lastUpdateTime, setLastUpdateTime] = useState(Date.now())

  // 性能数据收集
  const collectMetrics = useCallback(() => {
    const now = Date.now()
    const deltaTime = now - lastUpdateTime

    // FPS 计算
    const fps = deltaTime > 0 ? Math.round(1000 / deltaTime) : 0

    // 模拟其他性能指标 (实际项目中应该从真实API获取)
    const latency = websocketService?.getStats?.()?.latency || Math.random() * 100 + 20
    const cpuUsage = Math.random() * 60 + 20
    const memoryUsage = Math.random() * 40 + 30
    const networkStatus = websocketService?.isConnected?.() ? 
      (latency < 100 ? 'good' : 'poor') : 'disconnected'
    const recognitionAccuracy = Math.random() * 30 + 70
    const frameDropRate = Math.random() * 10
    const bufferHealth = Math.max(0, 100 - frameDropRate * 5)

    const newMetrics = {
      fps: Math.min(fps, 60), // 限制最大FPS显示
      latency: Math.round(latency),
      cpuUsage: Math.round(cpuUsage),
      memoryUsage: Math.round(memoryUsage),
      networkStatus,
      recognitionAccuracy: Math.round(recognitionAccuracy),
      frameDropRate: Math.round(frameDropRate * 10) / 10,
      bufferHealth: Math.round(bufferHealth),
    }

    setMetrics(newMetrics)
    setLastUpdateTime(now)

    // 更新性能历史 (用于趋势分析)
    setPerformanceHistory(prev => {
      const newHistory = [...prev, newMetrics.fps]
      return newHistory.slice(-30) // 保留最近30个数据点
    })

    // 性能问题检测
    if (newMetrics.fps < 15 && onPerformanceIssue) {
      onPerformanceIssue('低帧率警告：当前FPS过低，可能影响识别精度')
    }

    if (newMetrics.latency > 200 && onPerformanceIssue) {
      onPerformanceIssue('网络延迟警告：延迟过高，建议检查网络连接')
    }

    if (newMetrics.frameDropRate > 5 && onPerformanceIssue) {
      onPerformanceIssue('帧丢失警告：检测到帧丢失，可能影响识别连续性')
    }

  }, [websocketService, onPerformanceIssue, lastUpdateTime])

  // 定期收集性能数据
  useEffect(() => {
    const interval = setInterval(collectMetrics, 1000) // 每秒更新一次
    return () => clearInterval(interval)
  }, [collectMetrics])

  // 获取性能等级颜色
  const getPerformanceColor = (value: number, thresholds: [number, number]) => {
    if (value >= thresholds[1]) return 'success'
    if (value >= thresholds[0]) return 'warning'
    return 'error'
  }

  // 获取网络状态颜色和图标
  const getNetworkStatusIcon = () => {
    switch (metrics.networkStatus) {
      case 'good':
        return <CheckCircle color="success" />
      case 'poor':
        return <Warning color="warning" />
      case 'disconnected':
        return <ErrorIcon color="error" />
      default:
        return <NetworkCheck />
    }
  }

  // 紧凑模式渲染
  if (compact) {
    return (
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
        <Tooltip title={`FPS: ${metrics.fps}`}>
          <Chip
            icon={<Speed />}
            label={`${metrics.fps}fps`}
            color={getPerformanceColor(metrics.fps, [20, 30])}
            size="small"
          />
        </Tooltip>
        
        <Tooltip title={`延迟: ${metrics.latency}ms`}>
          <Chip
            label={`${metrics.latency}ms`}
            color={getPerformanceColor(100 - metrics.latency / 2, [40, 70])}
            size="small"
          />
        </Tooltip>

        <Tooltip title={`网络状态: ${metrics.networkStatus}`}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {getNetworkStatusIcon()}
          </Box>
        </Tooltip>
      </Box>
    )
  }

  return (
    <Card elevation={2} sx={{ mb: 2 }}>
      <CardContent sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Speed color="primary" />
            性能监控
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={metrics.networkStatus === 'good' ? '连接正常' : 
                     metrics.networkStatus === 'poor' ? '连接不稳定' : '连接断开'}
              color={metrics.networkStatus === 'good' ? 'success' : 
                     metrics.networkStatus === 'poor' ? 'warning' : 'error'}
              size="small"
              icon={getNetworkStatusIcon()}
            />
            
            <IconButton size="small" onClick={() => setExpanded(!expanded)}>
              {expanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
        </Box>

        <Collapse in={expanded}>
          <Grid container spacing={2}>
            {/* FPS 指标 */}
            <Grid item xs={6} sm={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  帧率 (FPS)
                </Typography>
                <Typography variant="h6" color={
                  metrics.fps >= 30 ? 'success.main' :
                  metrics.fps >= 20 ? 'warning.main' : 'error.main'
                }>
                  {metrics.fps}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(metrics.fps / 60 * 100, 100)}
                  color={getPerformanceColor(metrics.fps, [20, 30])}
                  sx={{ height: 4, borderRadius: 2 }}
                />
              </Box>
            </Grid>

            {/* 延迟指标 */}
            <Grid item xs={6} sm={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  延迟 (ms)
                </Typography>
                <Typography variant="h6" color={
                  metrics.latency <= 50 ? 'success.main' :
                  metrics.latency <= 100 ? 'warning.main' : 'error.main'
                }>
                  {metrics.latency}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.max(0, 100 - metrics.latency / 2)}
                  color={getPerformanceColor(100 - metrics.latency / 2, [40, 70])}
                  sx={{ height: 4, borderRadius: 2 }}
                />
              </Box>
            </Grid>

            {/* CPU 使用率 */}
            <Grid item xs={6} sm={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  CPU 使用率
                </Typography>
                <Typography variant="h6" color={
                  metrics.cpuUsage <= 50 ? 'success.main' :
                  metrics.cpuUsage <= 75 ? 'warning.main' : 'error.main'
                }>
                  {metrics.cpuUsage}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={metrics.cpuUsage}
                  color={getPerformanceColor(100 - metrics.cpuUsage, [25, 50])}
                  sx={{ height: 4, borderRadius: 2 }}
                />
              </Box>
            </Grid>

            {/* 内存使用率 */}
            <Grid item xs={6} sm={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  内存使用率
                </Typography>
                <Typography variant="h6" color={
                  metrics.memoryUsage <= 60 ? 'success.main' :
                  metrics.memoryUsage <= 80 ? 'warning.main' : 'error.main'
                }>
                  {metrics.memoryUsage}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={metrics.memoryUsage}
                  color={getPerformanceColor(100 - metrics.memoryUsage, [20, 40])}
                  sx={{ height: 4, borderRadius: 2 }}
                />
              </Box>
            </Grid>

            {/* 识别精度 */}
            <Grid item xs={6} sm={4}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  识别精度
                </Typography>
                <Typography variant="h6" color="success.main">
                  {metrics.recognitionAccuracy}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={metrics.recognitionAccuracy}
                  color="success"
                  sx={{ height: 4, borderRadius: 2 }}
                />
              </Box>
            </Grid>

            {/* 帧丢失率 */}
            <Grid item xs={6} sm={4}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  帧丢失率
                </Typography>
                <Typography variant="h6" color={
                  metrics.frameDropRate <= 2 ? 'success.main' :
                  metrics.frameDropRate <= 5 ? 'warning.main' : 'error.main'
                }>
                  {metrics.frameDropRate}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.max(0, 100 - metrics.frameDropRate * 10)}
                  color={getPerformanceColor(100 - metrics.frameDropRate * 10, [50, 80])}
                  sx={{ height: 4, borderRadius: 2 }}
                />
              </Box>
            </Grid>

            {/* 缓冲区健康度 */}
            <Grid item xs={6} sm={4}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  缓冲区健康度
                </Typography>
                <Typography variant="h6" color={
                  metrics.bufferHealth >= 80 ? 'success.main' :
                  metrics.bufferHealth >= 60 ? 'warning.main' : 'error.main'
                }>
                  {metrics.bufferHealth}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={metrics.bufferHealth}
                  color={getPerformanceColor(metrics.bufferHealth, [60, 80])}
                  sx={{ height: 4, borderRadius: 2 }}
                />
              </Box>
            </Grid>
          </Grid>

          {/* 性能警告 */}
          {(metrics.fps < 20 || metrics.latency > 150 || metrics.frameDropRate > 3) && (
            <Alert 
              severity="warning" 
              sx={{ mt: 2 }}
              action={
                <Typography variant="caption">
                  建议降低视频质量或检查网络连接
                </Typography>
              }
            >
              检测到性能问题，可能影响识别效果
            </Alert>
          )}

          {/* 性能优化建议 */}
          {metrics.cpuUsage > 80 && (
            <Alert severity="info" sx={{ mt: 1 }}>
              CPU使用率较高，建议关闭其他占用资源的程序
            </Alert>
          )}
        </Collapse>
      </CardContent>
    </Card>
  )
}

export default PerformanceMonitor
