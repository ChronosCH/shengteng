/**
 * 智能仪表板组件 - 系统总览和快速操作
 */

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Card,
  CardContent,
  Button,
  Stack,
  Chip,
  Avatar,
  LinearProgress,
  CircularProgress,
  Fade,
  Alert,
  IconButton,
  Tooltip,
  Divider,
  useTheme,
  Skeleton,
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  TrendingUp,
  Speed,
  Psychology,
  Visibility,
  School,
  Person,
  Science,
  PlayArrow,
  Refresh,
  Settings,
  NotificationsActive,
  CheckCircle,
  Warning,
  Error,
  Info,
  Timeline,
  Analytics,
  Memory,
  Storage,
  Wifi,
  Battery,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import PerformanceMonitor from '../components/PerformanceMonitor'
import { useSignLanguageRecognition } from '../hooks/useSignLanguageRecognition'

interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  icon: React.ReactNode
  color: string
  trend?: 'up' | 'down' | 'stable'
  loading?: boolean
}

const MetricCard = ({ title, value, change, icon, color, trend, loading }: MetricCardProps) => {
  const theme = useTheme()
  
  return (
    <Card
      sx={{
        height: '100%',
        background: `linear-gradient(135deg, ${color}15 0%, ${color}08 100%)`,
        border: `1px solid ${color}30`,
        borderRadius: 3,
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 8px 25px ${color}30`,
          border: `1px solid ${color}60`,
        },
      }}
    >
      <CardContent sx={{ p: 3 }}>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <Avatar
            sx={{
              bgcolor: color,
              width: 48,
              height: 48,
              boxShadow: `0 4px 12px ${color}40`,
            }}
          >
            {icon}
          </Avatar>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
              {title}
            </Typography>
            {loading ? (
              <Skeleton variant="text" width={80} height={32} />
            ) : (
              <Typography variant="h4" sx={{ fontWeight: 700, color: 'text.primary' }}>
                {value}
              </Typography>
            )}
          </Box>
        </Stack>
        
        {change !== undefined && !loading && (
          <Stack direction="row" spacing={1} alignItems="center">
            <TrendingUp 
              sx={{ 
                fontSize: 16, 
                color: trend === 'up' ? 'success.main' : trend === 'down' ? 'error.main' : 'text.secondary',
                transform: trend === 'down' ? 'rotate(180deg)' : 'none',
              }} 
            />
            <Typography 
              variant="caption" 
              sx={{ 
                color: trend === 'up' ? 'success.main' : trend === 'down' ? 'error.main' : 'text.secondary',
                fontWeight: 600,
              }}
            >
              {Math.abs(change)}% {trend === 'up' ? '提升' : trend === 'down' ? '下降' : '稳定'}
            </Typography>
          </Stack>
        )}
      </CardContent>
    </Card>
  )
}

function DashboardPage() {
  const navigate = useNavigate()
  const theme = useTheme()
  const [loading, setLoading] = useState(true)
  const [systemHealth, setSystemHealth] = useState('good')
  const [recentActivity, setRecentActivity] = useState<any[]>([])
  const [quickStats, setQuickStats] = useState({
    recognitionAccuracy: 0,
    totalRecognitions: 0,
    learningProgress: 0,
    systemLoad: 0,
  })

  const { isRecognizing, confidence, stats } = useSignLanguageRecognition()

  // 模拟数据加载
  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      setQuickStats({
        recognitionAccuracy: 94.8,
        totalRecognitions: 1247,
        learningProgress: 76.5,
        systemLoad: 28.3,
      })
      
      setRecentActivity([
        { type: 'recognition', text: '识别了手语：你好', time: '2分钟前', accuracy: 96 },
        { type: 'learning', text: '完成了基础手语练习', time: '15分钟前', progress: 85 },
        { type: 'avatar', text: '生成了3D Avatar动画', time: '32分钟前', duration: '2.3s' },
        { type: 'system', text: '系统性能优化完成', time: '1小时前', improvement: 12 },
      ])
      
      setLoading(false)
    }

    loadData()
  }, [])

  // 检查系统健康状况
  useEffect(() => {
    const checkHealth = () => {
      const accuracy = quickStats.recognitionAccuracy
      const load = quickStats.systemLoad
      
      if (accuracy > 90 && load < 50) {
        setSystemHealth('excellent')
      } else if (accuracy > 80 && load < 70) {
        setSystemHealth('good')
      } else if (accuracy > 70 && load < 85) {
        setSystemHealth('fair')
      } else {
        setSystemHealth('poor')
      }
    }

    if (!loading) {
      checkHealth()
    }
  }, [quickStats, loading])

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'excellent': return theme.palette.success.main
      case 'good': return theme.palette.info.main
      case 'fair': return theme.palette.warning.main
      case 'poor': return theme.palette.error.main
      default: return theme.palette.text.secondary
    }
  }

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'excellent': return <CheckCircle />
      case 'good': return <Info />
      case 'fair': return <Warning />
      case 'poor': return <Error />
      default: return <Info />
    }
  }

  const quickActions = [
    {
      title: '开始识别',
      description: '启动实时手语识别',
      icon: <Visibility />,
      color: '#FFB3BA',
      path: '/recognition',
      disabled: false,
    },
    {
      title: '学习训练',
      description: '进入学习模式',
      icon: <School />,
      color: '#B5EAD7',
      path: '/learning',
      disabled: false,
    },
    {
      title: '3D演示',
      description: '查看Avatar演示',
      icon: <Person />,
      color: '#FFDAB9',
      path: '/avatar',
      disabled: false,
    },
    {
      title: '实验功能',
      description: '体验前沿技术',
      icon: <Science />,
      color: '#C7CEDB',
      path: '/lab',
      disabled: false,
    },
  ]

  const systemMetrics = [
    {
      title: '识别准确率',
      value: loading ? 0 : quickStats.recognitionAccuracy,
      suffix: '%',
      icon: <Psychology />,
      color: theme.palette.success.main,
      change: 2.3,
      trend: 'up' as const,
    },
    {
      title: '总识别次数',
      value: loading ? 0 : quickStats.totalRecognitions,
      suffix: '',
      icon: <Analytics />,
      color: theme.palette.info.main,
      change: 12.7,
      trend: 'up' as const,
    },
    {
      title: '学习进度',
      value: loading ? 0 : quickStats.learningProgress,
      suffix: '%',
      icon: <Timeline />,
      color: theme.palette.warning.main,
      change: 5.2,
      trend: 'up' as const,
    },
    {
      title: '系统负载',
      value: loading ? 0 : quickStats.systemLoad,
      suffix: '%',
      icon: <Memory />,
      color: theme.palette.error.main,
      change: -3.1,
      trend: 'down' as const,
    },
  ]

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4 }}>
          <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
            <Avatar
              sx={{
                bgcolor: 'primary.main',
                width: 48,
                height: 48,
                boxShadow: '0 4px 12px rgba(25, 118, 210, 0.3)',
              }}
            >
              <DashboardIcon />
            </Avatar>
            <Box>
              <Typography variant="h3" sx={{ fontWeight: 700, color: 'text.primary' }}>
                智能仪表板
              </Typography>
              <Typography variant="body1" color="text.secondary">
                系统总览、实时监控和快速操作中心
              </Typography>
            </Box>
          </Stack>
          
          {/* 系统健康状态 */}
          <Alert
            severity={systemHealth === 'excellent' || systemHealth === 'good' ? 'success' : systemHealth === 'fair' ? 'warning' : 'error'}
            icon={getHealthIcon(systemHealth)}
            sx={{ 
              borderRadius: 3,
              '& .MuiAlert-message': { fontWeight: 500 },
            }}
          >
            系统运行状态：
            {systemHealth === 'excellent' && '优秀 - 所有功能运行正常'}
            {systemHealth === 'good' && '良好 - 系统运行稳定'}
            {systemHealth === 'fair' && '一般 - 建议检查系统设置'}
            {systemHealth === 'poor' && '需要注意 - 请检查系统配置'}
          </Alert>
        </Box>
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧主要内容 */}
        <Grid item xs={12} lg={8}>
          <Stack spacing={4}>
            {/* 关键指标卡片 */}
            <Fade in timeout={800}>
              <Box>
                <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                  关键指标
                </Typography>
                <Grid container spacing={3}>
                  {systemMetrics.map((metric, index) => (
                    <Grid item xs={12} sm={6} md={3} key={metric.title}>
                      <MetricCard
                        title={metric.title}
                        value={`${metric.value}${metric.suffix}`}
                        change={metric.change}
                        icon={metric.icon}
                        color={metric.color}
                        trend={metric.trend}
                        loading={loading}
                      />
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Fade>

            {/* 快速操作 */}
            <Fade in timeout={1000}>
              <Box>
                <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                  快速操作
                </Typography>
                <Grid container spacing={3}>
                  {quickActions.map((action, index) => (
                    <Grid item xs={12} sm={6} md={3} key={action.title}>
                      <Card
                        sx={{
                          height: '100%',
                          cursor: 'pointer',
                          background: `linear-gradient(135deg, ${action.color}15 0%, ${action.color}08 100%)`,
                          border: `1px solid ${action.color}30`,
                          borderRadius: 3,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-6px)',
                            boxShadow: `0 12px 30px ${action.color}30`,
                            border: `1px solid ${action.color}60`,
                          },
                        }}
                        onClick={() => navigate(action.path)}
                      >
                        <CardContent sx={{ p: 3, textAlign: 'center' }}>
                          <Avatar
                            sx={{
                              bgcolor: action.color,
                              width: 56,
                              height: 56,
                              mx: 'auto',
                              mb: 2,
                              boxShadow: `0 4px 12px ${action.color}40`,
                            }}
                          >
                            {action.icon}
                          </Avatar>
                          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                            {action.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {action.description}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Fade>

            {/* 最近活动 */}
            <Fade in timeout={1200}>
              <Paper sx={{ p: 4, borderRadius: 3 }}>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 3 }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, flexGrow: 1 }}>
                    最近活动
                  </Typography>
                  <Tooltip title="刷新">
                    <IconButton onClick={() => window.location.reload()}>
                      <Refresh />
                    </IconButton>
                  </Tooltip>
                </Stack>
                
                <Stack spacing={2}>
                  {loading ? (
                    // 加载骨架屏
                    Array.from({ length: 4 }).map((_, index) => (
                      <Box key={index} sx={{ display: 'flex', alignItems: 'center', p: 2 }}>
                        <Skeleton variant="circular" width={40} height={40} sx={{ mr: 2 }} />
                        <Box sx={{ flexGrow: 1 }}>
                          <Skeleton variant="text" width="60%" height={20} />
                          <Skeleton variant="text" width="40%" height={16} />
                        </Box>
                      </Box>
                    ))
                  ) : (
                    recentActivity.map((activity, index) => (
                      <Box
                        key={index}
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          p: 2,
                          borderRadius: 2,
                          backgroundColor: 'background.paper',
                          border: '1px solid',
                          borderColor: 'divider',
                          transition: 'all 0.2s ease',
                          '&:hover': {
                            backgroundColor: 'action.hover',
                            transform: 'translateX(4px)',
                          },
                        }}
                      >
                        <Avatar
                          sx={{
                            bgcolor: activity.type === 'recognition' ? '#FFB3BA' :
                                     activity.type === 'learning' ? '#B5EAD7' :
                                     activity.type === 'avatar' ? '#FFDAB9' : '#C7CEDB',
                            mr: 2,
                            width: 40,
                            height: 40,
                          }}
                        >
                          {activity.type === 'recognition' && <Visibility />}
                          {activity.type === 'learning' && <School />}
                          {activity.type === 'avatar' && <Person />}
                          {activity.type === 'system' && <Settings />}
                        </Avatar>
                        
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="body1" sx={{ fontWeight: 500 }}>
                            {activity.text}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {activity.time}
                          </Typography>
                        </Box>
                        
                        {activity.accuracy && (
                          <Chip
                            label={`${activity.accuracy}%`}
                            size="small"
                            color="success"
                            variant="outlined"
                          />
                        )}
                        {activity.progress && (
                          <Chip
                            label={`${activity.progress}%`}
                            size="small"
                            color="info"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    ))
                  )}
                </Stack>
              </Paper>
            </Fade>
          </Stack>
        </Grid>

        {/* 右侧边栏 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={4}>
            {/* 实时状态 */}
            <Fade in timeout={1400}>
              <Paper sx={{ p: 3, borderRadius: 3 }}>
                <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                  实时状态
                </Typography>
                
                <Stack spacing={3}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Wifi color={isRecognizing ? 'success' : 'disabled'} />
                      <Typography variant="body2">WebSocket连接</Typography>
                    </Stack>
                    <Chip
                      label={isRecognizing ? '已连接' : '未连接'}
                      size="small"
                      color={isRecognizing ? 'success' : 'default'}
                      variant="outlined"
                    />
                  </Box>
                  
                  {confidence !== null && (
                    <Box>
                      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                        <Psychology />
                        <Typography variant="body2">识别置信度</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {(confidence * 100).toFixed(1)}%
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={confidence * 100}
                        color={confidence > 0.8 ? 'success' : confidence > 0.6 ? 'warning' : 'error'}
                        sx={{ height: 6, borderRadius: 3 }}
                      />
                    </Box>
                  )}
                  
                  <Box>
                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                      <Memory />
                      <Typography variant="body2">系统负载</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {quickStats.systemLoad.toFixed(1)}%
                      </Typography>
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={quickStats.systemLoad}
                      color={quickStats.systemLoad < 50 ? 'success' : quickStats.systemLoad < 75 ? 'warning' : 'error'}
                      sx={{ height: 6, borderRadius: 3 }}
                    />
                  </Box>
                </Stack>
              </Paper>
            </Fade>

            {/* 性能监控 */}
            <Fade in timeout={1600}>
              <ErrorBoundary>
                <PerformanceMonitor isVisible={true} />
              </ErrorBoundary>
            </Fade>

            {/* 快捷设置 */}
            <Fade in timeout={1800}>
              <Paper sx={{ p: 3, borderRadius: 3 }}>
                <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                  快捷设置
                </Typography>
                
                <Stack spacing={2}>
                  <Button
                    variant="outlined"
                    startIcon={<Settings />}
                    onClick={() => navigate('/settings')}
                    fullWidth
                    sx={{ justifyContent: 'flex-start' }}
                  >
                    系统设置
                  </Button>
                  
                  <Button
                    variant="outlined"
                    startIcon={<NotificationsActive />}
                    fullWidth
                    sx={{ justifyContent: 'flex-start' }}
                  >
                    通知管理
                  </Button>
                  
                  <Button
                    variant="outlined"
                    startIcon={<Speed />}
                    fullWidth
                    sx={{ justifyContent: 'flex-start' }}
                  >
                    性能优化
                  </Button>
                </Stack>
              </Paper>
            </Fade>
          </Stack>
        </Grid>
      </Grid>
    </Container>
  )
}

export default DashboardPage