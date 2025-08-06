/**
 * 智能仪表板组件 - 系统概览和快捷操作
 */

import { useState, useEffect } from 'react'
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Stack,
  Avatar,
  Chip,
  LinearProgress,
  Paper,
  IconButton,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Badge,
  useTheme,
  alpha,
  Tooltip,
  CircularProgress,
} from '@mui/material'
import {
  Dashboard,
  TrendingUp,
  Speed,
  Psychology,
  School,
  Analytics,
  Notifications,
  Settings,
  Person,
  Calendar,
  Star,
  CheckCircle,
  PlayArrow,
  Pause,
  MoreVert,
  Launch,
  Timeline,
  EmojiEvents,
  Group,
  BookmarkBorder,
  AccessTime,
  Lightbulb,
  AutoAwesome,
  Favorite,
  Share,
} from '@mui/icons-material'
import { Line, Doughnut, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  ArcElement,
  BarElement,
} from 'chart.js'

// 注册Chart.js组件
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  ArcElement,
  BarElement
)

interface DashboardProps {
  onNavigate?: (path: string) => void
}

function Dashboard({ onNavigate }: DashboardProps) {
  const theme = useTheme()
  const [systemStatus, setSystemStatus] = useState({
    cpu: 45,
    memory: 62,
    gpu: 78,
    network: 89,
  })
  const [realtimeData, setRealtimeData] = useState({
    activeUsers: 1247,
    recognitionAccuracy: 95.2,
    processingSpeed: 23,
    totalSessions: 8934,
  })
  const [notifications, setNotifications] = useState([
    { id: 1, title: '新功能发布', message: '3D Avatar升级完成', time: '2分钟前', type: 'success' },
    { id: 2, title: '系统维护', message: '今晚23:00-01:00', time: '1小时前', type: 'warning' },
    { id: 3, title: '学习提醒', message: '您有未完成的课程', time: '3小时前', type: 'info' },
  ])

  useEffect(() => {
    // 模拟实时数据更新
    const interval = setInterval(() => {
      setSystemStatus(prev => ({
        cpu: Math.max(20, Math.min(80, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(30, Math.min(90, prev.memory + (Math.random() - 0.5) * 8)),
        gpu: Math.max(40, Math.min(95, prev.gpu + (Math.random() - 0.5) * 6)),
        network: Math.max(60, Math.min(100, prev.network + (Math.random() - 0.5) * 4)),
      }))
      
      setRealtimeData(prev => ({
        activeUsers: Math.max(800, Math.min(2000, prev.activeUsers + Math.floor((Math.random() - 0.5) * 50))),
        recognitionAccuracy: Math.max(90, Math.min(98, prev.recognitionAccuracy + (Math.random() - 0.5) * 2)),
        processingSpeed: Math.max(15, Math.min(35, prev.processingSpeed + (Math.random() - 0.5) * 5)),
        totalSessions: prev.totalSessions + Math.floor(Math.random() * 3),
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  // 图表数据配置
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        display: false,
      },
    },
    elements: {
      point: {
        radius: 0,
      },
    },
  }

  const performanceData = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
    datasets: [{
      data: [85, 87, 91, 95, 93, 96, 94],
      borderColor: '#B5EAD7',
      backgroundColor: alpha('#B5EAD7', 0.1),
      fill: true,
      tension: 0.4,
    }],
  }

  const usageData = {
    labels: ['识别', '学习', '分析', '其他'],
    datasets: [{
      data: [45, 30, 15, 10],
      backgroundColor: ['#B5EAD7', '#FFDAB9', '#FFB3BA', '#C7CEDB'],
      borderWidth: 0,
    }],
  }

  const learningProgressData = {
    labels: ['基础', '初级', '中级', '高级', '专业'],
    datasets: [{
      data: [100, 85, 60, 35, 15],
      backgroundColor: alpha('#B5EAD7', 0.8),
      borderColor: '#B5EAD7',
      borderWidth: 1,
    }],
  }

  const quickActions = [
    { title: '开始识别', icon: <PlayArrow />, color: '#B5EAD7', path: '/recognition' },
    { title: '学习训练', icon: <School />, color: '#FFDAB9', path: '/learning' },
    { title: '数据分析', icon: <Analytics />, color: '#FFB3BA', path: '/analytics' },
    { title: '3D演示', icon: <Psychology />, color: '#C7CEDB', path: '/avatar' },
  ]

  const recentActivities = [
    { title: '完成基础手语课程', time: '2小时前', icon: <CheckCircle />, color: '#B5EAD7' },
    { title: '识别准确率达到95%', time: '4小时前', icon: <TrendingUp />, color: '#FFDAB9' },
    { title: '参与社区讨论', time: '6小时前', icon: <Group />, color: '#FFB3BA' },
    { title: '保存学习笔记', time: '8小时前', icon: <BookmarkBorder />, color: '#C7CEDB' },
  ]

  const handleQuickAction = (path: string) => {
    if (onNavigate) {
      onNavigate(path)
    } else {
      window.location.href = path
    }
  }

  const getStatusColor = (value: number) => {
    if (value >= 80) return '#ff6b6b'
    if (value >= 60) return '#feca57'
    return '#48dbfb'
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题和欢迎信息 */}
      <Box sx={{ mb: 4 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
              欢迎回来！
            </Typography>
            <Typography variant="body1" color="text.secondary">
              今天是 {new Date().toLocaleDateString('zh-CN')} • 让我们继续您的手语学习之旅
            </Typography>
          </Box>
          <Stack direction="row" spacing={1}>
            <Tooltip title="设置">
              <IconButton>
                <Settings />
              </IconButton>
            </Tooltip>
            <Tooltip title="通知">
              <Badge badgeContent={notifications.length} color="error">
                <IconButton>
                  <Notifications />
                </IconButton>
              </Badge>
            </Tooltip>
          </Stack>
        </Stack>
      </Box>

      <Grid container spacing={3}>
        {/* 快捷操作 */}
        <Grid item xs={12}>
          <Card sx={{ borderRadius: 3, mb: 3 }}>
            <CardContent sx={{ pb: '16px !important' }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                快捷操作
              </Typography>
              <Grid container spacing={2}>
                {quickActions.map((action, index) => (
                  <Grid item xs={6} sm={3} key={action.title}>
                    <Button
                      fullWidth
                      variant="outlined"
                      startIcon={action.icon}
                      onClick={() => handleQuickAction(action.path)}
                      sx={{
                        py: 2,
                        borderColor: `${action.color}40`,
                        color: action.color,
                        backgroundColor: `${action.color}08`,
                        borderRadius: 2,
                        '&:hover': {
                          borderColor: action.color,
                          backgroundColor: `${action.color}15`,
                          transform: 'translateY(-2px)',
                        },
                        transition: 'all 0.3s ease',
                      }}
                    >
                      {action.title}
                    </Button>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* 实时数据概览 */}
        <Grid item xs={12} md={8}>
          <Grid container spacing={3}>
            {/* 关键指标 */}
            <Grid item xs={12}>
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                    实时数据概览
                  </Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Avatar
                          sx={{
                            width: 56,
                            height: 56,
                            bgcolor: '#B5EAD7',
                            mx: 'auto',
                            mb: 1,
                          }}
                        >
                          <Group />
                        </Avatar>
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {realtimeData.activeUsers.toLocaleString()}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          在线用户
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Avatar
                          sx={{
                            width: 56,
                            height: 56,
                            bgcolor: '#FFDAB9',
                            mx: 'auto',
                            mb: 1,
                          }}
                        >
                          <TrendingUp />
                        </Avatar>
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {realtimeData.recognitionAccuracy.toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          识别准确率
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Avatar
                          sx={{
                            width: 56,
                            height: 56,
                            bgcolor: '#FFB3BA',
                            mx: 'auto',
                            mb: 1,
                          }}
                        >
                          <Speed />
                        </Avatar>
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {realtimeData.processingSpeed}ms
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          处理速度
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Avatar
                          sx={{
                            width: 56,
                            height: 56,
                            bgcolor: '#C7CEDB',
                            mx: 'auto',
                            mb: 1,
                          }}
                        >
                          <EmojiEvents />
                        </Avatar>
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {realtimeData.totalSessions.toLocaleString()}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          总学习次数
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* 性能趋势图 */}
            <Grid item xs={12} md={6}>
              <Card sx={{ borderRadius: 3, height: 300 }}>
                <CardContent sx={{ height: '100%' }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                    性能趋势
                  </Typography>
                  <Box sx={{ height: 200 }}>
                    <Line data={performanceData} options={chartOptions} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* 使用分布 */}
            <Grid item xs={12} md={6}>
              <Card sx={{ borderRadius: 3, height: 300 }}>
                <CardContent sx={{ height: '100%' }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                    功能使用分布
                  </Typography>
                  <Box sx={{ height: 200, display: 'flex', justifyContent: 'center' }}>
                    <Doughnut data={usageData} options={{ ...chartOptions, maintainAspectRatio: true }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* 学习进度 */}
            <Grid item xs={12}>
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                    学习进度统计
                  </Typography>
                  <Box sx={{ height: 200 }}>
                    <Bar data={learningProgressData} options={chartOptions} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* 右侧边栏 */}
        <Grid item xs={12} md={4}>
          <Stack spacing={3}>
            {/* 系统状态 */}
            <Card sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                  系统状态
                </Typography>
                <Stack spacing={3}>
                  {Object.entries(systemStatus).map(([key, value]) => (
                    <Box key={key}>
                      <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                        <Typography variant="body2" sx={{ textTransform: 'uppercase' }}>
                          {key === 'cpu' ? 'CPU' : key === 'memory' ? '内存' : key === 'gpu' ? 'GPU' : '网络'}
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {value}%
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={value}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: alpha(getStatusColor(value), 0.2),
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getStatusColor(value),
                            borderRadius: 4,
                          },
                        }}
                      />
                    </Box>
                  ))}
                </Stack>
              </CardContent>
            </Card>

            {/* 通知中心 */}
            <Card sx={{ borderRadius: 3 }}>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    通知中心
                  </Typography>
                  <Badge badgeContent={notifications.length} color="error">
                    <Notifications />
                  </Badge>
                </Stack>
                <List sx={{ p: 0 }}>
                  {notifications.map((notification, index) => (
                    <ListItem
                      key={notification.id}
                      sx={{
                        px: 0,
                        borderBottom: index < notifications.length - 1 ? '1px solid' : 'none',
                        borderColor: 'divider',
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        <Avatar
                          sx={{
                            width: 32,
                            height: 32,
                            bgcolor: notification.type === 'success' ? '#B5EAD7' :
                                     notification.type === 'warning' ? '#FFDAB9' : '#C7CEDB',
                          }}
                        >
                          <AutoAwesome sx={{ fontSize: 16 }} />
                        </Avatar>
                      </ListItemIcon>
                      <ListItemText
                        primary={notification.title}
                        secondary={
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              {notification.message}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {notification.time}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>

            {/* 最近活动 */}
            <Card sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  最近活动
                </Typography>
                <List sx={{ p: 0 }}>
                  {recentActivities.map((activity, index) => (
                    <ListItem
                      key={index}
                      sx={{
                        px: 0,
                        borderBottom: index < recentActivities.length - 1 ? '1px solid' : 'none',
                        borderColor: 'divider',
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        <Avatar
                          sx={{
                            width: 32,
                            height: 32,
                            bgcolor: activity.color,
                          }}
                        >
                          <Box sx={{ color: 'white', fontSize: 16 }}>
                            {activity.icon}
                          </Box>
                        </Avatar>
                      </ListItemIcon>
                      <ListItemText
                        primary={activity.title}
                        secondary={activity.time}
                        primaryTypographyProps={{ variant: 'body2' }}
                        secondaryTypographyProps={{ variant: 'caption' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Stack>
        </Grid>
      </Grid>
    </Container>
  )
}

export default Dashboard