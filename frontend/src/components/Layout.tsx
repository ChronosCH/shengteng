/**
 * 优化的响应式主布局组件 - 增强版
 */

import { ReactNode, useState, useEffect, useMemo } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Stack,
  Chip,
  useTheme,
  useMediaQuery,
  Avatar,
  Badge,
  Tooltip,
  Collapse,
  Paper,
  LinearProgress,
  Dialog,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Brightness4,
  Brightness7,
  Home,
  Visibility,
  School,
  Person,
  Settings,
  Science,
  Handshake,
  Close,
  FiberManualRecord,
  ExpandLess,
  ExpandMore,
  Dashboard,
  Notifications,
  Help,
  AutoAwesome,
  Diamond,
  Face,
  Compare,
} from '@mui/icons-material'

import StatusIndicator from './StatusIndicator'
import SafeFade from './SafeFade'
import { useSignLanguageRecognition } from '../hooks/useSignLanguageRecognition'
import { useAuth } from '../contexts/AuthContext'
import AuthModal from './auth/AuthModal'
import UserProfile from './auth/UserProfile'

interface LayoutProps {
  children: ReactNode
  darkMode: boolean
  onToggleDarkMode: () => void
}

// 优化的导航结构，支持分组和子菜单
const navigationGroups = [
  {
    title: '主要功能',
    items: [
      { text: '总览', icon: <Dashboard />, path: '/', color: '#FFB3BA', description: '系统总览和快速访问' },
      { text: '手语识别', icon: <Visibility />, path: '/recognition', color: '#B5EAD7', description: '实时手语识别体验' },
      { text: '学习训练', icon: <School />, path: '/learning', color: '#C7CEDB', description: '系统化手语学习' },
    ]
  },
  {
    title: '3D Avatar系统',
    items: [
      { text: '基础Avatar', icon: <Person />, path: '/avatar', color: '#FFDAB9', description: '基础3D手语演示' },
      { text: '专业Avatar', icon: <AutoAwesome />, path: '/avatar-pro', color: '#FFE4B5', description: '专业手语Avatar' },
      { text: '高质量Avatar', icon: <Diamond />, path: '/avatar-hq', color: '#F0E68C', description: '高质量3D建模' },
      { text: '真人级Avatar', icon: <Face />, path: '/avatar-advanced', color: '#98FB98', description: '真人级3D Avatar' },
      { text: 'Avatar对比', icon: <Compare />, path: '/avatar-compare', color: '#87CEEB', description: 'Avatar版本对比' },
    ]
  },
  {
    title: '高级功能',
    items: [
      { text: '实验室', icon: <Science />, path: '/lab', color: '#E0E4CC', description: '前沿技术体验' },
      { text: '设置', icon: <Settings />, path: '/settings', color: '#FFD6CC', description: '个性化配置' },
    ]
  }
]

function Layout({ children, darkMode, onToggleDarkMode }: LayoutProps) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [expandedGroups, setExpandedGroups] = useState<string[]>(['主要功能'])
  const [notifications, setNotifications] = useState(3)
  const [isMounted, setIsMounted] = useState(false)
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const [profileModalOpen, setProfileModalOpen] = useState(false)

  const navigate = useNavigate()
  const location = useLocation()
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('lg'))
  const isTablet = useMediaQuery(theme.breakpoints.down('md'))

  const { isRecognizing, confidence, websocketService, stats } = useSignLanguageRecognition()
  const { isAuthenticated, user } = useAuth()
  const [isConnected, setIsConnected] = useState(false)

  // 确保组件完全挂载后再显示动画
  useEffect(() => {
    const timer = setTimeout(() => setIsMounted(true), 100)
    return () => clearTimeout(timer)
  }, [])

  // 优化的WebSocket连接状态监听
  useEffect(() => {
    if (websocketService) {
      const handleConnect = () => {
        setIsConnected(true)
        setNotifications(prev => Math.max(0, prev - 1))
      }
      const handleDisconnect = () => {
        setIsConnected(false)
        setNotifications(prev => prev + 1)
      }

      websocketService.on('connect', handleConnect)
      websocketService.on('disconnect', handleDisconnect)

      return () => {
        websocketService.off('connect', handleConnect)
        websocketService.off('disconnect', handleDisconnect)
      }
    }
  }, [websocketService])

  // 智能抽屉管理
  useEffect(() => {
    if (!isMobile && drawerOpen) {
      // 在大屏幕上保持侧边栏展开
      setDrawerOpen(true)
    } else if (isMobile && drawerOpen) {
      // 移动端导航后自动关闭
      const timer = setTimeout(() => setDrawerOpen(false), 200)
      return () => clearTimeout(timer)
    }
  }, [location.pathname, isMobile])

  const handleNavigation = (path: string) => {
    navigate(path)
    if (isMobile) {
      setDrawerOpen(false)
    }
  }

  const toggleGroup = (groupTitle: string) => {
    setExpandedGroups(prev => 
      prev.includes(groupTitle) 
        ? prev.filter(g => g !== groupTitle)
        : [...prev, groupTitle]
    )
  }

  const getCurrentPageInfo = useMemo(() => {
    for (const group of navigationGroups) {
      const item = group.items.find(item => item.path === location.pathname)
      if (item) return item
    }
    return { text: 'SignAvatar Web', description: '智能手语识别系统' }
  }, [location.pathname])

  // 优化的侧边栏内容
  const drawerContent = (
    <Box sx={{ width: 320, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 增强的Logo区域 */}
      <Box sx={{ 
        p: 3, 
        textAlign: 'center',
        background: `linear-gradient(135deg, ${theme.palette.primary.main}15 0%, ${theme.palette.secondary.main}15 100%)`,
        borderBottom: `1px solid ${theme.palette.divider}`,
        position: 'relative',
        overflow: 'hidden',
      }}>
        <Box
          sx={{
            position: 'absolute',
            top: -20,
            right: -20,
            width: 60,
            height: 60,
            background: `radial-gradient(circle, ${theme.palette.primary.light}20 0%, transparent 70%)`,
            borderRadius: '50%',
          }}
        />
        
        <Avatar
          sx={{
            width: 72,
            height: 72,
            mx: 'auto',
            mb: 2,
            background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
            boxShadow: '0 8px 24px rgba(255, 179, 186, 0.3)',
            border: '3px solid rgba(255,255,255,0.9)',
          }}
        >
          <Handshake sx={{ fontSize: 36, color: 'white' }} />
        </Avatar>
        
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5, color: 'primary.main' }}>
          SignAvatar
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.9rem', mb: 2 }}>
          智能手语识别系统
        </Typography>
        
        {/* 快速状态指示器 */}
        <Stack direction="row" spacing={1} justifyContent="center">
          <Chip
            icon={<FiberManualRecord sx={{ fontSize: '12px !important' }} />}
            label={isConnected ? '已连接' : '未连接'}
            size="small"
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
            sx={{ fontSize: '0.7rem' }}
          />
          {stats.totalRecognitions > 0 && (
            <Chip
              label={`${stats.totalRecognitions}次识别`}
              size="small"
              color="info"
              variant="outlined"
              sx={{ fontSize: '0.7rem' }}
            />
          )}
        </Stack>
      </Box>
      
      {/* 分组导航菜单 */}
      <Box sx={{ flex: 1, px: 2, py: 1, overflow: 'auto' }}>
        {navigationGroups.map((group, groupIndex) => (
          <Box key={group.title} sx={{ mb: 1 }}>
            <ListItemButton
              onClick={() => toggleGroup(group.title)}
              sx={{
                borderRadius: 2,
                mb: 0.5,
                py: 1,
                px: 2,
                backgroundColor: 'transparent',
                '&:hover': {
                  backgroundColor: 'primary.light',
                  '& .MuiTypography-root': { color: 'primary.contrastText' },
                },
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, flexGrow: 1 }}>
                {group.title}
              </Typography>
              {expandedGroups.includes(group.title) ? <ExpandLess /> : <ExpandMore />}
            </ListItemButton>
            
            <Collapse in={expandedGroups.includes(group.title)}>
              <List sx={{ pt: 0 }}>
                {group.items.map((item, index) => (
                  <SafeFade in={isMounted} timeout={200 + index * 50} key={`${group.title}-${item.text}`}>
                    <ListItem disablePadding sx={{ mb: 0.5 }}>
                      <Tooltip title={item.description} placement="right" arrow>
                        <ListItemButton
                          onClick={() => handleNavigation(item.path)}
                          selected={location.pathname === item.path}
                          sx={{
                            borderRadius: 2,
                            py: 1.5,
                            px: 2,
                            ml: 1,
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                            '&.Mui-selected': {
                              background: `linear-gradient(135deg, ${item.color}50 0%, ${item.color}30 100%)`,
                              color: 'primary.main',
                              transform: 'translateX(8px)',
                              boxShadow: `0 4px 20px ${item.color}30`,
                              border: `1px solid ${item.color}60`,
                              '& .MuiListItemIcon-root': {
                                color: 'primary.main',
                                transform: 'scale(1.1)',
                              },
                            },
                            '&:hover': {
                              transform: 'translateX(4px)',
                              backgroundColor: `${item.color}20`,
                              '& .MuiAvatar-root': {
                                transform: 'scale(1.05)',
                                boxShadow: `0 4px 12px ${item.color}40`,
                              },
                            },
                          }}
                        >
                          <ListItemIcon sx={{ minWidth: 44 }}>
                            <Avatar
                              sx={{
                                width: 32,
                                height: 32,
                                backgroundColor: location.pathname === item.path ? item.color : 'transparent',
                                color: location.pathname === item.path ? 'white' : 'text.secondary',
                                border: location.pathname !== item.path ? `2px solid ${item.color}50` : 'none',
                                transition: 'all 0.2s ease',
                              }}
                            >
                              {item.icon}
                            </Avatar>
                          </ListItemIcon>
                          <ListItemText 
                            primary={item.text} 
                            primaryTypographyProps={{
                              fontWeight: location.pathname === item.path ? 600 : 500,
                              fontSize: '0.95rem',
                            }}
                          />
                          {location.pathname === item.path && (
                            <Chip
                              label="当前"
                              size="small"
                              color="primary"
                              sx={{ fontSize: '0.6rem', height: 20 }}
                            />
                          )}
                        </ListItemButton>
                      </Tooltip>
                    </ListItem>
                  </SafeFade>
                ))}
              </List>
            </Collapse>
          </Box>
        ))}
      </Box>
      
      {/* 增强的状态信息面板 */}
      <Paper sx={{ 
        m: 2, 
        p: 2,
        borderRadius: 3,
        background: `linear-gradient(135deg, ${theme.palette.background.paper} 0%, ${theme.palette.primary.light}08 100%)`,
        border: `1px solid ${theme.palette.divider}`,
      }}>
        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600, color: 'text.primary' }}>
          系统状态
        </Typography>
        
        <Stack spacing={1.5}>
          {/* 连接状态 */}
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">
              服务器连接
            </Typography>
            <Badge 
              color={isConnected ? 'success' : 'error'} 
              variant="dot"
              sx={{ 
                '& .MuiBadge-dot': { 
                  animation: isConnected ? 'pulse 2s infinite' : 'none',
                } 
              }}
            >
              <Chip
                label={isConnected ? '正常' : '断开'}
                size="small"
                color={isConnected ? 'success' : 'error'}
                sx={{ fontSize: '0.7rem', minWidth: 60 }}
              />
            </Badge>
          </Box>
          
          {/* 识别状态 */}
          {isRecognizing && (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">
                识别状态
              </Typography>
              <Chip
                label="进行中"
                color="info"
                size="small"
                sx={{ 
                  fontSize: '0.7rem',
                  animation: 'pulse 1.5s infinite',
                }}
              />
            </Box>
          )}
          
          {/* 置信度 */}
          {confidence !== null && confidence > 0 && (
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" color="text.secondary">
                  识别置信度
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {(confidence * 100).toFixed(0)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={confidence * 100}
                color={confidence > 0.8 ? "success" : confidence > 0.6 ? "warning" : "error"}
                sx={{ 
                  height: 4, 
                  borderRadius: 2,
                  backgroundColor: 'rgba(0,0,0,0.1)',
                }}
              />
            </Box>
          )}
          
          {/* 使用统计 */}
          {stats.totalRecognitions > 0 && (
            <Box sx={{ pt: 1, borderTop: `1px solid ${theme.palette.divider}` }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                今日统计
              </Typography>
              <Stack direction="row" spacing={1}>
                <Chip
                  label={`${stats.totalRecognitions}次`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.6rem' }}
                />
                <Chip
                  label={`${Math.round(stats.averageConfidence * 100)}%准确`}
                  size="small"
                  variant="outlined"
                  color="success"
                  sx={{ fontSize: '0.6rem' }}
                />
              </Stack>
            </Box>
          )}
        </Stack>
      </Paper>
    </Box>
  )

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* 增强的顶部导航栏 */}
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: theme.zIndex.drawer + 1,
          background: `linear-gradient(135deg, ${theme.palette.background.paper} 0%, ${theme.palette.primary.light}10 100%)`,
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${theme.palette.divider}`,
          boxShadow: '0 2px 20px rgba(0,0,0,0.05)',
        }}
        elevation={0}
      >
        <Toolbar sx={{ px: { xs: 2, sm: 3 } }}>
          <IconButton
            color="inherit"
            aria-label="打开导航"
            onClick={() => setDrawerOpen(true)}
            edge="start"
            sx={{ 
              mr: 2,
              borderRadius: 2,
              p: 1.5,
              backgroundColor: 'primary.light',
              color: 'primary.contrastText',
              '&:hover': {
                backgroundColor: 'primary.main',
                transform: 'scale(1.05)',
              },
              transition: 'all 0.2s ease',
            }}
          >
            <MenuIcon />
          </IconButton>
          
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Box>
              <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 600, color: 'text.primary' }}>
                {getCurrentPageInfo.text}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: { xs: 'none', sm: 'block' } }}>
                {getCurrentPageInfo.description}
              </Typography>
            </Box>
          </Box>

          <Stack direction="row" spacing={1} alignItems="center">
            {/* 通知按钮 */}
            <Tooltip title="通知">
              <IconButton
                color="inherit"
                sx={{ 
                  borderRadius: 2,
                  position: 'relative',
                }}
              >
                <Badge badgeContent={notifications} color="error" max={9}>
                  <Notifications />
                </Badge>
              </IconButton>
            </Tooltip>
            
            {/* 帮助按钮 */}
            <Tooltip title="帮助">
              <IconButton
                color="inherit"
                onClick={() => navigate('/help')}
                sx={{ borderRadius: 2 }}
              >
                <Help />
              </IconButton>
            </Tooltip>
            
            {/* 状态指示器 */}
            <StatusIndicator 
              isConnected={isConnected}
              isRecognizing={isRecognizing}
              confidence={confidence}
            />

            {/* 用户认证区域 */}
            {isAuthenticated ? (
              <Tooltip title="个人资料">
                <IconButton
                  color="inherit"
                  onClick={() => setProfileModalOpen(true)}
                  sx={{
                    borderRadius: 2,
                    backgroundColor: 'primary.light',
                    color: 'primary.contrastText',
                    '&:hover': {
                      backgroundColor: 'primary.main',
                      transform: 'scale(1.05)',
                    },
                    transition: 'all 0.3s ease',
                    mr: 1,
                  }}
                >
                  <Avatar
                    sx={{
                      width: 28,
                      height: 28,
                      fontSize: '0.8rem',
                      bgcolor: 'primary.dark',
                    }}
                  >
                    {user?.username?.charAt(0).toUpperCase() || 'U'}
                  </Avatar>
                </IconButton>
              </Tooltip>
            ) : (
              <Tooltip title="登录">
                <IconButton
                  color="inherit"
                  onClick={() => setAuthModalOpen(true)}
                  sx={{
                    borderRadius: 2,
                    backgroundColor: 'success.light',
                    color: 'success.contrastText',
                    '&:hover': {
                      backgroundColor: 'success.main',
                      transform: 'scale(1.05)',
                    },
                    transition: 'all 0.3s ease',
                    mr: 1,
                  }}
                >
                  <Person />
                </IconButton>
              </Tooltip>
            )}

            {/* 主题切换 */}
            <Tooltip title={darkMode ? '切换到亮色模式' : '切换到暗色模式'}>
              <IconButton
                color="inherit"
                onClick={onToggleDarkMode}
                sx={{
                  borderRadius: 2,
                  backgroundColor: darkMode ? 'warning.light' : 'info.light',
                  color: darkMode ? 'warning.contrastText' : 'info.contrastText',
                  '&:hover': {
                    backgroundColor: darkMode ? 'warning.main' : 'info.main',
                    transform: 'rotate(180deg)',
                  },
                  transition: 'all 0.3s ease',
                }}
              >
                {darkMode ? <Brightness7 /> : <Brightness4 />}
              </IconButton>
            </Tooltip>
          </Stack>
        </Toolbar>
      </AppBar>

      {/* 优化的侧边导航抽屉 */}
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        ModalProps={{
          keepMounted: true,
        }}
        PaperProps={{
          sx: {
            borderRight: `1px solid ${theme.palette.divider}`,
            background: theme.palette.background.paper,
            backdropFilter: 'blur(20px)',
          }
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 2, pb: 0 }}>
          <IconButton 
            onClick={() => setDrawerOpen(false)}
            sx={{ 
              borderRadius: 2,
              backgroundColor: 'error.light',
              color: 'error.contrastText',
              '&:hover': {
                backgroundColor: 'error.main',
                transform: 'scale(1.1)',
              }
            }}
          >
            <Close />
          </IconButton>
        </Box>
        {drawerContent}
      </Drawer>

      {/* 优化的主内容区域 */}
      <Box 
        component="main" 
        sx={{ 
          flexGrow: 1, 
          pt: 12,
          minHeight: '100vh',
          background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, ${theme.palette.primary.light}05 100%)`,
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `
              radial-gradient(circle at 20% 80%, ${theme.palette.primary.light}08 0%, transparent 50%),
              radial-gradient(circle at 80% 20%, ${theme.palette.secondary.light}08 0%, transparent 50%),
              radial-gradient(circle at 40% 40%, ${theme.palette.info.light}05 0%, transparent 50%)
            `,
            pointerEvents: 'none',
            zIndex: -1,
          }
        }}
      >
        {children}
      </Box>

      {/* 添加增强的CSS动画 */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
          
          @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-3px); }
          }
          
          @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
        `}
      </style>

      {/* 认证模态框 */}
      <AuthModal
        open={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        initialMode="login"
      />

      {/* 用户资料模态框 */}
      <Dialog
        open={profileModalOpen}
        onClose={() => setProfileModalOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 3,
            maxHeight: '90vh',
          }
        }}
      >
        <UserProfile onLogout={() => setProfileModalOpen(false)} />
      </Dialog>
    </Box>
  )
}

export default Layout