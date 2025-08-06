/**
 * 优化的首页组件 - 增强用户体验和视觉设计
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
  Fade,
  Grow,
  Paper,
  Chip,
  IconButton,
  useTheme,
  alpha,
  Slide,
} from '@mui/material'
import {
  PlayArrow,
  School,
  Analytics,
  Visibility,
  Psychology,
  TrendingUp,
  Star,
  ArrowForward,
  Speed,
  Security,
  CloudUpload,
  Groups,
  EmojiEvents,
  Lightbulb,
  AutoAwesome,
  Rocket,
} from '@mui/icons-material'
import { keyframes } from '@emotion/react'

// 动画定义
const float = keyframes`
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
`

const pulse = keyframes`
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.05); opacity: 0.8; }
`

const slideIn = keyframes`
  from { transform: translateX(-100px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
`

function HomePage() {
  const theme = useTheme()
  const [visibleFeatures, setVisibleFeatures] = useState(0)
  const [statsVisible, setStatsVisible] = useState(false)

  // 主要功能特性 - 移到useEffect之前
  const features = [
    {
      icon: <Visibility />,
      title: '实时手语识别',
      description: '基于深度学习的实时手语动作识别，准确率高达95%',
      color: '#B5EAD7',
      path: '/recognition',
      gradient: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
    },
    {
      icon: <School />,
      title: '智能学习训练',
      description: '个性化学习路径，从零基础到专业水平的系统化训练',
      color: '#FFDAB9',
      path: '/learning',
      gradient: 'linear-gradient(135deg, #FFDAB9 0%, #FFE7CC 100%)',
    },
    {
      icon: <Psychology />,
      title: '3D Avatar演示',
      description: '逼真的3D虚拟人物实时演示手语动作，提升学习体验',
      color: '#FFB3BA',
      path: '/avatar',
      gradient: 'linear-gradient(135deg, #FFB3BA 0%, #FF9AA2 100%)',
    },
    {
      icon: <Analytics />,
      title: '数据分析洞察',
      description: '详细的学习进度分析和个人能力评估报告',
      color: '#C7CEDB',
      path: '/analytics',
      gradient: 'linear-gradient(135deg, #C7CEDB 0%, #D6DCE5 100%)',
    },
  ]

  useEffect(() => {
    // 逐个显示功能特性
    const timer = setInterval(() => {
      setVisibleFeatures(prev => {
        if (prev < features.length) {
          return prev + 1
        }
        clearInterval(timer)
        return prev
      })
    }, 300)

    // 延迟显示统计数据
    setTimeout(() => setStatsVisible(true), 1500)

    return () => clearInterval(timer)
  }, [features.length])

  // 技术亮点
  const highlights = [
    {
      icon: <Speed />,
      title: '毫秒级响应',
      description: '先进的算法优化，实现超低延迟的实时处理',
      color: '#B5EAD7',
    },
    {
      icon: <Security />,
      title: '隐私保护',
      description: '本地化处理，用户数据安全得到充分保障',
      color: '#FFDAB9',
    },
    {
      icon: <CloudUpload />,
      title: '云端同步',
      description: '学习进度云端备份，随时随地继续学习',
      color: '#FFB3BA',
    },
    {
      icon: <Groups />,
      title: '社区互动',
      description: '与全球手语学习者交流分享学习心得',
      color: '#C7CEDB',
    },
  ]

  // 统计数据
  const stats = [
    { label: '用户总数', value: '50,000+', icon: <Groups />, color: '#B5EAD7' },
    { label: '识别准确率', value: '95%', icon: <TrendingUp />, color: '#FFDAB9' },
    { label: '学习课程', value: '200+', icon: <School />, color: '#FFB3BA' },
    { label: '用户满意度', value: '4.9/5', icon: <Star />, color: '#C7CEDB' },
  ]

  const handleGetStarted = () => {
    window.location.href = '/recognition'
  }

  const handleLearnMore = () => {
    window.location.href = '/learning'
  }

  return (
    <Box sx={{ minHeight: '100vh', overflow: 'hidden' }}>
      {/* 英雄区域 */}
      <Container maxWidth="xl" sx={{ pt: 8, pb: 12 }}>
        <Grid container spacing={6} alignItems="center" sx={{ minHeight: '70vh' }}>
          {/* 左侧内容 */}
          <Grid item xs={12} lg={6}>
            <Fade in timeout={800}>
              <Box>
                <Chip
                  label="🚀 SignAvatar 2.0 正式发布"
                  sx={{
                    mb: 3,
                    px: 2,
                    py: 1,
                    fontSize: '0.9rem',
                    fontWeight: 600,
                    background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                    color: 'white',
                    '&:hover': { background: 'linear-gradient(135deg, #9BC1BC 0%, #B5EAD7 100%)' },
                    animation: `${pulse} 3s infinite`,
                  }}
                />
                
                <Typography
                  variant="h1"
                  sx={{
                    fontSize: { xs: '2.5rem', md: '3.5rem', lg: '4rem' },
                    fontWeight: 800,
                    lineHeight: 1.1,
                    mb: 3,
                    background: 'linear-gradient(135deg, #2D3748 0%, #4A5568 50%, #718096 100%)',
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  重新定义
                  <br />
                  <Box
                    component="span"
                    sx={{
                      background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      position: 'relative',
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        bottom: '-8px',
                        left: 0,
                        right: 0,
                        height: '4px',
                        background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                        borderRadius: '2px',
                      },
                    }}
                  >
                    手语交流
                  </Box>
                </Typography>

                <Typography
                  variant="h5"
                  color="text.secondary"
                  sx={{
                    mb: 4,
                    lineHeight: 1.6,
                    maxWidth: '600px',
                    fontWeight: 400,
                  }}
                >
                  通过人工智能技术，让手语学习变得简单高效。
                  实时识别、3D演示、个性化训练，开启全新的无障碍沟通体验。
                </Typography>

                <Stack
                  direction={{ xs: 'column', sm: 'row' }}
                  spacing={3}
                  sx={{ mb: 6 }}
                >
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={<PlayArrow />}
                    onClick={handleGetStarted}
                    sx={{
                      px: 4,
                      py: 2,
                      fontSize: '1.1rem',
                      fontWeight: 600,
                      borderRadius: 4,
                      background: 'linear-gradient(135deg, #B5EAD7 0%, #9BC1BC 100%)',
                      boxShadow: '0 8px 25px rgba(181, 234, 215, 0.4)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #9BC1BC 0%, #8CB0A6 100%)',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 12px 35px rgba(181, 234, 215, 0.5)',
                      },
                      transition: 'all 0.3s ease',
                    }}
                  >
                    立即体验
                  </Button>
                  
                  <Button
                    variant="outlined"
                    size="large"
                    endIcon={<ArrowForward />}
                    onClick={handleLearnMore}
                    sx={{
                      px: 4,
                      py: 2,
                      fontSize: '1.1rem',
                      fontWeight: 600,
                      borderRadius: 4,
                      borderColor: '#B5EAD7',
                      color: '#B5EAD7',
                      '&:hover': {
                        borderColor: '#9BC1BC',
                        color: '#9BC1BC',
                        background: 'rgba(181, 234, 215, 0.05)',
                        transform: 'translateY(-2px)',
                      },
                      transition: 'all 0.3s ease',
                    }}
                  >
                    了解更多
                  </Button>
                </Stack>

                {/* 快速统计 */}
                <Slide in={statsVisible} direction="up" timeout={800}>
                  <Grid container spacing={4}>
                    {stats.map((stat, index) => (
                      <Grid item xs={6} sm={3} key={stat.label}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Avatar
                            sx={{
                              width: 48,
                              height: 48,
                              bgcolor: stat.color,
                              mx: 'auto',
                              mb: 1,
                              boxShadow: `0 4px 15px ${stat.color}40`,
                            }}
                          >
                            {stat.icon}
                          </Avatar>
                          <Typography
                            variant="h4"
                            sx={{
                              fontWeight: 700,
                              color: 'text.primary',
                              mb: 0.5,
                            }}
                          >
                            {stat.value}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {stat.label}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </Slide>
              </Box>
            </Fade>
          </Grid>

          {/* 右侧视觉元素 */}
          <Grid item xs={12} lg={6}>
            <Fade in timeout={1200}>
              <Box
                sx={{
                  position: 'relative',
                  height: { xs: '300px', md: '500px' },
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {/* 背景装饰圆圈 */}
                <Box
                  sx={{
                    position: 'absolute',
                    width: '400px',
                    height: '400px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #B5EAD720 0%, #C7F0DB10 100%)',
                    animation: `${float} 6s ease-in-out infinite`,
                    zIndex: 1,
                  }}
                />
                <Box
                  sx={{
                    position: 'absolute',
                    width: '300px',
                    height: '300px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #FFDAB920 0%, #FFE7CC10 100%)',
                    animation: `${float} 4s ease-in-out infinite 1s`,
                    zIndex: 2,
                  }}
                />
                
                {/* 中心主要元素 */}
                <Paper
                  elevation={0}
                  sx={{
                    width: '200px',
                    height: '200px',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                    boxShadow: '0 20px 60px rgba(181, 234, 215, 0.4)',
                    zIndex: 3,
                    position: 'relative',
                    animation: `${pulse} 4s ease-in-out infinite`,
                  }}
                >
                  <AutoAwesome
                    sx={{
                      fontSize: 80,
                      color: 'white',
                      filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))',
                    }}
                  />
                </Paper>

                {/* 浮动装饰元素 */}
                { [
                  {
                    icon: <Visibility />,
                    top: '10%',
                    left: '10%',
                    delay: 0,
                  },
                  {
                    icon: <School />,
                    top: '20%',
                    right: '10%',
                    delay: 1,
                  },
                  {
                    icon: <Psychology />,
                    bottom: '20%',
                    left: '15%',
                    delay: 2,
                  },
                  {
                    icon: <Analytics />,
                    bottom: '10%',
                    right: '15%',
                    delay: 3,
                  }
                ].map((item, index) => (
                  <Grow
                    key={index}
                    in
                    timeout={1000}
                    style={{ transitionDelay: `${1500 + item.delay * 200}ms` }}
                  >
                    <Paper
                      elevation={0}
                      sx={{
                        position: 'absolute',
                        top: item.top,
                        left: item.left,
                        right: item.right,
                        bottom: item.bottom,
                        width: 56,
                        height: 56,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: `linear-gradient(135deg, ${features[index].color} 0%, ${features[index].color}CC 100%)`,
                        boxShadow: `0 8px 25px ${features[index].color}40`,
                        zIndex: 4,
                        animation: `${float} ${3 + index}s ease-in-out infinite ${item.delay}s`,
                        cursor: 'pointer',
                        '&:hover': {
                          transform: 'scale(1.1)',
                          boxShadow: `0 12px 35px ${features[index].color}60`,
                        },
                        transition: 'all 0.3s ease',
                      }}
                    >
                      <Box sx={{ color: 'white', fontSize: 24 }}>
                        {item.icon}
                      </Box>
                    </Paper>
                  </Grow>
                ))}
              </Box>
            </Fade>
          </Grid>
        </Grid>
      </Container>

      {/* 功能特性展示 */}
      <Container maxWidth="xl" sx={{ py: 8 }}>
        <Fade in timeout={1000}>
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography
              variant="h2"
              sx={{
                fontWeight: 700,
                mb: 3,
                background: 'linear-gradient(135deg, #2D3748 0%, #4A5568 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              核心功能
            </Typography>
            <Typography
              variant="h6"
              color="text.secondary"
              sx={{ maxWidth: '600px', mx: 'auto', lineHeight: 1.6 }}
            >
              基于最新AI技术，提供全方位的手语学习和识别解决方案
            </Typography>
          </Box>
        </Fade>

        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} lg={3} key={feature.title}>
              <Grow
                in={index < visibleFeatures}
                timeout={800}
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <Card
                  sx={{
                    height: '100%',
                    borderRadius: 4,
                    background: `linear-gradient(135deg, ${feature.color}15 0%, ${feature.color}05 100%)`,
                    border: `1px solid ${feature.color}30`,
                    transition: 'all 0.4s ease',
                    cursor: 'pointer',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: `0 15px 40px ${feature.color}30`,
                      background: `linear-gradient(135deg, ${feature.color}20 0%, ${feature.color}10 100%)`,
                    },
                  }}
                  onClick={() => window.location.href = feature.path}
                >
                  <CardContent sx={{ p: 4, textAlign: 'center' }}>
                    <Avatar
                      sx={{
                        width: 80,
                        height: 80,
                        mx: 'auto',
                        mb: 3,
                        background: feature.gradient,
                        boxShadow: `0 8px 25px ${feature.color}40`,
                      }}
                    >
                      <Box sx={{ color: 'white', fontSize: 40 }}>
                        {feature.icon}
                      </Box>
                    </Avatar>
                    
                    <Typography
                      variant="h5"
                      sx={{
                        fontWeight: 600,
                        mb: 2,
                        color: 'text.primary',
                      }}
                    >
                      {feature.title}
                    </Typography>
                    
                    <Typography
                      variant="body1"
                      color="text.secondary"
                      sx={{
                        lineHeight: 1.6,
                        mb: 3,
                      }}
                    >
                      {feature.description}
                    </Typography>

                    <Button
                      variant="text"
                      endIcon={<ArrowForward />}
                      sx={{
                        color: feature.color,
                        fontWeight: 600,
                        '&:hover': {
                          background: `${feature.color}10`,
                        },
                      }}
                    >
                      了解详情
                    </Button>
                  </CardContent>
                </Card>
              </Grow>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* 技术亮点 */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 100%)',
          py: 8,
        }}
      >
        <Container maxWidth="xl">
          <Fade in timeout={1200}>
            <Box sx={{ textAlign: 'center', mb: 6 }}>
              <Typography
                variant="h3"
                sx={{
                  fontWeight: 700,
                  mb: 2,
                  color: 'text.primary',
                }}
              >
                技术优势
              </Typography>
              <Typography variant="h6" color="text.secondary">
                领先的技术实现，卓越的用户体验
              </Typography>
            </Box>
          </Fade>

          <Grid container spacing={4}>
            {highlights.map((highlight, index) => (
              <Grid item xs={12} sm={6} md={3} key={highlight.title}>
                <Fade in timeout={1000} style={{ transitionDelay: `${(index + 1) * 200}ms` }}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 4,
                      height: '100%',
                      textAlign: 'center',
                      borderRadius: 4,
                      background: 'white',
                      border: '1px solid',
                      borderColor: 'divider',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: `0 12px 30px ${highlight.color}20`,
                        borderColor: `${highlight.color}40`,
                      },
                    }}
                  >
                    <Avatar
                      sx={{
                        width: 64,
                        height: 64,
                        mx: 'auto',
                        mb: 2,
                        bgcolor: highlight.color,
                        boxShadow: `0 6px 20px ${highlight.color}40`,
                      }}
                    >
                      {highlight.icon}
                    </Avatar>
                    
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 600,
                        mb: 1,
                        color: 'text.primary',
                      }}
                    >
                      {highlight.title}
                    </Typography>
                    
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ lineHeight: 1.6 }}
                    >
                      {highlight.description}
                    </Typography>
                  </Paper>
                </Fade>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* CTA区域 */}
      <Container maxWidth="xl" sx={{ py: 10 }}>
        <Fade in timeout={1400}>
          <Paper
            elevation={0}
            sx={{
              p: 8,
              textAlign: 'center',
              borderRadius: 6,
              background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
              color: 'white',
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: -50,
                right: -50,
                width: 100,
                height: 100,
                background: 'radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%)',
                borderRadius: '50%',
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                bottom: -30,
                left: -30,
                width: 80,
                height: 80,
                background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
                borderRadius: '50%',
              },
            }}
          >
            <Box sx={{ position: 'relative', zIndex: 1 }}>
              <Avatar
                sx={{
                  width: 80,
                  height: 80,
                  mx: 'auto',
                  mb: 3,
                  bgcolor: 'rgba(255,255,255,0.2)',
                  backdropFilter: 'blur(10px)',
                }}
              >
                <Rocket sx={{ fontSize: 40 }} />
              </Avatar>
              
              <Typography
                variant="h3"
                sx={{
                  fontWeight: 700,
                  mb: 2,
                }}
              >
                开始您的手语学习之旅
              </Typography>
              
              <Typography
                variant="h6"
                sx={{
                  mb: 4,
                  opacity: 0.9,
                  maxWidth: '600px',
                  mx: 'auto',
                  lineHeight: 1.6,
                }}
              >
                加入我们的社区，与全球数万名学习者一起，探索手语的魅力，
                建立更广阔的沟通桥梁。
              </Typography>
              
              <Stack
                direction={{ xs: 'column', sm: 'row' }}
                spacing={3}
                justifyContent="center"
              >
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<PlayArrow />}
                  onClick={handleGetStarted}
                  sx={{
                    px: 5,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 600,
                    borderRadius: 4,
                    bgcolor: 'white',
                    color: '#B5EAD7',
                    boxShadow: '0 8px 25px rgba(255,255,255,0.3)',
                    '&:hover': {
                      bgcolor: 'rgba(255,255,255,0.95)',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 12px 35px rgba(255,255,255,0.4)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  免费开始学习
                </Button>
                
                <Button
                  variant="outlined"
                  size="large"
                  endIcon={<EmojiEvents />}
                  sx={{
                    px: 5,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 600,
                    borderRadius: 4,
                    borderColor: 'rgba(255,255,255,0.5)',
                    color: 'white',
                    '&:hover': {
                      borderColor: 'white',
                      background: 'rgba(255,255,255,0.1)',
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  查看成就系统
                </Button>
              </Stack>
            </Box>
          </Paper>
        </Fade>
      </Container>
    </Box>
  )
}

export default HomePage