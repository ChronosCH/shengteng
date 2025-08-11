/**
 * Avatar对比展示页面
 * 同时展示不同版本的Avatar，方便对比效果
 */

import { useState, useCallback } from 'react'
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
  Tabs,
  Tab,
  Fade,
  Alert,
  List,
  ListItem,
  ListItemText,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Compare,
  Star,
  Diamond,
  AutoAwesome,
  Person,
  Face,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import ThreeAvatar from '../components/ThreeAvatar'
import RealisticHumanAvatar from '../components/RealisticHumanAvatar'
import AdvancedRealisticAvatar from '../components/AdvancedRealisticAvatar'
import { 
  PROFESSIONAL_SIGN_LANGUAGE_LIBRARY, 
  ProfessionalSignLanguagePlayer,
  type ProfessionalSignLanguageKeypoint
} from '../data/ProfessionalSignLanguageLibrary'

function AvatarComparisonPage() {
  const [selectedGesture, setSelectedGesture] = useState('hello')
  const [currentText, setCurrentText] = useState('你好')
  const [isPerforming, setIsPerforming] = useState(false)
  const [currentKeypoints, setCurrentKeypoints] = useState<{
    left: ProfessionalSignLanguageKeypoint[]
    right: ProfessionalSignLanguageKeypoint[]
  }>({ left: [], right: [] })
  const [selectedTab, setSelectedTab] = useState(0)

  const player = new ProfessionalSignLanguagePlayer()

  // 播放手语手势
  const handlePlayGesture = useCallback(() => {
    if (isPerforming) return

    const gesture = PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture]
    if (!gesture) return

    setCurrentText(gesture.name)
    setIsPerforming(true)

    // 播放手语动作序列
    player.playSequence(
      gesture.keyframes,
      gesture.duration,
      (frame) => {
        setCurrentKeypoints({
          left: frame.leftHand || [],
          right: frame.rightHand || []
        })
      },
      () => {
        setIsPerforming(false)
      }
    )
  }, [selectedGesture, isPerforming, player])

  const handleStop = useCallback(() => {
    setIsPerforming(false)
    setCurrentKeypoints({ left: [], right: [] })
  }, [])

  const avatarVersions = [
    {
      id: 'basic',
      name: '基础版本',
      icon: <Person />,
      color: '#FFDAB9',
      component: ThreeAvatar,
      features: ['简单几何建模', '基础材质', '简单动画', '火柴人造型'],
      description: '最初版本，使用简单的几何形状拼接'
    },
    {
      id: 'realistic',
      name: '写实版本',
      icon: <AutoAwesome />,
      color: '#FFE4B5',
      component: RealisticHumanAvatar,
      features: ['改进建模', 'PBR材质', '面部特征', '人体比例'],
      description: '第二代版本，增加了基本的人体特征'
    },
    {
      id: 'advanced',
      name: '真人级版本',
      icon: <Diamond />,
      color: '#98FB98',
      component: AdvancedRealisticAvatar,
      features: ['解剖学建模', '次表面散射', '微表情', '电影级渲染'],
      description: '最新版本，真人级别的3D Avatar'
    }
  ]

  const CurrentAvatar = avatarVersions[selectedTab].component

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 2, justifyContent: 'center' }}>
            <Compare sx={{ fontSize: 40, color: 'primary.main' }} />
            Avatar进化对比展示
          </Typography>
          
          <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
            从简单几何体到真人级建模的进化历程
          </Typography>
          
          <Alert 
            severity="info" 
            sx={{ 
              mt: 2, 
              maxWidth: 800, 
              mx: 'auto',
              borderRadius: 3
            }}
            icon={<Star />}
          >
            <Typography variant="body1" sx={{ fontWeight: 500 }}>
              🚀 见证Avatar系统的完整进化过程 - 从"鸡爪"手型到真人级别的突破
            </Typography>
          </Alert>
        </Box>
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧控制面板 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            {/* 版本选择 */}
            <Card>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <Face sx={{ mr: 1 }} />
                  Avatar版本
                </Typography>
                
                <Tabs
                  value={selectedTab}
                  onChange={(_, newValue) => setSelectedTab(newValue)}
                  orientation="vertical"
                  sx={{ width: '100%' }}
                >
                  {avatarVersions.map((version, index) => (
                    <Tab
                      key={version.id}
                      icon={version.icon}
                      label={
                        <Box sx={{ textAlign: 'left', ml: 1 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {version.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {version.description}
                          </Typography>
                        </Box>
                      }
                      sx={{
                        flexDirection: 'row',
                        justifyContent: 'flex-start',
                        alignItems: 'center',
                        textAlign: 'left',
                        minHeight: 80,
                        border: '2px solid transparent',
                        borderRadius: 2,
                        mb: 1,
                        transition: 'all 0.3s ease',
                        '&.Mui-selected': {
                          backgroundColor: `${version.color}30`,
                          borderColor: version.color,
                          color: 'text.primary'
                        },
                        '&:hover': {
                          backgroundColor: `${version.color}20`,
                        }
                      }}
                    />
                  ))}
                </Tabs>
              </CardContent>
            </Card>

            {/* 当前版本特性 */}
            <Card sx={{ border: `2px solid ${avatarVersions[selectedTab].color}` }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  {avatarVersions[selectedTab].icon}
                  <Box sx={{ ml: 1 }}>
                    {avatarVersions[selectedTab].name}
                  </Box>
                </Typography>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {avatarVersions[selectedTab].description}
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  核心特性：
                </Typography>
                <List dense>
                  {avatarVersions[selectedTab].features.map((feature, index) => (
                    <ListItem key={index} sx={{ py: 0.5, px: 0 }}>
                      <ListItemText 
                        primary={feature}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>

            {/* 演示控制 */}
            <Card>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  演示控制
                </Typography>
                
                {/* 快速手势选择 */}
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  {['hello', 'thank_you', 'i_love_you', 'goodbye'].map((id) => {
                    const gesture = PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[id]
                    return (
                      <Grid item xs={6} key={id}>
                        <Button
                          fullWidth
                          variant={selectedGesture === id ? "contained" : "outlined"}
                          onClick={() => {
                            setSelectedGesture(id)
                            if (!isPerforming) {
                              handlePlayGesture()
                            }
                          }}
                          startIcon={<span style={{ fontSize: '16px' }}>{gesture?.icon || '👋'}</span>}
                          sx={{ 
                            py: 1,
                            flexDirection: 'column',
                            gap: 0.5,
                            height: 60
                          }}
                        >
                          <Typography variant="caption">
                            {gesture?.name || id}
                          </Typography>
                        </Button>
                      </Grid>
                    )
                  })}
                </Grid>

                {/* 控制按钮 */}
                <Stack direction="row" spacing={2}>
                  <Button
                    variant="contained"
                    onClick={handlePlayGesture}
                    disabled={isPerforming}
                    startIcon={<PlayArrow />}
                    fullWidth
                    size="large"
                  >
                    开始演示
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleStop}
                    disabled={!isPerforming}
                    startIcon={<Stop />}
                    color="error"
                  >
                    停止
                  </Button>
                </Stack>
                
                {isPerforming && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="primary" gutterBottom>
                      正在演示: {currentText}
                    </Typography>
                    <Chip 
                      label="演示中"
                      color="success"
                      size="small"
                      sx={{ 
                        animation: 'pulse 2s infinite',
                        '@keyframes pulse': {
                          '0%': { opacity: 1 },
                          '50%': { opacity: 0.7 },
                          '100%': { opacity: 1 },
                        }
                      }}
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        {/* 右侧Avatar显示区域 */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ height: '80vh', position: 'relative' }}>
            <CardContent sx={{ height: '100%', p: 0 }}>
              {/* 版本指示器 */}
              <Box sx={{ 
                position: 'absolute', 
                top: 16, 
                left: 16, 
                zIndex: 10,
                display: 'flex',
                flexDirection: 'column',
                gap: 1
              }}>
                <Box sx={{
                  bgcolor: 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: 2,
                  p: 2,
                  border: `2px solid ${avatarVersions[selectedTab].color}`,
                  minWidth: 200
                }}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    {avatarVersions[selectedTab].icon}
                    <Box>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {avatarVersions[selectedTab].name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {avatarVersions[selectedTab].description}
                      </Typography>
                    </Box>
                  </Stack>
                </Box>
              </Box>

              {/* Avatar显示区域 */}
              <Box 
                sx={{ 
                  width: '100%',
                  height: '100%',
                  borderRadius: 3,
                  overflow: 'hidden',
                  position: 'relative',
                  background: `linear-gradient(135deg, ${avatarVersions[selectedTab].color}30 0%, ${avatarVersions[selectedTab].color}10 100%)`,
                }}
              >
                <ErrorBoundary>
                  {selectedTab === 0 && (
                    <ThreeAvatar
                      text={currentText}
                      isActive={isPerforming}
                      animationType="手语"
                      leftHandKeypoints={currentKeypoints.left}
                      rightHandKeypoints={currentKeypoints.right}
                    />
                  )}
                  {selectedTab === 1 && (
                    <RealisticHumanAvatar
                      signText={currentText}
                      isPerforming={isPerforming}
                      leftHandKeypoints={currentKeypoints.left}
                      rightHandKeypoints={currentKeypoints.right}
                      realisticMode={true}
                    />
                  )}
                  {selectedTab === 2 && (
                    <AdvancedRealisticAvatar
                      signText={currentText}
                      isPerforming={isPerforming}
                      leftHandKeypoints={currentKeypoints.left}
                      rightHandKeypoints={currentKeypoints.right}
                      realisticMode={true}
                    />
                  )}
                </ErrorBoundary>
              </Box>

              {/* 版本对比信息 */}
              <Box sx={{ 
                position: 'absolute', 
                bottom: 16, 
                right: 16, 
                zIndex: 10
              }}>
                <Box sx={{
                  bgcolor: 'rgba(0, 0, 0, 0.8)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: 2,
                  p: 2,
                  color: 'white',
                  minWidth: 180
                }}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
                    技术规格
                  </Typography>
                  <Typography variant="caption" sx={{ opacity: 0.9, lineHeight: 1.4 }}>
                    {selectedTab === 0 && "• 基础几何建模\n• 简单材质系统\n• 固定动画"}
                    {selectedTab === 1 && "• 改进人体建模\n• PBR材质系统\n• 面部特征"}
                    {selectedTab === 2 && "• 解剖学精度\n• 次表面散射\n• 电影级渲染"}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* 进化历程总结 */}
      <Fade in timeout={1600}>
        <Box sx={{ mt: 6 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, textAlign: 'center' }}>
            🚀 Avatar进化历程
          </Typography>
          <Grid container spacing={3}>
            {avatarVersions.map((version, index) => (
              <Grid item xs={12} md={4} key={version.id}>
                <Card 
                  sx={{ 
                    height: '100%', 
                    border: `2px solid ${version.color}`,
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: `0 8px 25px ${version.color}40`
                    }
                  }}
                  onClick={() => setSelectedTab(index)}
                >
                  <CardContent sx={{ textAlign: 'center', py: 4 }}>
                    <Box sx={{ mb: 2 }}>
                      {version.icon}
                    </Box>
                    <Typography variant="h6" gutterBottom>
                      {version.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {version.description}
                    </Typography>
                    <List dense>
                      {version.features.map((feature, idx) => (
                        <ListItem key={idx} sx={{ justifyContent: 'center', py: 0.5 }}>
                          <ListItemText 
                            primary={feature}
                            primaryTypographyProps={{ 
                              variant: 'caption',
                              textAlign: 'center'
                            }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      </Fade>
    </Container>
  )
}

export default AvatarComparisonPage
