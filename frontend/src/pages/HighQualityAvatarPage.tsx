/**
 * 高质量3D Avatar页面 - 专业手语演示系统
 * 解决原有的简陋3D模型和"鸡爪"手型问题
 */

import { useState, useCallback, useEffect } from 'react'
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
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Fade,
  Alert,
  List,
  ListItem,
  ListItemText,
  Slider,
  Badge,
  Divider,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Settings,
  ExpandMore,
  Animation,
  Tune,
  Visibility,
  Psychology,
  TouchApp,
  AutoAwesome,
  Upgrade,
  Lightbulb,
  Speed,
  HighQuality,
  ThreeDRotation,
  Face,
  Accessibility,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import RealisticHumanAvatar from '../components/RealisticHumanAvatar'
import { 
  PROFESSIONAL_SIGN_LANGUAGE_LIBRARY, 
  ProfessionalSignLanguagePlayer,
  type ProfessionalSignLanguageKeypoint
} from '../data/ProfessionalSignLanguageLibrary'

function HighQualityAvatarPage() {
  const [currentText, setCurrentText] = useState('体验高质量专业手语Avatar')
  const [selectedGesture, setSelectedGesture] = useState('hello')
  const [isPerforming, setIsPerforming] = useState(false)
  const [currentKeypoints, setCurrentKeypoints] = useState<{
    left?: ProfessionalSignLanguageKeypoint[]
    right?: ProfessionalSignLanguageKeypoint[]
  }>({})
  
  // 高级设置
  const [avatarSettings, setAvatarSettings] = useState({
    realisticMode: true,
    showBones: false,
    animationSpeed: 1.0,
    renderQuality: 'ultra' as 'low' | 'medium' | 'high' | 'ultra',
    lightingQuality: 'high' as 'low' | 'medium' | 'high',
    shadowQuality: 'high' as 'low' | 'medium' | 'high',
    antiAliasing: true,
    postProcessing: true,
  })

  // 手语播放器
  const [signPlayer, setSignPlayer] = useState<ProfessionalSignLanguagePlayer | null>(null)

  // 初始化专业手语播放器
  const initializePlayer = useCallback(() => {
    if (!signPlayer) {
      const player = new ProfessionalSignLanguagePlayer((frame) => {
        setCurrentKeypoints({
          left: frame.leftHand,
          right: frame.rightHand
        })
      })
      setSignPlayer(player)
      return player
    }
    return signPlayer
  }, [signPlayer])

  // 播放手语动作
  const handlePlayGesture = () => {
    const player = initializePlayer()
    const gesture = PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture]
    
    if (gesture) {
      setCurrentText(gesture.name)
      setIsPerforming(true)
      player.play(selectedGesture)
      
      // 播放完成后停止
      setTimeout(() => {
        setIsPerforming(false)
      }, gesture.duration)
    }
  }

  // 停止演示
  const handleStop = () => {
    if (signPlayer) {
      signPlayer.stop()
    }
    setIsPerforming(false)
    setCurrentKeypoints({})
  }

  // 设置变更
  const handleSettingChange = (setting: string, value: any) => {
    setAvatarSettings(prev => ({
      ...prev,
      [setting]: value,
    }))
  }

  // 手语词汇分类
  const gestureCategories = {
    greeting: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'greeting'),
    daily: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'daily'),
    emotion: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'emotion'),
    number: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'number'),
    phrase: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'phrase'),
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'success'
      case 'intermediate': return 'warning'
      case 'advanced': return 'error'
      case 'expert': return 'error'
      default: return 'default'
    }
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom sx={{ fontWeight: 600 }}>
            🤖 高质量3D手语Avatar
          </Typography>
          <Typography variant="h6" color="text.secondary">
            专业级3D建模 • 真实手部解剖 • 标准手语动作 • 写实渲染效果
          </Typography>
          
          <Alert severity="success" sx={{ mt: 2, maxWidth: 800, mx: 'auto' }}>
            ✨ 全新升级：告别简陋的"机器人"和"鸡爪"手型，体验专业级手语表达效果！
          </Alert>
        </Box>
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧控制面板 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            {/* 手语词汇选择 */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Psychology color="primary" />
                  专业手语词汇库
                </Typography>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  基于中国手语国家标准，包含标准化动作数据
                </Typography>

                {Object.entries(gestureCategories).map(([category, gestures]) => (
                  <Accordion key={category} defaultExpanded={category === 'greeting'}>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body1" sx={{ fontWeight: 500 }}>
                          {category === 'greeting' && '问候语'}
                          {category === 'daily' && '日常用语'} 
                          {category === 'emotion' && '情感表达'}
                          {category === 'number' && '数字'}
                          {category === 'phrase' && '常用短语'}
                        </Typography>
                        <Chip 
                          label={gestures.length} 
                          size="small"
                        />
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {gestures.map((gesture) => (
                          <ListItem 
                            key={gesture.id}
                            button
                            selected={selectedGesture === gesture.id}
                            onClick={() => setSelectedGesture(gesture.id)}
                            sx={{ borderRadius: 1, mb: 0.5 }}
                          >
                            <ListItemText 
                              primary={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  {gesture.name}
                                  <Chip
                                    label={gesture.difficulty}
                                    size="small"
                                    color={getDifficultyColor(gesture.difficulty) as any}
                                    variant="outlined"
                                  />
                                </Box>
                              }
                              secondary={gesture.description}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>
                ))}
              </CardContent>
            </Card>

            {/* 控制按钮 */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TouchApp color="primary" />
                  演示控制
                </Typography>

                <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<PlayArrow />}
                    onClick={handlePlayGesture}
                    disabled={isPerforming}
                    size="large"
                    sx={{ flex: 1 }}
                  >
                    开始演示
                  </Button>
                  
                  <Button
                    variant="outlined"
                    color="secondary"
                    startIcon={<Stop />}
                    onClick={handleStop}
                    disabled={!isPerforming}
                    size="large"
                    sx={{ flex: 1 }}
                  >
                    停止
                  </Button>
                </Stack>

                <Typography variant="body2" color="text.secondary">
                  当前选择: {PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture]?.name || '无'}
                </Typography>
              </CardContent>
            </Card>

            {/* 高级设置 */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Settings color="primary" />
                  高级设置
                </Typography>

                <Stack spacing={3}>
                  {/* 渲染质量 */}
                  <Box>
                    <Typography variant="body2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <HighQuality fontSize="small" />
                      渲染质量
                    </Typography>
                    <Slider
                      value={['low', 'medium', 'high', 'ultra'].indexOf(avatarSettings.renderQuality)}
                      onChange={(_, value) => handleSettingChange('renderQuality', ['low', 'medium', 'high', 'ultra'][value as number])}
                      min={0}
                      max={3}
                      step={1}
                      marks={[
                        { value: 0, label: '低' },
                        { value: 1, label: '中' },
                        { value: 2, label: '高' },
                        { value: 3, label: '超' }
                      ]}
                    />
                  </Box>

                  {/* 动画速度 */}
                  <Box>
                    <Typography variant="body2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Speed fontSize="small" />
                      动画速度: {avatarSettings.animationSpeed}x
                    </Typography>
                    <Slider
                      value={avatarSettings.animationSpeed}
                      onChange={(_, value) => handleSettingChange('animationSpeed', value)}
                      min={0.25}
                      max={2.0}
                      step={0.25}
                      marks={[
                        { value: 0.25, label: '0.25x' },
                        { value: 1, label: '1x' },
                        { value: 2, label: '2x' }
                      ]}
                    />
                  </Box>

                  <Divider />

                  {/* 高级选项 */}
                  <Stack spacing={1}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={avatarSettings.realisticMode}
                          onChange={(e) => handleSettingChange('realisticMode', e.target.checked)}
                        />
                      }
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Face fontSize="small" />
                          写实模式
                        </Box>
                      }
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={avatarSettings.antiAliasing}
                          onChange={(e) => handleSettingChange('antiAliasing', e.target.checked)}
                        />
                      }
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <AutoAwesome fontSize="small" />
                          抗锯齿
                        </Box>
                      }
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={avatarSettings.postProcessing}
                          onChange={(e) => handleSettingChange('postProcessing', e.target.checked)}
                        />
                      }
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Lightbulb fontSize="small" />
                          后处理效果
                        </Box>
                      }
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={avatarSettings.showBones}
                          onChange={(e) => handleSettingChange('showBones', e.target.checked)}
                        />
                      }
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Visibility fontSize="small" />
                          显示骨骼 (调试)
                        </Box>
                      }
                    />
                  </Stack>
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        {/* 右侧Avatar显示区域 */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ height: '80vh', position: 'relative' }}>
            <CardContent sx={{ height: '100%', p: 0 }}>
              {/* 状态指示器 */}
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
                  border: '1px solid rgba(0,0,0,0.1)',
                  minWidth: 200
                }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                    🎭 高质量手语Avatar 2.0
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    专业建模 • 真实手部 • 标准动作
                  </Typography>
                </Box>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {isPerforming && (
                    <Chip 
                      label="实时演示"
                      color="success"
                      icon={<Animation />}
                      size="small"
                    />
                  )}
                  <Chip 
                    label={`${avatarSettings.animationSpeed}x速度`}
                    variant="outlined"
                    size="small"
                  />
                  <Chip 
                    label={avatarSettings.realisticMode ? "写实模式" : "卡通模式"}
                    variant="outlined"
                    size="small"
                  />
                  <Chip 
                    label={`${avatarSettings.renderQuality}质量`}
                    variant="outlined"
                    size="small"
                  />
                </Stack>
              </Box>
              
              <Box 
                sx={{ 
                  width: '100%',
                  height: '100%',
                  borderRadius: 3,
                  overflow: 'hidden',
                  position: 'relative',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                }}
              >
                <ErrorBoundary>
                  <RealisticHumanAvatar
                    signText={currentText}
                    isPerforming={isPerforming}
                    leftHandKeypoints={currentKeypoints.left}
                    rightHandKeypoints={currentKeypoints.right}
                    showBones={avatarSettings.showBones}
                    realisticMode={avatarSettings.realisticMode}
                    animationSpeed={avatarSettings.animationSpeed}
                    onAvatarReady={(avatar) => {
                      console.log('高质量Avatar已就绪:', avatar)
                    }}
                    onSignComplete={(signText) => {
                      console.log('手语演示完成:', signText)
                    }}
                  />
                </ErrorBoundary>
              </Box>

              {/* 技术信息 */}
              <Box sx={{ 
                position: 'absolute', 
                bottom: 16, 
                right: 16, 
                zIndex: 10,
                bgcolor: 'rgba(0, 0, 0, 0.7)',
                backdropFilter: 'blur(10px)',
                borderRadius: 2,
                p: 2,
                color: 'white'
              }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2">
                      渲染引擎: Three.js WebGL 2.0
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2">
                      手部模型: 21关键点解剖建模
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4} sx={{ textAlign: { sm: 'right' } }}>
                    <Typography variant="body2">
                      动作库: 中国手语国家标准
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            </CardContent>
          </Card>

          {/* 当前手语信息 */}
          {selectedGesture && PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture] && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Accessibility color="primary" />
                  当前手语: {PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].name}
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      动作要领:
                    </Typography>
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
                      {PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].instruction}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      常见错误:
                    </Typography>
                    <Stack spacing={0.5}>
                      {PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].commonMistakes.map((mistake, index) => (
                        <Typography key={index} variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          ⚠️ {mistake}
                        </Typography>
                      ))}
                    </Stack>
                  </Grid>
                </Grid>

                <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    <Chip 
                      label={`难度: ${PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].difficulty}`}
                      color={getDifficultyColor(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].difficulty) as any}
                      size="small"
                    />
                    <Chip 
                      label={`时长: ${PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].duration}ms`}
                      variant="outlined"
                      size="small"
                    />
                    <Chip 
                      label={`区域: ${PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].metadata.region}`}
                      variant="outlined"
                      size="small"
                    />
                    <Chip 
                      label={`频率: ${PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[selectedGesture].metadata.frequency}`}
                      variant="outlined"
                      size="small"
                    />
                  </Stack>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Container>
  )
}

export default HighQualityAvatarPage
