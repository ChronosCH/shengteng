/**
 * 专业手语Avatar演示页面 - 替换原有的粗糙3D小人
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
  TextField,
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
  ListItemButton,
  Divider,
  Slider,
} from '@mui/material'
import {
  Person,
  PlayArrow,
  Stop,
  Settings,
  ExpandMore,
  ThreeDRotation,
  Animation,
  Tune,
  Visibility,
  Psychology,
  TouchApp,
  AutoAwesome,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import ProfessionalSignLanguageAvatar from '../components/ProfessionalSignLanguageAvatar'
import { 
  CHINESE_SIGN_LANGUAGE_LIBRARY, 
  SignLanguagePlayer,
  type SignLanguageKeypoint
} from '../data/ChineseSignLanguageLibrary'

function ProfessionalAvatarPage() {
  const [currentText, setCurrentText] = useState('你好，欢迎体验专业手语Avatar')
  const [selectedGesture, setSelectedGesture] = useState('hello')
  const [isPerforming, setIsPerforming] = useState(false)
  const [currentKeypoints, setCurrentKeypoints] = useState<{
    left?: SignLanguageKeypoint[]
    right?: SignLanguageKeypoint[]
  }>({})
  
  // 专业设置
  const [avatarSettings, setAvatarSettings] = useState({
    realisticMode: true,
    showBones: false,
    showWireframe: false,
    animationSpeed: 1.0,
    handDetail: 'high',
    lightingQuality: 'ultra',
    shadowQuality: 'high',
  })

  // 手语播放器
  const [signPlayer, setSignPlayer] = useState<SignLanguagePlayer | null>(null)

  // 初始化手语播放器
  const initializePlayer = useCallback(() => {
    if (!signPlayer) {
      const player = new SignLanguagePlayer((frame) => {
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
    const gesture = CHINESE_SIGN_LANGUAGE_LIBRARY[selectedGesture]
    
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
    greeting: Object.values(CHINESE_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'greeting'),
    daily: Object.values(CHINESE_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'daily'),
    emotion: Object.values(CHINESE_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'emotion'),
    number: Object.values(CHINESE_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'number'),
    phrase: Object.values(CHINESE_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'phrase'),
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom sx={{ fontWeight: 600 }}>
            🤖 专业手语识别Avatar
          </Typography>
          <Typography variant="h6" color="text.secondary">
            高精度3D手部建模 • 专业手语动作库 • 实时表达演示
          </Typography>
          
          <Alert severity="success" sx={{ mt: 2, maxWidth: 600, mx: 'auto' }}>
            ✨ 全新升级：告别"鸡爪"手型，体验专业级手语表达效果
          </Alert>
        </Box>
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧控制面板 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            {/* 手语词汇选择 */}
            <Fade in timeout={800}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Psychology sx={{ mr: 1 }} />
                    手语词汇库
                  </Typography>
                  
                  {Object.entries(gestureCategories).map(([category, gestures]) => (
                    <Accordion key={category} sx={{ mb: 1 }}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="subtitle2">
                          {category === 'greeting' && '问候语'}
                          {category === 'daily' && '日常用语'}
                          {category === 'emotion' && '情感表达'}
                          {category === 'number' && '数字'}
                          {category === 'phrase' && '短语'}
                          {` (${gestures.length})`}
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails sx={{ pt: 0 }}>
                        <List dense>
                          {gestures.map((gesture) => (
                            <ListItemButton
                              key={gesture.id}
                              selected={selectedGesture === gesture.id}
                              onClick={() => setSelectedGesture(gesture.id)}
                              sx={{ borderRadius: 1, mb: 0.5 }}
                            >
                              <ListItemText
                                primary={gesture.name}
                                secondary={gesture.description}
                              />
                              <Chip 
                                label={gesture.difficulty}
                                size="small"
                                color={
                                  gesture.difficulty === 'easy' ? 'success' :
                                  gesture.difficulty === 'medium' ? 'warning' : 'error'
                                }
                              />
                            </ListItemButton>
                          ))}
                        </List>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </CardContent>
              </Card>
            </Fade>

            {/* 播放控制 */}
            <Fade in timeout={1000}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <TouchApp sx={{ mr: 1 }} />
                    演示控制
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      当前选择: {CHINESE_SIGN_LANGUAGE_LIBRARY[selectedGesture]?.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {CHINESE_SIGN_LANGUAGE_LIBRARY[selectedGesture]?.description}
                    </Typography>
                  </Box>
                  
                  <Stack direction="row" spacing={2}>
                    <Button
                      variant="contained"
                      onClick={handlePlayGesture}
                      disabled={isPerforming}
                      startIcon={<PlayArrow />}
                      fullWidth
                      size="large"
                      color="primary"
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
                        icon={<Animation />}
                        size="small"
                      />
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Fade>

            {/* 专业设置 */}
            <Fade in timeout={1200}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Settings sx={{ mr: 1 }} />
                    专业设置
                  </Typography>
                  
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Visibility sx={{ mr: 1 }} />
                        <Typography>渲染质量</Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Stack spacing={2}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={avatarSettings.realisticMode}
                              onChange={(e) => handleSettingChange('realisticMode', e.target.checked)}
                            />
                          }
                          label="写实模式"
                        />
                        <FormControlLabel
                          control={
                            <Switch
                              checked={avatarSettings.showBones}
                              onChange={(e) => handleSettingChange('showBones', e.target.checked)}
                            />
                          }
                          label="显示骨骼（调试）"
                        />
                        <FormControlLabel
                          control={
                            <Switch
                              checked={avatarSettings.showWireframe}
                              onChange={(e) => handleSettingChange('showWireframe', e.target.checked)}
                            />
                          }
                          label="线框模式"
                        />
                      </Stack>
                    </AccordionDetails>
                  </Accordion>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Tune sx={{ mr: 1 }} />
                        <Typography>动画参数</Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Stack spacing={2}>
                        <Box>
                          <Typography variant="body2" gutterBottom>
                            动画速度: {avatarSettings.animationSpeed.toFixed(1)}x
                          </Typography>
                          <Slider
                            value={avatarSettings.animationSpeed}
                            onChange={(_, value) => handleSettingChange('animationSpeed', value)}
                            min={0.5}
                            max={2.0}
                            step={0.1}
                            size="small"
                          />
                        </Box>
                      </Stack>
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            </Fade>
          </Stack>
        </Grid>

        {/* 主要3D显示区域 */}
        <Grid item xs={12} lg={8}>
          <Fade in timeout={600}>
            <Paper 
              sx={{ 
                p: 3, 
                height: '800px', 
                display: 'flex', 
                flexDirection: 'column',
                background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <AutoAwesome sx={{ mr: 2, color: 'primary.main', fontSize: 28 }} />
                  <Box>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                      专业手语Avatar
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      高精度手部建模 • 实时动作捕捉 • 专业表达效果
                    </Typography>
                  </Box>
                </Box>
                <Stack direction="row" spacing={1}>
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
                </Stack>
              </Box>
              
              <Box 
                sx={{ 
                  flex: 1, 
                  minHeight: 0,
                  borderRadius: 3,
                  overflow: 'hidden',
                  position: 'relative',
                  border: '2px solid rgba(0,0,0,0.1)',
                }}
              >
                <ErrorBoundary>
                  <ProfessionalSignLanguageAvatar
                    signText={currentText}
                    isPerforming={isPerforming}
                    leftHandKeypoints={currentKeypoints.left}
                    rightHandKeypoints={currentKeypoints.right}
                    showBones={avatarSettings.showBones}
                    showWireframe={avatarSettings.showWireframe}
                    realisticMode={avatarSettings.realisticMode}
                    animationSpeed={avatarSettings.animationSpeed}
                    onAvatarReady={(avatar) => {
                      console.log('专业Avatar已就绪:', avatar)
                    }}
                    onSignComplete={(signText) => {
                      console.log('手语演示完成:', signText)
                    }}
                  />
                </ErrorBoundary>
              </Box>

              <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" color="text.secondary">
                      渲染引擎: Three.js WebGL 2.0
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" color="text.secondary">
                      手部模型: 21关键点精确建模
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4} sx={{ textAlign: { sm: 'right' } }}>
                    <Typography variant="body2" color="text.secondary">
                      动作库: {Object.keys(CHINESE_SIGN_LANGUAGE_LIBRARY).length} 个专业手语
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            </Paper>
          </Fade>
        </Grid>
      </Grid>

      {/* 功能介绍 */}
      <Fade in timeout={1400}>
        <Box sx={{ mt: 6 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            🚀 技术特色
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <ThreeDRotation sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    高精度手部建模
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    21个关键点精确映射，解剖学级别的手指关节建模，告别"鸡爪"效果
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Psychology sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    专业手语动作库
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    中国手语标准动作，精确的时序控制和动作插值，真实还原手语表达
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Animation sx={{ fontSize: 48, color: 'warning.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    实时动作识别
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    MediaPipe Holistic集成，实时手语识别与3D Avatar同步演示
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <AutoAwesome sx={{ fontSize: 48, color: 'error.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    写实渲染效果
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    PBR材质系统，环境光照，阴影效果，专业级3D渲染质量
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      </Fade>

      {/* 使用说明 */}
      <Fade in timeout={1600}>
        <Box sx={{ mt: 6 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            📖 使用指南
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    基础操作
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="选择手语词汇"
                        secondary="从左侧词汇库中选择要演示的手语动作"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="开始演示"
                        secondary="点击演示按钮观看3D Avatar执行手语动作"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="调整视角"
                        secondary="鼠标拖拽旋转，滚轮缩放，右键平移视角"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="success">
                    高级功能
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="骨骼调试模式"
                        secondary="显示手部骨骼结构，用于动作分析和调试"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="写实/卡通切换"
                        secondary="支持不同渲染风格，适应各种应用场景"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="动画速度调节"
                        secondary="0.5x到2x速度调节，便于学习和观察细节"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      </Fade>
    </Container>
  )
}

export default ProfessionalAvatarPage