/**
 * 升级版3D手语Avatar页面 - 使用专业手语系统
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
  ListItemButton,
  Slider,
  Badge,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  Person,
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
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import ProfessionalSignLanguageAvatar from '../components/ProfessionalSignLanguageAvatar'
import { 
  CHINESE_SIGN_LANGUAGE_LIBRARY, 
  SignLanguagePlayer,
  type SignLanguageKeypoint
} from '../data/ChineseSignLanguageLibrary'

function AvatarPage() {
  const [currentText, setCurrentText] = useState('体验全新专业手语Avatar')
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

  // 快速演示
  const quickDemos = [
    { id: 'hello', name: '你好', icon: '👋' },
    { id: 'thank_you', name: '谢谢', icon: '🙏' },
    { id: 'i_love_you', name: '我爱你', icon: '❤️' },
    { id: 'goodbye', name: '再见', icon: '👋' },
  ]

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题和升级提示 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Badge badgeContent="升级版" color="primary" sx={{ mb: 2 }}>
            <Typography variant="h3" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 2, justifyContent: 'center' }}>
              <AutoAwesome sx={{ fontSize: 40, color: 'primary.main' }} />
              3D手语Avatar - 专业版
            </Typography>
          </Badge>
          
          <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
            高精度手部建模 • 告别"鸡爪"效果 • 专业手语表达
          </Typography>
          
          <Alert 
            severity="success" 
            sx={{ 
              mt: 2, 
              maxWidth: 800, 
              mx: 'auto',
              background: 'linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 100%)',
              border: '1px solid #4caf50'
            }}
            icon={<Upgrade />}
          >
            <Typography variant="body1" sx={{ fontWeight: 500 }}>
              🎉 系统已升级！全新专业级3D手语Avatar，解决了粗糙建模问题，现在拥有解剖学级精度
            </Typography>
          </Alert>
        </Box>
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧控制面板 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            {/* 快速演示 */}
            <Fade in timeout={800}>
              <Card sx={{ background: 'linear-gradient(135deg, #fff 0%, #f8f9fa 100%)' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Lightbulb sx={{ mr: 1, color: 'warning.main' }} />
                    快速体验
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {quickDemos.map((demo) => (
                      <Grid item xs={6} key={demo.id}>
                        <Button
                          fullWidth
                          variant={selectedGesture === demo.id ? "contained" : "outlined"}
                          onClick={() => {
                            setSelectedGesture(demo.id)
                            if (!isPerforming) {
                              handlePlayGesture()
                            }
                          }}
                          startIcon={<span style={{ fontSize: '20px' }}>{demo.icon}</span>}
                          sx={{ 
                            py: 1.5,
                            flexDirection: 'column',
                            gap: 0.5,
                            height: 80
                          }}
                        >
                          <Typography variant="body2">{demo.name}</Typography>
                        </Button>
                      </Grid>
                    ))}
                  </Grid>
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
            </Fade>

            {/* 高级设置 */}
            <Fade in timeout={1200}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Settings sx={{ mr: 1 }} />
                    高级设置
                  </Typography>
                  
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Visibility sx={{ mr: 1 }} />
                        <Typography>视觉效果</Typography>
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
                          label="显示骨骼结构"
                        />
                      </Stack>
                    </AccordionDetails>
                  </Accordion>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Speed sx={{ mr: 1 }} />
                        <Typography>动画控制</Typography>
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
                            marks={[
                              { value: 0.5, label: '0.5x' },
                              { value: 1.0, label: '1x' },
                              { value: 2.0, label: '2x' }
                            ]}
                          />
                        </Box>
                      </Stack>
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            </Fade>

            {/* 专业特性展示 */}
            <Fade in timeout={1400}>
              <Card sx={{ 
                background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)',
                border: '1px solid rgba(33, 150, 243, 0.2)'
              }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Psychology sx={{ mr: 1, color: 'primary.main' }} />
                    专业特性
                  </Typography>
                  
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="21关键点精确建模"
                        secondary="MediaPipe Holistic级别的手部精度"
                      />
                      <Chip label="NEW" color="primary" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="解剖学级别手部结构"
                        secondary="包含掌骨、关节、肌肉群建模"
                      />
                      <Chip label="PRO" color="success" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="专业手语动作库"
                        secondary="标准中国手语词汇和动作序列"
                      />
                      <Chip label="HOT" color="warning" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="实时动作插值"
                        secondary="平滑的手语动作过渡效果"
                      />
                      <Chip label="AI" color="info" size="small" />
                    </ListItem>
                  </List>
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
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              {/* 装饰性背景元素 */}
              <Box
                sx={{
                  position: 'absolute',
                  top: -50,
                  right: -50,
                  width: 200,
                  height: 200,
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, rgba(255, 154, 162, 0.1) 0%, rgba(255, 179, 186, 0.1) 100%)',
                  zIndex: 0
                }}
              />
              
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3, position: 'relative', zIndex: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <AutoAwesome sx={{ mr: 2, color: 'primary.main', fontSize: 32 }} />
                  <Box>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                      专业手语Avatar 2.0
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      高精度建模 • 实时识别 • 专业表达
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
                      sx={{ 
                        animation: 'glow 2s infinite',
                        '@keyframes glow': {
                          '0%': { boxShadow: '0 0 5px rgba(76, 175, 80, 0.5)' },
                          '50%': { boxShadow: '0 0 20px rgba(76, 175, 80, 0.8)' },
                          '100%': { boxShadow: '0 0 5px rgba(76, 175, 80, 0.5)' },
                        }
                      }}
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
                  border: '3px solid rgba(255, 255, 255, 0.8)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
                  zIndex: 1
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

              <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(0,0,0,0.1)', position: 'relative', zIndex: 1 }}>
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

      {/* 对比展示 */}
      <Fade in timeout={1600}>
        <Box sx={{ mt: 6 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, textAlign: 'center' }}>
            🔥 升级对比
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%', border: '2px solid #f44336' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="h6" color="error" gutterBottom>
                    ❌ 旧版本问题
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="粗糙的火柴人造型" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="'鸡爪'式手部建模" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="简单的几何图形拼接" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="基础材质和光照" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="有限的动作表达" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%', border: '2px solid #4caf50' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="h6" color="success" gutterBottom>
                    ✅ 新版本优势
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="解剖学级别的真实建模" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="21关键点精确手部结构" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="专业PBR材质系统" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Studio级环境光照" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="专业手语动作库" />
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

export default AvatarPage

  const presetTexts = [
    '你好',
    '谢谢',
    '再见',
    '我爱你',
    '祝你好运',
    '生日快乐',
    '新年快乐',
    '欢迎光临',
  ]

  const handleTextChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentText(event.target.value)
  }

  const handlePresetClick = (text: string) => {
    setCurrentText(text)
  }

  const handlePlayStop = () => {
    setIsPlaying(!isPlaying)
  }

  const handleSettingChange = (setting: string, value: any) => {
    setAvatarSettings(prev => ({
      ...prev,
      [setting]: value,
    }))
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom sx={{ fontWeight: 600 }}>
            3D Avatar演示
          </Typography>
          <Typography variant="h6" color="text.secondary">
            观看逼真的3D手语Avatar，体验沉浸式手语演示
          </Typography>
        </Box>
      </Fade>

      <Grid container spacing={4}>
        {/* 左侧控制面板 */}
        <Grid item xs={12} lg={3}>
          <Stack spacing={3}>
            {/* 文本输入 */}
            <Fade in timeout={800}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    文本输入
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={3}
                    value={currentText}
                    onChange={handleTextChange}
                    placeholder="输入要演示的文本..."
                    sx={{ mb: 2 }}
                  />
                  <Button
                    variant="contained"
                    onClick={handlePlayStop}
                    startIcon={isPlaying ? <Stop /> : <PlayArrow />}
                    fullWidth
                    size="large"
                    color={isPlaying ? "error" : "primary"}
                  >
                    {isPlaying ? '停止演示' : '开始演示'}
                  </Button>
                </CardContent>
              </Card>
            </Fade>

            {/* 预设文本 */}
            <Fade in timeout={1000}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    常用词汇
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    点击下方词汇快速演示
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
                    {presetTexts.map((text) => (
                      <Chip
                        key={text}
                        label={text}
                        onClick={() => handlePresetClick(text)}
                        clickable
                        variant={currentText === text ? "filled" : "outlined"}
                        color={currentText === text ? "primary" : "default"}
                      />
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Fade>

            {/* Avatar设置 */}
            <Fade in timeout={1200}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Avatar设置
                  </Typography>
                  
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Animation sx={{ mr: 1 }} />
                        <Typography>动画设置</Typography>
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
                        <FormControlLabel
                          control={
                            <Switch
                              checked={avatarSettings.autoRotate}
                              onChange={(e) => handleSettingChange('autoRotate', e.target.checked)}
                            />
                          }
                          label="自动旋转"
                        />
                      </Stack>
                    </AccordionDetails>
                  </Accordion>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Palette sx={{ mr: 1 }} />
                        <Typography>外观设置</Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Stack spacing={2}>
                        <Box>
                          <Typography variant="body2" gutterBottom>
                            Avatar缩放: {avatarSettings.avatarScale.toFixed(1)}x
                          </Typography>
                          <Slider
                            value={avatarSettings.avatarScale}
                            onChange={(_, value) => handleSettingChange('avatarScale', value)}
                            min={0.5}
                            max={2.0}
                            step={0.1}
                            size="small"
                          />
                        </Box>
                        <Box>
                          <Typography variant="body2" gutterBottom>
                            光照强度: {avatarSettings.lightIntensity.toFixed(1)}
                          </Typography>
                          <Slider
                            value={avatarSettings.lightIntensity}
                            onChange={(_, value) => handleSettingChange('lightIntensity', value)}
                            min={0.3}
                            max={2.0}
                            step={0.1}
                            size="small"
                          />
                        </Box>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={avatarSettings.showWireframe}
                              onChange={(e) => handleSettingChange('showWireframe', e.target.checked)}
                            />
                          }
                          label="显示线框"
                        />
                      </Stack>
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            </Fade>
          </Stack>
        </Grid>

        {/* 主要3D显示区域 */}
        <Grid item xs={12} lg={9}>
          <Fade in timeout={600}>
            <Paper 
              sx={{ 
                p: 3, 
                height: '800px', 
                display: 'flex', 
                flexDirection: 'column',
                background: `linear-gradient(135deg, ${avatarSettings.backgroundColor} 0%, rgba(240,255,240,0.7) 100%)`,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Person sx={{ mr: 2, color: 'primary.main', fontSize: 28 }} />
                  <Box>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                      3D手语Avatar
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      当前演示: {currentText}
                    </Typography>
                  </Box>
                </Box>
                <Stack direction="row" spacing={1}>
                  {isPlaying && (
                    <Chip 
                      label="演示中"
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
                  <AvatarViewer
                    text={currentText}
                    isActive={isPlaying}
                    settings={avatarSettings}
                  />
                </ErrorBoundary>
                
                {/* 控制覆盖层 */}
                <Box
                  sx={{
                    position: 'absolute',
                    top: 16,
                    right: 16,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 1,
                  }}
                >
                  <Chip
                    icon={<ThreeDRotation />}
                    label="鼠标拖拽旋转"
                    size="small"
                    sx={{ bgcolor: 'rgba(255,255,255,0.9)' }}
                  />
                  <Chip
                    icon={<Speed />}
                    label="滚轮缩放"
                    size="small"
                    sx={{ bgcolor: 'rgba(255,255,255,0.9)' }}
                  />
                </Box>
              </Box>

              {/* 底部信息栏 */}
              <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} sm={6}>
                    <Typography variant="body2" color="text.secondary">
                      文本长度: {currentText.length} 字符
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} sx={{ textAlign: { sm: 'right' } }}>
                    <Typography variant="body2" color="text.secondary">
                      渲染引擎: Three.js WebGL
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
            功能特色
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <ThreeDRotation sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    360°全方位查看
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    支持鼠标拖拽旋转、缩放，多角度观察手语动作细节
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Animation sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    流畅动画效果
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    高质量3D动画，真实还原手语动作的连贯性和流畅性
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Tune sx={{ fontSize: 48, color: 'warning.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    个性化设置
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    丰富的自定义选项，调整动画速度、外观效果等
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      </Fade>
    </Container>
  )
}

export default AvatarPage