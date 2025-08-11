/**
 * 高级写实Avatar演示页面
 * 展示真人级别的3D Avatar效果
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
  Star,
  Diamond,
  Camera,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import AdvancedRealisticAvatar from '../components/AdvancedRealisticAvatar'
import { 
  PROFESSIONAL_SIGN_LANGUAGE_LIBRARY, 
  ProfessionalSignLanguagePlayer,
  type ProfessionalSignLanguageKeypoint
} from '../data/ProfessionalSignLanguageLibrary'

function AdvancedAvatarPage() {
  const [selectedGesture, setSelectedGesture] = useState('hello')
  const [currentText, setCurrentText] = useState('你好')
  const [isPerforming, setIsPerforming] = useState(false)
  const [currentKeypoints, setCurrentKeypoints] = useState<{
    left: ProfessionalSignLanguageKeypoint[]
    right: ProfessionalSignLanguageKeypoint[]
  }>({ left: [], right: [] })

  const [avatarSettings, setAvatarSettings] = useState({
    realisticMode: true,
    animationSpeed: 1.0,
    renderQuality: 'ultra' as 'low' | 'medium' | 'high' | 'ultra',
    antiAliasing: true,
    postProcessing: true,
    showBones: false,
    advancedLighting: true,
    skinTone: '#f4c2a1',
    clothingStyle: 'professional'
  })

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
        console.log('手语演示完成:', gesture.name)
      }
    )
  }, [selectedGesture, isPerforming, player])

  const handleStop = useCallback(() => {
    setIsPerforming(false)
    setCurrentKeypoints({ left: [], right: [] })
  }, [])

  const handleSettingChange = (setting: string, value: any) => {
    setAvatarSettings(prev => ({
      ...prev,
      [setting]: value,
    }))
  }

  // 按类别分组的手势
  const gesturesByCategory = {
    greeting: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'greeting'),
    emotion: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'emotion'),
    daily: Object.values(PROFESSIONAL_SIGN_LANGUAGE_LIBRARY).filter(g => g.category === 'daily'),
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

  const getSkinToneOptions = () => [
    { value: '#f4c2a1', label: '浅色' },
    { value: '#deb887', label: '小麦色' },
    { value: '#d2b48c', label: '暖色' },
    { value: '#c8956d', label: '橄榄色' },
    { value: '#8b4513', label: '深色' }
  ]

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Badge badgeContent="革命性" color="error" sx={{ mb: 2 }}>
            <Typography variant="h3" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 2, justifyContent: 'center' }}>
              <Diamond sx={{ fontSize: 40, color: 'primary.main' }} />
              真人级3D Avatar 3.0
            </Typography>
          </Badge>
          
          <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
            解剖学级精度建模 • 真实人类外观 • 专业级渲染质量
          </Typography>
          
          <Alert 
            severity="success" 
            sx={{ 
              mt: 2, 
              maxWidth: 900, 
              mx: 'auto',
              background: 'linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 100%)',
              border: '2px solid #4caf50',
              borderRadius: 3
            }}
            icon={<Star />}
          >
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              🎉 突破性升级！告别所有不真实感，体验真人级别的3D Avatar效果 - 堪比电影级CG品质
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
              <Card sx={{ background: 'linear-gradient(135deg, #fff 0%, #f8f9fa 100%)', border: '2px solid rgba(76, 175, 80, 0.2)' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Star sx={{ mr: 1, color: 'warning.main' }} />
                    精选展示
                  </Typography>
                  
                  <Grid container spacing={2}>
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
                            startIcon={<span style={{ fontSize: '20px' }}>{gesture?.icon || '👋'}</span>}
                            sx={{ 
                              py: 1.5,
                              flexDirection: 'column',
                              gap: 0.5,
                              height: 80,
                              borderRadius: 2
                            }}
                          >
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {gesture?.name || id}
                            </Typography>
                          </Button>
                        </Grid>
                      )
                    })}
                  </Grid>

                  <Divider sx={{ my: 2 }} />

                  {/* 控制按钮 */}
                  <Stack direction="row" spacing={2}>
                    <Button
                      variant="contained"
                      onClick={handlePlayGesture}
                      disabled={isPerforming}
                      startIcon={<PlayArrow />}
                      fullWidth
                      size="large"
                      color="primary"
                      sx={{ borderRadius: 2 }}
                    >
                      开始演示
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={handleStop}
                      disabled={!isPerforming}
                      startIcon={<Stop />}
                      color="error"
                      sx={{ borderRadius: 2 }}
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
                        label="真人级演示中"
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
            <Fade in timeout={1000}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Camera sx={{ mr: 1 }} />
                    渲染设置
                  </Typography>
                  
                  {/* 渲染质量 */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="body2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <HighQuality fontSize="small" />
                      渲染质量: {avatarSettings.renderQuality}
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
                        { value: 3, label: '真人级' }
                      ]}
                    />
                  </Box>

                  {/* 动画速度 */}
                  <Box sx={{ mb: 3 }}>
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

                  {/* 肤色选择 */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="body2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Face fontSize="small" />
                      肤色选择
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      {getSkinToneOptions().map((tone) => (
                        <Button
                          key={tone.value}
                          variant={avatarSettings.skinTone === tone.value ? "contained" : "outlined"}
                          onClick={() => handleSettingChange('skinTone', tone.value)}
                          size="small"
                          sx={{ 
                            minWidth: 60, 
                            backgroundColor: tone.value,
                            '&:hover': {
                              backgroundColor: tone.value,
                              opacity: 0.8
                            }
                          }}
                        >
                          {tone.label}
                        </Button>
                      ))}
                    </Stack>
                  </Box>

                  <Divider sx={{ my: 2 }} />

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
                          真人模式
                        </Box>
                      }
                    />

                    <FormControlLabel
                      control={
                        <Switch
                          checked={avatarSettings.advancedLighting}
                          onChange={(e) => handleSettingChange('advancedLighting', e.target.checked)}
                        />
                      }
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Lightbulb fontSize="small" />
                          高级光照
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
                          <Camera fontSize="small" />
                          后期处理
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
                </CardContent>
              </Card>
            </Fade>

            {/* 技术特性 */}
            <Fade in timeout={1200}>
              <Card sx={{ 
                background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)',
                border: '1px solid rgba(33, 150, 243, 0.2)'
              }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Psychology sx={{ mr: 1, color: 'primary.main' }} />
                    革命性特性
                  </Typography>
                  
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="解剖学级精度建模"
                        secondary="基于真实人体解剖学的3D建模"
                      />
                      <Chip label="革命性" color="error" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="真人级面部表情"
                        secondary="包含眨眼、微表情、肌肉运动"
                      />
                      <Chip label="NEW" color="primary" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="高级材质系统"
                        secondary="PBR材质、次表面散射、各向异性"
                      />
                      <Chip label="PRO" color="success" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="电影级光照"
                        secondary="三点布光、环境遮蔽、全局光照"
                      />
                      <Chip label="ULTRA" color="warning" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="自然身体语言"
                        secondary="呼吸、重心转移、微动作"
                      />
                      <Chip label="AI" color="info" size="small" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Fade>
          </Stack>
        </Grid>

        {/* 右侧Avatar显示区域 */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ height: '85vh', position: 'relative' }}>
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
                  minWidth: 220
                }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                    💎 真人级Avatar 3.0
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    解剖学建模 • 真实外观 • 电影品质
                  </Typography>
                </Box>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {isPerforming && (
                    <Chip 
                      label="真人级演示"
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
                    label={avatarSettings.realisticMode ? "真人模式" : "卡通模式"}
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
                  <AdvancedRealisticAvatar
                    signText={currentText}
                    isPerforming={isPerforming}
                    leftHandKeypoints={currentKeypoints.left}
                    rightHandKeypoints={currentKeypoints.right}
                    showBones={avatarSettings.showBones}
                    realisticMode={avatarSettings.realisticMode}
                    animationSpeed={avatarSettings.animationSpeed}
                    onAvatarReady={(avatar) => {
                      console.log('高级真人Avatar已就绪:', avatar)
                    }}
                    onSignComplete={(signText) => {
                      console.log('手语演示完成:', signText)
                    }}
                  />
                </ErrorBoundary>
              </Box>

              {/* 技术规格 */}
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
                  minWidth: 200
                }}>
                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    🎬 电影级渲染引擎
                  </Typography>
                  <Typography variant="caption" sx={{ opacity: 0.8 }}>
                    • PBR材质系统<br/>
                    • 次表面散射<br/>
                    • 全局光照<br/>
                    • 4K材质贴图
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* 对比展示 */}
          <Box sx={{ mt: 2 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">
                  渲染品质: 电影级CG
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">
                  建模精度: 解剖学级
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4} sx={{ textAlign: { sm: 'right' } }}>
                <Typography variant="body2" color="text.secondary">
                  材质系统: PBR物理材质
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </Grid>
      </Grid>

      {/* 升级对比 */}
      <Fade in timeout={1600}>
        <Box sx={{ mt: 6 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, textAlign: 'center' }}>
            🔥 革命性升级
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', border: '2px solid #f44336' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="h6" color="error" gutterBottom>
                    ❌ 旧版本问题
                  </Typography>
                  <List dense>
                    <ListItem><ListItemText primary="粗糙的火柴人造型" /></ListItem>
                    <ListItem><ListItemText primary="'鸡爪'式手部建模" /></ListItem>
                    <ListItem><ListItemText primary="简单的几何图形拼接" /></ListItem>
                    <ListItem><ListItemText primary="基础材质和光照" /></ListItem>
                    <ListItem><ListItemText primary="不像人类的外观" /></ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', border: '2px solid #ff9800' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="h6" color="warning" gutterBottom>
                    ⚡ 中级版本
                  </Typography>
                  <List dense>
                    <ListItem><ListItemText primary="基本的人体建模" /></ListItem>
                    <ListItem><ListItemText primary="简化的手部结构" /></ListItem>
                    <ListItem><ListItemText primary="基础PBR材质" /></ListItem>
                    <ListItem><ListItemText primary="简单的面部特征" /></ListItem>
                    <ListItem><ListItemText primary="卡通化外观" /></ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', border: '2px solid #4caf50' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="h6" color="success" gutterBottom>
                    ✅ 真人级Avatar 3.0
                  </Typography>
                  <List dense>
                    <ListItem><ListItemText primary="解剖学级精确建模" /></ListItem>
                    <ListItem><ListItemText primary="真人级面部表情系统" /></ListItem>
                    <ListItem><ListItemText primary="电影级PBR材质" /></ListItem>
                    <ListItem><ListItemText primary="高级光照和阴影" /></ListItem>
                    <ListItem><ListItemText primary="完全像真人的外观" /></ListItem>
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

export default AdvancedAvatarPage
