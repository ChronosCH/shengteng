import { useState } from 'react'
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
  Slider,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Fade,
  Divider,
} from '@mui/material'
import {
  Person,
  PlayArrow,
  Stop,
  Settings,
  Palette,
  Speed,
  ExpandMore,
  ThreeDRotation,
  Animation,
  Tune,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import AvatarViewer from '../components/AvatarViewer'
import ThreeAvatar from '../components/ThreeAvatar'

function AvatarPage() {
  const [currentText, setCurrentText] = useState('你好，欢迎来到SignAvatar')
  const [isPlaying, setIsPlaying] = useState(false)
  const [avatarSettings, setAvatarSettings] = useState({
    animationSpeed: 1.0,
    showWireframe: false,
    autoRotate: false,
    backgroundColor: '#f0f8ff',
    avatarScale: 1.0,
    lightIntensity: 1.0,
  })

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