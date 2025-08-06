import { useState } from 'react'
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Card,
  CardContent,
  FormControl,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  Slider,
  TextField,
  Button,
  Divider,
  Stack,
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Fade,
} from '@mui/material'
import {
  Settings,
  Videocam,
  Volume,
  Accessibility,
  Security,
  Palette,
  Speed,
  ExpandMore,
  Save,
  Restore,
  Info,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import PerformanceMonitor from '../components/PerformanceMonitor'
import AccessibilityPanel from '../components/AccessibilityPanel'

function SettingsPage() {
  const [settings, setSettings] = useState({
    // 视频设置
    videoQuality: 'high',
    frameRate: 30,
    enableMirror: true,
    autoAdjustExposure: true,
    
    // 音频设置
    enableAudio: false,
    audioVolume: 0.7,
    enableSoundEffects: true,
    
    // 识别设置
    recognitionMode: 'realtime',
    confidenceThreshold: 0.6,
    enablePreprocessing: true,
    batchSize: 1,
    
    // 界面设置
    theme: 'light',
    language: 'zh-CN',
    showPerformanceMonitor: false,
    enableAnimations: true,
    compactMode: false,
    
    // 隐私设置
    dataCollection: false,
    localProcessing: true,
    saveHistory: false,
    anonymousUsage: true,
    
    // 高级设置
    debugMode: false,
    apiEndpoint: 'ws://localhost:8000',
    maxRetries: 3,
    connectionTimeout: 5000,
  })

  const [showSaveAlert, setShowSaveAlert] = useState(false)

  const handleSettingChange = (section: string, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value,
    }))
  }

  const handleSaveSettings = () => {
    // 保存设置到本地存储
    localStorage.setItem('signAvatar_settings', JSON.stringify(settings))
    setShowSaveAlert(true)
    setTimeout(() => setShowSaveAlert(false), 3000)
  }

  const handleResetSettings = () => {
    // 重置为默认设置
    if (window.confirm('确定要重置所有设置吗？此操作不可撤销。')) {
      localStorage.removeItem('signAvatar_settings')
      window.location.reload()
    }
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom sx={{ fontWeight: 600 }}>
            系统设置
          </Typography>
          <Typography variant="h6" color="text.secondary">
            个性化配置您的手语识别体验
          </Typography>
        </Box>
      </Fade>

      {/* 保存提示 */}
      {showSaveAlert && (
        <Fade in>
          <Alert severity="success" sx={{ mb: 3 }}>
            设置已保存成功！
          </Alert>
        </Fade>
      )}

      <Grid container spacing={4}>
        {/* 左侧设置面板 */}
        <Grid item xs={12} lg={8}>
          <Stack spacing={3}>
            {/* 视频设置 */}
            <Fade in timeout={800}>
              <Card>
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Videocam sx={{ mr: 2, color: 'primary.main' }} />
                      <Typography variant="h6">视频设置</Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <Typography variant="body2" gutterBottom>
                            视频质量
                          </Typography>
                          <Select
                            value={settings.videoQuality}
                            onChange={(e) => handleSettingChange('video', 'videoQuality', e.target.value)}
                            size="small"
                          >
                            <MenuItem value="low">低质量 (480p)</MenuItem>
                            <MenuItem value="medium">中等质量 (720p)</MenuItem>
                            <MenuItem value="high">高质量 (1080p)</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" gutterBottom>
                          帧率: {settings.frameRate} FPS
                        </Typography>
                        <Slider
                          value={settings.frameRate}
                          onChange={(_, value) => handleSettingChange('video', 'frameRate', value)}
                          min={15}
                          max={60}
                          step={15}
                          marks
                          size="small"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.enableMirror}
                              onChange={(e) => handleSettingChange('video', 'enableMirror', e.target.checked)}
                            />
                          }
                          label="镜像显示"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.autoAdjustExposure}
                              onChange={(e) => handleSettingChange('video', 'autoAdjustExposure', e.target.checked)}
                            />
                          }
                          label="自动曝光调节"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Card>
            </Fade>

            {/* 识别设置 */}
            <Fade in timeout={1000}>
              <Card>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Settings sx={{ mr: 2, color: 'success.main' }} />
                      <Typography variant="h6">识别设置</Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <Typography variant="body2" gutterBottom>
                            识别模式
                          </Typography>
                          <Select
                            value={settings.recognitionMode}
                            onChange={(e) => handleSettingChange('recognition', 'recognitionMode', e.target.value)}
                            size="small"
                          >
                            <MenuItem value="realtime">实时识别</MenuItem>
                            <MenuItem value="batch">批量识别</MenuItem>
                            <MenuItem value="precise">精确模式</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" gutterBottom>
                          置信度阈值: {(settings.confidenceThreshold * 100).toFixed(0)}%
                        </Typography>
                        <Slider
                          value={settings.confidenceThreshold}
                          onChange={(_, value) => handleSettingChange('recognition', 'confidenceThreshold', value)}
                          min={0.3}
                          max={0.9}
                          step={0.1}
                          size="small"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.enablePreprocessing}
                              onChange={(e) => handleSettingChange('recognition', 'enablePreprocessing', e.target.checked)}
                            />
                          }
                          label="启用预处理"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" gutterBottom>
                          批处理大小: {settings.batchSize}
                        </Typography>
                        <Slider
                          value={settings.batchSize}
                          onChange={(_, value) => handleSettingChange('recognition', 'batchSize', value)}
                          min={1}
                          max={10}
                          step={1}
                          size="small"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Card>
            </Fade>

            {/* 界面设置 */}
            <Fade in timeout={1200}>
              <Card>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Palette sx={{ mr: 2, color: 'warning.main' }} />
                      <Typography variant="h6">界面设置</Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <Typography variant="body2" gutterBottom>
                            主题
                          </Typography>
                          <Select
                            value={settings.theme}
                            onChange={(e) => handleSettingChange('ui', 'theme', e.target.value)}
                            size="small"
                          >
                            <MenuItem value="light">亮色主题</MenuItem>
                            <MenuItem value="dark">暗色主题</MenuItem>
                            <MenuItem value="auto">跟随系统</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <Typography variant="body2" gutterBottom>
                            语言
                          </Typography>
                          <Select
                            value={settings.language}
                            onChange={(e) => handleSettingChange('ui', 'language', e.target.value)}
                            size="small"
                          >
                            <MenuItem value="zh-CN">简体中文</MenuItem>
                            <MenuItem value="zh-TW">繁体中文</MenuItem>
                            <MenuItem value="en-US">English</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.enableAnimations}
                              onChange={(e) => handleSettingChange('ui', 'enableAnimations', e.target.checked)}
                            />
                          }
                          label="启用动画效果"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.compactMode}
                              onChange={(e) => handleSettingChange('ui', 'compactMode', e.target.checked)}
                            />
                          }
                          label="紧凑模式"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Card>
            </Fade>

            {/* 隐私设置 */}
            <Fade in timeout={1400}>
              <Card>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Security sx={{ mr: 2, color: 'error.main' }} />
                      <Typography variant="h6">隐私与安全</Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={3}>
                      <Grid item xs={12}>
                        <Alert severity="info" sx={{ mb: 2 }}>
                          我们重视您的隐私，所有视频数据默认在本地处理
                        </Alert>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.localProcessing}
                              onChange={(e) => handleSettingChange('privacy', 'localProcessing', e.target.checked)}
                            />
                          }
                          label="本地数据处理"
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          视频数据不会上传到服务器
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.dataCollection}
                              onChange={(e) => handleSettingChange('privacy', 'dataCollection', e.target.checked)}
                            />
                          }
                          label="允许数据收集"
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          用于改进服务质量
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.saveHistory}
                              onChange={(e) => handleSettingChange('privacy', 'saveHistory', e.target.checked)}
                            />
                          }
                          label="保存识别历史"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.anonymousUsage}
                              onChange={(e) => handleSettingChange('privacy', 'anonymousUsage', e.target.checked)}
                            />
                          }
                          label="匿名使用统计"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Card>
            </Fade>

            {/* 高级设置 */}
            <Fade in timeout={1600}>
              <Card>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Speed sx={{ mr: 2, color: 'secondary.main' }} />
                      <Typography variant="h6">高级设置</Typography>
                      <Chip label="专家选项" size="small" sx={{ ml: 2 }} />
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={3}>
                      <Grid item xs={12}>
                        <Alert severity="warning" sx={{ mb: 2 }}>
                          <Typography variant="body2">
                            <strong>警告:</strong> 修改这些设置可能影响系统性能和稳定性
                          </Typography>
                        </Alert>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField
                          fullWidth
                          label="API端点"
                          value={settings.apiEndpoint}
                          onChange={(e) => handleSettingChange('advanced', 'apiEndpoint', e.target.value)}
                          size="small"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" gutterBottom>
                          连接超时: {settings.connectionTimeout}ms
                        </Typography>
                        <Slider
                          value={settings.connectionTimeout}
                          onChange={(_, value) => handleSettingChange('advanced', 'connectionTimeout', value)}
                          min={1000}
                          max={10000}
                          step={1000}
                          size="small"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" gutterBottom>
                          最大重试次数: {settings.maxRetries}
                        </Typography>
                        <Slider
                          value={settings.maxRetries}
                          onChange={(_, value) => handleSettingChange('advanced', 'maxRetries', value)}
                          min={1}
                          max={10}
                          step={1}
                          size="small"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.debugMode}
                              onChange={(e) => handleSettingChange('advanced', 'debugMode', e.target.checked)}
                            />
                          }
                          label="调试模式"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Card>
            </Fade>
          </Stack>
        </Grid>

        {/* 右侧信息面板 */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            {/* 操作按钮 */}
            <Fade in timeout={1800}>
              <Card>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="h6" gutterBottom>
                    设置操作
                  </Typography>
                  <Stack spacing={2}>
                    <Button
                      variant="contained"
                      startIcon={<Save />}
                      onClick={handleSaveSettings}
                      fullWidth
                      size="large"
                    >
                      保存设置
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<Restore />}
                      onClick={handleResetSettings}
                      fullWidth
                      color="warning"
                    >
                      重置设置
                    </Button>
                  </Stack>
                </CardContent>
              </Card>
            </Fade>

            {/* 系统信息 */}
            <Fade in timeout={2000}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Info sx={{ mr: 1, verticalAlign: 'middle' }} />
                    系统信息
                  </Typography>
                  <Divider sx={{ my: 2 }} />
                  <Stack spacing={1}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">版本:</Typography>
                      <Typography variant="body2">v1.0.0</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">浏览器:</Typography>
                      <Typography variant="body2">{navigator.userAgent.split(' ')[0]}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">分辨率:</Typography>
                      <Typography variant="body2">{window.screen.width}×{window.screen.height}</Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Fade>

            {/* 性能监控 */}
            <Fade in timeout={2200}>
              <Box>
                <ErrorBoundary>
                  <PerformanceMonitor isVisible={true} />
                </ErrorBoundary>
              </Box>
            </Fade>
          </Stack>
        </Grid>
      </Grid>
    </Container>
  )
}

export default SettingsPage