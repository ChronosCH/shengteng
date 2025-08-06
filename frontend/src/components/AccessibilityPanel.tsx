/**
 * 可访问性面板组件 - 提供无障碍功能设置
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  Switch,
  FormControlLabel,
  Slider,
  Button,
  Divider,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
} from '@mui/material'
import {
  Accessibility,
  TextFields,
  Contrast,
  VolumeUp,
  Keyboard,
  ZoomIn,
} from '@mui/icons-material'

interface AccessibilitySettings {
  highContrast: boolean
  largeText: boolean
  reducedMotion: boolean
  screenReaderSupport: boolean
  keyboardNavigation: boolean
  fontSize: number
  speechRate: number
  colorBlindSupport: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia'
  autoSpeak: boolean
}

interface AccessibilityPanelProps {
  isVisible: boolean
  onSettingsChange: (settings: AccessibilitySettings) => void
}

const AccessibilityPanel: React.FC<AccessibilityPanelProps> = ({
  isVisible,
  onSettingsChange,
}) => {
  const [settings, setSettings] = useState<AccessibilitySettings>({
    highContrast: false,
    largeText: false,
    reducedMotion: false,
    screenReaderSupport: true,
    keyboardNavigation: true,
    fontSize: 16,
    speechRate: 1.0,
    colorBlindSupport: 'none',
    autoSpeak: true,
  })

  // 从localStorage加载设置
  useEffect(() => {
    const savedSettings = localStorage.getItem('accessibility-settings')
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings)
        setSettings(parsed)
        onSettingsChange(parsed)
      } catch (error) {
        console.error('加载可访问性设置失败:', error)
      }
    }
  }, [onSettingsChange])

  // 保存设置到localStorage
  useEffect(() => {
    localStorage.setItem('accessibility-settings', JSON.stringify(settings))
    onSettingsChange(settings)
    applySettings(settings)
  }, [settings, onSettingsChange])

  // 应用设置到DOM
  const applySettings = (newSettings: AccessibilitySettings) => {
    const root = document.documentElement

    // 高对比度
    if (newSettings.highContrast) {
      root.style.setProperty('--text-primary', '#000000')
      root.style.setProperty('--text-secondary', '#333333')
      root.style.setProperty('--background-default', '#ffffff')
    } else {
      root.style.removeProperty('--text-primary')
      root.style.removeProperty('--text-secondary')
      root.style.removeProperty('--background-default')
    }

    // 字体大小
    root.style.fontSize = `${newSettings.fontSize}px`

    // 减少动画
    if (newSettings.reducedMotion) {
      root.style.setProperty('--animation-duration', '0.01s')
    } else {
      root.style.removeProperty('--animation-duration')
    }

    // 色盲支持
    if (newSettings.colorBlindSupport !== 'none') {
      root.classList.add(`colorblind-${newSettings.colorBlindSupport}`)
    } else {
      root.classList.remove('colorblind-protanopia', 'colorblind-deuteranopia', 'colorblind-tritanopia')
    }
  }

  const handleSettingChange = (key: keyof AccessibilitySettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  const resetSettings = () => {
    const defaultSettings: AccessibilitySettings = {
      highContrast: false,
      largeText: false,
      reducedMotion: false,
      screenReaderSupport: true,
      keyboardNavigation: true,
      fontSize: 16,
      speechRate: 1.0,
      colorBlindSupport: 'none',
      autoSpeak: true,
    }
    setSettings(defaultSettings)
  }

  const testSpeech = () => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance('这是语音测试，您可以听到这段话吗？')
      utterance.rate = settings.speechRate
      utterance.lang = 'zh-CN'
      window.speechSynthesis.speak(utterance)
    }
  }

  if (!isVisible) return null

  return (
    <Paper 
      sx={{ 
        position: 'fixed',
        top: 80,
        left: 16,
        width: 350,
        maxHeight: 'calc(100vh - 100px)',
        overflowY: 'auto',
        p: 2,
        zIndex: 1000,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Accessibility color="primary" />
        <Typography variant="h6">无障碍设置</Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 2 }}>
        这些设置将帮助不同需求的用户更好地使用本应用
      </Alert>

      {/* 视觉辅助 */}
      <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
        <Contrast sx={{ mr: 1, verticalAlign: 'middle' }} />
        视觉辅助
      </Typography>

      <FormControlLabel
        control={
          <Switch
            checked={settings.highContrast}
            onChange={(e) => handleSettingChange('highContrast', e.target.checked)}
          />
        }
        label="高对比度模式"
      />

      <FormControlLabel
        control={
          <Switch
            checked={settings.largeText}
            onChange={(e) => handleSettingChange('largeText', e.target.checked)}
          />
        }
        label="大字体模式"
      />

      <Box sx={{ mt: 2, mb: 2 }}>
        <Typography gutterBottom>
          <TextFields sx={{ mr: 1, verticalAlign: 'middle' }} />
          字体大小: {settings.fontSize}px
        </Typography>
        <Slider
          value={settings.fontSize}
          onChange={(_: any, value: any) => handleSettingChange('fontSize', value)}
          min={12}
          max={24}
          step={1}
          marks={[
            { value: 12, label: '小' },
            { value: 16, label: '中' },
            { value: 20, label: '大' },
            { value: 24, label: '特大' },
          ]}
        />
      </Box>

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>色盲支持</InputLabel>
        <Select
          value={settings.colorBlindSupport}
          onChange={(e) => handleSettingChange('colorBlindSupport', e.target.value)}
        >
          <MenuItem value="none">无</MenuItem>
          <MenuItem value="protanopia">红色盲</MenuItem>
          <MenuItem value="deuteranopia">绿色盲</MenuItem>
          <MenuItem value="tritanopia">蓝色盲</MenuItem>
        </Select>
      </FormControl>

      <Divider sx={{ my: 2 }} />

      {/* 语音辅助 */}
      <Typography variant="subtitle1" gutterBottom>
        <VolumeUp sx={{ mr: 1, verticalAlign: 'middle' }} />
        语音辅助
      </Typography>

      <FormControlLabel
        control={
          <Switch
            checked={settings.autoSpeak}
            onChange={(e) => handleSettingChange('autoSpeak', e.target.checked)}
          />
        }
        label="自动语音播报"
      />

      <Box sx={{ mt: 2, mb: 2 }}>
        <Typography gutterBottom>
          语音速度: {settings.speechRate.toFixed(1)}x
        </Typography>
        <Slider
          value={settings.speechRate}
          onChange={(_: any, value: any) => handleSettingChange('speechRate', value)}
          min={0.5}
          max={2.0}
          step={0.1}
          marks={[
            { value: 0.5, label: '慢' },
            { value: 1.0, label: '正常' },
            { value: 1.5, label: '快' },
            { value: 2.0, label: '很快' },
          ]}
        />
      </Box>

      <Button variant="outlined" onClick={testSpeech} fullWidth sx={{ mb: 2 }}>
        测试语音
      </Button>

      <Divider sx={{ my: 2 }} />

      {/* 交互辅助 */}
      <Typography variant="subtitle1" gutterBottom>
        <Keyboard sx={{ mr: 1, verticalAlign: 'middle' }} />
        交互辅助
      </Typography>

      <FormControlLabel
        control={
          <Switch
            checked={settings.keyboardNavigation}
            onChange={(e) => handleSettingChange('keyboardNavigation', e.target.checked)}
          />
        }
        label="键盘导航支持"
      />

      <FormControlLabel
        control={
          <Switch
            checked={settings.reducedMotion}
            onChange={(e) => handleSettingChange('reducedMotion', e.target.checked)}
          />
        }
        label="减少动画效果"
      />

      <FormControlLabel
        control={
          <Switch
            checked={settings.screenReaderSupport}
            onChange={(e) => handleSettingChange('screenReaderSupport', e.target.checked)}
          />
        }
        label="屏幕阅读器支持"
      />

      <Divider sx={{ my: 2 }} />

      {/* 快捷键说明 */}
      <Typography variant="subtitle2" gutterBottom>
        键盘快捷键:
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
        <Chip label="空格键: 开始/停止识别" size="small" />
        <Chip label="Esc: 关闭面板" size="small" />
        <Chip label="Tab: 切换焦点" size="small" />
      </Box>

      {/* 重置按钮 */}
      <Button 
        variant="outlined" 
        color="secondary" 
        onClick={resetSettings}
        fullWidth
      >
        重置为默认设置
      </Button>
    </Paper>
  )
}

export default AccessibilityPanel
