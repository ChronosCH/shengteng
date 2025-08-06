/**
 * 手语测试面板 - 用于测试精细手部模型和手语预设
 */

import React, { useState } from 'react'
import {
  Box,
  Typography,
  Button,
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Stack,
} from '@mui/material'
import { 
  getAvailableSignPresets, 
  getSignPreset, 
  createSignSequence,
  SIGN_NUMBERS,
  SIGN_WORDS 
} from './SignLanguagePresets'
import AvatarViewer from './AvatarViewer'

const HandSignTestPanel: React.FC = () => {
  const [selectedSign, setSelectedSign] = useState<string>('')
  const [currentText, setCurrentText] = useState<string>('')
  const [isActive, setIsActive] = useState<boolean>(true)
  const [testMode, setTestMode] = useState<'single' | 'sequence'>('single')

  const availablePresets = getAvailableSignPresets()
  const numberPresets = Object.keys(SIGN_NUMBERS)
  const wordPresets = Object.keys(SIGN_WORDS)

  // 处理手语选择
  const handleSignSelect = (sign: string) => {
    setSelectedSign(sign)
    setCurrentText(sign)
  }

  // 测试数字序列
  const testNumberSequence = () => {
    const numbers = ['1', '2', '3', '4', '5']
    setCurrentText(numbers.join(' '))
    setTestMode('sequence')
  }

  // 测试词汇序列
  const testWordSequence = () => {
    const words = ['你好', '谢谢']
    setCurrentText(words.join(' '))
    setTestMode('sequence')
  }

  // 重置测试
  const resetTest = () => {
    setSelectedSign('')
    setCurrentText('')
    setTestMode('single')
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        手语小人测试面板
      </Typography>
      
      <Grid container spacing={3}>
        {/* 控制面板 */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              测试控制
            </Typography>
            
            {/* 测试模式选择 */}
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>测试模式</InputLabel>
              <Select
                value={testMode}
                label="测试模式"
                onChange={(e) => setTestMode(e.target.value as 'single' | 'sequence')}
              >
                <MenuItem value="single">单个手语</MenuItem>
                <MenuItem value="sequence">手语序列</MenuItem>
              </Select>
            </FormControl>

            {/* 单个手语选择 */}
            {testMode === 'single' && (
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>选择手语</InputLabel>
                <Select
                  value={selectedSign}
                  label="选择手语"
                  onChange={(e) => handleSignSelect(e.target.value)}
                >
                  {availablePresets.map((preset) => (
                    <MenuItem key={preset} value={preset}>
                      {preset}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            {/* 快速测试按钮 */}
            <Stack spacing={1} sx={{ mb: 2 }}>
              <Button 
                variant="outlined" 
                onClick={testNumberSequence}
                fullWidth
              >
                测试数字序列 (1-5)
              </Button>
              <Button 
                variant="outlined" 
                onClick={testWordSequence}
                fullWidth
              >
                测试词汇序列
              </Button>
              <Button 
                variant="outlined" 
                onClick={resetTest}
                fullWidth
              >
                重置测试
              </Button>
            </Stack>

            {/* 激活状态控制 */}
            <Button
              variant={isActive ? "contained" : "outlined"}
              onClick={() => setIsActive(!isActive)}
              fullWidth
              sx={{ mb: 2 }}
            >
              {isActive ? '激活状态' : '非激活状态'}
            </Button>

            {/* 当前状态显示 */}
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                当前状态:
              </Typography>
              <Typography variant="body2" color="text.secondary">
                文本: {currentText || '无'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                选中手语: {selectedSign || '无'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                模式: {testMode === 'single' ? '单个' : '序列'}
              </Typography>
            </Box>
          </Paper>

          {/* 可用预设显示 */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              数字手语预设
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {numberPresets.map((preset) => (
                <Chip
                  key={preset}
                  label={preset}
                  onClick={() => handleSignSelect(preset)}
                  color={selectedSign === preset ? "primary" : "default"}
                  variant={selectedSign === preset ? "filled" : "outlined"}
                  size="small"
                />
              ))}
            </Stack>

            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              词汇手语预设
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {wordPresets.map((preset) => (
                <Chip
                  key={preset}
                  label={preset}
                  onClick={() => handleSignSelect(preset)}
                  color={selectedSign === preset ? "primary" : "default"}
                  variant={selectedSign === preset ? "filled" : "outlined"}
                  size="small"
                />
              ))}
            </Stack>
          </Paper>
        </Grid>

        {/* Avatar显示区域 */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '600px' }}>
            <Typography variant="h6" gutterBottom>
              手语小人显示
            </Typography>
            
            <Box sx={{ height: '550px', border: '1px solid #e0e0e0', borderRadius: 1 }}>
              <AvatarViewer
                text={currentText}
                isActive={isActive}
                onAvatarMeshReady={(mesh) => {
                  console.log('Avatar mesh ready:', mesh)
                }}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* 说明信息 */}
      <Paper sx={{ p: 2, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          功能说明
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          • <strong>精细手部模型</strong>: 新的手语小人支持21个手部关键点，能够精确表现手指动作
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          • <strong>手语预设</strong>: 内置数字(0-5)和常用词汇(你好、谢谢)的手语动作
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          • <strong>自动识别</strong>: 当输入文本包含预设的手语时，会自动显示对应的手势
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          • <strong>双手协调</strong>: 支持左右手独立控制和双手协调动作
        </Typography>
        <Typography variant="body2" color="text.secondary">
          • <strong>实时更新</strong>: 手部模型会根据MediaPipe检测到的关键点实时更新
        </Typography>
      </Paper>
    </Box>
  )
}

export default HandSignTestPanel
