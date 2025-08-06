/**
 * 简化的手语测试组件 - 展示精细手部模型改进
 */

import React, { useState } from 'react'
import {
  Box,
  Typography,
  Button,
  Grid,
  Paper,
  Chip,
  Stack,
} from '@mui/material'
import AvatarViewer from './AvatarViewer'

const SimpleHandSignTest: React.FC = () => {
  const [currentText, setCurrentText] = useState<string>('')
  const [isActive, setIsActive] = useState<boolean>(true)

  // 测试不同的文本
  const testTexts = ['1', '2', '5', '你好', '谢谢']

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        手语小人精细化改进演示
      </Typography>
      
      <Grid container spacing={3}>
        {/* 控制面板 */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              测试控制
            </Typography>
            
            {/* 测试文本选择 */}
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              选择测试文本:
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
              {testTexts.map((text) => (
                <Chip
                  key={text}
                  label={text}
                  onClick={() => setCurrentText(text)}
                  color={currentText === text ? "primary" : "default"}
                  variant={currentText === text ? "filled" : "outlined"}
                />
              ))}
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

            {/* 重置按钮 */}
            <Button
              variant="outlined"
              onClick={() => setCurrentText('')}
              fullWidth
            >
              重置
            </Button>

            {/* 当前状态显示 */}
            <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                当前状态:
              </Typography>
              <Typography variant="body2" color="text.secondary">
                文本: {currentText || '无'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                状态: {isActive ? '激活' : '非激活'}
              </Typography>
            </Box>
          </Paper>

          {/* 改进说明 */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              改进内容
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              ✅ <strong>精细手部模型</strong>: 支持21个手部关键点
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              ✅ <strong>手指关节控制</strong>: 每个手指都有独立的关节
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              ✅ <strong>手语预设</strong>: 内置数字和常用词汇手势
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              ✅ <strong>双手协调</strong>: 支持左右手独立控制
            </Typography>
            <Typography variant="body2" color="text.secondary">
              ✅ <strong>自动识别</strong>: 根据文本自动显示对应手势
            </Typography>
          </Paper>
        </Grid>

        {/* Avatar显示区域 */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '600px' }}>
            <Typography variant="h6" gutterBottom>
              精细手语小人
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

      {/* 对比说明 */}
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          改进前后对比
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom color="error">
              改进前的问题:
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              • 手臂只是简单的方块，无法表现手指细节
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              • 没有手部关节，无法做出精确的手语动作
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              • 缺乏手语专用特征，不适合手语识别应用
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 无法区分不同的手指状态和手势
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom color="success.main">
              改进后的优势:
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              • 精细的手部模型，支持21个关键点
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              • 每个手指都有独立的关节控制
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              • 内置手语预设，支持常用手势
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 能够准确表现复杂的手语动作
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  )
}

export default SimpleHandSignTest
