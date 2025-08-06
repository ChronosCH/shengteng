/**
 * 手语改进演示组件 - 展示精细手部模型的概念
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
  Card,
  CardContent,
  Divider,
} from '@mui/material'
import {
  PanTool,
  Fingerprint,
  TouchApp,
  Gesture,
} from '@mui/icons-material'

const HandSignDemo: React.FC = () => {
  const [selectedDemo, setSelectedDemo] = useState<string>('overview')

  const demoSections = [
    { id: 'overview', label: '总览', icon: <PanTool /> },
    { id: 'joints', label: '关节控制', icon: <Fingerprint /> },
    { id: 'gestures', label: '手语预设', icon: <TouchApp /> },
    { id: 'comparison', label: '对比', icon: <Gesture /> },
  ]

  const renderOverview = () => (
    <Box>
      <Typography variant="h5" gutterBottom>
        手语小人精细化改进总览
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" color="error" gutterBottom>
                改进前的问题
              </Typography>
              <Box sx={{ textAlign: 'center', mb: 2 }}>
                <Box
                  sx={{
                    width: 60,
                    height: 100,
                    bgcolor: 'grey.300',
                    mx: 'auto',
                    mb: 1,
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <Typography variant="caption">简单方块</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  手臂只是简单的几何体
                </Typography>
              </Box>
              <Typography variant="body2" paragraph>
                • 无法表现手指细节
              </Typography>
              <Typography variant="body2" paragraph>
                • 缺乏关节控制
              </Typography>
              <Typography variant="body2">
                • 不适合手语识别
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" color="success.main" gutterBottom>
                改进后的优势
              </Typography>
              <Box sx={{ textAlign: 'center', mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5, mb: 1 }}>
                  {[1,2,3,4,5].map(i => (
                    <Box
                      key={i}
                      sx={{
                        width: 8,
                        height: 40,
                        bgcolor: 'primary.main',
                        borderRadius: 1,
                      }}
                    />
                  ))}
                </Box>
                <Typography variant="body2" color="text.secondary">
                  精细的手指模型
                </Typography>
              </Box>
              <Typography variant="body2" paragraph>
                • 21个手部关键点
              </Typography>
              <Typography variant="body2" paragraph>
                • 独立的手指关节
              </Typography>
              <Typography variant="body2">
                • 专业手语支持
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )

  const renderJoints = () => (
    <Box>
      <Typography variant="h5" gutterBottom>
        关节控制系统
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              21个手部关键点
            </Typography>
            
            <Grid container spacing={2}>
              {[
                { name: '手腕', points: 1, color: 'primary.main' },
                { name: '拇指', points: 4, color: 'secondary.main' },
                { name: '食指', points: 4, color: 'success.main' },
                { name: '中指', points: 4, color: 'warning.main' },
                { name: '无名指', points: 4, color: 'error.main' },
                { name: '小指', points: 4, color: 'info.main' },
              ].map((finger) => (
                <Grid item xs={6} sm={4} key={finger.name}>
                  <Box sx={{ textAlign: 'center', p: 2, border: 1, borderColor: 'grey.200', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      {finger.name}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5, mb: 1 }}>
                      {Array.from({ length: finger.points }).map((_, i) => (
                        <Box
                          key={i}
                          sx={{
                            width: 8,
                            height: 8,
                            bgcolor: finger.color,
                            borderRadius: '50%',
                          }}
                        />
                      ))}
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {finger.points} 个关键点
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              技术特性
            </Typography>
            <Typography variant="body2" paragraph>
              • <strong>实时映射</strong>: MediaPipe关键点直接映射到3D模型
            </Typography>
            <Typography variant="body2" paragraph>
              • <strong>独立控制</strong>: 每个关节都可以独立旋转和定位
            </Typography>
            <Typography variant="body2" paragraph>
              • <strong>平滑动画</strong>: 关键点之间的插值确保流畅的动作
            </Typography>
            <Typography variant="body2">
              • <strong>双手协调</strong>: 左右手可以同时进行不同的动作
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )

  const renderGestures = () => (
    <Box>
      <Typography variant="h5" gutterBottom>
        手语预设系统
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              数字手语 (0-5)
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {['0', '1', '2', '3', '4', '5'].map((num) => (
                <Chip
                  key={num}
                  label={num}
                  variant="outlined"
                  sx={{ fontSize: '1.2rem', p: 1 }}
                />
              ))}
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              每个数字都有对应的精确手指配置
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              常用词汇
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {['你好', '谢谢', '再见', '请', '对不起'].map((word) => (
                <Chip
                  key={word}
                  label={word}
                  variant="outlined"
                  color="primary"
                />
              ))}
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              基础交流词汇的标准手语动作
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              自动识别功能
            </Typography>
            <Typography variant="body2" paragraph>
              当用户输入文本时，系统会自动检测是否包含预设的手语词汇，并自动显示对应的手势。
              这大大提高了手语表达的准确性和一致性。
            </Typography>
            <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1, mt: 2 }}>
              <Typography variant="body2" fontFamily="monospace">
                输入: "你好" → 自动显示: 挥手手势<br/>
                输入: "123" → 自动显示: 数字1、2、3的手势序列
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )

  const renderComparison = () => (
    <Box>
      <Typography variant="h5" gutterBottom>
        改进前后对比
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              技术对比表
            </Typography>
            
            <Box sx={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #e0e0e0' }}>特性</th>
                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #e0e0e0', color: '#f44336' }}>改进前</th>
                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #e0e0e0', color: '#4caf50' }}>改进后</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { feature: '手部模型', before: '简单方块', after: '21关键点精细模型' },
                    { feature: '手指控制', before: '无', after: '5指独立控制' },
                    { feature: '关节数量', before: '0', after: '21个关键点' },
                    { feature: '手语预设', before: '无', after: '数字+词汇预设' },
                    { feature: '双手协调', before: '不支持', after: '完全支持' },
                    { feature: '实时映射', before: '不支持', after: 'MediaPipe集成' },
                    { feature: '手语识别适用性', before: '不适用', after: '专业级支持' },
                  ].map((row, index) => (
                    <tr key={index} style={{ backgroundColor: index % 2 === 0 ? '#f9f9f9' : 'white' }}>
                      <td style={{ padding: '12px', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>{row.feature}</td>
                      <td style={{ padding: '12px', borderBottom: '1px solid #e0e0e0', textAlign: 'center', color: '#f44336' }}>❌ {row.before}</td>
                      <td style={{ padding: '12px', borderBottom: '1px solid #e0e0e0', textAlign: 'center', color: '#4caf50' }}>✅ {row.after}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )

  const renderContent = () => {
    switch (selectedDemo) {
      case 'joints': return renderJoints()
      case 'gestures': return renderGestures()
      case 'comparison': return renderComparison()
      default: return renderOverview()
    }
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* 导航 */}
      <Paper sx={{ mb: 3 }}>
        <Stack direction="row" spacing={1} sx={{ p: 2 }}>
          {demoSections.map((section) => (
            <Button
              key={section.id}
              variant={selectedDemo === section.id ? "contained" : "outlined"}
              startIcon={section.icon}
              onClick={() => setSelectedDemo(section.id)}
            >
              {section.label}
            </Button>
          ))}
        </Stack>
      </Paper>

      {/* 内容 */}
      {renderContent()}
    </Box>
  )
}

export default HandSignDemo
