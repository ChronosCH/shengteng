/**
 * 控制面板组件 - 主要控制按钮和设置
 */

import React, { useState } from 'react'
import {
  Box,
  Button,
  Paper,
  Typography,
  Collapse,
  Grid,
  Slider,
  FormControlLabel,
  Switch,
  Divider,
  Alert,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Settings,
  ExpandMore,
  ExpandLess,
} from '@mui/icons-material'

interface ControlPanelProps {
  isRecognizing: boolean
  isConnected: boolean
  onStartRecognition: () => Promise<void>
  onStopRecognition: () => void
  showSettings: boolean
  onToggleSettings: () => void
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  isRecognizing,
  isConnected,
  onStartRecognition,
  onStopRecognition,
  showSettings,
  onToggleSettings,
}) => {
  const [isStarting, setIsStarting] = useState(false)
  const [confidenceThreshold, setConfidenceThreshold] = useState(60)
  const [autoStart, setAutoStart] = useState(false)
  const [showLandmarks, setShowLandmarks] = useState(true)
  const [enableTTS, setEnableTTS] = useState(true)

  const handleStartRecognition = async () => {
    setIsStarting(true)
    try {
      await onStartRecognition()
    } catch (error) {
      console.error('启动识别失败:', error)
    } finally {
      setIsStarting(false)
    }
  }

  const handleStopRecognition = () => {
    onStopRecognition()
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Grid container spacing={2} alignItems="center">
        {/* 主控制按钮 */}
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {!isRecognizing ? (
              <Button
                variant="contained"
                size="large"
                startIcon={<PlayArrow />}
                onClick={handleStartRecognition}
                disabled={!isConnected || isStarting}
                fullWidth
                sx={{ py: 1.5 }}
              >
                {isStarting ? '启动中...' : '开始识别'}
              </Button>
            ) : (
              <Button
                variant="outlined"
                size="large"
                startIcon={<Stop />}
                onClick={handleStopRecognition}
                fullWidth
                sx={{ py: 1.5 }}
                color="error"
              >
                停止识别
              </Button>
            )}
          </Box>
        </Grid>

        {/* 状态信息 */}
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ textAlign: { xs: 'left', sm: 'center' } }}>
            <Typography variant="body2" color="text.secondary">
              状态: {isRecognizing ? '🟢 识别中' : '⚪ 待机'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              连接: {isConnected ? '🟢 已连接' : '🔴 未连接'}
            </Typography>
          </Box>
        </Grid>

        {/* 设置按钮 */}
        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', justifyContent: { xs: 'flex-start', md: 'flex-end' } }}>
            <Button
              variant="text"
              startIcon={<Settings />}
              endIcon={showSettings ? <ExpandLess /> : <ExpandMore />}
              onClick={onToggleSettings}
            >
              高级设置
            </Button>
          </Box>
        </Grid>

        {/* 高级设置面板 */}
        <Grid item xs={12}>
          <Collapse in={showSettings}>
            <Box sx={{ mt: 2 }}>
              <Divider sx={{ mb: 2 }} />
              
              <Typography variant="h6" gutterBottom>
                识别设置
              </Typography>

              <Grid container spacing={3}>
                {/* 置信度阈值 */}
                <Grid item xs={12} sm={6}>
                  <Typography gutterBottom>
                    置信度阈值: {confidenceThreshold}%
                  </Typography>
                  <Slider
                    value={confidenceThreshold}
                    onChange={(_: any, value: any) => setConfidenceThreshold(value as number)}
                    min={0}
                    max={100}
                    step={5}
                    marks={[
                      { value: 0, label: '0%' },
                      { value: 50, label: '50%' },
                      { value: 100, label: '100%' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                  <Typography variant="caption" color="text.secondary">
                    低于此阈值的识别结果将被过滤
                  </Typography>
                </Grid>

                {/* 功能开关 */}
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={autoStart}
                          onChange={(e) => setAutoStart(e.target.checked)}
                        />
                      }
                      label="自动开始识别"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={showLandmarks}
                          onChange={(e) => setShowLandmarks(e.target.checked)}
                        />
                      }
                      label="显示关键点"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={enableTTS}
                          onChange={(e) => setEnableTTS(e.target.checked)}
                        />
                      }
                      label="语音播报"
                    />
                  </Box>
                </Grid>

                {/* 性能信息 */}
                <Grid item xs={12}>
                  <Alert severity="info" sx={{ mt: 1 }}>
                    <Typography variant="body2">
                      💡 <strong>性能提示:</strong> 
                      关闭关键点显示可以提高性能；
                      调整置信度阈值可以平衡准确性和响应速度。
                    </Typography>
                  </Alert>
                </Grid>
              </Grid>
            </Box>
          </Collapse>
        </Grid>
      </Grid>
    </Paper>
  )
}

export default ControlPanel
