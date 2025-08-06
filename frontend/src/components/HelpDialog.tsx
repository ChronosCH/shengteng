/**
 * 帮助对话框组件 - 提供使用教程和帮助信息
 */

import React, { useState } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from '@mui/material'
import {
  ExpandMore,
  Videocam,
  Wifi,
  RecordVoiceOver,
  ThreeDRotation,
  Settings,
  Help,
  CheckCircle,
  Warning,
  Info,
} from '@mui/icons-material'

interface HelpDialogProps {
  open: boolean
  onClose: () => void
}

const HelpDialog: React.FC<HelpDialogProps> = ({ open, onClose }) => {
  const [activeStep, setActiveStep] = useState(0)

  const tutorialSteps = [
    {
      label: '启动摄像头',
      icon: <Videocam />,
      content: '点击"启动摄像头"按钮，允许浏览器访问您的摄像头。确保您的手部在摄像头视野范围内。',
      tips: ['确保光线充足', '保持手部清晰可见', '避免背景干扰'],
    },
    {
      label: '连接服务器',
      icon: <Wifi />,
      content: '系统会自动连接到手语识别服务器。等待连接状态显示为"已连接"。',
      tips: ['检查网络连接', '确保服务器正在运行', '查看状态指示器'],
    },
    {
      label: '开始识别',
      icon: <RecordVoiceOver />,
      content: '点击"开始识别"按钮，开始进行手语识别。系统会实时分析您的手语动作。',
      tips: ['动作要清晰标准', '保持适当的速度', '注意置信度指示'],
    },
    {
      label: '查看结果',
      icon: <ThreeDRotation />,
      content: '识别结果会显示在字幕区域，同时3D虚拟人会播放相应的手语动画。',
      tips: ['可以调整字体大小', '支持语音播报', '可以复制识别文本'],
    },
  ]

  const faqData = [
    {
      question: '为什么摄像头无法启动？',
      answer: '请检查浏览器权限设置，确保允许网站访问摄像头。在Chrome中，点击地址栏左侧的摄像头图标进行设置。',
      severity: 'warning' as const,
    },
    {
      question: '识别准确率不高怎么办？',
      answer: '确保光线充足，手部动作清晰标准，避免背景干扰。可以在设置中调整置信度阈值。',
      severity: 'info' as const,
    },
    {
      question: '系统支持哪些手语？',
      answer: '目前主要支持中国手语(CSL)的常用词汇和短语。我们正在不断扩展词汇库。',
      severity: 'info' as const,
    },
    {
      question: '如何提高识别速度？',
      answer: '关闭关键点显示，降低视频质量，或者在性能监控中查看系统建议。',
      severity: 'success' as const,
    },
    {
      question: '数据隐私如何保护？',
      answer: '我们只上传关键点数据，原始视频完全在本地处理，不会上传到服务器。',
      severity: 'success' as const,
    },
  ]

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1)
  }

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1)
  }

  const handleReset = () => {
    setActiveStep(0)
  }

  const getSeverityIcon = (severity: 'success' | 'warning' | 'info') => {
    switch (severity) {
      case 'success':
        return <CheckCircle color="success" />
      case 'warning':
        return <Warning color="warning" />
      case 'info':
        return <Info color="info" />
    }
  }

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      scroll="paper"
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Help color="primary" />
          <Typography variant="h6">使用帮助</Typography>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {/* 快速开始教程 */}
        <Typography variant="h6" gutterBottom>
          📚 快速开始教程
        </Typography>

        <Stepper activeStep={activeStep} orientation="vertical">
          {tutorialSteps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel icon={step.icon}>
                {step.label}
              </StepLabel>
              <StepContent>
                <Typography paragraph>
                  {step.content}
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    💡 小贴士:
                  </Typography>
                  <List dense>
                    {step.tips.map((tip, tipIndex) => (
                      <ListItem key={tipIndex} sx={{ py: 0 }}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <CheckCircle color="success" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={tip} />
                      </ListItem>
                    ))}
                  </List>
                </Box>

                <Box sx={{ mb: 1 }}>
                  <Button
                    variant="contained"
                    onClick={handleNext}
                    sx={{ mt: 1, mr: 1 }}
                    disabled={index === tutorialSteps.length - 1}
                  >
                    {index === tutorialSteps.length - 1 ? '完成' : '下一步'}
                  </Button>
                  <Button
                    disabled={index === 0}
                    onClick={handleBack}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    上一步
                  </Button>
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>

        {activeStep === tutorialSteps.length && (
          <Paper square elevation={0} sx={{ p: 3, mt: 2 }}>
            <Typography>🎉 恭喜！您已经完成了所有教程步骤。</Typography>
            <Button onClick={handleReset} sx={{ mt: 1, mr: 1 }}>
              重新开始
            </Button>
          </Paper>
        )}

        {/* 常见问题 */}
        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          ❓ 常见问题
        </Typography>

        {faqData.map((faq, index) => (
          <Accordion key={index}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getSeverityIcon(faq.severity)}
                <Typography>{faq.question}</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography>{faq.answer}</Typography>
            </AccordionDetails>
          </Accordion>
        ))}

        {/* 键盘快捷键 */}
        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          ⌨️ 键盘快捷键
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          <Chip label="空格键 - 开始/停止识别" />
          <Chip label="Esc - 关闭对话框" />
          <Chip label="Tab - 切换焦点" />
          <Chip label="Enter - 确认操作" />
          <Chip label="F11 - 全屏模式" />
        </Box>

        {/* 技术信息 */}
        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          🔧 技术信息
        </Typography>

        <List>
          <ListItem>
            <ListItemText 
              primary="手语识别模型" 
              secondary="ST-Transformer-CTC，支持连续手语识别"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="关键点提取" 
              secondary="MediaPipe Holistic，提取543个关键点"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="3D渲染" 
              secondary="Three.js + React Three Fiber"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="实时通信" 
              secondary="WebSocket，目标延迟 ≤ 150ms"
            />
          </ListItem>
        </List>

        {/* 联系信息 */}
        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          📞 获取帮助
        </Typography>

        <Typography paragraph>
          如果您遇到问题或有建议，请通过以下方式联系我们：
        </Typography>

        <List>
          <ListItem>
            <ListItemText 
              primary="技术支持" 
              secondary="support@signavatar.com"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="用户反馈" 
              secondary="feedback@signavatar.com"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="项目主页" 
              secondary="https://github.com/signavatar/web"
            />
          </ListItem>
        </List>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} variant="contained">
          关闭
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default HelpDialog
