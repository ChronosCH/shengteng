import { useState } from 'react'
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Tab,
  Tabs,
  Paper,
  Chip,
  Stack,
  Alert,
  Switch,
  FormControlLabel,
  Slider,
  Fade,
  Badge,
} from '@mui/material'
import {
  Science,
  CloudSync,
  Vibration,
  Psychology,
  Gamepad,
  ViewInAr,
  BugReport,
  NewReleases,
  Warning,
  Settings,
} from '@mui/icons-material'

import ErrorBoundary from '../components/ErrorBoundary'
import FederatedLearningPanel from '../components/FederatedLearningPanel'
import HapticPanel from '../components/HapticPanel'
import DiffusionPanel from '../components/DiffusionPanel'
import MultimodalSensorPanel from '../components/MultimodalSensorPanel'
import GamePlaygroundPanel from '../components/GamePlaygroundPanel'
import WebXRPanel from '../components/WebXRPanel'
import AttentionVisualizationPanel from '../components/AttentionVisualizationPanel'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`lab-tabpanel-${index}`}
      aria-labelledby={`lab-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  )
}

function LabPage() {
  const [currentTab, setCurrentTab] = useState(0)
  const [experimentalFeatures, setExperimentalFeatures] = useState({
    enableFederatedLearning: false,
    enableHapticFeedback: false,
    enableDiffusionModel: false,
    enableMultimodalSensor: false,
    enableGameMode: false,
    enableWebXR: false,
    enableAttentionVisualization: false,
  })

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue)
  }

  const handleFeatureToggle = (feature: string, enabled: boolean) => {
    setExperimentalFeatures(prev => ({
      ...prev,
      [feature]: enabled,
    }))
  }

  // 处理扩散模型生成的手语序列
  const handleSequenceGenerated = (sequence: any) => {
    console.log('Generated sign sequence:', sequence)
    // 这里可以添加处理生成序列的逻辑
  }

  const experimentalTabs = [
    {
      label: '联邦学习',
      icon: <CloudSync />,
      component: <FederatedLearningPanel />,
      description: '分布式协作学习',
      status: 'beta',
      feature: 'enableFederatedLearning',
    },
    {
      label: '触觉反馈',
      icon: <Vibration />,
      component: <HapticPanel />,
      description: '触觉交互体验',
      status: 'alpha',
      feature: 'enableHapticFeedback',
    },
    {
      label: '扩散模型',
      icon: <Psychology />,
      component: <DiffusionPanel onSequenceGenerated={handleSequenceGenerated} />,
      description: '生成式AI增强',
      status: 'experimental',
      feature: 'enableDiffusionModel',
    },
    {
      label: '多模态传感',
      icon: <Settings />,
      component: <MultimodalSensorPanel />,
      description: '多传感器融合',
      status: 'beta',
      feature: 'enableMultimodalSensor',
    },
    {
      label: '游戏模式',
      icon: <Gamepad />,
      component: <GamePlaygroundPanel />,
      description: '游戏化学习体验',
      status: 'stable',
      feature: 'enableGameMode',
    },
    {
      label: 'WebXR',
      icon: <ViewInAr />,
      component: <WebXRPanel />,
      description: '虚拟现实体验',
      status: 'experimental',
      feature: 'enableWebXR',
    },
    {
      label: '注意力可视化',
      icon: <Psychology />,
      component: <AttentionVisualizationPanel />,
      description: 'AI注意力机制展示',
      status: 'alpha',
      feature: 'enableAttentionVisualization',
    },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'stable': return 'success'
      case 'beta': return 'info'
      case 'alpha': return 'warning'
      case 'experimental': return 'error'
      default: return 'default'
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'stable': return '稳定版'
      case 'beta': return 'Beta'
      case 'alpha': return 'Alpha'
      case 'experimental': return '实验性'
      default: return ''
    }
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 页面标题 */}
      <Fade in timeout={600}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom sx={{ fontWeight: 600 }}>
            <Science sx={{ mr: 2, fontSize: 'inherit', verticalAlign: 'middle' }} />
            实验室
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
            探索前沿AI技术，体验未来的手语识别功能
          </Typography>
          
          <Alert severity="warning" sx={{ maxWidth: 800, mx: 'auto' }}>
            <Typography variant="body2">
              <strong>注意：</strong> 这里的功能处于实验阶段，可能不稳定或有变化。
              请在测试环境中谨慎使用。
            </Typography>
          </Alert>
        </Box>
      </Fade>

      {/* 实验性功能概览 */}
      <Fade in timeout={800}>
        <Paper sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            实验性功能
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            启用下面的实验性功能来体验最新的AI技术
          </Typography>
          
          <Grid container spacing={3}>
            {experimentalTabs.map((tab, index) => (
              <Grid item xs={12} sm={6} md={4} key={tab.label}>
                <Card
                  sx={{
                    height: '100%',
                    transition: 'all 0.2s',
                    opacity: experimentalFeatures[tab.feature as keyof typeof experimentalFeatures] ? 1 : 0.7,
                    '&:hover': { transform: 'translateY(-2px)' },
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Badge
                        badgeContent={<NewReleases sx={{ fontSize: 16 }} />}
                        color={getStatusColor(tab.status) as any}
                        sx={{ mr: 2 }}
                      >
                        {tab.icon}
                      </Badge>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {tab.label}
                        </Typography>
                        <Chip
                          label={getStatusLabel(tab.status)}
                          size="small"
                          color={getStatusColor(tab.status) as any}
                          sx={{ mt: 0.5 }}
                        />
                      </Box>
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {tab.description}
                    </Typography>
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={experimentalFeatures[tab.feature as keyof typeof experimentalFeatures]}
                          onChange={(e) => handleFeatureToggle(tab.feature, e.target.checked)}
                          size="small"
                        />
                      }
                      label="启用功能"
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Fade>

      {/* 实验性功能标签页 */}
      <Fade in timeout={1000}>
        <Paper sx={{ mb: 4 }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              '& .MuiTab-root': { py: 2, minHeight: 64 }
            }}
          >
            {experimentalTabs.map((tab, index) => (
              <Tab
                key={tab.label}
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {tab.icon}
                    <span>{tab.label}</span>
                    <Chip
                      label={getStatusLabel(tab.status)}
                      size="small"
                      color={getStatusColor(tab.status) as any}
                    />
                  </Box>
                }
                disabled={!experimentalFeatures[tab.feature as keyof typeof experimentalFeatures]}
              />
            ))}
          </Tabs>

          {experimentalTabs.map((tab, index) => (
            <TabPanel key={index} value={currentTab} index={index}>
              <Box sx={{ p: 3 }}>
                {experimentalFeatures[tab.feature as keyof typeof experimentalFeatures] ? (
                  <ErrorBoundary>
                    {tab.component}
                  </ErrorBoundary>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Warning sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      功能未启用
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      请先在上方启用 "{tab.label}" 功能
                    </Typography>
                    <Button
                      variant="contained"
                      onClick={() => handleFeatureToggle(tab.feature, true)}
                    >
                      启用 {tab.label}
                    </Button>
                  </Box>
                )}
              </Box>
            </TabPanel>
          ))}
        </Paper>
      </Fade>

      {/* 开发者信息 */}
      <Fade in timeout={1200}>
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 4, bgcolor: 'primary.main', color: 'white' }}>
              <Typography variant="h6" gutterBottom>
                <BugReport sx={{ mr: 1, verticalAlign: 'middle' }} />
                反馈与建议
              </Typography>
              <Typography variant="body2" sx={{ mb: 3, opacity: 0.9 }}>
                在使用实验性功能时遇到问题？或者有改进建议？
                我们非常欢迎您的反馈！
              </Typography>
              <Stack spacing={2}>
                <Typography variant="body2">
                  • 发现Bug时请记录详细的复现步骤
                </Typography>
                <Typography variant="body2">
                  • 功能建议请说明具体的使用场景
                </Typography>
                <Typography variant="body2">
                  • 性能问题请提供设备和浏览器信息
                </Typography>
              </Stack>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 4, bgcolor: 'success.main', color: 'white' }}>
              <Typography variant="h6" gutterBottom>
                🚀 即将推出
              </Typography>
              <Typography variant="body2" sx={{ mb: 3, opacity: 0.9 }}>
                我们正在开发更多令人兴奋的功能：
              </Typography>
              <Stack spacing={2}>
                <Typography variant="body2">
                  • 🧠 神经网络可视化
                </Typography>
                <Typography variant="body2">
                  • 🎯 个性化学习路径
                </Typography>
                <Typography variant="body2">
                  • 🌐 多语言手语支持
                </Typography>
                <Typography variant="body2">
                  • 📱 移动端增强现实
                </Typography>
              </Stack>
            </Paper>
          </Grid>
        </Grid>
      </Fade>

      {/* 免责声明 */}
      <Fade in timeout={1400}>
        <Alert severity="info" sx={{ mt: 4 }}>
          <Typography variant="body2">
            <strong>免责声明：</strong> 
            实验室功能可能包含不稳定的代码，建议仅在测试环境中使用。
            我们不对实验性功能的稳定性和数据安全性承担责任。
            使用前请确保已备份重要数据。
          </Typography>
        </Alert>
      </Fade>
    </Container>
  )
}

export default LabPage