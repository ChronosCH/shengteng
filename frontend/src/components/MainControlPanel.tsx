/**
 * 主控制面板 - 集成所有创新功能
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Tabs,
  Tab,
  Typography,
  Paper,
  Chip,
  Alert,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  AutoAwesome as DiffusionIcon,
  Security as PrivacyIcon,
  Sensors as SensorIcon,
  Accessibility as HapticIcon,
  ViewInAr as ARIcon,
  School as FederatedIcon,
  Visibility as AttentionIcon,
  SportsEsports as GameIcon,
} from '@mui/icons-material';

import DiffusionPanel from './DiffusionPanel';
import PrivacyPanel from './PrivacyPanel';
import MultimodalSensorPanel from './MultimodalSensorPanel';
import HapticPanel from './HapticPanel';
import WebXRPanel from './WebXRPanel';
import FederatedLearningPanel from './FederatedLearningPanel';
import AttentionVisualizationPanel from './AttentionVisualizationPanel';
import GamePlaygroundPanel from './GamePlaygroundPanel';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`control-tabpanel-${index}`}
      aria-labelledby={`control-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

interface MainControlPanelProps {
  landmarks?: number[][];
  isActive?: boolean;
  text?: string;
  isRecognizing: boolean;
  isConnected: boolean;
  onStartRecognition: () => Promise<void>;
  onStopRecognition: () => void;
  onConnect: () => Promise<void>;
}

const MainControlPanel: React.FC<MainControlPanelProps> = ({
  landmarks,
  isActive = false,
  text = "",
  isRecognizing,
  isConnected,
  onStartRecognition,
  onStopRecognition,
  onConnect,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [currentSignSequence, setCurrentSignSequence] = useState<any>(null);
  const [avatarMesh, setAvatarMesh] = useState<THREE.Object3D | null>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleSequenceGenerated = useCallback((sequence: any) => {
    setCurrentSignSequence(sequence);
  }, []);

  const handleAvatarMeshReady = useCallback((mesh: THREE.Object3D) => {
    setAvatarMesh(mesh);
  }, []);

  const handleStartStop = async () => {
    if (isRecognizing) {
      onStopRecognition()
    } else {
      try {
        await onStartRecognition()
      } catch (error) {
        console.error('启动识别失败:', error)
      }
    }
  }

  const tabs = [
    {
      label: "Diffusion 生成",
      icon: <DiffusionIcon />,
      component: <DiffusionPanel onSequenceGenerated={handleSequenceGenerated} />
    },
    {
      label: "隐私保护",
      icon: <PrivacyIcon />,
      component: <PrivacyPanel />
    },
    {
      label: "多模态传感",
      icon: <SensorIcon />,
      component: <MultimodalSensorPanel />
    },
    {
      label: "触觉反馈",
      icon: <HapticIcon />,
      component: <HapticPanel />
    },
    {
      label: "WebXR AR",
      icon: <ARIcon />,
      component: <WebXRPanel avatarMesh={avatarMesh || undefined} />
    },
    {
      label: "联邦学习",
      icon: <FederatedIcon />,
      component: <FederatedLearningPanel />
    },
    {
      label: "注意力可视化",
      icon: <AttentionIcon />,
      component: <AttentionVisualizationPanel landmarks={landmarks} />
    },
    {
      label: "游戏化学习",
      icon: <GameIcon />,
      component: <GamePlaygroundPanel />
    }
  ];

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 标题栏 */}
      <Paper sx={{ p: 2, mb: 1 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DashboardIcon color="primary" />
          SignAvatar Web - 创新功能控制台
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip label="8项创新功能" color="primary" />
          <Chip label="实时手语识别" color="success" />
          <Chip label="隐私保护" color="warning" />
          <Chip label="多模态融合" color="info" />
          <Chip label="WebXR支持" color="secondary" />
        </Box>
      </Paper>

      <Grid container spacing={2} sx={{ flex: 1, overflow: 'hidden' }}>
        {/* 左侧控制面板 */}
        <Grid item xs={12} md={4} sx={{ height: '100%', overflow: 'hidden' }}>
          <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 功能选项卡 */}
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              {tabs.map((tab, index) => (
                <Tab
                  key={index}
                  icon={tab.icon}
                  label={tab.label}
                  iconPosition="start"
                  sx={{ minHeight: 48 }}
                />
              ))}
            </Tabs>

            {/* 选项卡内容 */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
              {tabs.map((tab, index) => (
                <TabPanel key={index} value={activeTab} index={index}>
                  {tab.component}
                </TabPanel>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* 右侧显示区域 */}
        <Grid item xs={12} md={8} sx={{ height: '100%' }}>
          <Paper sx={{ height: '100%', p: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography variant="h6" color="text.secondary">
              {text || '手语播报显示区域'}
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* 底部状态栏 */}
      <Paper sx={{ p: 1, mt: 1 }}>
        <Alert severity="info" sx={{ mb: 0 }}>
          <Typography variant="body2">
            💡 SignAvatar Web 集成了8项创新功能：
            ① Diffusion生成式手语 ② 隐私保护数据采集 ③ 多模态传感融合 ④ 触觉反馈系统 
            ⑤ WebXR AR叠加 ⑥ 联邦学习 ⑦ 注意力可视化 ⑧ 游戏化学习
          </Typography>
        </Alert>
      </Paper>
    </Box>
  );
};

export default MainControlPanel;
