/**
 * 多模态传感器控制面板
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Sensors as SensorsIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  Visibility as VisibilityIcon,
  TouchApp as TouchIcon,
  RotateRight as RotateIcon,
  Analytics as AnalyticsIcon,
  SignalCellularAlt as SignalIcon,
} from '@mui/icons-material';

interface SensorConfig {
  emgEnabled: boolean;
  imuEnabled: boolean;
  visualEnabled: boolean;
  fusionMode: 'early' | 'late' | 'hybrid';
}

interface SensorStats {
  totalSamples: number;
  emgSamples: number;
  imuSamples: number;
  visualSamples: number;
  fusionPredictions: number;
  averageLatency: number;
  isCollecting: boolean;
  bufferSizes: {
    emg: number;
    imu: number;
    visual: number;
  };
}

interface MultimodalSensorPanelProps {
  onPredictionResult?: (result: any) => void;
}

const MultimodalSensorPanel: React.FC<MultimodalSensorPanelProps> = ({
  onPredictionResult,
}) => {
  const [config, setConfig] = useState<SensorConfig>({
    emgEnabled: true,
    imuEnabled: true,
    visualEnabled: true,
    fusionMode: 'early',
  });

  const [isCollecting, setIsCollecting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<SensorStats | null>(null);
  const [lastPrediction, setLastPrediction] = useState<any>(null);

  // 获取统计信息
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/stats');
      const result = await response.json();
      
      if (result.success) {
        setStats(result.data);
        setIsCollecting(result.data.isCollecting);
      }
    } catch (err) {
      console.error('获取统计信息失败:', err);
    }
  }, []);

  // 定期更新统计信息
  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  // 开始数据收集
  const handleStartCollection = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/multimodal/start-collection', {
        method: 'POST',
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      setIsCollecting(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动收集失败');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 停止数据收集
  const handleStopCollection = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/multimodal/stop-collection', {
        method: 'POST',
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      setIsCollecting(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : '停止收集失败');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 执行预测
  const handlePredict = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/multimodal/predict', {
        method: 'POST',
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      setLastPrediction(result.data);
      onPredictionResult?.(result.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '预测失败');
    } finally {
      setIsLoading(false);
    }
  }, [onPredictionResult]);

  // 更新配置
  const handleConfigUpdate = useCallback(async (newConfig: Partial<SensorConfig>) => {
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);

    try {
      const response = await fetch('/api/multimodal/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          emg_enabled: updatedConfig.emgEnabled,
          imu_enabled: updatedConfig.imuEnabled,
          visual_enabled: updatedConfig.visualEnabled,
          fusion_mode: updatedConfig.fusionMode,
        }),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '配置更新失败');
      // 回滚配置
      setConfig(config);
    }
  }, [config]);

  const getModalityIcon = (modality: string) => {
    switch (modality) {
      case 'emg': return <TouchIcon />;
      case 'imu': return <RotateIcon />;
      case 'visual': return <VisibilityIcon />;
      default: return <SensorsIcon />;
    }
  };

  const getModalityName = (modality: string) => {
    switch (modality) {
      case 'emg': return '肌电信号';
      case 'imu': return '惯性测量';
      case 'visual': return '视觉关键点';
      default: return modality;
    }
  };

  const getFusionModeDescription = (mode: string) => {
    switch (mode) {
      case 'early': return '早期融合 - 特征级融合';
      case 'late': return '后期融合 - 决策级融合';
      case 'hybrid': return '混合融合 - 多级融合';
      default: return mode;
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SensorsIcon color="primary" />
          多模态传感器
        </Typography>

        {/* 传感器配置 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            传感器配置
          </Typography>
          <Grid container spacing={1}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.emgEnabled}
                    onChange={(e) => handleConfigUpdate({ emgEnabled: e.target.checked })}
                    disabled={isCollecting || isLoading}
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TouchIcon fontSize="small" />
                    EMG 肌电信号
                  </Box>
                }
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.imuEnabled}
                    onChange={(e) => handleConfigUpdate({ imuEnabled: e.target.checked })}
                    disabled={isCollecting || isLoading}
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <RotateIcon fontSize="small" />
                    IMU 惯性测量
                  </Box>
                }
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.visualEnabled}
                    onChange={(e) => handleConfigUpdate({ visualEnabled: e.target.checked })}
                    disabled={isCollecting || isLoading}
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <VisibilityIcon fontSize="small" />
                    视觉关键点
                  </Box>
                }
              />
            </Grid>
          </Grid>
        </Box>

        {/* 融合模式 */}
        <Box sx={{ mb: 2 }}>
          <FormControl fullWidth disabled={isCollecting || isLoading}>
            <InputLabel>融合模式</InputLabel>
            <Select
              value={config.fusionMode}
              onChange={(e) => handleConfigUpdate({ fusionMode: e.target.value as any })}
            >
              <MenuItem value="early">早期融合</MenuItem>
              <MenuItem value="late">后期融合</MenuItem>
              <MenuItem value="hybrid">混合融合</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            {getFusionModeDescription(config.fusionMode)}
          </Typography>
        </Box>

        {/* 控制按钮 */}
        <Grid container spacing={1} sx={{ mb: 2 }}>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant={isCollecting ? "outlined" : "contained"}
              onClick={isCollecting ? handleStopCollection : handleStartCollection}
              disabled={isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : (isCollecting ? <StopIcon /> : <PlayIcon />)}
              color={isCollecting ? "error" : "primary"}
            >
              {isCollecting ? '停止收集' : '开始收集'}
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="outlined"
              onClick={handlePredict}
              disabled={!isCollecting || isLoading}
              startIcon={<AnalyticsIcon />}
            >
              执行预测
            </Button>
          </Grid>
        </Grid>

        {/* 统计信息 */}
        {stats && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SignalIcon />
              实时统计
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  总样本: {stats.totalSamples}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  预测次数: {stats.fusionPredictions}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  EMG: {stats.emgSamples}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  IMU: {stats.imuSamples}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  平均延迟: {(stats.averageLatency * 1000).toFixed(1)}ms
                </Typography>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* 最近预测结果 */}
        {lastPrediction && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              最近预测
            </Typography>
            <Alert severity="info">
              <Typography variant="body2">
                <strong>{lastPrediction.prediction}</strong>
              </Typography>
              <Typography variant="caption" color="text.secondary">
                置信度: {(lastPrediction.confidence * 100).toFixed(1)}% | 
                使用模态: {lastPrediction.modalities_used?.join(', ')}
              </Typography>
            </Alert>
          </Box>
        )}

        {/* 错误提示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* 状态指示 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip
            label={isCollecting ? "收集中" : "已停止"}
            color={isCollecting ? "success" : "default"}
            size="small"
            icon={isCollecting ? <PlayIcon /> : <StopIcon />}
          />
          {stats && (
            <Chip
              label={`缓冲区: ${stats.bufferSizes.emg + stats.bufferSizes.imu + stats.bufferSizes.visual}`}
              size="small"
              variant="outlined"
            />
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default MultimodalSensorPanel;
