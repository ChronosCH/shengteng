/**
 * 触觉反馈控制面板
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
} from '@mui/material';
import {
  Vibration as VibrationIcon,
  TouchApp as TouchIcon,
  Stop as StopIcon,
  Warning as WarningIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayIcon,
  Accessibility as AccessibilityIcon,
  Fingerprint as FingerprintIcon,
  DeviceHub as DeviceHubIcon,
} from '@mui/icons-material';

interface HapticStats {
  totalMessages: number;
  successfulOutputs: number;
  averageLatency: number;
  deviceStatus: string;
  isActive: boolean;
  queueSize: number;
  currentMessage: string | null;
  deviceConfig: {
    numActuators: number;
    brailleCells: number;
    semanticPatterns: number;
    brailleCharacters: number;
  };
}

interface HapticPanelProps {
  onHapticSent?: (result: any) => void;
}

const HapticPanel: React.FC<HapticPanelProps> = ({
  onHapticSent,
}) => {
  const [message, setMessage] = useState('');
  const [semanticType, setSemanticType] = useState('你好');
  const [intensity, setIntensity] = useState('medium');
  const [useBraille, setUseBraille] = useState(true);
  const [useHaptic, setUseHaptic] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<HapticStats | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<any>(null);

  // 预定义的语义类型
  const semanticTypes = [
    '你好', '谢谢', '再见', '开心', '难过', '生气',
    '左', '右', '上', '下', '1', '2', '3', '4', '5'
  ];

  // 获取统计信息
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch('/api/haptic/stats');
      const result = await response.json();
      
      if (result.success) {
        setStats(result.data);
      }
    } catch (err) {
      console.error('获取触觉统计信息失败:', err);
    }
  }, []);

  // 定期更新统计信息
  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 3000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  // 发送触觉消息
  const handleSendMessage = useCallback(async () => {
    if (!message.trim()) {
      setError('请输入要发送的消息');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/haptic/send-message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: message,
          use_braille: useBraille,
          use_haptic: useHaptic,
        }),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      onHapticSent?.(result.data);
      setMessage(''); // 清空输入
    } catch (err) {
      setError(err instanceof Error ? err.message : '发送失败');
    } finally {
      setIsLoading(false);
    }
  }, [message, useBraille, useHaptic, onHapticSent]);

  // 发送语义反馈
  const handleSendSemantic = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/haptic/send-semantic', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          semantic_type: semanticType,
          intensity: intensity,
        }),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      onHapticSent?.(result.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '发送语义反馈失败');
    } finally {
      setIsLoading(false);
    }
  }, [semanticType, intensity, onHapticSent]);

  // 发送紧急警报
  const handleEmergencyAlert = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/haptic/emergency-alert', {
        method: 'POST',
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '发送紧急警报失败');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 停止播放
  const handleStopPlayback = useCallback(async () => {
    try {
      const response = await fetch('/api/haptic/stop-playback', {
        method: 'POST',
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '停止播放失败');
    }
  }, []);

  // 测试设备
  const handleTestDevices = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/haptic/test-devices', {
        method: 'POST',
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      setDeviceStatus(result.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '设备测试失败');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getDeviceStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'success';
      case 'mock': return 'warning';
      case 'disconnected': return 'error';
      default: return 'default';
    }
  };

  const getIntensityColor = (level: string) => {
    switch (level) {
      case 'low': return 'info';
      case 'medium': return 'warning';
      case 'high': return 'error';
      default: return 'default';
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AccessibilityIcon color="primary" />
          触觉反馈系统
        </Typography>

        {/* 设备状态 */}
        {stats && (
          <Box sx={{ mb: 2 }}>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Chip
                  label={`设备: ${stats.deviceStatus}`}
                  color={getDeviceStatusColor(stats.deviceStatus)}
                  size="small"
                  icon={<DeviceHubIcon />}
                />
              </Grid>
              <Grid item xs={6}>
                <Chip
                  label={stats.isActive ? "播放中" : "空闲"}
                  color={stats.isActive ? "success" : "default"}
                  size="small"
                  icon={stats.isActive ? <PlayIcon /> : <StopIcon />}
                />
              </Grid>
            </Grid>
          </Box>
        )}

        {/* 文本消息输入 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            文本消息
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={2}
            label="输入要转换为触觉的文本"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            disabled={isLoading}
            placeholder="例如：你好，欢迎使用触觉反馈系统"
          />
          
          {/* 输出选项 */}
          <Box sx={{ mt: 1, display: 'flex', gap: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={useHaptic}
                  onChange={(e) => setUseHaptic(e.target.checked)}
                  disabled={isLoading}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <VibrationIcon fontSize="small" />
                  触觉模式
                </Box>
              }
            />
            <FormControlLabel
              control={
                <Switch
                  checked={useBraille}
                  onChange={(e) => setUseBraille(e.target.checked)}
                  disabled={isLoading}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <FingerprintIcon fontSize="small" />
                  盲文输出
                </Box>
              }
            />
          </Box>

          <Button
            fullWidth
            variant="contained"
            onClick={handleSendMessage}
            disabled={isLoading || !message.trim()}
            sx={{ mt: 1 }}
            startIcon={isLoading ? <CircularProgress size={20} /> : <TouchIcon />}
          >
            发送触觉消息
          </Button>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* 语义反馈 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            快速语义反馈
          </Typography>
          <Grid container spacing={1}>
            <Grid item xs={8}>
              <FormControl fullWidth size="small">
                <InputLabel>语义类型</InputLabel>
                <Select
                  value={semanticType}
                  onChange={(e) => setSemanticType(e.target.value)}
                  disabled={isLoading}
                >
                  {semanticTypes.map((type) => (
                    <MenuItem key={type} value={type}>
                      {type}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={4}>
              <FormControl fullWidth size="small">
                <InputLabel>强度</InputLabel>
                <Select
                  value={intensity}
                  onChange={(e) => setIntensity(e.target.value)}
                  disabled={isLoading}
                >
                  <MenuItem value="low">低</MenuItem>
                  <MenuItem value="medium">中</MenuItem>
                  <MenuItem value="high">高</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          
          <Button
            fullWidth
            variant="outlined"
            onClick={handleSendSemantic}
            disabled={isLoading}
            sx={{ mt: 1 }}
            startIcon={<VibrationIcon />}
          >
            发送语义反馈
          </Button>
        </Box>

        {/* 控制按钮 */}
        <Grid container spacing={1} sx={{ mb: 2 }}>
          <Grid item xs={4}>
            <Button
              fullWidth
              variant="outlined"
              color="error"
              onClick={handleEmergencyAlert}
              disabled={isLoading}
              startIcon={<WarningIcon />}
              size="small"
            >
              紧急警报
            </Button>
          </Grid>
          <Grid item xs={4}>
            <Button
              fullWidth
              variant="outlined"
              onClick={handleStopPlayback}
              disabled={isLoading}
              startIcon={<StopIcon />}
              size="small"
            >
              停止播放
            </Button>
          </Grid>
          <Grid item xs={4}>
            <Button
              fullWidth
              variant="outlined"
              onClick={handleTestDevices}
              disabled={isLoading}
              startIcon={<SettingsIcon />}
              size="small"
            >
              测试设备
            </Button>
          </Grid>
        </Grid>

        {/* 设备测试结果 */}
        {deviceStatus && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              设备测试结果
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Chip
                  label={`触觉设备: ${deviceStatus.haptic_device ? '正常' : '异常'}`}
                  color={deviceStatus.haptic_device ? 'success' : 'error'}
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <Chip
                  label={`盲文设备: ${deviceStatus.braille_device ? '正常' : '异常'}`}
                  color={deviceStatus.braille_device ? 'success' : 'error'}
                  size="small"
                />
              </Grid>
            </Grid>
          </Box>
        )}

        {/* 统计信息 */}
        {stats && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              运行统计
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  总消息: {stats.totalMessages}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  成功输出: {stats.successfulOutputs}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  队列大小: {stats.queueSize}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  平均延迟: {(stats.averageLatency * 1000).toFixed(1)}ms
                </Typography>
              </Grid>
              {stats.currentMessage && (
                <Grid item xs={12}>
                  <Typography variant="body2" color="primary">
                    当前播放: {stats.currentMessage}
                  </Typography>
                </Grid>
              )}
            </Grid>
          </Box>
        )}

        {/* 错误提示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* 使用说明 */}
        <Alert severity="info">
          <Typography variant="body2">
            💡 触觉反馈系统支持文本转触觉和盲文输出，适用于盲聋用户。
            系统会自动将识别到的手语转换为触觉反馈。
          </Typography>
        </Alert>
      </CardContent>
    </Card>
  );
};

export default HapticPanel;
