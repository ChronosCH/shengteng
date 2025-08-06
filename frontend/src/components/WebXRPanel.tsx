/**
 * WebXR AR 控制面板
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ViewInAr as ViewInArIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  CenterFocusStrong as CenterFocusIcon,
  ThreeDRotation as ThreeDRotationIcon,
  Opacity as OpacityIcon,
  ZoomIn as ZoomInIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

import { webxrService, XRSessionInfo, AROverlayConfig } from '../services/webxrService';

interface WebXRPanelProps {
  onSessionStart?: () => void;
  onSessionEnd?: () => void;
  avatarMesh?: THREE.Object3D;
}

const WebXRPanel: React.FC<WebXRPanelProps> = ({
  onSessionStart,
  onSessionEnd,
  avatarMesh,
}) => {
  const [isSupported, setIsSupported] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionInfo, setSessionInfo] = useState<XRSessionInfo | null>(null);
  const [overlayConfig, setOverlayConfig] = useState<AROverlayConfig>({
    avatarScale: 1.0,
    avatarPosition: { x: 0, y: 0, z: -2 },
    opacity: 0.8,
    followCamera: true,
    showBackground: false,
  });

  // 检查 WebXR 支持
  useEffect(() => {
    const checkSupport = async () => {
      try {
        const supported = await webxrService.checkWebXRSupport();
        setIsSupported(supported);
      } catch (err) {
        console.error('Error checking WebXR support:', err);
        setIsSupported(false);
      }
    };

    checkSupport();
  }, []);

  // 更新会话信息
  const updateSessionInfo = useCallback(() => {
    const info = webxrService.getSessionInfo();
    setSessionInfo(info);
  }, []);

  // 定期更新会话信息
  useEffect(() => {
    updateSessionInfo();
    const interval = setInterval(updateSessionInfo, 1000);
    return () => clearInterval(interval);
  }, [updateSessionInfo]);

  // 开始 AR 会话
  const handleStartAR = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await webxrService.initializeSession({
        mode: 'immersive-ar',
        requiredFeatures: ['local'],
        optionalFeatures: ['dom-overlay', 'hit-test', 'anchors']
      });

      // 添加 Avatar 到场景
      if (avatarMesh) {
        webxrService.addSignLanguageAvatar(avatarMesh, overlayConfig);
      }

      // 开始渲染循环
      webxrService.startRenderLoop();

      onSessionStart?.();
      updateSessionInfo();
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动 AR 会话失败');
    } finally {
      setIsLoading(false);
    }
  }, [avatarMesh, overlayConfig, onSessionStart, updateSessionInfo]);

  // 结束 AR 会话
  const handleEndAR = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await webxrService.endSession();
      onSessionEnd?.();
      updateSessionInfo();
    } catch (err) {
      setError(err instanceof Error ? err.message : '结束 AR 会话失败');
    } finally {
      setIsLoading(false);
    }
  }, [onSessionEnd, updateSessionInfo]);

  // 更新叠加配置
  const handleConfigChange = useCallback((newConfig: Partial<AROverlayConfig>) => {
    const updatedConfig = { ...overlayConfig, ...newConfig };
    setOverlayConfig(updatedConfig);

    // 如果会话活跃，立即应用配置
    if (sessionInfo?.isActive) {
      webxrService.updateOverlayConfig(newConfig);
    }
  }, [overlayConfig, sessionInfo]);

  // 重置 Avatar 位置
  const handleResetPosition = useCallback(() => {
    handleConfigChange({
      avatarPosition: { x: 0, y: 0, z: -2 },
      avatarScale: 1.0,
    });
  }, [handleConfigChange]);

  const getSessionStatusColor = (isActive: boolean) => {
    return isActive ? 'success' : 'default';
  };

  const getSessionStatusText = (isActive: boolean) => {
    return isActive ? 'AR 会话活跃' : 'AR 会话未启动';
  };

  if (!isSupported) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ViewInArIcon color="disabled" />
            WebXR AR 叠加
          </Typography>
          <Alert severity="warning">
            您的浏览器不支持 WebXR。请使用支持 WebXR 的现代浏览器，如 Chrome、Edge 或 Firefox Reality。
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ViewInArIcon color="primary" />
          WebXR AR 叠加
        </Typography>

        {/* 会话状态 */}
        <Box sx={{ mb: 2 }}>
          <Grid container spacing={1} alignItems="center">
            <Grid item>
              <Chip
                label={getSessionStatusText(sessionInfo?.isActive || false)}
                color={getSessionStatusColor(sessionInfo?.isActive || false)}
                size="small"
                icon={sessionInfo?.isActive ? <ViewInArIcon /> : <StopIcon />}
              />
            </Grid>
            {sessionInfo?.mode && (
              <Grid item>
                <Chip
                  label={`模式: ${sessionInfo.mode}`}
                  size="small"
                  variant="outlined"
                />
              </Grid>
            )}
          </Grid>
        </Box>

        {/* 控制按钮 */}
        <Box sx={{ mb: 2 }}>
          {!sessionInfo?.isActive ? (
            <Button
              fullWidth
              variant="contained"
              onClick={handleStartAR}
              disabled={isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : <ViewInArIcon />}
              color="primary"
            >
              {isLoading ? '启动中...' : '开始 AR 体验'}
            </Button>
          ) : (
            <Button
              fullWidth
              variant="outlined"
              onClick={handleEndAR}
              disabled={isLoading}
              startIcon={<StopIcon />}
              color="error"
            >
              结束 AR 会话
            </Button>
          )}
        </Box>

        {/* AR 叠加配置 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SettingsIcon />
            叠加配置
          </Typography>

          {/* Avatar 缩放 */}
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ZoomInIcon fontSize="small" />
              Avatar 大小: {overlayConfig.avatarScale.toFixed(1)}
            </Typography>
            <Slider
              value={overlayConfig.avatarScale}
              onChange={(_, value) => handleConfigChange({ avatarScale: value as number })}
              min={0.1}
              max={3.0}
              step={0.1}
              disabled={isLoading}
              marks={[
                { value: 0.5, label: '0.5x' },
                { value: 1.0, label: '1.0x' },
                { value: 2.0, label: '2.0x' },
              ]}
            />
          </Box>

          {/* 透明度 */}
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <OpacityIcon fontSize="small" />
              透明度: {(overlayConfig.opacity * 100).toFixed(0)}%
            </Typography>
            <Slider
              value={overlayConfig.opacity}
              onChange={(_, value) => handleConfigChange({ opacity: value as number })}
              min={0.1}
              max={1.0}
              step={0.1}
              disabled={isLoading}
              marks={[
                { value: 0.3, label: '30%' },
                { value: 0.5, label: '50%' },
                { value: 0.8, label: '80%' },
                { value: 1.0, label: '100%' },
              ]}
            />
          </Box>

          {/* 位置控制 */}
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>Avatar 位置</Typography>
            <Grid container spacing={1}>
              <Grid item xs={4}>
                <Typography variant="caption">X: {overlayConfig.avatarPosition.x.toFixed(1)}</Typography>
                <Slider
                  value={overlayConfig.avatarPosition.x}
                  onChange={(_, value) => handleConfigChange({
                    avatarPosition: { ...overlayConfig.avatarPosition, x: value as number }
                  })}
                  min={-5}
                  max={5}
                  step={0.1}
                  disabled={isLoading}
                  orientation="vertical"
                  sx={{ height: 60 }}
                />
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption">Y: {overlayConfig.avatarPosition.y.toFixed(1)}</Typography>
                <Slider
                  value={overlayConfig.avatarPosition.y}
                  onChange={(_, value) => handleConfigChange({
                    avatarPosition: { ...overlayConfig.avatarPosition, y: value as number }
                  })}
                  min={-3}
                  max={3}
                  step={0.1}
                  disabled={isLoading}
                  orientation="vertical"
                  sx={{ height: 60 }}
                />
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption">Z: {overlayConfig.avatarPosition.z.toFixed(1)}</Typography>
                <Slider
                  value={overlayConfig.avatarPosition.z}
                  onChange={(_, value) => handleConfigChange({
                    avatarPosition: { ...overlayConfig.avatarPosition, z: value as number }
                  })}
                  min={-10}
                  max={0}
                  step={0.1}
                  disabled={isLoading}
                  orientation="vertical"
                  sx={{ height: 60 }}
                />
              </Grid>
            </Grid>
          </Box>

          {/* 其他选项 */}
          <Box sx={{ mb: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={overlayConfig.followCamera}
                  onChange={(e) => handleConfigChange({ followCamera: e.target.checked })}
                  disabled={isLoading}
                />
              }
              label="跟随相机"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={overlayConfig.showBackground}
                  onChange={(e) => handleConfigChange({ showBackground: e.target.checked })}
                  disabled={isLoading}
                />
              }
              label="显示背景"
            />
          </Box>

          {/* 重置按钮 */}
          <Button
            fullWidth
            variant="outlined"
            onClick={handleResetPosition}
            disabled={isLoading}
            startIcon={<CenterFocusIcon />}
            size="small"
          >
            重置位置
          </Button>
        </Box>

        {/* 会话信息 */}
        {sessionInfo && sessionInfo.isActive && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <InfoIcon />
              会话信息
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  环境混合: {sessionInfo.environmentBlendMode || 'unknown'}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  输入源: {sessionInfo.inputSources.length} 个
                </Typography>
              </Grid>
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
            💡 WebXR AR 功能可以在真实环境中叠加手语 Avatar。
            需要支持 WebXR 的设备和浏览器。建议在光线充足的环境中使用。
          </Typography>
        </Alert>
      </CardContent>
    </Card>
  );
};

export default WebXRPanel;
