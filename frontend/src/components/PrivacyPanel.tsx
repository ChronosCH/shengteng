/**
 * 隐私保护控制面板
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Security as SecurityIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Shield as ShieldIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

interface PrivacyPanelProps {
  onAnonymizationComplete?: (result: any) => void;
}

interface AnonymizationConfig {
  level: 'low' | 'medium' | 'high';
  preserveGesture: boolean;
  preserveExpression: boolean;
  blurBackground: boolean;
  addNoise: boolean;
  seed?: number;
}

interface PrivacyMetrics {
  anonymizationScore: number;
  utilityScore: number;
  processingTime: number;
  dataSizeReduction: number;
}

const PrivacyPanel: React.FC<PrivacyPanelProps> = ({
  onAnonymizationComplete,
}) => {
  const [config, setConfig] = useState<AnonymizationConfig>({
    level: 'medium',
    preserveGesture: true,
    preserveExpression: false,
    blurBackground: true,
    addNoise: true,
  });
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMetrics, setLastMetrics] = useState<PrivacyMetrics | null>(null);
  const [privacyMode, setPrivacyMode] = useState(false);

  const handleAnonymize = useCallback(async () => {
    setIsProcessing(true);
    setError(null);

    try {
      const response = await fetch('/api/privacy/anonymize-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data_type: 'image',
          level: config.level,
          preserve_gesture: config.preserveGesture,
          preserve_expression: config.preserveExpression,
          blur_background: config.blurBackground,
          add_noise: config.addNoise,
          seed: config.seed,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      setLastMetrics(result.metrics);
      onAnonymizationComplete?.(result.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '匿名化失败');
    } finally {
      setIsProcessing(false);
    }
  }, [config, onAnonymizationComplete]);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'low': return 'warning';
      case 'medium': return 'info';
      case 'high': return 'success';
      default: return 'default';
    }
  };

  const getLevelDescription = (level: string) => {
    switch (level) {
      case 'low': return '轻度匿名化 - 基本隐私保护';
      case 'medium': return '中度匿名化 - 平衡隐私与可用性';
      case 'high': return '高度匿名化 - 最强隐私保护';
      default: return '';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SecurityIcon color="primary" />
          隐私保护设置
        </Typography>

        {/* 隐私模式开关 */}
        <Box sx={{ mb: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={privacyMode}
                onChange={(e) => setPrivacyMode(e.target.checked)}
                color="primary"
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {privacyMode ? <VisibilityOffIcon /> : <VisibilityIcon />}
                {privacyMode ? '隐私模式已启用' : '启用隐私模式'}
              </Box>
            }
          />
        </Box>

        {privacyMode && (
          <>
            {/* 匿名化级别 */}
            <Box sx={{ mb: 2 }}>
              <FormControl fullWidth>
                <InputLabel>匿名化级别</InputLabel>
                <Select
                  value={config.level}
                  onChange={(e) => setConfig(prev => ({ ...prev, level: e.target.value as any }))}
                  disabled={isProcessing}
                >
                  <MenuItem value="low">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip label="低" color="warning" size="small" />
                      轻度匿名化
                    </Box>
                  </MenuItem>
                  <MenuItem value="medium">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip label="中" color="info" size="small" />
                      中度匿名化
                    </Box>
                  </MenuItem>
                  <MenuItem value="high">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip label="高" color="success" size="small" />
                      高度匿名化
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                {getLevelDescription(config.level)}
              </Typography>
            </Box>

            {/* 保护选项 */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                数据保留选项
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.preserveGesture}
                        onChange={(e) => setConfig(prev => ({ ...prev, preserveGesture: e.target.checked }))}
                        disabled={isProcessing}
                      />
                    }
                    label="保留手势动作"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.preserveExpression}
                        onChange={(e) => setConfig(prev => ({ ...prev, preserveExpression: e.target.checked }))}
                        disabled={isProcessing}
                      />
                    }
                    label="保留面部表情"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.blurBackground}
                        onChange={(e) => setConfig(prev => ({ ...prev, blurBackground: e.target.checked }))}
                        disabled={isProcessing}
                      />
                    }
                    label="模糊背景"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.addNoise}
                        onChange={(e) => setConfig(prev => ({ ...prev, addNoise: e.target.checked }))}
                        disabled={isProcessing}
                      />
                    }
                    label="添加噪声"
                  />
                </Grid>
              </Grid>
            </Box>

            {/* 隐私指标显示 */}
            {lastMetrics && (
              <Accordion sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AnalyticsIcon />
                    隐私保护指标
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" gutterBottom>
                        匿名化得分
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={lastMetrics.anonymizationScore * 100}
                          color={getScoreColor(lastMetrics.anonymizationScore)}
                          sx={{ flex: 1 }}
                        />
                        <Typography variant="body2">
                          {(lastMetrics.anonymizationScore * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" gutterBottom>
                        数据可用性
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={lastMetrics.utilityScore * 100}
                          color={getScoreColor(lastMetrics.utilityScore)}
                          sx={{ flex: 1 }}
                        />
                        <Typography variant="body2">
                          {(lastMetrics.utilityScore * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        处理时间: {lastMetrics.processingTime.toFixed(3)}s
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        数据减少: {(lastMetrics.dataSizeReduction * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            )}

            {/* 错误提示 */}
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {/* 应用按钮 */}
            <Button
              fullWidth
              variant="contained"
              onClick={handleAnonymize}
              disabled={isProcessing}
              startIcon={isProcessing ? <CircularProgress size={20} /> : <ShieldIcon />}
              color="primary"
            >
              {isProcessing ? '处理中...' : '应用隐私保护'}
            </Button>
          </>
        )}

        {!privacyMode && (
          <Alert severity="info" sx={{ mt: 2 }}>
            启用隐私模式以保护您的数据安全。系统将自动对视频和图像进行匿名化处理。
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default PrivacyPanel;
