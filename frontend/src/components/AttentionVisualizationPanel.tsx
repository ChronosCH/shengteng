/**
 * 注意力可视化教学面板
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  Chip,
  Grid,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  School as SchoolIcon,
  Palette as PaletteIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';

interface AttentionData {
  keypoints: number[][];
  attention_weights: number[][];
  saliency_map: number[][];
  timestamp: number;
}

interface HeatmapConfig {
  opacity: number;
  colormap: 'hot' | 'viridis' | 'plasma' | 'cool';
  threshold: number;
  showKeypoints: boolean;
  showConnections: boolean;
}

interface AttentionVisualizationPanelProps {
  landmarks?: number[][];
  onConfigChange?: (config: HeatmapConfig) => void;
}

const AttentionVisualizationPanel: React.FC<AttentionVisualizationPanelProps> = ({
  landmarks,
  onConfigChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isEnabled, setIsEnabled] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [attentionData, setAttentionData] = useState<AttentionData[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [config, setConfig] = useState<HeatmapConfig>({
    opacity: 0.7,
    colormap: 'hot',
    threshold: 0.3,
    showKeypoints: true,
    showConnections: true,
  });

  // 生成模拟注意力数据
  const generateMockAttentionData = useCallback((landmarks: number[][]) => {
    if (!landmarks || landmarks.length === 0) return null;

    const numKeypoints = landmarks.length;
    
    // 生成注意力权重 (模拟手部关键点有更高的注意力)
    const attention_weights = landmarks.map((_, i) => {
      // 手部关键点 (468-542) 有更高的注意力
      if (i >= 468 && i <= 542) {
        return Math.random() * 0.8 + 0.2; // 0.2-1.0
      }
      // 面部关键点有中等注意力
      else if (i < 468) {
        return Math.random() * 0.5 + 0.1; // 0.1-0.6
      }
      // 其他关键点有较低注意力
      else {
        return Math.random() * 0.3; // 0.0-0.3
      }
    });

    // 生成显著性图 (基于关键点运动)
    const saliency_map = landmarks.map((point, i) => {
      const [x, y] = point;
      // 模拟基于位置的显著性
      const centerX = 0.5, centerY = 0.5;
      const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
      const saliency = Math.max(0, 1 - distance * 2) * attention_weights[i];
      return [saliency];
    });

    return {
      keypoints: landmarks,
      attention_weights: [attention_weights],
      saliency_map: saliency_map,
      timestamp: Date.now(),
    };
  }, []);

  // 更新注意力数据
  useEffect(() => {
    if (isEnabled && landmarks) {
      const newData = generateMockAttentionData(landmarks);
      if (newData) {
        setAttentionData(prev => [...prev.slice(-29), newData]); // 保持最近30帧
      }
    }
  }, [isEnabled, landmarks, generateMockAttentionData]);

  // 绘制注意力热图
  const drawAttentionHeatmap = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || attentionData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const data = attentionData[currentFrame] || attentionData[attentionData.length - 1];
    if (!data) return;

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { keypoints, attention_weights, saliency_map } = data;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    // 绘制热图背景
    if (saliency_map && saliency_map.length > 0) {
      const imageData = ctx.createImageData(canvasWidth, canvasHeight);
      
      for (let y = 0; y < canvasHeight; y++) {
        for (let x = 0; x < canvasWidth; x++) {
          const normalizedX = x / canvasWidth;
          const normalizedY = y / canvasHeight;
          
          // 找到最近的关键点
          let maxSaliency = 0;
          keypoints.forEach((point, i) => {
            if (point.length >= 2) {
              const distance = Math.sqrt(
                (normalizedX - point[0]) ** 2 + (normalizedY - point[1]) ** 2
              );
              const influence = Math.exp(-distance * 10); // 高斯衰减
              const saliency = saliency_map[i] ? saliency_map[i][0] * influence : 0;
              maxSaliency = Math.max(maxSaliency, saliency);
            }
          });

          // 应用阈值
          if (maxSaliency > config.threshold) {
            const color = getHeatmapColor(maxSaliency, config.colormap);
            const pixelIndex = (y * canvasWidth + x) * 4;
            imageData.data[pixelIndex] = color.r;
            imageData.data[pixelIndex + 1] = color.g;
            imageData.data[pixelIndex + 2] = color.b;
            imageData.data[pixelIndex + 3] = color.a * config.opacity * 255;
          }
        }
      }
      
      ctx.putImageData(imageData, 0, 0);
    }

    // 绘制关键点
    if (config.showKeypoints && keypoints) {
      keypoints.forEach((point, i) => {
        if (point.length >= 2) {
          const x = point[0] * canvasWidth;
          const y = point[1] * canvasHeight;
          const attention = attention_weights[0] ? attention_weights[0][i] : 0;
          
          if (attention > config.threshold) {
            ctx.beginPath();
            ctx.arc(x, y, 3 + attention * 5, 0, 2 * Math.PI);
            ctx.fillStyle = `rgba(255, 255, 255, ${attention})`;
            ctx.fill();
            ctx.strokeStyle = `rgba(0, 0, 0, ${attention})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      });
    }

    // 绘制连接线
    if (config.showConnections && keypoints && attention_weights[0]) {
      const connections = getKeypointConnections();
      connections.forEach(([start, end]) => {
        if (start < keypoints.length && end < keypoints.length) {
          const startPoint = keypoints[start];
          const endPoint = keypoints[end];
          const startAttention = attention_weights[0][start];
          const endAttention = attention_weights[0][end];
          const avgAttention = (startAttention + endAttention) / 2;
          
          if (avgAttention > config.threshold && startPoint.length >= 2 && endPoint.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(startPoint[0] * canvasWidth, startPoint[1] * canvasHeight);
            ctx.lineTo(endPoint[0] * canvasWidth, endPoint[1] * canvasHeight);
            ctx.strokeStyle = `rgba(255, 255, 0, ${avgAttention * config.opacity})`;
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }
      });
    }
  }, [attentionData, currentFrame, config]);

  // 获取热图颜色
  const getHeatmapColor = (value: number, colormap: string) => {
    const clampedValue = Math.max(0, Math.min(1, value));
    
    switch (colormap) {
      case 'hot':
        return {
          r: Math.floor(255 * Math.min(1, clampedValue * 3)),
          g: Math.floor(255 * Math.max(0, Math.min(1, clampedValue * 3 - 1))),
          b: Math.floor(255 * Math.max(0, Math.min(1, clampedValue * 3 - 2))),
          a: 1
        };
      case 'viridis':
        return {
          r: Math.floor(255 * (0.267 + 0.005 * clampedValue)),
          g: Math.floor(255 * (0.004 + 0.632 * clampedValue)),
          b: Math.floor(255 * (0.329 + 0.528 * clampedValue)),
          a: 1
        };
      case 'plasma':
        return {
          r: Math.floor(255 * (0.050 + 0.839 * clampedValue)),
          g: Math.floor(255 * (0.030 + 0.718 * clampedValue)),
          b: Math.floor(255 * (0.527 + 0.415 * clampedValue)),
          a: 1
        };
      case 'cool':
        return {
          r: Math.floor(255 * clampedValue),
          g: Math.floor(255 * (1 - clampedValue)),
          b: 255,
          a: 1
        };
      default:
        return { r: 255, g: 0, b: 0, a: 1 };
    }
  };

  // 获取关键点连接关系
  const getKeypointConnections = () => {
    // 简化的手部连接关系
    const connections: [number, number][] = [];
    
    // 左手连接 (468-488)
    for (let i = 468; i < 488; i++) {
      if (i < 487) connections.push([i, i + 1]);
    }
    
    // 右手连接 (489-509)
    for (let i = 489; i < 509; i++) {
      if (i < 508) connections.push([i, i + 1]);
    }
    
    return connections;
  };

  // 重绘画布
  useEffect(() => {
    drawAttentionHeatmap();
  }, [drawAttentionHeatmap]);

  // 播放控制
  useEffect(() => {
    if (isPlaying && attentionData.length > 1) {
      const interval = setInterval(() => {
        setCurrentFrame(prev => (prev + 1) % attentionData.length);
      }, 100); // 10 FPS
      
      return () => clearInterval(interval);
    }
  }, [isPlaying, attentionData.length]);

  // 配置更改处理
  const handleConfigChange = (newConfig: Partial<HeatmapConfig>) => {
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);
    onConfigChange?.(updatedConfig);
  };

  // 导出热图
  const handleExportHeatmap = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const link = document.createElement('a');
      link.download = `attention_heatmap_${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SchoolIcon color="primary" />
          注意力可视化教学
        </Typography>

        {/* 启用开关 */}
        <Box sx={{ mb: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={isEnabled}
                onChange={(e) => setIsEnabled(e.target.checked)}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <VisibilityIcon />
                启用注意力可视化
              </Box>
            }
          />
        </Box>

        {isEnabled && (
          <>
            {/* 热图画布 */}
            <Paper sx={{ mb: 2, p: 1, bgcolor: 'grey.100' }}>
              <canvas
                ref={canvasRef}
                width={400}
                height={300}
                style={{
                  width: '100%',
                  height: 'auto',
                  border: '1px solid #ccc',
                  borderRadius: '4px'
                }}
              />
            </Paper>

            {/* 播放控制 */}
            <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <IconButton
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={attentionData.length <= 1}
              >
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
              </IconButton>
              
              <Typography variant="body2" sx={{ flex: 1 }}>
                帧: {currentFrame + 1} / {attentionData.length}
              </Typography>
              
              <Tooltip title="导出热图">
                <IconButton onClick={handleExportHeatmap}>
                  <DownloadIcon />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="重置">
                <IconButton onClick={() => setAttentionData([])}>
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
            </Box>

            {/* 配置选项 */}
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <Typography gutterBottom>透明度: {config.opacity.toFixed(1)}</Typography>
                <Slider
                  value={config.opacity}
                  onChange={(_, value) => handleConfigChange({ opacity: value as number })}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={6}>
                <Typography gutterBottom>阈值: {config.threshold.toFixed(1)}</Typography>
                <Slider
                  value={config.threshold}
                  onChange={(_, value) => handleConfigChange({ threshold: value as number })}
                  min={0.0}
                  max={1.0}
                  step={0.1}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControl fullWidth size="small">
                  <InputLabel>颜色映射</InputLabel>
                  <Select
                    value={config.colormap}
                    onChange={(e) => handleConfigChange({ colormap: e.target.value as any })}
                  >
                    <MenuItem value="hot">热力图</MenuItem>
                    <MenuItem value="viridis">Viridis</MenuItem>
                    <MenuItem value="plasma">Plasma</MenuItem>
                    <MenuItem value="cool">冷色调</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>

            {/* 显示选项 */}
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.showKeypoints}
                    onChange={(e) => handleConfigChange({ showKeypoints: e.target.checked })}
                    size="small"
                  />
                }
                label="显示关键点"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={config.showConnections}
                    onChange={(e) => handleConfigChange({ showConnections: e.target.checked })}
                    size="small"
                  />
                }
                label="显示连接线"
              />
            </Box>

            {/* 状态信息 */}
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip
                label={`数据帧: ${attentionData.length}`}
                size="small"
                color={attentionData.length > 0 ? 'success' : 'default'}
              />
              <Chip
                label={`颜色: ${config.colormap}`}
                size="small"
                icon={<PaletteIcon />}
              />
              {isPlaying && (
                <Chip
                  label="播放中"
                  size="small"
                  color="primary"
                  icon={<PlayIcon />}
                />
              )}
            </Box>
          </>
        )}

        {!isEnabled && (
          <Alert severity="info">
            启用注意力可视化以查看模型关注的关键点区域。
            热图将显示模型在识别手语时的注意力分布。
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default AttentionVisualizationPanel;
