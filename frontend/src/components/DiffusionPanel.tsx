/**
 * Diffusion 生成式手语控制面板
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  AutoAwesome as AutoAwesomeIcon,
  Psychology as PsychologyIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

import {
  diffusionService,
  EmotionType,
  SigningSpeed,
  DiffusionConfig,
  SignSequence,
} from '../services/diffusionService';

interface DiffusionPanelProps {
  onSequenceGenerated: (sequence: SignSequence) => void;
  isGenerating?: boolean;
}

const DiffusionPanel: React.FC<DiffusionPanelProps> = ({
  onSequenceGenerated,
  isGenerating = false,
}) => {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState<EmotionType>(EmotionType.NEUTRAL);
  const [speed, setSpeed] = useState<SigningSpeed>(SigningSpeed.NORMAL);
  const [numSteps, setNumSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [useSeed, setUseSeed] = useState(false);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastGenerated, setLastGenerated] = useState<SignSequence | null>(null);

  const handleGenerate = useCallback(async () => {
    if (!text.trim()) {
      setError('请输入要生成的文本');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const config: Partial<DiffusionConfig> = {
        emotion,
        speed,
        numInferenceSteps: numSteps,
        guidanceScale,
        seed: useSeed ? seed : undefined,
      };

      const sequence = await diffusionService.generateSignSequence(text, config);
      setLastGenerated(sequence);
      onSequenceGenerated(sequence);
    } catch (err) {
      setError(err instanceof Error ? err.message : '生成失败');
    } finally {
      setLoading(false);
    }
  }, [text, emotion, speed, numSteps, guidanceScale, useSeed, seed, onSequenceGenerated]);

  const handlePresetEmotion = useCallback((emotionType: EmotionType) => {
    const presets = diffusionService.getEmotionPresets();
    const preset = presets[emotionType];
    
    setEmotion(emotionType);
    if (preset.speed) setSpeed(preset.speed);
    if (preset.guidanceScale) setGuidanceScale(preset.guidanceScale);
  }, []);

  const isDisabled = loading || isGenerating;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AutoAwesomeIcon color="primary" />
          Diffusion 生成式手语
        </Typography>

        {/* 文本输入 */}
        <TextField
          fullWidth
          multiline
          rows={3}
          label="输入文本"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={isDisabled}
          placeholder="请输入要转换为手语的文本..."
          sx={{ mb: 2 }}
        />

        {/* 快速情绪预设 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            快速情绪预设
          </Typography>
          <Grid container spacing={1}>
            {Object.values(EmotionType).map((emotionType) => (
              <Grid item key={emotionType}>
                <Chip
                  label={diffusionService.getEmotionDisplayName(emotionType)}
                  onClick={() => handlePresetEmotion(emotionType)}
                  color={emotion === emotionType ? 'primary' : 'default'}
                  variant={emotion === emotionType ? 'filled' : 'outlined'}
                  disabled={isDisabled}
                />
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* 基础设置 */}
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={6}>
            <FormControl fullWidth disabled={isDisabled}>
              <InputLabel>情绪</InputLabel>
              <Select
                value={emotion}
                onChange={(e) => setEmotion(e.target.value as EmotionType)}
                startAdornment={<PsychologyIcon sx={{ mr: 1, color: 'action.active' }} />}
              >
                {Object.values(EmotionType).map((emotionType) => (
                  <MenuItem key={emotionType} value={emotionType}>
                    {diffusionService.getEmotionDisplayName(emotionType)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <FormControl fullWidth disabled={isDisabled}>
              <InputLabel>速度</InputLabel>
              <Select
                value={speed}
                onChange={(e) => setSpeed(e.target.value as SigningSpeed)}
                startAdornment={<SpeedIcon sx={{ mr: 1, color: 'action.active' }} />}
              >
                {Object.values(SigningSpeed).map((speedType) => (
                  <MenuItem key={speedType} value={speedType}>
                    {diffusionService.getSpeedDisplayName(speedType)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>

        {/* 高级设置 */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SettingsIcon />
              高级设置
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography gutterBottom>推理步数: {numSteps}</Typography>
                <Slider
                  value={numSteps}
                  onChange={(_, value) => setNumSteps(value as number)}
                  min={10}
                  max={50}
                  step={5}
                  disabled={isDisabled}
                  marks={[
                    { value: 10, label: '10' },
                    { value: 20, label: '20' },
                    { value: 30, label: '30' },
                    { value: 50, label: '50' },
                  ]}
                />
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>引导强度: {guidanceScale}</Typography>
                <Slider
                  value={guidanceScale}
                  onChange={(_, value) => setGuidanceScale(value as number)}
                  min={1.0}
                  max={15.0}
                  step={0.5}
                  disabled={isDisabled}
                  marks={[
                    { value: 1.0, label: '1.0' },
                    { value: 7.5, label: '7.5' },
                    { value: 15.0, label: '15.0' },
                  ]}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={useSeed}
                      onChange={(e) => setUseSeed(e.target.checked)}
                      disabled={isDisabled}
                    />
                  }
                  label="使用固定种子"
                />
                {useSeed && (
                  <TextField
                    fullWidth
                    type="number"
                    label="种子值"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                    disabled={isDisabled}
                    sx={{ mt: 1 }}
                  />
                )}
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* 错误提示 */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {/* 生成结果信息 */}
        {lastGenerated && (
          <Alert severity="success" sx={{ mt: 2 }}>
            成功生成 {lastGenerated.numFrames} 帧手语动画，时长 {lastGenerated.duration.toFixed(2)} 秒
          </Alert>
        )}

        {/* 生成按钮 */}
        <Button
          fullWidth
          variant="contained"
          onClick={handleGenerate}
          disabled={isDisabled || !text.trim()}
          sx={{ mt: 2 }}
          startIcon={loading ? <CircularProgress size={20} /> : <AutoAwesomeIcon />}
        >
          {loading ? '生成中...' : '生成手语'}
        </Button>
      </CardContent>
    </Card>
  );
};

export default DiffusionPanel;
