/**
 * 联邦学习控制面板
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
  LinearProgress,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  School as SchoolIcon,
  Security as SecurityIcon,
  Analytics as AnalyticsIcon,
  Visibility as VisibilityIcon,
  Group as GroupIcon,
  TrendingUp as TrendingUpIcon,
  Shield as ShieldIcon,
  PlayArrow as PlayIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

interface FederatedStats {
  totalRounds: number;
  successfulUpdates: number;
  averageRoundTime: number;
  privacyBudgetUsed: number;
  explanationRequests: number;
  clientInfo: {
    clientId: string;
    role: string;
    dataSize: number;
    modelVersion: number;
    privacyBudget: number;
    contributionScore: number;
  };
  currentRound: number;
  isTraining: boolean;
  differentialPrivacy: boolean;
  privacyBudgetRemaining: number;
}

interface ExplanationSummary {
  totalExplanations: number;
  averageConfidence: number;
  cacheSize: number;
  latestExplanationTime: number;
}

interface FederatedLearningPanelProps {
  onTrainingStart?: () => void;
  onExplanationGenerated?: (explanation: any) => void;
}

const FederatedLearningPanel: React.FC<FederatedLearningPanelProps> = ({
  onTrainingStart,
  onExplanationGenerated,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<FederatedStats | null>(null);
  const [explanationSummary, setExplanationSummary] = useState<ExplanationSummary | null>(null);
  const [lastTrainingResult, setLastTrainingResult] = useState<any>(null);

  // 获取统计信息
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch('/api/federated/stats');
      const result = await response.json();
      
      if (result.success) {
        setStats(result.data);
      }
    } catch (err) {
      console.error('获取联邦学习统计信息失败:', err);
    }
  }, []);

  // 获取解释摘要
  const fetchExplanationSummary = useCallback(async () => {
    try {
      const response = await fetch('/api/federated/explanation-summary');
      const result = await response.json();
      
      if (result.success) {
        setExplanationSummary(result.data);
      }
    } catch (err) {
      console.error('获取解释摘要失败:', err);
    }
  }, []);

  // 定期更新数据
  useEffect(() => {
    fetchStats();
    fetchExplanationSummary();
    
    const interval = setInterval(() => {
      fetchStats();
      fetchExplanationSummary();
    }, 3000);
    
    return () => clearInterval(interval);
  }, [fetchStats, fetchExplanationSummary]);

  // 开始联邦学习训练
  const handleStartTraining = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/federated/start-training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          config: {
            // 可以添加训练配置
          }
        }),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      setLastTrainingResult(result.data);
      onTrainingStart?.();
      
      // 刷新统计信息
      await fetchStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动联邦学习失败');
    } finally {
      setIsLoading(false);
    }
  }, [onTrainingStart, fetchStats]);

  // 生成模型解释
  const handleGenerateExplanation = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // 生成模拟输入数据
      const mockInputData = Array.from({ length: 30 }, () =>
        Array.from({ length: 543 }, () => [
          Math.random() * 2 - 1,
          Math.random() * 2 - 1,
          Math.random() * 2 - 1
        ])
      );

      const response = await fetch('/api/federated/generate-explanation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_data: mockInputData,
          prediction: {
            text: "你好",
            confidence: 0.85
          }
        }),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      onExplanationGenerated?.(result.data);
      
      // 刷新解释摘要
      await fetchExplanationSummary();
    } catch (err) {
      setError(err instanceof Error ? err.message : '生成模型解释失败');
    } finally {
      setIsLoading(false);
    }
  }, [onExplanationGenerated, fetchExplanationSummary]);

  const getPrivacyBudgetColor = (remaining: number) => {
    if (remaining > 0.7) return 'success';
    if (remaining > 0.3) return 'warning';
    return 'error';
  };

  const getRoleDisplayName = (role: string) => {
    const roleMap: Record<string, string> = {
      'trainer': '训练者',
      'validator': '验证者',
      'observer': '观察者'
    };
    return roleMap[role] || role;
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SchoolIcon color="primary" />
          联邦学习系统
        </Typography>

        {/* 客户端状态 */}
        {stats && (
          <Box sx={{ mb: 2 }}>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Chip
                  label={`角色: ${getRoleDisplayName(stats.clientInfo.role)}`}
                  color="primary"
                  size="small"
                  icon={<GroupIcon />}
                />
              </Grid>
              <Grid item xs={6}>
                <Chip
                  label={stats.isTraining ? "训练中" : "空闲"}
                  color={stats.isTraining ? "success" : "default"}
                  size="small"
                  icon={stats.isTraining ? <PlayIcon /> : <InfoIcon />}
                />
              </Grid>
            </Grid>
          </Box>
        )}

        {/* 隐私保护状态 */}
        {stats && stats.differentialPrivacy && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SecurityIcon />
              隐私保护状态
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="body2">
                隐私预算剩余: {(stats.privacyBudgetRemaining * 100).toFixed(1)}%
              </Typography>
              <Chip
                label="差分隐私"
                color="success"
                size="small"
                icon={<ShieldIcon />}
              />
            </Box>
            <LinearProgress
              variant="determinate"
              value={stats.privacyBudgetRemaining * 100}
              color={getPrivacyBudgetColor(stats.privacyBudgetRemaining)}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
        )}

        {/* 控制按钮 */}
        <Grid container spacing={1} sx={{ mb: 2 }}>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              onClick={handleStartTraining}
              disabled={isLoading || (stats?.isTraining || false)}
              startIcon={isLoading ? <CircularProgress size={20} /> : <PlayIcon />}
            >
              {stats?.isTraining ? '训练中...' : '开始训练'}
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="outlined"
              onClick={handleGenerateExplanation}
              disabled={isLoading}
              startIcon={<VisibilityIcon />}
            >
              生成解释
            </Button>
          </Grid>
        </Grid>

        {/* 最近训练结果 */}
        {lastTrainingResult && (
          <Alert severity="success" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>第 {lastTrainingResult.round_number} 轮训练完成</strong>
            </Typography>
            {lastTrainingResult.latest_update && (
              <Typography variant="caption" color="text.secondary">
                损失: {lastTrainingResult.latest_update.loss?.toFixed(4)} | 
                准确率: {(lastTrainingResult.latest_update.accuracy * 100)?.toFixed(1)}% |
                隐私噪声: {lastTrainingResult.latest_update.privacy_noise?.toFixed(4)}
              </Typography>
            )}
          </Alert>
        )}

        {/* 训练统计 */}
        {stats && (
          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUpIcon />
                训练统计
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    总轮次: {stats.totalRounds}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    成功更新: {stats.successfulUpdates}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    当前轮次: {stats.currentRound}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    平均轮次时间: {stats.averageRoundTime.toFixed(2)}s
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    数据大小: {stats.clientInfo.dataSize}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    贡献分数: {stats.clientInfo.contributionScore.toFixed(3)}
                  </Typography>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        )}

        {/* 解释性统计 */}
        {explanationSummary && (
          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AnalyticsIcon />
                解释性统计
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    总解释数: {explanationSummary.totalExplanations}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    平均置信度: {(explanationSummary.averageConfidence * 100).toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">
                    缓存大小: {explanationSummary.cacheSize}
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

        {/* 使用说明 */}
        <Alert severity="info">
          <Typography variant="body2">
            💡 联邦学习系统支持隐私保护的分布式训练。
            系统使用差分隐私技术保护用户数据，同时提供模型解释功能。
          </Typography>
        </Alert>
      </CardContent>
    </Card>
  );
};

export default FederatedLearningPanel;
