/**
 * 游戏化手语学习平台
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Chip,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  SportsEsports as GameIcon,
  Star as StarIcon,
  EmojiEvents as TrophyIcon,
  Timer as TimerIcon,
  PlayArrow as PlayIcon,
  CheckCircle as CheckIcon,
} from '@mui/icons-material';

interface GameLevel {
  id: number;
  name: string;
  description: string;
  difficulty: 'easy' | 'medium' | 'hard';
  words: string[];
  timeLimit: number;
  stars: number;
  completed: boolean;
}

interface PlayerStats {
  level: number;
  experience: number;
  totalStars: number;
  completedLevels: number;
  accuracy: number;
  streak: number;
}

const GamePlaygroundPanel: React.FC = () => {
  const [currentLevel, setCurrentLevel] = useState<GameLevel | null>(null);
  const [gameActive, setGameActive] = useState(false);
  const [timeLeft, setTimeLeft] = useState(0);
  const [score, setScore] = useState(0);
  const [playerStats, setPlayerStats] = useState<PlayerStats>({
    level: 1,
    experience: 150,
    totalStars: 12,
    completedLevels: 4,
    accuracy: 85.5,
    streak: 3
  });

  const gameLevels: GameLevel[] = [
    {
      id: 1,
      name: "基础问候",
      description: "学习基本的问候手语",
      difficulty: 'easy',
      words: ['你好', '再见', '谢谢'],
      timeLimit: 60,
      stars: 3,
      completed: true
    },
    {
      id: 2,
      name: "数字练习",
      description: "练习数字1-10的手语",
      difficulty: 'easy',
      words: ['1', '2', '3', '4', '5'],
      timeLimit: 90,
      stars: 2,
      completed: true
    },
    {
      id: 3,
      name: "情感表达",
      description: "学习表达情感的手语",
      difficulty: 'medium',
      words: ['开心', '难过', '生气', '惊讶'],
      timeLimit: 120,
      stars: 1,
      completed: false
    },
    {
      id: 4,
      name: "日常对话",
      description: "练习日常对话手语",
      difficulty: 'hard',
      words: ['请问', '不客气', '对不起', '没关系'],
      timeLimit: 150,
      stars: 0,
      completed: false
    }
  ];

  const leaderboard = [
    { rank: 1, name: "手语达人", score: 2850, avatar: "🏆" },
    { rank: 2, name: "学习之星", score: 2640, avatar: "⭐" },
    { rank: 3, name: "练习高手", score: 2420, avatar: "🎯" },
    { rank: 4, name: "你", score: 2180, avatar: "👤" },
    { rank: 5, name: "新手小白", score: 1950, avatar: "🌱" }
  ];

  const startLevel = (level: GameLevel) => {
    setCurrentLevel(level);
    setGameActive(true);
    setTimeLeft(level.timeLimit);
    setScore(0);
  };

  const completeLevel = () => {
    if (currentLevel) {
      const newStats = {
        ...playerStats,
        experience: playerStats.experience + 50,
        totalStars: playerStats.totalStars + 3,
        completedLevels: playerStats.completedLevels + 1,
        streak: playerStats.streak + 1
      };
      setPlayerStats(newStats);
    }
    setGameActive(false);
    setCurrentLevel(null);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'success';
      case 'medium': return 'warning';
      case 'hard': return 'error';
      default: return 'default';
    }
  };

  // 游戏计时器
  useEffect(() => {
    if (gameActive && timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
      return () => clearTimeout(timer);
    } else if (timeLeft === 0 && gameActive) {
      setGameActive(false);
    }
  }, [gameActive, timeLeft]);

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <GameIcon color="primary" />
          游戏化学习平台
        </Typography>

        {/* 玩家统计 */}
        <Box sx={{ mb: 2, p: 2, bgcolor: 'primary.light', borderRadius: 1, color: 'white' }}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2">等级 {playerStats.level}</Typography>
              <LinearProgress 
                variant="determinate" 
                value={(playerStats.experience % 200) / 2} 
                sx={{ mt: 0.5, bgcolor: 'rgba(255,255,255,0.3)' }}
              />
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">⭐ {playerStats.totalStars} 星</Typography>
              <Typography variant="body2">🔥 {playerStats.streak} 连胜</Typography>
            </Grid>
          </Grid>
        </Box>

        {/* 关卡列表 */}
        <Typography variant="subtitle1" gutterBottom>
          学习关卡
        </Typography>
        <List dense>
          {gameLevels.map((level) => (
            <ListItem
              key={level.id}
              sx={{
                border: 1,
                borderColor: 'divider',
                borderRadius: 1,
                mb: 1,
                bgcolor: level.completed ? 'success.light' : 'background.paper'
              }}
            >
              <ListItemIcon>
                {level.completed ? <CheckIcon color="success" /> : <PlayIcon />}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {level.name}
                    <Chip 
                      label={level.difficulty} 
                      size="small" 
                      color={getDifficultyColor(level.difficulty)}
                    />
                    <Box sx={{ display: 'flex' }}>
                      {[...Array(3)].map((_, i) => (
                        <StarIcon 
                          key={i} 
                          sx={{ 
                            fontSize: 16, 
                            color: i < level.stars ? 'gold' : 'grey.300' 
                          }} 
                        />
                      ))}
                    </Box>
                  </Box>
                }
                secondary={level.description}
              />
              <Button
                variant={level.completed ? "outlined" : "contained"}
                size="small"
                onClick={() => startLevel(level)}
                disabled={gameActive}
              >
                {level.completed ? '重玩' : '开始'}
              </Button>
            </ListItem>
          ))}
        </List>

        {/* 排行榜 */}
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
          排行榜
        </Typography>
        <List dense>
          {leaderboard.slice(0, 5).map((player) => (
            <ListItem key={player.rank}>
              <ListItemIcon>
                <Avatar sx={{ width: 24, height: 24, fontSize: 12 }}>
                  {player.rank}
                </Avatar>
              </ListItemIcon>
              <ListItemText
                primary={player.name}
                secondary={`${player.score} 分`}
              />
              <Typography variant="body2">{player.avatar}</Typography>
            </ListItem>
          ))}
        </List>

        {/* 游戏对话框 */}
        <Dialog open={gameActive} maxWidth="sm" fullWidth>
          <DialogTitle>
            {currentLevel?.name}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 1 }}>
              <Chip icon={<TimerIcon />} label={`${timeLeft}s`} color="primary" />
              <Chip icon={<StarIcon />} label={`分数: ${score}`} />
            </Box>
          </DialogTitle>
          <DialogContent>
            <Typography gutterBottom>
              练习以下手语词汇：
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
              {currentLevel?.words.map((word, index) => (
                <Chip key={index} label={word} variant="outlined" />
              ))}
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={timeLeft / (currentLevel?.timeLimit || 1) * 100} 
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setGameActive(false)}>
              退出
            </Button>
            <Button onClick={completeLevel} variant="contained">
              完成关卡
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default GamePlaygroundPanel;
