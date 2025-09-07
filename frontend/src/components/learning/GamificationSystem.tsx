/**
 * 游戏化系统组件
 * 提供积分、等级、徽章等游戏化元素
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Avatar,
  Stack,
  Chip,
  LinearProgress,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Paper,
  Badge,
  Tooltip,
  Zoom,
} from '@mui/material'
import {
  EmojiEvents,
  Star,
  LocalFireDepartment,
  Speed,
  School,
  Psychology,
  Timer,
  TrendingUp,
  Assignment,
  CheckCircle,
  AutoAwesome,
  Celebration,
  Close,
  Lock,
  LockOpen,
} from '@mui/icons-material'

interface Achievement {
  id: string
  title: string
  description: string
  icon: React.ReactNode
  color: string
  category: 'learning' | 'streak' | 'speed' | 'accuracy' | 'special'
  unlocked: boolean
  unlockedAt?: string
  progress: number
  maxProgress: number
  xpReward: number
  rarity: 'common' | 'rare' | 'epic' | 'legendary'
}

interface UserLevel {
  level: number
  currentXP: number
  nextLevelXP: number
  title: string
  perks: string[]
}

interface GamificationSystemProps {
  userStats: {
    totalXP: number
    level: number
    currentStreak: number
    completedLessons: number
    totalLearningTime: number
  }
  onClaimReward?: (achievementId: string) => void
}

const GamificationSystem: React.FC<GamificationSystemProps> = ({
  userStats,
  onClaimReward
}) => {
  const [achievements, setAchievements] = useState<Achievement[]>([])
  const [selectedAchievement, setSelectedAchievement] = useState<Achievement | null>(null)
  const [showNewAchievement, setShowNewAchievement] = useState<Achievement | null>(null)
  const [userLevel, setUserLevel] = useState<UserLevel>({
    level: userStats.level,
    currentXP: userStats.totalXP,
    nextLevelXP: userStats.level * 100,
    title: getLevelTitle(userStats.level),
    perks: getLevelPerks(userStats.level)
  })

  useEffect(() => {
    // 初始化成就系统
    const allAchievements: Achievement[] = [
      {
        id: 'first_lesson',
        title: '初学者',
        description: '完成第一个课程',
        icon: <School />,
        color: '#4CAF50',
        category: 'learning',
        unlocked: userStats.completedLessons >= 1,
        unlockedAt: userStats.completedLessons >= 1 ? '2024-01-15' : undefined,
        progress: Math.min(userStats.completedLessons, 1),
        maxProgress: 1,
        xpReward: 50,
        rarity: 'common',
      },
      {
        id: 'streak_7',
        title: '连续学习者',
        description: '连续学习7天',
        icon: <LocalFireDepartment />,
        color: '#FF5722',
        category: 'streak',
        unlocked: userStats.currentStreak >= 7,
        unlockedAt: userStats.currentStreak >= 7 ? '2024-01-14' : undefined,
        progress: Math.min(userStats.currentStreak, 7),
        maxProgress: 7,
        xpReward: 100,
        rarity: 'rare',
      },
      {
        id: 'lessons_10',
        title: '勤奋学习者',
        description: '完成10个课程',
        icon: <Assignment />,
        color: '#2196F3',
        category: 'learning',
        unlocked: userStats.completedLessons >= 10,
        unlockedAt: userStats.completedLessons >= 10 ? '2024-01-13' : undefined,
        progress: Math.min(userStats.completedLessons, 10),
        maxProgress: 10,
        xpReward: 150,
        rarity: 'rare',
      },
      {
        id: 'time_60',
        title: '时间管理大师',
        description: '累计学习60分钟',
        icon: <Timer />,
        color: '#9C27B0',
        category: 'learning',
        unlocked: userStats.totalLearningTime >= 60,
        unlockedAt: userStats.totalLearningTime >= 60 ? '2024-01-12' : undefined,
        progress: Math.min(userStats.totalLearningTime, 60),
        maxProgress: 60,
        xpReward: 80,
        rarity: 'common',
      },
      {
        id: 'speed_master',
        title: '速度大师',
        description: '在5分钟内完成一个课程',
        icon: <Speed />,
        color: '#FF9800',
        category: 'speed',
        unlocked: false,
        progress: 0,
        maxProgress: 1,
        xpReward: 200,
        rarity: 'epic',
      },
      {
        id: 'perfectionist',
        title: '完美主义者',
        description: '连续5个课程都获得100%准确率',
        icon: <Star />,
        color: '#FFD700',
        category: 'accuracy',
        unlocked: false,
        progress: 0,
        maxProgress: 5,
        xpReward: 300,
        rarity: 'legendary',
      },
      {
        id: 'streak_30',
        title: '坚持不懈',
        description: '连续学习30天',
        icon: <TrendingUp />,
        color: '#E91E63',
        category: 'streak',
        unlocked: false,
        progress: Math.min(userStats.currentStreak, 30),
        maxProgress: 30,
        xpReward: 500,
        rarity: 'legendary',
      },
      {
        id: 'lessons_50',
        title: '学习专家',
        description: '完成50个课程',
        icon: <Psychology />,
        color: '#3F51B5',
        category: 'learning',
        unlocked: false,
        progress: Math.min(userStats.completedLessons, 50),
        maxProgress: 50,
        xpReward: 400,
        rarity: 'epic',
      },
    ]

    setAchievements(allAchievements)

    // 检查是否有新解锁的成就
    const newlyUnlocked = allAchievements.find(
      achievement => achievement.unlocked && !achievement.unlockedAt
    )
    if (newlyUnlocked) {
      setShowNewAchievement(newlyUnlocked)
    }
  }, [userStats])

  function getLevelTitle(level: number): string {
    if (level < 5) return '新手学习者'
    if (level < 10) return '初级学习者'
    if (level < 20) return '中级学习者'
    if (level < 30) return '高级学习者'
    if (level < 50) return '专家学习者'
    return '大师级学习者'
  }

  function getLevelPerks(level: number): string[] {
    const perks = ['基础学习功能']
    if (level >= 5) perks.push('解锁高级练习')
    if (level >= 10) perks.push('个性化推荐')
    if (level >= 20) perks.push('专属学习路径')
    if (level >= 30) perks.push('导师模式')
    if (level >= 50) perks.push('创建自定义课程')
    return perks
  }

  const getRarityColor = (rarity: string) => {
    switch (rarity) {
      case 'common': return '#9E9E9E'
      case 'rare': return '#2196F3'
      case 'epic': return '#9C27B0'
      case 'legendary': return '#FF9800'
      default: return '#9E9E9E'
    }
  }

  const getRarityLabel = (rarity: string) => {
    switch (rarity) {
      case 'common': return '普通'
      case 'rare': return '稀有'
      case 'epic': return '史诗'
      case 'legendary': return '传说'
      default: return '未知'
    }
  }

  const levelProgress = ((userLevel.currentXP % 100) / 100) * 100
  const unlockedAchievements = achievements.filter(a => a.unlocked)
  const lockedAchievements = achievements.filter(a => !a.unlocked)

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
        🎮 游戏化系统
      </Typography>

      <Grid container spacing={3}>
        {/* 等级信息 */}
        <Grid item xs={12} md={6}>
          <Card sx={{ borderRadius: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <CardContent sx={{ color: 'white' }}>
              <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                <Avatar
                  sx={{
                    width: 60,
                    height: 60,
                    bgcolor: 'rgba(255,255,255,0.2)',
                    fontSize: '1.5rem',
                  }}
                >
                  {userLevel.level}
                </Avatar>
                <Box flex={1}>
                  <Typography variant="h5" fontWeight="bold">
                    等级 {userLevel.level}
                  </Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>
                    {userLevel.title}
                  </Typography>
                </Box>
              </Stack>

              <Box sx={{ mb: 2 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    经验值进度
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    {userLevel.currentXP}/{userLevel.nextLevelXP}
                  </Typography>
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={levelProgress}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    bgcolor: 'rgba(255,255,255,0.2)',
                    '& .MuiLinearProgress-bar': {
                      bgcolor: 'white',
                    }
                  }}
                />
              </Box>

              <Typography variant="body2" sx={{ opacity: 0.9, mb: 1 }}>
                等级特权：
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" gap={0.5}>
                {userLevel.perks.map((perk, index) => (
                  <Chip
                    key={index}
                    label={perk}
                    size="small"
                    sx={{
                      bgcolor: 'rgba(255,255,255,0.2)',
                      color: 'white',
                      fontSize: '0.7rem',
                    }}
                  />
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* 成就统计 */}
        <Grid item xs={12} md={6}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#FFD700', mx: 'auto', mb: 1 }}>
                  <EmojiEvents />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {unlockedAchievements.length}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  已解锁成就
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#B5EAD7', mx: 'auto', mb: 1 }}>
                  <Star />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {userLevel.currentXP}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  总经验值
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#FFB3BA', mx: 'auto', mb: 1 }}>
                  <LocalFireDepartment />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {userStats.currentStreak}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  连续天数
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center', borderRadius: 3 }}>
                <Avatar sx={{ bgcolor: '#C7CEDB', mx: 'auto', mb: 1 }}>
                  <CheckCircle />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  {Math.round((unlockedAchievements.length / achievements.length) * 100)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  完成度
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Grid>

        {/* 已解锁成就 */}
        <Grid item xs={12}>
          <Card sx={{ borderRadius: 3 }}>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                🏆 已解锁成就
              </Typography>
              <Grid container spacing={2}>
                {unlockedAchievements.map((achievement) => (
                  <Grid item xs={12} sm={6} md={4} key={achievement.id}>
                    <Tooltip title={achievement.description}>
                      <Card
                        sx={{
                          borderRadius: 2,
                          border: `2px solid ${achievement.color}`,
                          cursor: 'pointer',
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: `0 4px 20px ${achievement.color}40`,
                          }
                        }}
                        onClick={() => setSelectedAchievement(achievement)}
                      >
                        <CardContent sx={{ textAlign: 'center', py: 2 }}>
                          <Badge
                            badgeContent={<CheckCircle fontSize="small" />}
                            color="success"
                            overlap="circular"
                          >
                            <Avatar
                              sx={{
                                bgcolor: achievement.color,
                                width: 48,
                                height: 48,
                                mx: 'auto',
                                mb: 1,
                              }}
                            >
                              {achievement.icon}
                            </Avatar>
                          </Badge>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                            {achievement.title}
                          </Typography>
                          <Chip
                            label={getRarityLabel(achievement.rarity)}
                            size="small"
                            sx={{
                              bgcolor: getRarityColor(achievement.rarity),
                              color: 'white',
                              fontSize: '0.7rem',
                            }}
                          />
                        </CardContent>
                      </Card>
                    </Tooltip>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* 进行中的成就 */}
        <Grid item xs={12}>
          <Card sx={{ borderRadius: 3 }}>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                🎯 进行中的成就
              </Typography>
              <Grid container spacing={2}>
                {lockedAchievements.filter(a => a.progress > 0).map((achievement) => (
                  <Grid item xs={12} sm={6} md={4} key={achievement.id}>
                    <Card
                      sx={{
                        borderRadius: 2,
                        border: `2px dashed ${achievement.color}60`,
                        cursor: 'pointer',
                      }}
                      onClick={() => setSelectedAchievement(achievement)}
                    >
                      <CardContent sx={{ textAlign: 'center', py: 2 }}>
                        <Avatar
                          sx={{
                            bgcolor: achievement.color + '40',
                            width: 48,
                            height: 48,
                            mx: 'auto',
                            mb: 1,
                          }}
                        >
                          {achievement.icon}
                        </Avatar>
                        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                          {achievement.title}
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={(achievement.progress / achievement.maxProgress) * 100}
                          sx={{
                            height: 6,
                            borderRadius: 3,
                            bgcolor: achievement.color + '20',
                            '& .MuiLinearProgress-bar': {
                              bgcolor: achievement.color,
                            }
                          }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                          {achievement.progress}/{achievement.maxProgress}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* 成就详情对话框 */}
      <Dialog
        open={!!selectedAchievement}
        onClose={() => setSelectedAchievement(null)}
        maxWidth="sm"
        fullWidth
      >
        {selectedAchievement && (
          <>
            <DialogTitle sx={{ textAlign: 'center', pb: 1 }}>
              <Avatar
                sx={{
                  bgcolor: selectedAchievement.color,
                  width: 80,
                  height: 80,
                  mx: 'auto',
                  mb: 2,
                  fontSize: '2rem',
                }}
              >
                {selectedAchievement.icon}
              </Avatar>
              <Typography variant="h5" fontWeight="bold">
                {selectedAchievement.title}
              </Typography>
              <Chip
                label={getRarityLabel(selectedAchievement.rarity)}
                sx={{
                  bgcolor: getRarityColor(selectedAchievement.rarity),
                  color: 'white',
                  mt: 1,
                }}
              />
            </DialogTitle>
            <DialogContent>
              <Typography variant="body1" textAlign="center" sx={{ mb: 2 }}>
                {selectedAchievement.description}
              </Typography>
              
              {selectedAchievement.unlocked ? (
                <Paper sx={{ p: 2, bgcolor: 'success.50', borderRadius: 2, textAlign: 'center' }}>
                  <CheckCircle color="success" sx={{ mb: 1 }} />
                  <Typography variant="body2" color="success.main" fontWeight="bold">
                    已解锁！获得 {selectedAchievement.xpReward} XP
                  </Typography>
                  {selectedAchievement.unlockedAt && (
                    <Typography variant="caption" color="text.secondary">
                      解锁时间：{selectedAchievement.unlockedAt}
                    </Typography>
                  )}
                </Paper>
              ) : (
                <Paper sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 2, textAlign: 'center' }}>
                  <Lock color="action" sx={{ mb: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    进度：{selectedAchievement.progress}/{selectedAchievement.maxProgress}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(selectedAchievement.progress / selectedAchievement.maxProgress) * 100}
                    sx={{ mt: 1, height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    完成后可获得 {selectedAchievement.xpReward} XP
                  </Typography>
                </Paper>
              )}
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedAchievement(null)}>
                关闭
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* 新成就解锁动画 */}
      <Dialog
        open={!!showNewAchievement}
        onClose={() => setShowNewAchievement(null)}
        maxWidth="sm"
        fullWidth
        TransitionComponent={Zoom}
      >
        {showNewAchievement && (
          <>
            <DialogTitle sx={{ textAlign: 'center', bgcolor: 'primary.main', color: 'white' }}>
              <Celebration sx={{ fontSize: 60, mb: 1 }} />
              <Typography variant="h4" fontWeight="bold">
                🎉 新成就解锁！
              </Typography>
            </DialogTitle>
            <DialogContent sx={{ textAlign: 'center', py: 4 }}>
              <Avatar
                sx={{
                  bgcolor: showNewAchievement.color,
                  width: 100,
                  height: 100,
                  mx: 'auto',
                  mb: 2,
                  fontSize: '3rem',
                }}
              >
                {showNewAchievement.icon}
              </Avatar>
              <Typography variant="h5" fontWeight="bold" gutterBottom>
                {showNewAchievement.title}
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                {showNewAchievement.description}
              </Typography>
              <Chip
                label={`+${showNewAchievement.xpReward} XP`}
                color="primary"
                sx={{ fontSize: '1rem', py: 2 }}
              />
            </DialogContent>
            <DialogActions sx={{ justifyContent: 'center', pb: 3 }}>
              <Button
                variant="contained"
                size="large"
                onClick={() => {
                  setShowNewAchievement(null)
                  onClaimReward?.(showNewAchievement.id)
                }}
                sx={{ borderRadius: 3, px: 4 }}
              >
                领取奖励
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  )
}

export default GamificationSystem
