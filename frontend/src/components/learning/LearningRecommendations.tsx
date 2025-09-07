/**
 * 个性化学习推荐组件
 * 基于用户学习进度和偏好提供智能推荐
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Grid,
  Chip,
  Avatar,
  Stack,
  IconButton,
  Tooltip,
  LinearProgress,
  Divider,
  Alert,
} from '@mui/material'
import {
  PlayArrow,
  BookmarkBorder,
  Bookmark,
  Share,
  School,
  Assignment,
  AutoAwesome,
  Timer,
  TrendingUp,
  Psychology,
  Speed,
  EmojiEvents,
  Lightbulb,
  Star,
  AccessTime,
} from '@mui/icons-material'

import { useAuth } from '../../contexts/AuthContext'

interface LearningRecommendation {
  id: string
  title: string
  description: string
  type: 'lesson' | 'practice' | 'review' | 'challenge'
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  estimatedTime: number
  xpReward: number
  completionRate: number
  tags: string[]
  icon: React.ReactNode
  color: string
  reason: string
  priority: 'high' | 'medium' | 'low'
}

interface LearningRecommendationsProps {
  userLevel: number
  completedLessons: number
  currentStreak: number
  onStartLesson?: (lessonId: string) => void
  onBookmarkLesson?: (lessonId: string) => void
}

const LearningRecommendations: React.FC<LearningRecommendationsProps> = ({
  userLevel,
  completedLessons,
  currentStreak,
  onStartLesson,
  onBookmarkLesson
}) => {
  const { isAuthenticated } = useAuth()
  const [recommendations, setRecommendations] = useState<LearningRecommendation[]>([])
  const [bookmarkedLessons, setBookmarkedLessons] = useState<Set<string>>(new Set())

  useEffect(() => {
    // 基于用户数据生成个性化推荐
    const generateRecommendations = () => {
      const baseRecommendations: LearningRecommendation[] = [
        {
          id: 'rec-1',
          title: '数字手语进阶',
          description: '学习更复杂的数字表达和计算相关手语',
          type: 'lesson',
          difficulty: userLevel < 10 ? 'beginner' : 'intermediate',
          estimatedTime: 15,
          xpReward: 50,
          completionRate: 87,
          tags: ['数字', '计算', '进阶'],
          icon: <School />,
          color: '#B5EAD7',
          reason: '基于您在基础数字课程中的优秀表现',
          priority: 'high',
        },
        {
          id: 'rec-2',
          title: '日常对话强化练习',
          description: '通过实际对话场景巩固已学手语技能',
          type: 'practice',
          difficulty: 'beginner',
          estimatedTime: 12,
          xpReward: 35,
          completionRate: 92,
          tags: ['对话', '实践', '巩固'],
          icon: <Assignment />,
          color: '#FFB3BA',
          reason: '练习有助于巩固您已掌握的基础技能',
          priority: 'medium',
        },
        {
          id: 'rec-3',
          title: '手语语法规则',
          description: '深入理解手语的语法结构和表达规则',
          type: 'lesson',
          difficulty: 'intermediate',
          estimatedTime: 20,
          xpReward: 75,
          completionRate: 78,
          tags: ['语法', '规则', '理论'],
          icon: <Psychology />,
          color: '#FFDAB9',
          reason: '您的学习进度表明可以开始学习更深层的理论知识',
          priority: 'medium',
        },
        {
          id: 'rec-4',
          title: '速度挑战赛',
          description: '测试您的手语识别和表达速度',
          type: 'challenge',
          difficulty: 'advanced',
          estimatedTime: 10,
          xpReward: 100,
          completionRate: 65,
          tags: ['挑战', '速度', '测试'],
          icon: <Speed />,
          color: '#C7CEDB',
          reason: `您已连续学习${currentStreak}天，可以尝试挑战自己`,
          priority: currentStreak >= 7 ? 'high' : 'low',
        },
        {
          id: 'rec-5',
          title: '复习基础手语',
          description: '温习之前学过的基础手语内容',
          type: 'review',
          difficulty: 'beginner',
          estimatedTime: 8,
          xpReward: 25,
          completionRate: 95,
          tags: ['复习', '基础', '巩固'],
          icon: <AutoAwesome />,
          color: '#E0E4CC',
          reason: '定期复习有助于长期记忆',
          priority: 'low',
        },
        {
          id: 'rec-6',
          title: '情感表达手语',
          description: '学习如何用手语表达各种情感和感受',
          type: 'lesson',
          difficulty: 'intermediate',
          estimatedTime: 18,
          xpReward: 60,
          completionRate: 83,
          tags: ['情感', '表达', '交流'],
          icon: <EmojiEvents />,
          color: '#FFD6CC',
          reason: '丰富您的表达能力，让交流更生动',
          priority: 'medium',
        },
      ]

      // 根据用户等级和完成课程数筛选推荐
      const filteredRecommendations = baseRecommendations.filter(rec => {
        if (rec.difficulty === 'advanced' && userLevel < 15) return false
        if (rec.difficulty === 'intermediate' && userLevel < 8) return false
        return true
      })

      // 按优先级排序
      const priorityOrder = { high: 3, medium: 2, low: 1 }
      filteredRecommendations.sort((a, b) => priorityOrder[b.priority] - priorityOrder[a.priority])

      setRecommendations(filteredRecommendations.slice(0, 6))
    }

    generateRecommendations()
  }, [userLevel, completedLessons, currentStreak])

  const handleBookmark = (lessonId: string) => {
    if (!isAuthenticated) return

    const newBookmarked = new Set(bookmarkedLessons)
    if (newBookmarked.has(lessonId)) {
      newBookmarked.delete(lessonId)
    } else {
      newBookmarked.add(lessonId)
    }
    setBookmarkedLessons(newBookmarked)
    onBookmarkLesson?.(lessonId)
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '#4CAF50'
      case 'intermediate': return '#FF9800'
      case 'advanced': return '#F44336'
      default: return '#2196F3'
    }
  }

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'lesson': return '课程'
      case 'practice': return '练习'
      case 'review': return '复习'
      case 'challenge': return '挑战'
      default: return '学习'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return '#F44336'
      case 'medium': return '#FF9800'
      case 'low': return '#4CAF50'
      default: return '#2196F3'
    }
  }

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
        🎯 为您推荐
      </Typography>

      {!isAuthenticated && (
        <Alert severity="info" sx={{ mb: 3, borderRadius: 3 }}>
          <Typography variant="body2">
            登录后可获得基于您学习进度的个性化推荐内容
          </Typography>
        </Alert>
      )}

      <Grid container spacing={3}>
        {recommendations.map((recommendation) => (
          <Grid item xs={12} sm={6} md={4} key={recommendation.id}>
            <Card
              sx={{
                height: '100%',
                borderRadius: 3,
                border: `2px solid ${recommendation.color}30`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: `0 8px 25px ${recommendation.color}40`,
                  border: `2px solid ${recommendation.color}60`,
                }
              }}
            >
              <CardContent sx={{ pb: 1 }}>
                {/* 优先级标识 */}
                {recommendation.priority === 'high' && (
                  <Chip
                    label="推荐"
                    size="small"
                    sx={{
                      position: 'absolute',
                      top: 8,
                      right: 8,
                      bgcolor: getPriorityColor(recommendation.priority),
                      color: 'white',
                      fontWeight: 600,
                    }}
                  />
                )}

                {/* 图标和标题 */}
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
                  <Avatar
                    sx={{
                      bgcolor: recommendation.color,
                      width: 48,
                      height: 48,
                    }}
                  >
                    {recommendation.icon}
                  </Avatar>
                  <Box flex={1}>
                    <Typography variant="h6" fontWeight="bold" gutterBottom>
                      {recommendation.title}
                    </Typography>
                    <Stack direction="row" spacing={1}>
                      <Chip
                        label={getTypeLabel(recommendation.type)}
                        size="small"
                        sx={{ bgcolor: recommendation.color + '30' }}
                      />
                      <Chip
                        label={recommendation.difficulty === 'beginner' ? '初级' : 
                              recommendation.difficulty === 'intermediate' ? '中级' : '高级'}
                        size="small"
                        sx={{
                          bgcolor: getDifficultyColor(recommendation.difficulty),
                          color: 'white',
                        }}
                      />
                    </Stack>
                  </Box>
                </Stack>

                {/* 描述 */}
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {recommendation.description}
                </Typography>

                {/* 推荐理由 */}
                <Typography variant="caption" color="primary" sx={{ mb: 2, display: 'block' }}>
                  💡 {recommendation.reason}
                </Typography>

                {/* 统计信息 */}
                <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
                  <Stack direction="row" spacing={0.5} alignItems="center">
                    <AccessTime fontSize="small" color="action" />
                    <Typography variant="caption">
                      {recommendation.estimatedTime}分钟
                    </Typography>
                  </Stack>
                  <Stack direction="row" spacing={0.5} alignItems="center">
                    <Star fontSize="small" color="action" />
                    <Typography variant="caption">
                      {recommendation.xpReward} XP
                    </Typography>
                  </Stack>
                </Stack>

                {/* 完成率 */}
                <Box sx={{ mb: 2 }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      完成率
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {recommendation.completionRate}%
                    </Typography>
                  </Stack>
                  <LinearProgress
                    variant="determinate"
                    value={recommendation.completionRate}
                    sx={{
                      height: 4,
                      borderRadius: 2,
                      bgcolor: recommendation.color + '20',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: recommendation.color,
                      }
                    }}
                  />
                </Box>

                {/* 标签 */}
                <Stack direction="row" spacing={1} flexWrap="wrap" gap={0.5}>
                  {recommendation.tags.map((tag) => (
                    <Chip
                      key={tag}
                      label={tag}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: '0.7rem' }}
                    />
                  ))}
                </Stack>
              </CardContent>

              <CardActions sx={{ px: 2, pb: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={() => onStartLesson?.(recommendation.id)}
                  sx={{
                    flex: 1,
                    borderRadius: 2,
                    bgcolor: recommendation.color,
                    '&:hover': {
                      bgcolor: recommendation.color + 'CC',
                    }
                  }}
                >
                  开始学习
                </Button>
                <Tooltip title={bookmarkedLessons.has(recommendation.id) ? '取消收藏' : '收藏'}>
                  <IconButton
                    onClick={() => handleBookmark(recommendation.id)}
                    disabled={!isAuthenticated}
                  >
                    {bookmarkedLessons.has(recommendation.id) ? <Bookmark /> : <BookmarkBorder />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="分享">
                  <IconButton>
                    <Share />
                  </IconButton>
                </Tooltip>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  )
}

export default LearningRecommendations
