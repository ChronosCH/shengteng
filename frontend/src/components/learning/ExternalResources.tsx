/**
 * 外部学习资源组件
 * 提供精选的外部手语学习网站和资源链接
 */

import React, { useState } from 'react'
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
  TextField,
  InputAdornment,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Paper,
  Rating,
} from '@mui/material'
import {
  OpenInNew,
  Search,
  Language,
  VideoLibrary,
  MenuBook,
  Group,
  School,
  Psychology,
  Assignment,
  Star,
  Bookmark,
  BookmarkBorder,
  Share,
  ThumbUp,
  PlayCircleOutline,
  Article,
  Forum,
  Download,
} from '@mui/icons-material'

interface ExternalResource {
  id: string
  title: string
  description: string
  url: string
  type: 'website' | 'video' | 'course' | 'community' | 'app' | 'book'
  category: 'beginner' | 'intermediate' | 'advanced' | 'general'
  language: 'chinese' | 'english' | 'international'
  rating: number
  reviews: number
  tags: string[]
  icon: React.ReactNode
  color: string
  featured: boolean
  free: boolean
}

interface ExternalResourcesProps {
  onBookmark?: (resourceId: string) => void
}

const ExternalResources: React.FC<ExternalResourcesProps> = ({ onBookmark }) => {
  const [currentTab, setCurrentTab] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [bookmarkedResources, setBookmarkedResources] = useState<Set<string>>(new Set())

  const resources: ExternalResource[] = [
    {
      id: 'csl-online',
      title: '中国手语在线学习平台',
      description: '官方认证的中国手语学习网站，提供系统化的课程和认证',
      url: 'https://www.csl-online.org.cn',
      type: 'website',
      category: 'general',
      language: 'chinese',
      rating: 4.8,
      reviews: 1250,
      tags: ['官方', '认证', '系统化', '中文'],
      icon: <School />,
      color: '#2196F3',
      featured: true,
      free: false,
    },
    {
      id: 'signlanguage101',
      title: 'Sign Language 101',
      description: '国际手语学习社区，包含多种手语体系的学习资源',
      url: 'https://www.signlanguage101.com',
      type: 'website',
      category: 'beginner',
      language: 'international',
      rating: 4.6,
      reviews: 890,
      tags: ['国际', '社区', '多语言', '初学者'],
      icon: <Language />,
      color: '#4CAF50',
      featured: true,
      free: true,
    },
    {
      id: 'asl-university',
      title: 'ASL University',
      description: '美国手语大学，提供免费的ASL学习课程和词典',
      url: 'https://www.lifeprint.com',
      type: 'course',
      category: 'general',
      language: 'english',
      rating: 4.7,
      reviews: 2100,
      tags: ['ASL', '免费', '词典', '大学'],
      icon: <MenuBook />,
      color: '#FF9800',
      featured: true,
      free: true,
    },
    {
      id: 'handtalk-app',
      title: 'HandTalk 手语翻译',
      description: '智能手语翻译应用，支持实时文字转手语动画',
      url: 'https://www.handtalk.me',
      type: 'app',
      category: 'general',
      language: 'international',
      rating: 4.4,
      reviews: 5600,
      tags: ['翻译', '应用', '实时', '动画'],
      icon: <Psychology />,
      color: '#9C27B0',
      featured: false,
      free: true,
    },
    {
      id: 'deaf-community',
      title: '聋人社区论坛',
      description: '聋人和手语学习者的交流社区，分享经验和资源',
      url: 'https://www.deafcommunity.org',
      type: 'community',
      category: 'general',
      language: 'chinese',
      rating: 4.5,
      reviews: 780,
      tags: ['社区', '交流', '经验分享', '聋人'],
      icon: <Group />,
      color: '#E91E63',
      featured: false,
      free: true,
    },
    {
      id: 'youtube-asl',
      title: 'YouTube ASL 频道合集',
      description: '精选的YouTube手语学习频道，包含大量免费视频教程',
      url: 'https://www.youtube.com/playlist?list=ASL_Learning',
      type: 'video',
      category: 'beginner',
      language: 'english',
      rating: 4.3,
      reviews: 1890,
      tags: ['YouTube', '视频', '免费', '教程'],
      icon: <VideoLibrary />,
      color: '#F44336',
      featured: false,
      free: true,
    },
    {
      id: 'signschool-pro',
      title: 'SignSchool Pro',
      description: '专业手语培训机构的在线课程，提供认证和就业指导',
      url: 'https://www.signschool.pro',
      type: 'course',
      category: 'advanced',
      language: 'chinese',
      rating: 4.9,
      reviews: 450,
      tags: ['专业', '认证', '就业', '培训'],
      icon: <Assignment />,
      color: '#3F51B5',
      featured: true,
      free: false,
    },
    {
      id: 'sign-dictionary',
      title: '手语词典大全',
      description: '最全面的中英文手语词典，包含视频演示和详细说明',
      url: 'https://www.signdictionary.com',
      type: 'website',
      category: 'general',
      language: 'chinese',
      rating: 4.6,
      reviews: 1120,
      tags: ['词典', '视频', '中英文', '全面'],
      icon: <Article />,
      color: '#607D8B',
      featured: false,
      free: true,
    },
  ]

  const categories = [
    { label: '全部', value: 'all' },
    { label: '初学者', value: 'beginner' },
    { label: '中级', value: 'intermediate' },
    { label: '高级', value: 'advanced' },
    { label: '通用', value: 'general' },
  ]

  const types = [
    { label: '全部', value: 'all', icon: <Star /> },
    { label: '网站', value: 'website', icon: <Language /> },
    { label: '视频', value: 'video', icon: <VideoLibrary /> },
    { label: '课程', value: 'course', icon: <School /> },
    { label: '社区', value: 'community', icon: <Group /> },
    { label: '应用', value: 'app', icon: <Psychology /> },
    { label: '书籍', value: 'book', icon: <MenuBook /> },
  ]

  const filteredResources = resources.filter(resource => {
    const matchesSearch = resource.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         resource.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         resource.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesTab = currentTab === 0 || 
                      (currentTab === 1 && resource.featured) ||
                      (currentTab === 2 && resource.free) ||
                      (currentTab === 3 && resource.language === 'chinese')
    
    return matchesSearch && matchesTab
  })

  const handleBookmark = (resourceId: string) => {
    const newBookmarked = new Set(bookmarkedResources)
    if (newBookmarked.has(resourceId)) {
      newBookmarked.delete(resourceId)
    } else {
      newBookmarked.add(resourceId)
    }
    setBookmarkedResources(newBookmarked)
    onBookmark?.(resourceId)
  }

  const handleOpenResource = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  const getTypeIcon = (type: string) => {
    const typeObj = types.find(t => t.value === type)
    return typeObj?.icon || <Star />
  }

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'website': return '网站'
      case 'video': return '视频'
      case 'course': return '课程'
      case 'community': return '社区'
      case 'app': return '应用'
      case 'book': return '书籍'
      default: return '资源'
    }
  }

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'beginner': return '初学者'
      case 'intermediate': return '中级'
      case 'advanced': return '高级'
      case 'general': return '通用'
      default: return '未知'
    }
  }

  const getLanguageLabel = (language: string) => {
    switch (language) {
      case 'chinese': return '中文'
      case 'english': return '英文'
      case 'international': return '国际'
      default: return '未知'
    }
  }

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
        🌐 外部学习资源
      </Typography>

      {/* 搜索和筛选 */}
      <Paper sx={{ p: 3, mb: 3, borderRadius: 3 }}>
        <Stack spacing={3}>
          <TextField
            fullWidth
            placeholder="搜索资源..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 3,
              }
            }}
          />

          <Tabs
            value={currentTab}
            onChange={(_, newValue) => setCurrentTab(newValue)}
            variant="fullWidth"
            sx={{
              '& .MuiTab-root': {
                borderRadius: 2,
                mx: 0.5,
              }
            }}
          >
            <Tab label="全部资源" />
            <Tab label="精选推荐" />
            <Tab label="免费资源" />
            <Tab label="中文资源" />
          </Tabs>
        </Stack>
      </Paper>

      {/* 资源列表 */}
      <Grid container spacing={3}>
        {filteredResources.map((resource) => (
          <Grid item xs={12} sm={6} md={4} key={resource.id}>
            <Card
              sx={{
                height: '100%',
                borderRadius: 3,
                border: resource.featured ? `2px solid ${resource.color}` : '1px solid #e0e0e0',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: `0 8px 25px ${resource.color}40`,
                }
              }}
            >
              <CardContent sx={{ pb: 1 }}>
                {/* 资源标识 */}
                <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 2 }}>
                  <Avatar
                    sx={{
                      bgcolor: resource.color,
                      width: 48,
                      height: 48,
                    }}
                  >
                    {resource.icon}
                  </Avatar>
                  <Stack direction="row" spacing={0.5}>
                    {resource.featured && (
                      <Chip
                        label="精选"
                        size="small"
                        sx={{
                          bgcolor: '#FFD700',
                          color: 'white',
                          fontWeight: 600,
                        }}
                      />
                    )}
                    {resource.free && (
                      <Chip
                        label="免费"
                        size="small"
                        color="success"
                      />
                    )}
                  </Stack>
                </Stack>

                {/* 标题和描述 */}
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  {resource.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 40 }}>
                  {resource.description}
                </Typography>

                {/* 评分和评论 */}
                <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                  <Rating value={resource.rating} precision={0.1} size="small" readOnly />
                  <Typography variant="caption" color="text.secondary">
                    {resource.rating} ({resource.reviews} 评论)
                  </Typography>
                </Stack>

                {/* 标签 */}
                <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                  <Chip
                    label={getTypeLabel(resource.type)}
                    size="small"
                    icon={getTypeIcon(resource.type)}
                    variant="outlined"
                  />
                  <Chip
                    label={getCategoryLabel(resource.category)}
                    size="small"
                    sx={{
                      bgcolor: resource.color + '20',
                      color: resource.color,
                    }}
                  />
                  <Chip
                    label={getLanguageLabel(resource.language)}
                    size="small"
                    variant="outlined"
                  />
                </Stack>

                {/* 标签云 */}
                <Stack direction="row" spacing={0.5} flexWrap="wrap" gap={0.5}>
                  {resource.tags.slice(0, 3).map((tag) => (
                    <Chip
                      key={tag}
                      label={tag}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: '0.7rem' }}
                    />
                  ))}
                  {resource.tags.length > 3 && (
                    <Chip
                      label={`+${resource.tags.length - 3}`}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: '0.7rem' }}
                    />
                  )}
                </Stack>
              </CardContent>

              <CardActions sx={{ px: 2, pb: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<OpenInNew />}
                  onClick={() => handleOpenResource(resource.url)}
                  sx={{
                    flex: 1,
                    borderRadius: 2,
                    bgcolor: resource.color,
                    '&:hover': {
                      bgcolor: resource.color + 'CC',
                    }
                  }}
                >
                  访问资源
                </Button>
                <Tooltip title={bookmarkedResources.has(resource.id) ? '取消收藏' : '收藏'}>
                  <IconButton
                    onClick={() => handleBookmark(resource.id)}
                    color={bookmarkedResources.has(resource.id) ? 'primary' : 'default'}
                  >
                    {bookmarkedResources.has(resource.id) ? <Bookmark /> : <BookmarkBorder />}
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

      {filteredResources.length === 0 && (
        <Paper sx={{ p: 6, textAlign: 'center', borderRadius: 3 }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            没有找到匹配的资源
          </Typography>
          <Typography variant="body2" color="text.secondary">
            尝试调整搜索关键词或筛选条件
          </Typography>
        </Paper>
      )}

      {/* 资源统计 */}
      <Paper sx={{ p: 3, mt: 4, borderRadius: 3, bgcolor: 'primary.50' }}>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          📊 资源统计
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Typography variant="h4" fontWeight="bold" color="primary">
              {resources.length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              总资源数
            </Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="h4" fontWeight="bold" color="success.main">
              {resources.filter(r => r.free).length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              免费资源
            </Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="h4" fontWeight="bold" color="warning.main">
              {resources.filter(r => r.featured).length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              精选推荐
            </Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="h4" fontWeight="bold" color="info.main">
              {resources.filter(r => r.language === 'chinese').length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              中文资源
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  )
}

export default ExternalResources
