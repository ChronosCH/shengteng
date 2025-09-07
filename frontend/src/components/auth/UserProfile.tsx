/**
 * 用户个人资料组件
 * 显示和编辑用户个人信息
 */

import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Avatar,
  Button,
  TextField,
  Stack,
  Divider,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  Grid,
  Switch,
  FormControlLabel,
} from '@mui/material'
import {
  Edit,
  Save,
  Cancel,
  Person,
  Email,
  Badge,
  Settings,
  Security,
  Logout,
  VpnKey,
} from '@mui/icons-material'

import { useAuth } from '../../contexts/AuthContext'

interface UserProfileProps {
  onLogout?: () => void
}

const UserProfile: React.FC<UserProfileProps> = ({ onLogout }) => {
  const { user, logout, updateProfile, loading } = useAuth()
  const [isEditing, setIsEditing] = useState(false)
  const [showPasswordDialog, setShowPasswordDialog] = useState(false)
  const [editData, setEditData] = useState({
    full_name: user?.full_name || '',
    email: user?.email || '',
  })
  const [preferences, setPreferences] = useState({
    notifications: user?.preferences?.notifications ?? true,
    learning_reminders: user?.preferences?.learning_reminders ?? true,
    theme: user?.preferences?.theme || 'light',
  })
  const [accessibility, setAccessibility] = useState({
    high_contrast: user?.accessibility_settings?.high_contrast ?? false,
    large_text: user?.accessibility_settings?.large_text ?? false,
    reduced_motion: user?.accessibility_settings?.reduced_motion ?? false,
  })
  const [error, setError] = useState<string | null>(null)

  if (!user) {
    return null
  }

  const handleEdit = () => {
    setIsEditing(true)
    setError(null)
  }

  const handleCancel = () => {
    setIsEditing(false)
    setEditData({
      full_name: user.full_name || '',
      email: user.email || '',
    })
    setError(null)
  }

  const handleSave = async () => {
    try {
      setError(null)
      await updateProfile({
        ...editData,
        preferences,
        accessibility_settings: accessibility,
      })
      setIsEditing(false)
    } catch (error) {
      setError(error instanceof Error ? error.message : '更新失败')
    }
  }

  const handleLogout = async () => {
    try {
      await logout()
      onLogout?.()
    } catch (error) {
      console.error('登出失败:', error)
    }
  }

  const handlePreferenceChange = (key: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setPreferences(prev => ({
      ...prev,
      [key]: event.target.checked
    }))
  }

  const handleAccessibilityChange = (key: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setAccessibility(prev => ({
      ...prev,
      [key]: event.target.checked
    }))
  }

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 2 }}>
      <Card elevation={4} sx={{ borderRadius: 3 }}>
        <CardContent sx={{ p: 4 }}>
          {/* 头部信息 */}
          <Box display="flex" alignItems="center" mb={4}>
            <Avatar
              sx={{
                width: 80,
                height: 80,
                bgcolor: 'primary.main',
                fontSize: '2rem',
                mr: 3,
              }}
            >
              {user.username.charAt(0).toUpperCase()}
            </Avatar>
            <Box flex={1}>
              <Typography variant="h4" fontWeight="bold" gutterBottom>
                {user.full_name || user.username}
              </Typography>
              <Typography variant="body1" color="text.secondary" gutterBottom>
                @{user.username}
              </Typography>
              <Stack direction="row" spacing={1} mt={1}>
                <Chip
                  label={user.is_admin ? '管理员' : '用户'}
                  color={user.is_admin ? 'secondary' : 'primary'}
                  size="small"
                />
                <Chip
                  label={user.is_active ? '活跃' : '禁用'}
                  color={user.is_active ? 'success' : 'error'}
                  size="small"
                />
              </Stack>
            </Box>
            <Box>
              {!isEditing ? (
                <IconButton onClick={handleEdit} color="primary">
                  <Edit />
                </IconButton>
              ) : (
                <Stack direction="row" spacing={1}>
                  <IconButton onClick={handleSave} color="primary" disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : <Save />}
                  </IconButton>
                  <IconButton onClick={handleCancel} disabled={loading}>
                    <Cancel />
                  </IconButton>
                </Stack>
              )}
            </Box>
          </Box>

          {/* 错误提示 */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {/* 基本信息 */}
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Person /> 基本信息
          </Typography>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="姓名"
                value={editData.full_name}
                onChange={(e) => setEditData(prev => ({ ...prev, full_name: e.target.value }))}
                disabled={!isEditing}
                InputProps={{
                  startAdornment: <Badge sx={{ mr: 1, color: 'action.active' }} />,
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="邮箱"
                value={editData.email}
                onChange={(e) => setEditData(prev => ({ ...prev, email: e.target.value }))}
                disabled={!isEditing}
                InputProps={{
                  startAdornment: <Email sx={{ mr: 1, color: 'action.active' }} />,
                }}
              />
            </Grid>
          </Grid>

          <Divider sx={{ my: 3 }} />

          {/* 偏好设置 */}
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Settings /> 偏好设置
          </Typography>
          <Stack spacing={2} sx={{ mb: 4 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={preferences.notifications}
                  onChange={handlePreferenceChange('notifications')}
                  disabled={!isEditing}
                />
              }
              label="接收通知"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={preferences.learning_reminders}
                  onChange={handlePreferenceChange('learning_reminders')}
                  disabled={!isEditing}
                />
              }
              label="学习提醒"
            />
          </Stack>

          <Divider sx={{ my: 3 }} />

          {/* 无障碍设置 */}
          <Typography variant="h6" gutterBottom>
            无障碍设置
          </Typography>
          <Stack spacing={2} sx={{ mb: 4 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={accessibility.high_contrast}
                  onChange={handleAccessibilityChange('high_contrast')}
                  disabled={!isEditing}
                />
              }
              label="高对比度"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={accessibility.large_text}
                  onChange={handleAccessibilityChange('large_text')}
                  disabled={!isEditing}
                />
              }
              label="大字体"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={accessibility.reduced_motion}
                  onChange={handleAccessibilityChange('reduced_motion')}
                  disabled={!isEditing}
                />
              }
              label="减少动画"
            />
          </Stack>

          <Divider sx={{ my: 3 }} />

          {/* 操作按钮 */}
          <Stack direction="row" spacing={2} justifyContent="flex-end">
            <Button
              variant="outlined"
              startIcon={<VpnKey />}
              onClick={() => setShowPasswordDialog(true)}
            >
              修改密码
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<Logout />}
              onClick={handleLogout}
            >
              登出
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {/* 修改密码对话框 */}
      <Dialog open={showPasswordDialog} onClose={() => setShowPasswordDialog(false)}>
        <DialogTitle>修改密码</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            此功能将在后续版本中实现
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPasswordDialog(false)}>关闭</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default UserProfile
