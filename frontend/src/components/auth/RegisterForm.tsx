/**
 * 注册表单组件
 * 提供用户注册功能，包含表单验证
 */

import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
  Link,
  Divider,
  Stack,
  FormHelperText,
} from '@mui/material'
import {
  Visibility,
  VisibilityOff,
  Person,
  Email,
  Lock,
  PersonAdd,
  Badge,
} from '@mui/icons-material'

interface RegisterFormProps {
  onRegister: (userData: RegisterData) => Promise<void>
  onSwitchToLogin: () => void
  loading?: boolean
  error?: string
}

interface RegisterData {
  username: string
  email: string
  password: string
  confirmPassword: string
  full_name?: string
}

const RegisterForm: React.FC<RegisterFormProps> = ({
  onRegister,
  onSwitchToLogin,
  loading = false,
  error
}) => {
  const [userData, setUserData] = useState<RegisterData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    full_name: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [validationErrors, setValidationErrors] = useState<{[key: string]: string}>({})

  const validateForm = (): boolean => {
    const errors: {[key: string]: string} = {}

    // 用户名验证
    if (!userData.username.trim()) {
      errors.username = '请输入用户名'
    } else if (userData.username.length < 3) {
      errors.username = '用户名至少需要3个字符'
    } else if (userData.username.length > 50) {
      errors.username = '用户名不能超过50个字符'
    }

    // 邮箱验证
    if (!userData.email.trim()) {
      errors.email = '请输入邮箱'
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(userData.email)) {
      errors.email = '请输入有效的邮箱地址'
    }

    // 密码验证
    if (!userData.password) {
      errors.password = '请输入密码'
    } else if (userData.password.length < 6) {
      errors.password = '密码至少需要6个字符'
    }

    // 确认密码验证
    if (!userData.confirmPassword) {
      errors.confirmPassword = '请确认密码'
    } else if (userData.password !== userData.confirmPassword) {
      errors.confirmPassword = '两次输入的密码不一致'
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!validateForm()) {
      return
    }

    try {
      const { confirmPassword, ...registerData } = userData
      await onRegister(registerData)
    } catch (error) {
      // 错误处理由父组件处理
    }
  }

  const handleInputChange = (field: keyof RegisterData) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const value = e.target.value
    setUserData(prev => ({
      ...prev,
      [field]: value
    }))
    
    // 清除对应字段的验证错误
    if (validationErrors[field]) {
      setValidationErrors(prev => ({
        ...prev,
        [field]: ''
      }))
    }

    // 如果修改密码，同时清除确认密码的错误
    if (field === 'password' && validationErrors.confirmPassword) {
      setValidationErrors(prev => ({
        ...prev,
        confirmPassword: ''
      }))
    }
  }

  return (
    <Card
      elevation={8}
      sx={{
        maxWidth: 450,
        width: '100%',
        borderRadius: 4,
        background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
      }}
    >
      <CardContent sx={{ p: 4 }}>
        {/* 标题 */}
        <Box textAlign="center" mb={3}>
          <PersonAdd 
            sx={{ 
              fontSize: 48, 
              color: 'primary.main',
              mb: 1
            }} 
          />
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            注册
          </Typography>
          <Typography variant="body2" color="text.secondary">
            创建您的手语学习账户
          </Typography>
        </Box>

        {/* 错误提示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* 注册表单 */}
        <Box component="form" onSubmit={handleSubmit}>
          <Stack spacing={3}>
            {/* 用户名输入 */}
            <TextField
              fullWidth
              label="用户名"
              value={userData.username}
              onChange={handleInputChange('username')}
              error={!!validationErrors.username}
              helperText={validationErrors.username}
              disabled={loading}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Person color="action" />
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                }
              }}
            />

            {/* 邮箱输入 */}
            <TextField
              fullWidth
              label="邮箱"
              type="email"
              value={userData.email}
              onChange={handleInputChange('email')}
              error={!!validationErrors.email}
              helperText={validationErrors.email}
              disabled={loading}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Email color="action" />
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                }
              }}
            />

            {/* 姓名输入（可选） */}
            <TextField
              fullWidth
              label="姓名（可选）"
              value={userData.full_name}
              onChange={handleInputChange('full_name')}
              disabled={loading}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Badge color="action" />
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                }
              }}
            />

            {/* 密码输入 */}
            <TextField
              fullWidth
              label="密码"
              type={showPassword ? 'text' : 'password'}
              value={userData.password}
              onChange={handleInputChange('password')}
              error={!!validationErrors.password}
              helperText={validationErrors.password}
              disabled={loading}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword(!showPassword)}
                      edge="end"
                      disabled={loading}
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                }
              }}
            />

            {/* 确认密码输入 */}
            <TextField
              fullWidth
              label="确认密码"
              type={showConfirmPassword ? 'text' : 'password'}
              value={userData.confirmPassword}
              onChange={handleInputChange('confirmPassword')}
              error={!!validationErrors.confirmPassword}
              helperText={validationErrors.confirmPassword}
              disabled={loading}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      edge="end"
                      disabled={loading}
                    >
                      {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                }
              }}
            />

            {/* 注册按钮 */}
            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <PersonAdd />}
              sx={{
                borderRadius: 2,
                py: 1.5,
                background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #A0D4C4 0%, #B4E3C8 100%)',
                }
              }}
            >
              {loading ? '注册中...' : '注册'}
            </Button>
          </Stack>
        </Box>

        {/* 分割线 */}
        <Divider sx={{ my: 3 }}>
          <Typography variant="body2" color="text.secondary">
            或
          </Typography>
        </Divider>

        {/* 登录链接 */}
        <Box textAlign="center">
          <Typography variant="body2" color="text.secondary">
            已有账户？{' '}
            <Link
              component="button"
              type="button"
              onClick={onSwitchToLogin}
              disabled={loading}
              sx={{
                textDecoration: 'none',
                fontWeight: 'bold',
                color: 'primary.main',
                '&:hover': {
                  textDecoration: 'underline',
                }
              }}
            >
              立即登录
            </Link>
          </Typography>
        </Box>
      </CardContent>
    </Card>
  )
}

export default RegisterForm
