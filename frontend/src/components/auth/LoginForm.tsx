/**
 * 登录表单组件
 * 提供用户登录功能，支持记住我选项
 */

import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  FormControlLabel,
  Checkbox,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
  Link,
  Divider,
  Stack,
} from '@mui/material'
import {
  Visibility,
  VisibilityOff,
  Person,
  Lock,
  Login as LoginIcon,
} from '@mui/icons-material'
import { validateUsername, sanitizeInput } from '../../utils/inputValidation'

interface LoginFormProps {
  onLogin: (credentials: LoginCredentials) => Promise<void>
  onSwitchToRegister: () => void
  loading?: boolean
  error?: string
}

interface LoginCredentials {
  username: string
  password: string
  remember_me: boolean
}

const LoginForm: React.FC<LoginFormProps> = ({
  onLogin,
  onSwitchToRegister,
  loading = false,
  error
}) => {
  const [credentials, setCredentials] = useState<LoginCredentials>({
    username: '',
    password: '',
    remember_me: false
  })
  const [showPassword, setShowPassword] = useState(false)
  const [validationErrors, setValidationErrors] = useState<{[key: string]: string}>({})

  const validateForm = (): boolean => {
    const errors: {[key: string]: string} = {}

    // 验证用户名
    const usernameResult = validateUsername(credentials.username)
    if (!usernameResult.isValid) {
      errors.username = usernameResult.errors[0]
    }

    // 验证密码（基本检查）
    if (!credentials.password) {
      errors.password = '请输入密码'
    } else if (credentials.password.length > 128) {
      errors.password = '密码过长'
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
      await onLogin(credentials)
    } catch (error) {
      // 错误处理由父组件处理
    }
  }

  const handleInputChange = (field: keyof LoginCredentials) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    let value: string | boolean = field === 'remember_me' ? e.target.checked : e.target.value

    // 对字符串输入进行基本清理
    if (typeof value === 'string') {
      value = sanitizeInput(value, field === 'username' ? 50 : 128)
    }

    setCredentials(prev => ({
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
  }

  return (
    <Card
      elevation={8}
      sx={{
        maxWidth: 400,
        width: '100%',
        borderRadius: 4,
        background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
      }}
    >
      <CardContent sx={{ p: 4 }}>
        {/* 标题 */}
        <Box textAlign="center" mb={3}>
          <LoginIcon 
            sx={{ 
              fontSize: 48, 
              color: 'primary.main',
              mb: 1
            }} 
          />
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            登录
          </Typography>
          <Typography variant="body2" color="text.secondary">
            登录手语学习训练系统
          </Typography>
        </Box>

        {/* 错误提示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* 登录表单 */}
        <Box component="form" onSubmit={handleSubmit}>
          <Stack spacing={3}>
            {/* 用户名输入 */}
            <TextField
              fullWidth
              label="用户名"
              value={credentials.username}
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

            {/* 密码输入 */}
            <TextField
              fullWidth
              label="密码"
              type={showPassword ? 'text' : 'password'}
              value={credentials.password}
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

            {/* 记住我选项 */}
            <FormControlLabel
              control={
                <Checkbox
                  checked={credentials.remember_me}
                  onChange={handleInputChange('remember_me')}
                  disabled={loading}
                  color="primary"
                />
              }
              label="记住我"
            />

            {/* 登录按钮 */}
            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <LoginIcon />}
              sx={{
                borderRadius: 2,
                py: 1.5,
                background: 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #A0D4C4 0%, #B4E3C8 100%)',
                }
              }}
            >
              {loading ? '登录中...' : '登录'}
            </Button>
          </Stack>
        </Box>

        {/* 分割线 */}
        <Divider sx={{ my: 3 }}>
          <Typography variant="body2" color="text.secondary">
            或
          </Typography>
        </Divider>

        {/* 注册链接 */}
        <Box textAlign="center">
          <Typography variant="body2" color="text.secondary">
            还没有账户？{' '}
            <Link
              component="button"
              type="button"
              onClick={onSwitchToRegister}
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
              立即注册
            </Link>
          </Typography>
        </Box>
      </CardContent>
    </Card>
  )
}

export default LoginForm
