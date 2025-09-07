/**
 * 认证模态框组件
 * 集成登录和注册功能的模态框
 */

import React, { useState } from 'react'
import {
  Dialog,
  DialogContent,
  Box,
  IconButton,
  Slide,
  useTheme,
  useMediaQuery,
} from '@mui/material'
import { Close } from '@mui/icons-material'
import { TransitionProps } from '@mui/material/transitions'

import LoginForm from './LoginForm'
import RegisterForm from './RegisterForm'
import { useAuth } from '../../contexts/AuthContext'
import { LoginCredentials, RegisterData } from '../../services/authService'

interface AuthModalProps {
  open: boolean
  onClose: () => void
  initialMode?: 'login' | 'register'
}

const Transition = React.forwardRef(function Transition(
  props: TransitionProps & {
    children: React.ReactElement<any, any>
  },
  ref: React.Ref<unknown>,
) {
  return <Slide direction="up" ref={ref} {...props} />
})

const AuthModal: React.FC<AuthModalProps> = ({
  open,
  onClose,
  initialMode = 'login'
}) => {
  const [mode, setMode] = useState<'login' | 'register'>(initialMode)
  const { login, register, loading, error, clearError } = useAuth()
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'))

  // 重置模式当模态框关闭时
  React.useEffect(() => {
    if (!open) {
      setMode(initialMode)
      clearError()
    }
  }, [open, initialMode, clearError])

  const handleLogin = async (credentials: LoginCredentials) => {
    try {
      await login(credentials)
      onClose() // 登录成功后关闭模态框
    } catch (error) {
      // 错误已经在context中处理
    }
  }

  const handleRegister = async (userData: RegisterData) => {
    try {
      await register(userData)
      // 注册成功后切换到登录模式
      setMode('login')
    } catch (error) {
      // 错误已经在context中处理
    }
  }

  const handleSwitchToRegister = () => {
    setMode('register')
    clearError()
  }

  const handleSwitchToLogin = () => {
    setMode('login')
    clearError()
  }

  const handleClose = () => {
    clearError()
    onClose()
  }

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      TransitionComponent={Transition}
      maxWidth="sm"
      fullWidth
      fullScreen={isMobile}
      PaperProps={{
        sx: {
          borderRadius: isMobile ? 0 : 4,
          background: 'transparent',
          boxShadow: 'none',
          overflow: 'visible',
        }
      }}
      sx={{
        '& .MuiBackdrop-root': {
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          backdropFilter: 'blur(8px)',
        }
      }}
    >
      <DialogContent
        sx={{
          p: 0,
          position: 'relative',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: isMobile ? '100vh' : 'auto',
          background: isMobile ? 'linear-gradient(135deg, #B5EAD7 0%, #C7F0DB 100%)' : 'transparent',
        }}
      >
        {/* 关闭按钮 */}
        <IconButton
          onClick={handleClose}
          sx={{
            position: 'absolute',
            top: isMobile ? 16 : -16,
            right: isMobile ? 16 : -16,
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            color: 'text.primary',
            zIndex: 1,
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 1)',
            },
            boxShadow: theme.shadows[4],
          }}
        >
          <Close />
        </IconButton>

        {/* 认证表单容器 */}
        <Box
          sx={{
            width: '100%',
            maxWidth: 450,
            px: isMobile ? 2 : 0,
            py: isMobile ? 4 : 0,
          }}
        >
          {mode === 'login' ? (
            <LoginForm
              onLogin={handleLogin}
              onSwitchToRegister={handleSwitchToRegister}
              loading={loading}
              error={error}
            />
          ) : (
            <RegisterForm
              onRegister={handleRegister}
              onSwitchToLogin={handleSwitchToLogin}
              loading={loading}
              error={error}
            />
          )}
        </Box>
      </DialogContent>
    </Dialog>
  )
}

export default AuthModal
