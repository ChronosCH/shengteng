/**
 * 认证上下文
 * 提供全局的用户认证状态管理
 */

import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react'
import authService, { UserInfo, LoginCredentials, RegisterData } from '../services/authService'

interface AuthState {
  isAuthenticated: boolean
  user: UserInfo | null
  loading: boolean
  error: string | null
}

type AuthAction =
  | { type: 'AUTH_START' }
  | { type: 'AUTH_SUCCESS'; payload: UserInfo }
  | { type: 'AUTH_FAILURE'; payload: string }
  | { type: 'AUTH_LOGOUT' }
  | { type: 'CLEAR_ERROR' }
  | { type: 'UPDATE_USER'; payload: UserInfo }

interface AuthContextType extends AuthState {
  login: (credentials: LoginCredentials) => Promise<void>
  register: (userData: RegisterData) => Promise<void>
  logout: () => Promise<void>
  updateProfile: (profileData: any) => Promise<void>
  clearError: () => void
  checkAuthStatus: () => Promise<void>
}

const initialState: AuthState = {
  isAuthenticated: false,
  user: null,
  loading: false,
  error: null,
}

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case 'AUTH_START':
      return {
        ...state,
        loading: true,
        error: null,
      }
    case 'AUTH_SUCCESS':
      return {
        ...state,
        isAuthenticated: true,
        user: action.payload,
        loading: false,
        error: null,
      }
    case 'AUTH_FAILURE':
      return {
        ...state,
        isAuthenticated: false,
        user: null,
        loading: false,
        error: action.payload,
      }
    case 'AUTH_LOGOUT':
      return {
        ...state,
        isAuthenticated: false,
        user: null,
        loading: false,
        error: null,
      }
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      }
    case 'UPDATE_USER':
      return {
        ...state,
        user: action.payload,
      }
    default:
      return state
  }
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState)

  // 检查认证状态
  const checkAuthStatus = async () => {
    if (authService.isAuthenticated()) {
      try {
        dispatch({ type: 'AUTH_START' })
        
        // 验证令牌是否仍然有效
        await authService.verifyToken()
        
        const user = authService.getCurrentUser()
        if (user) {
          dispatch({ type: 'AUTH_SUCCESS', payload: user })
        } else {
          dispatch({ type: 'AUTH_FAILURE', payload: '用户信息获取失败' })
        }
      } catch (error) {
        console.error('认证状态检查失败:', error)
        dispatch({ type: 'AUTH_FAILURE', payload: '认证状态验证失败' })
        // 清除无效的认证信息
        await authService.logout()
      }
    }
  }

  // 用户登录
  const login = async (credentials: LoginCredentials) => {
    try {
      dispatch({ type: 'AUTH_START' })
      
      const response = await authService.login(credentials)
      
      if (response.success && response.data) {
        dispatch({ type: 'AUTH_SUCCESS', payload: response.data.user_info })
      } else {
        dispatch({ type: 'AUTH_FAILURE', payload: response.message || '登录失败' })
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '登录失败'
      dispatch({ type: 'AUTH_FAILURE', payload: errorMessage })
      throw error
    }
  }

  // 用户注册
  const register = async (userData: RegisterData) => {
    try {
      dispatch({ type: 'AUTH_START' })
      
      const response = await authService.register(userData)
      
      if (response.success) {
        // 注册成功后不自动登录，让用户手动登录
        dispatch({ type: 'AUTH_LOGOUT' })
      } else {
        dispatch({ type: 'AUTH_FAILURE', payload: response.message || '注册失败' })
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '注册失败'
      dispatch({ type: 'AUTH_FAILURE', payload: errorMessage })
      throw error
    }
  }

  // 用户登出
  const logout = async () => {
    try {
      await authService.logout()
    } catch (error) {
      console.error('登出失败:', error)
    } finally {
      dispatch({ type: 'AUTH_LOGOUT' })
    }
  }

  // 更新用户资料
  const updateProfile = async (profileData: any) => {
    try {
      const response = await authService.updateProfile(profileData)
      
      if (response.success) {
        // 重新获取用户信息
        const profileResponse = await authService.getProfile()
        if (profileResponse.success && profileResponse.data) {
          dispatch({ type: 'UPDATE_USER', payload: profileResponse.data })
        }
      } else {
        throw new Error(response.message || '更新资料失败')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '更新资料失败'
      dispatch({ type: 'AUTH_FAILURE', payload: errorMessage })
      throw error
    }
  }

  // 清除错误
  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' })
  }

  // 组件挂载时检查认证状态
  useEffect(() => {
    checkAuthStatus()
  }, [])

  // 设置令牌刷新定时器
  useEffect(() => {
    if (state.isAuthenticated) {
      const refreshInterval = setInterval(async () => {
        try {
          await authService.refreshAccessToken()
        } catch (error) {
          console.error('令牌刷新失败:', error)
          // 如果刷新失败，登出用户
          logout()
        }
      }, 30 * 60 * 1000) // 每30分钟刷新一次

      return () => clearInterval(refreshInterval)
    }
  }, [state.isAuthenticated])

  const contextValue: AuthContextType = {
    ...state,
    login,
    register,
    logout,
    updateProfile,
    clearError,
    checkAuthStatus,
  }

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  )
}

// 自定义Hook
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext
