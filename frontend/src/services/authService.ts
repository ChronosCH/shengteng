/**
 * 认证服务
 * 处理用户认证相关的API调用
 */

const API_BASE_URL = 'http://localhost:8000'

export interface LoginCredentials {
  username: string
  password: string
  remember_me: boolean
}

export interface RegisterData {
  username: string
  email: string
  password: string
  full_name?: string
}

export interface UserInfo {
  id: number
  username: string
  email: string
  full_name?: string
  is_active: boolean
  is_admin: boolean
  preferences: Record<string, any>
  accessibility_settings: Record<string, any>
}

export interface AuthResponse {
  success: boolean
  message: string
  data?: any
}

export interface LoginResponse extends AuthResponse {
  data: {
    access_token: string
    token_type: string
    expires_in: number
    refresh_token: string
    user_info: UserInfo
  }
}

export interface ProfileUpdateData {
  full_name?: string
  email?: string
  preferences?: Record<string, any>
  accessibility_settings?: Record<string, any>
}

export interface PasswordChangeData {
  current_password: string
  new_password: string
}

class AuthService {
  private token: string | null = null
  private refreshToken: string | null = null
  private userInfo: UserInfo | null = null

  constructor() {
    // 从localStorage恢复认证状态
    this.loadAuthState()
  }

  private loadAuthState() {
    try {
      this.token = localStorage.getItem('access_token')
      this.refreshToken = localStorage.getItem('refresh_token')
      const userInfoStr = localStorage.getItem('user_info')
      if (userInfoStr) {
        this.userInfo = JSON.parse(userInfoStr)
      }
    } catch (error) {
      console.error('加载认证状态失败:', error)
      this.clearAuthState()
    }
  }

  private saveAuthState() {
    try {
      if (this.token) {
        localStorage.setItem('access_token', this.token)
      }
      if (this.refreshToken) {
        localStorage.setItem('refresh_token', this.refreshToken)
      }
      if (this.userInfo) {
        localStorage.setItem('user_info', JSON.stringify(this.userInfo))
      }
    } catch (error) {
      console.error('保存认证状态失败:', error)
    }
  }

  private clearAuthState() {
    this.token = null
    this.refreshToken = null
    this.userInfo = null
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    localStorage.removeItem('user_info')
  }

  private async makeRequest(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<any> {
    const url = `${API_BASE_URL}${endpoint}`
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    }

    // 添加认证头
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`
    }

    const response = await fetch(url, {
      ...options,
      headers,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  // 用户注册
  async register(userData: RegisterData): Promise<AuthResponse> {
    const response = await this.makeRequest('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    })
    return response
  }

  // 用户登录
  async login(credentials: LoginCredentials): Promise<LoginResponse> {
    const response = await this.makeRequest('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    })

    if (response.success && response.data) {
      this.token = response.data.access_token
      this.refreshToken = response.data.refresh_token
      this.userInfo = response.data.user_info
      this.saveAuthState()
    }

    return response
  }

  // 用户登出
  async logout(): Promise<AuthResponse> {
    try {
      const response = await this.makeRequest('/api/auth/logout', {
        method: 'POST',
      })
      return response
    } finally {
      // 无论API调用是否成功，都清除本地状态
      this.clearAuthState()
    }
  }

  // 获取用户个人资料
  async getProfile(): Promise<AuthResponse> {
    const response = await this.makeRequest('/api/auth/profile')
    if (response.success && response.data) {
      this.userInfo = response.data
      this.saveAuthState()
    }
    return response
  }

  // 更新用户个人资料
  async updateProfile(profileData: ProfileUpdateData): Promise<AuthResponse> {
    const response = await this.makeRequest('/api/auth/profile', {
      method: 'PUT',
      body: JSON.stringify(profileData),
    })
    
    if (response.success) {
      // 重新获取用户信息
      await this.getProfile()
    }
    
    return response
  }

  // 修改密码
  async changePassword(passwordData: PasswordChangeData): Promise<AuthResponse> {
    return this.makeRequest('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(passwordData),
    })
  }

  // 刷新访问令牌
  async refreshAccessToken(): Promise<AuthResponse> {
    if (!this.refreshToken) {
      throw new Error('没有刷新令牌')
    }

    const response = await this.makeRequest('/api/auth/refresh-token', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: this.refreshToken }),
    })

    if (response.success && response.data) {
      this.token = response.data.access_token
      this.saveAuthState()
    }

    return response
  }

  // 验证令牌
  async verifyToken(): Promise<AuthResponse> {
    return this.makeRequest('/api/auth/verify-token')
  }

  // 获取用户会话
  async getSessions(): Promise<AuthResponse> {
    return this.makeRequest('/api/auth/sessions')
  }

  // 终止会话
  async terminateSession(sessionId: string): Promise<AuthResponse> {
    return this.makeRequest(`/api/auth/sessions/${sessionId}`, {
      method: 'DELETE',
    })
  }

  // 检查是否已登录
  isAuthenticated(): boolean {
    return !!this.token && !!this.userInfo
  }

  // 获取当前用户信息
  getCurrentUser(): UserInfo | null {
    return this.userInfo
  }

  // 获取访问令牌
  getAccessToken(): string | null {
    return this.token
  }

  // 获取刷新令牌
  getRefreshToken(): string | null {
    return this.refreshToken
  }
}

// 创建单例实例
const authService = new AuthService()

export default authService
