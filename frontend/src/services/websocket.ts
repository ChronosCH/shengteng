/**
 * WebSocket服务 - 处理与后端的实时通信
 */

// WebSocket service for real-time communication

export interface LandmarkData {
  landmarks: number[][]
  timestamp: number
  frameId: number
}

export interface RecognitionResult {
  text: string
  confidence: number
  glossSequence: string[]
  timestamp: number
  frameId: number
}

export interface WebSocketMessage {
  type: string
  payload: any
}

export class WebSocketService {
  private socket: WebSocket | null = null
  private eventListeners: Map<string, Function[]> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private isConnecting = false

  // 性能优化相关
  private messageQueue: WebSocketMessage[] = []
  private isProcessingQueue = false
  private lastSendTime = 0
  private sendInterval = 33 // 30fps, 约33ms间隔
  private compressionEnabled = true
  private batchSize = 1 // 批处理大小

  constructor(private url?: string) {
    if (!this.url) {
      const envUrl = (import.meta as any)?.env?.VITE_WS_URL
      if (envUrl) {
        this.url = envUrl
      } else if (import.meta.env.VITE_WS_BASE_URL) {
        // 使用环境变量配置的WebSocket URL
        this.url = `${import.meta.env.VITE_WS_BASE_URL}/ws/sign-recognition`
      } else if (typeof window !== 'undefined') {
        // 浏览器环境，根据当前页面URL动态生成
        const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws'
        if (window.location.port === '5173') {
          this.url = `${wsProto}://${window.location.hostname}:8000/ws/sign-recognition`
        } else {
          this.url = `${wsProto}://${window.location.host}/ws/sign-recognition`
        }
      } else {
        this.url = 'ws://localhost:8000/ws/sign-recognition'
      }
    }
  }

  /**
   * 连接WebSocket
   */
  async connect(): Promise<void> {
    if (this.socket?.readyState === WebSocket.OPEN || this.isConnecting) {
      return
    }

    this.isConnecting = true

    try {
      this.socket = new WebSocket(this.url!)

      this.socket.onopen = () => {
        console.log('WebSocket连接已建立')
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.emit('connect')
      }

      this.socket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          this.handleMessage(message)
        } catch (error) {
          console.error('解析WebSocket消息失败:', error)
        }
      }

      this.socket.onclose = (event) => {
        console.log('WebSocket连接已关闭:', event.code, event.reason)
        this.isConnecting = false
        this.emit('disconnect')
      }

      this.socket.onerror = (error) => {
        console.error('WebSocket错误:', error)
        this.isConnecting = false
        this.emit('error', 'WebSocket连接错误')
      }

    } catch (error) {
      this.isConnecting = false
      console.error('WebSocket连接失败:', error)
      this.emit('error', '无法连接到服务器')
      throw error
    }
  }

  /**
   * 断开WebSocket连接
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.close(1000, '用户主动断开')
      this.socket = null
    }
  }

  /**
   * 发送关键点数据 (优化版本)
   */
  sendLandmarks(landmarks: number[][]): void {
    if (!this.isConnected()) {
      console.warn('WebSocket未连接，无法发送关键点数据')
      return
    }

    const now = Date.now()

    // 限制发送频率以减少延迟
    if (now - this.lastSendTime < this.sendInterval) {
      return
    }

    const message: WebSocketMessage = {
      type: 'landmarks',
      payload: {
        landmarks: this.compressionEnabled ? this.compressLandmarks(landmarks) : landmarks,
        timestamp: now,
        frameId: Math.floor(Math.random() * 1000000),
      }
    }

    this.sendOptimized(message)
    this.lastSendTime = now
  }

  /**
   * 压缩关键点数据
   */
  private compressLandmarks(landmarks: number[][]): number[][] {
    // 简单的数据压缩：保留关键关键点，降低精度
    return landmarks.map(point => [
      Math.round(point[0] * 1000) / 1000, // 保留3位小数
      Math.round(point[1] * 1000) / 1000,
      Math.round(point[2] * 1000) / 1000,
    ])
  }

  /**
   * 优化的发送方法
   */
  private sendOptimized(message: WebSocketMessage): void {
    if (this.batchSize > 1) {
      // 批处理模式
      this.messageQueue.push(message)
      if (!this.isProcessingQueue) {
        this.processMessageQueue()
      }
    } else {
      // 直接发送模式 (低延迟)
      this.send(message)
    }
  }

  /**
   * 处理消息队列
   */
  private async processMessageQueue(): Promise<void> {
    this.isProcessingQueue = true

    while (this.messageQueue.length > 0) {
      const batch = this.messageQueue.splice(0, this.batchSize)

      if (batch.length === 1) {
        this.send(batch[0])
      } else {
        // 批量发送
        const batchMessage: WebSocketMessage = {
          type: 'batch',
          payload: { messages: batch }
        }
        this.send(batchMessage)
      }

      // 避免阻塞
      await new Promise(resolve => setTimeout(resolve, 0))
    }

    this.isProcessingQueue = false
  }

  /**
   * 发送配置更新
   */
  sendConfig(config: any): void {
    if (!this.isConnected()) {
      console.warn('WebSocket未连接，无法发送配置')
      return
    }

    const message: WebSocketMessage = {
      type: 'config',
      payload: config
    }

    this.send(message)
  }

  /**
   * 检查连接状态
   */
  isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN
  }

  /**
   * 添加事件监听器
   */
  on(event: string, callback: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, [])
    }
    this.eventListeners.get(event)!.push(callback)
  }

  /**
   * 移除事件监听器
   */
  off(event: string, callback: Function): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      const index = listeners.indexOf(callback)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  /**
   * 触发事件
   */
  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach(callback => callback(data))
    }
  }

  /**
   * 发送消息
   */
  private send(message: WebSocketMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message))
    }
  }

  /**
   * 处理接收到的消息
   */
  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'connection_established':
        console.log('连接建立确认:', message.payload)
        break

      case 'recognition_result':
        this.emit('recognition_result', message.payload as RecognitionResult)
        break

      case 'config_updated':
        this.emit('config_updated', message.payload)
        break

      case 'error':
        console.error('服务器错误:', message.payload.message)
        this.emit('error', message.payload.message)
        break

      default:
        console.warn('未知消息类型:', message.type)
    }
  }

  /**
   * 安排重连
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    
    console.log(`${delay}ms后尝试第${this.reconnectAttempts}次重连...`)
    
    setTimeout(() => {
      this.connect().catch(error => {
        console.error('重连失败:', error)
      })
    }, delay)
  }

  /**
   * 获取连接统计信息
   */
  getStats(): any {
    return {
      isConnected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      url: this.url,
    }
  }
}

// 创建全局WebSocket服务实例（使用动态地址）
export const websocketService = new WebSocketService()
