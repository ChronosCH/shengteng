/**
 * 增强版CE-CSL手语识别服务
 * 集成训练好的enhanced_cecsl_final_model.ckpt模型
 */

export interface LandmarkData {
  landmarks: number[][]
  description?: string
}

export interface PredictionResult {
  text: string
  confidence: number
  gloss_sequence: string[]
  inference_time: number
  timestamp: number
  status: string
  error?: string
}

export interface TestResponse {
  success: boolean
  message: string
  prediction?: PredictionResult
  stats?: {
    predictions: number
    errors: number
    total_inference_time: number
    avg_inference_time: number
  }
}

export interface StatsResponse {
  success: boolean
  stats: {
    predictions: number
    errors: number
    total_inference_time: number
    avg_inference_time: number
  }
  model_info: {
    model_path: string
    vocab_path: string
    vocab_size: number
    is_loaded: boolean
  }
}

export interface VideoProcessResult {
  task_id: string
  video_path: string
  frame_count: number
  fps: number
  duration: number
  landmarks_extracted: boolean
  recognition_result?: PredictionResult
  processing_time: number
  status: 'processing' | 'completed' | 'error'
  error?: string
}

class EnhancedCECSLService {
  private baseUrl: string

  constructor() {
    this.baseUrl = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000'
  }

  /**
   * 获取服务健康状态
   */
  async getHealth(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/api/health`)
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`)
    }
    return response.json()
  }

  /**
   * 获取增强版CE-CSL服务统计信息
   */
  async getStats(): Promise<StatsResponse> {
    const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken')
    
    const response = await fetch(`${this.baseUrl}/api/enhanced-cecsl/stats`, {
      headers: {
        ...(token && { 'Authorization': `Bearer ${token}` })
      }
    })
    
    if (!response.ok) {
      throw new Error(`Failed to get stats: ${response.status}`)
    }
    return response.json()
  }

  /**
   * 测试模型预测（使用关键点数据）
   */
  async testPrediction(landmarks: number[][], description?: string): Promise<TestResponse> {
    const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken')
    
    const response = await fetch(`${this.baseUrl}/api/enhanced-cecsl/test`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` })
      },
      body: JSON.stringify({
        landmarks,
        description
      })
    })

    if (!response.ok) {
      throw new Error(`Prediction failed: ${response.status}`)
    }

    return response.json()
  }

  /**
   * 上传视频文件进行手语识别
   */
  async uploadVideoForRecognition(file: File, onProgress?: (progress: number) => void): Promise<{
    success: boolean
    task_id?: string
    message: string
  }> {
    const formData = new FormData()
    formData.append('file', file)  // 修改为 'file' 以匹配后端期望

    // 获取认证token
    const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken')

    const xhr = new XMLHttpRequest()
    
    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = event.loaded / event.total
          onProgress(progress * 0.3) // 上传占总进度的30%
        }
      })

      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          try {
            const result = JSON.parse(xhr.responseText)
            resolve(result)
          } catch (e) {
            reject(new Error('Invalid response format'))
          }
        } else {
          try {
            const errorData = JSON.parse(xhr.responseText)
            reject(new Error(errorData.detail || `Upload failed: ${xhr.status}`))
          } catch (e) {
            reject(new Error(`Upload failed: ${xhr.status}`))
          }
        }
      })

      xhr.addEventListener('error', () => {
        reject(new Error('Upload failed'))
      })

      xhr.open('POST', `${this.baseUrl}/api/enhanced-cecsl/upload-video`)
      
      // 添加认证头
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      }
      
      xhr.send(formData)
    })
  }

  /**
   * 检查视频处理状态
   */
  async getVideoProcessStatus(taskId: string): Promise<{
    status: 'processing' | 'completed' | 'error'
    progress: number
    result?: VideoProcessResult
    error?: string
  }> {
    const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken')
    
    const response = await fetch(`${this.baseUrl}/api/enhanced-cecsl/video-status/${taskId}`, {
      headers: {
        ...(token && { 'Authorization': `Bearer ${token}` })
      }
    })
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('任务不存在')
      }
      throw new Error(`Failed to get status: ${response.status}`)
    }

    return response.json()
  }

  /**
   * 轮询视频处理结果
   */
  async pollVideoResult(
    taskId: string, 
    onProgress?: (progress: number) => void,
    intervalMs: number = 2000,
    timeoutMs: number = 300000 // 5分钟超时
  ): Promise<VideoProcessResult> {
    const startTime = Date.now()

    while (Date.now() - startTime < timeoutMs) {
      try {
        const status = await this.getVideoProcessStatus(taskId)
        
        if (onProgress) {
          onProgress(0.3 + status.progress * 0.7) // 处理占剩余70%进度
        }

        if (status.status === 'completed' && status.result) {
          return status.result
        }

        if (status.status === 'error') {
          throw new Error(status.error || 'Video processing failed')
        }

        // 等待下次轮询
        await new Promise(resolve => setTimeout(resolve, intervalMs))
      } catch (error) {
        if (Date.now() - startTime >= timeoutMs) {
          throw new Error('Video processing timeout')
        }
        throw error
      }
    }

    throw new Error('Video processing timeout')
  }

  /**
   * 完整的视频识别流程
   */
  async recognizeVideo(
    file: File,
    onProgress?: (progress: number, status: string) => void
  ): Promise<VideoProcessResult> {
    try {
      // 1. 上传视频
      onProgress?.(0, '正在上传视频...')
      const uploadResult = await this.uploadVideoForRecognition(file, (progress) => {
        onProgress?.(progress, '正在上传视频...')
      })

      if (!uploadResult.success || !uploadResult.task_id) {
        throw new Error(uploadResult.message || 'Upload failed')
      }

      // 2. 处理视频并获取结果
      onProgress?.(0.3, '正在处理视频...')
      const result = await this.pollVideoResult(
        uploadResult.task_id,
        (progress) => {
          onProgress?.(progress, '正在识别手语...')
        }
      )

      onProgress?.(1, '识别完成')
      return result

    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Unknown error')
    }
  }

  /**
   * 生成测试关键点数据
   */
  generateTestLandmarks(numFrames: number = 30): number[][] {
    const landmarks: number[][] = []
    const featuresPerFrame = 543 * 3 // MediaPipe 关键点数据

    for (let frame = 0; frame < numFrames; frame++) {
      const frameData: number[] = []
      for (let i = 0; i < featuresPerFrame; i++) {
        // 生成归一化的随机坐标 (0.1-0.9)
        frameData.push(Math.random() * 0.8 + 0.1)
      }
      landmarks.push(frameData)
    }

    return landmarks
  }
}

export default new EnhancedCECSLService()

// 已下线：增强版 CE-CSL 模拟服务。保留此文件作为占位以避免历史导入报错。
// 如需使用真实的连续手语识别，请改用 continuousSignRecognitionService。
