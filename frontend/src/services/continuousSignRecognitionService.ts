/**
 * 连续手语识别服务
 * 对接真正的CSLR模型API
 */

export interface ContinuousRecognitionResult {
  task_id: string
  file_path: string
  gloss_sequence: string[]
  text: string
  segments: Array<{
    gloss_sequence: string[]
    start_frame: number
    end_frame: number
    confidence: number
    start_time: number
    end_time: number
  }>
  overall_confidence: number
  frame_count: number
  fps: number
  duration: number
  srt_path?: string
  created_at: number
}

export interface ContinuousUploadResponse {
  success: boolean
  task_id?: string
  message: string
  status: string
}

export interface ContinuousStatusResponse {
  status: string
  progress?: number
  result?: ContinuousRecognitionResult
  error?: string
}

class ContinuousSignRecognitionService {
  private baseUrl: string

  constructor() {
    this.baseUrl = (import.meta.env?.VITE_API_URL as string) || 'http://localhost:8000'
  }

  /**
   * 上传视频进行连续手语识别
   */
  async uploadVideo(file: File, onProgress?: (progress: number) => void): Promise<ContinuousUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    
    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = event.loaded / event.total
          onProgress(progress * 0.2) // 上传占总进度的20%
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

      xhr.open('POST', `${this.baseUrl}/api/sign-recognition/upload-video`)
      xhr.send(formData)
    })
  }

  /**
   * 获取任务状态
   */
  async getTaskStatus(taskId: string): Promise<ContinuousStatusResponse> {
    const response = await fetch(`${this.baseUrl}/api/sign-recognition/status/${taskId}`)
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    return response.json()
  }

  /**
   * 轮询获取结果
   */
  async pollResult(
    taskId: string, 
    pollInterval: number = 2000,
    onProgress?: (progress: number) => void,
    signal?: AbortSignal
  ): Promise<ContinuousRecognitionResult> {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          if (signal?.aborted) {
            reject(new Error('Operation aborted'))
            return
          }

          const status = await this.getTaskStatus(taskId)
          
          if (status.progress !== undefined && onProgress) {
            // 上传占20%，处理占80%
            onProgress(0.2 + status.progress * 0.8)
          }

          const st = (status.status || '').toLowerCase()
          if ((st === 'completed' || st === 'finished') && status.result) {
            resolve(status.result)
          } else if (st === 'error') {
            reject(new Error(status.error || '处理失败'))
          } else {
            // 继续轮询
            setTimeout(poll, pollInterval)
          }
        } catch (error) {
          reject(error)
        }
      }

      poll()
    })
  }

  /**
   * 完整的识别流程
   */
  async recognizeVideo(
    file: File,
    onProgress?: (progress: number, status: string) => void
  ): Promise<ContinuousRecognitionResult> {
    try {
      // 1. 上传视频
      onProgress?.(0, '正在上传视频...')
      const uploadResult = await this.uploadVideo(file, (progress) => {
        onProgress?.(progress, '正在上传视频...')
      })

      if (!uploadResult.success || !uploadResult.task_id) {
        throw new Error(uploadResult.message || 'Upload failed')
      }

      // 2. 处理视频并获取结果
      onProgress?.(0.2, '正在进行连续手语识别...')
      const result = await this.pollResult(
        uploadResult.task_id,
        2000,
        (progress) => {
          onProgress?.(progress, '正在进行连续手语识别...')
        }
      )

      onProgress?.(1, '识别完成')
      return result

    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Unknown error')
    }
  }
}

// 导出单例
const continuousSignRecognitionService = new ContinuousSignRecognitionService()
export default continuousSignRecognitionService
