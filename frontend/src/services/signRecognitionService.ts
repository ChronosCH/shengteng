/**
 * 离线视频手语识别任务服务
 * 对接后端: /api/sign-recognition/upload-video -> /api/sign-recognition/status/{taskId}
 */

export interface UploadVideoResponse {
  success: boolean
  message: string
  data?: {
    file_hash: string
    file_path: string
    processing_status: string
  }
}

export interface StatusResponse {
  status: string
  progress?: number
  error?: string
}

export interface SegmentResult {
  gloss_sequence: string[]
  start_frame: number
  end_frame: number
  confidence: number
  start_time: number
  end_time: number
}

export interface RecognitionResultData {
  task_id: string
  file_path: string
  gloss_sequence: string[]
  text: string
  segments: SegmentResult[]
  overall_confidence: number
  frame_count: number
  fps: number
  duration: number
  srt_path?: string
  created_at: number
}

export interface ResultResponse {
  status: string
  result?: RecognitionResultData
}

class SignRecognitionService {
  private baseUrl: string
  private getAuthHeaders(): HeadersInit {
    const token = localStorage.getItem('auth_token')
    return token ? { 'Authorization': `Bearer ${token}` } : {}
  }
  constructor() {
    this.baseUrl = (import.meta.env.VITE_API_URL as string) || (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000'
  }
  async uploadVideo(file: File): Promise<UploadVideoResponse> {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${this.baseUrl}/api/sign-recognition/upload-video`, {
      method: 'POST',
      body: form,
      headers: this.getAuthHeaders(),
    })
    if (!res.ok) throw new Error(`上传失败: ${res.status}`)
    const data = await res.json()
    // 兼容返回结构：返回 success + task_id
    if (data && data.task_id) {
      return { success: true, message: data.message || 'ok', data: { file_hash: data.task_id, file_path: '', processing_status: 'uploaded' } }
    }
    return data
  }
  async startTask(fileHash: string): Promise<string> {
    // 后端已在上传时创建任务，直接返回 fileHash 作为 taskId
    return fileHash
  }
  async getStatus(taskId: string): Promise<StatusResponse> {
    const res = await fetch(`${this.baseUrl}/api/sign-recognition/status/${taskId}`, { headers: this.getAuthHeaders() })
    if (!res.ok) throw new Error(`查询状态失败: ${res.status}`)
    const data = await res.json()
    // 统一状态字段
    const st = (data.status || '').toLowerCase()
    if (st === 'processing' || st === 'queued' || st === 'uploaded') {
      return { status: 'processing', progress: data.progress }
    }
    if (st === 'finished' || st === 'completed') {
      return { status: 'finished', progress: 1 }
    }
    if (st === 'error') {
      return { status: 'error', error: data.error }
    }
    return { status: data.status || 'unknown', progress: data.progress }
  }
  async getResult(taskId: string): Promise<ResultResponse> {
    // 后端在 status 中已携带 result，我们重用 getStatus 再取一次
    const res = await fetch(`${this.baseUrl}/api/sign-recognition/status/${taskId}`, { headers: this.getAuthHeaders() })
    if (!res.ok) throw new Error(`查询结果失败: ${res.status}`)
    const data = await res.json()
    return { status: data.status, result: data.result }
  }
  async pollResult(taskId: string, intervalMs = 2000, onProgress?: (p: number) => void, signal?: AbortSignal): Promise<RecognitionResultData> {
    while (true) {
      if (signal?.aborted) throw new Error('已取消')
      const status = await this.getStatus(taskId)
      if (status.status === 'finished') {
        const result = await this.getResult(taskId)
        if (result.status === 'finished' && result.result) return result.result
      } else if (status.status === 'error') {
        throw new Error(status.error || '任务失败')
      } else if (status.progress !== undefined && onProgress) {
        onProgress(status.progress)
      }
      await new Promise(r => setTimeout(r, intervalMs))
    }
  }
  getSrtDownloadUrl(srtPath: string): string {
    if (!srtPath) return ''
    const norm = srtPath.replace(/\\/g, '/').replace(/^temp\/sign_results\//, '')
    return `${this.baseUrl}/sign_results/${norm}`
  }
}

export const signRecognitionService = new SignRecognitionService()
export default signRecognitionService
