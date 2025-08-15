/**
 * 离线视频手语识别任务服务
 * 对接后端: /api/files/upload-video-for-recognition -> /api/sign/start -> /api/sign/status/{task_id} -> /api/sign/result/{task_id}
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

export interface StartTaskResponse {
  success: boolean
  task_id?: string
  message: string
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
    this.baseUrl = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000'
  }
  async uploadVideo(file: File): Promise<UploadVideoResponse> {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${this.baseUrl}/api/files/upload-video-for-recognition`, {
      method: 'POST',
      body: form,
      headers: this.getAuthHeaders(),
    })
    if (!res.ok) throw new Error(`上传失败: ${res.status}`)
    return res.json()
  }
  async startTask(fileHash: string): Promise<string> {
    const res = await fetch(`${this.baseUrl}/api/sign/start?file_hash=${encodeURIComponent(fileHash)}`, {
      method: 'POST',
      headers: { ...this.getAuthHeaders() }
    })
    if (!res.ok) throw new Error(`启动任务失败: ${res.status}`)
    const data: StartTaskResponse = await res.json()
    if (!data.success || !data.task_id) throw new Error(data.message || '任务启动失败')
    return data.task_id
  }
  async getStatus(taskId: string): Promise<StatusResponse> {
    const res = await fetch(`${this.baseUrl}/api/sign/status/${taskId}`, { headers: this.getAuthHeaders() })
    if (!res.ok) throw new Error(`查询状态失败: ${res.status}`)
    return res.json()
  }
  async getResult(taskId: string): Promise<ResultResponse> {
    const res = await fetch(`${this.baseUrl}/api/sign/result/${taskId}`, { headers: this.getAuthHeaders() })
    if (!res.ok) throw new Error(`查询结果失败: ${res.status}`)
    return res.json()
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
