export interface IsolatedUploadResponse {
  success: boolean
  file_path: string
  filename: string
}

export interface IsolatedPredictionResponse {
  success: boolean
  prediction: {
    gloss: string | null
    confidence: number
    logits: number[]
  }
  feedback?: {
    recognized_sign: string
    accuracy_level: string
    message: string
    tips: string[]
    next_steps: string[]
  }
}

class IsolatedSignLearningService {
  private baseUrl: string

  constructor() {
    this.baseUrl = (import.meta.env.VITE_API_URL as string) || (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000'
  }

  private getAuthHeaders(): HeadersInit {
    const token = localStorage.getItem('access_token')
    return token ? { Authorization: `Bearer ${token}` } : {}
  }

  async uploadIsolatedVideo(file: File): Promise<IsolatedUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch(`${this.baseUrl}/api/learning/isolated-sign/upload`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: formData,
    })

    const text = await res.text()
    if (!res.ok) {
      try {
        const errorData = text ? JSON.parse(text) : null
        throw new Error(errorData?.detail || `上传失败: ${res.status}`)
      } catch (parseErr) {
        throw new Error(`上传失败: ${res.status}`)
      }
    }

    const data = text ? JSON.parse(text) : null
    if (!data?.success) {
      throw new Error(data?.detail || '上传失败')
    }

    return data
  }

  async predictIsolatedVideo(filePath: string): Promise<IsolatedPredictionResponse> {
    const res = await fetch(`${this.baseUrl}/api/learning/isolated-sign/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify({ file_path: filePath }),
    })

    if (!res.ok) {
      throw new Error(`推理失败: ${res.status}`)
    }

    const data = await res.json()
    if (!data?.success) {
      throw new Error(data?.detail || '推理失败')
    }

    return data
  }
}

export const isolatedSignLearningService = new IsolatedSignLearningService()
export default isolatedSignLearningService
