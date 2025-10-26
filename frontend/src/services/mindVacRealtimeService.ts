/**
 * Mind-VAC 实时识别服务
 * 直接调用后端 Mind-VAC 推理接口
 */

export interface MindVacRealtimeResult {
  text: string
  gloss_sequence: string[]
  raw_gloss_text: string
  confidence: number
  frame_count: number
  duration: number
  baseline_text?: string
  llm_result?: Record<string, unknown>
}

class MindVacRealtimeService {
  private readonly baseUrl: string

  constructor() {
    this.baseUrl = (import.meta.env?.VITE_API_URL as string) || 'http://localhost:8000'
  }

  async recognizeFrames(frames: string[], fps: number, useLlm: boolean = true): Promise<MindVacRealtimeResult> {
    const payload = {
      frames,
      fps,
      use_llm: useLlm,
    }

    const response = await fetch(`${this.baseUrl}/api/mind-vac/recognize-frames`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`
      try {
        const errorData = await response.json()
        if (errorData?.detail) {
          detail = errorData.detail as string
        }
      } catch (error) {
        // ignore json parse error
      }
      throw new Error(detail)
    }

    const result = (await response.json()) as MindVacRealtimeResult
    return result
  }
}

const mindVacRealtimeService = new MindVacRealtimeService()
export default mindVacRealtimeService
