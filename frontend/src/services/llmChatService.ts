/**
 * LLM 聊天服务
 * 提供基于识别结果的对话功能
 */

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
}

export interface ChatContext {
  recognitionResult?: string
  glossSequence?: string[]
  baselineText?: string
}

export interface ChatRequest {
  message: string
  context?: ChatContext
  history?: ChatMessage[]
}

export interface ChatResponse {
  success: boolean
  message: string
  error?: string
}

class LLMChatService {
  private apiBaseUrl: string

  constructor() {
    this.apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
  }

  /**
   * 发送聊天消息
   */
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/llm/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('LLM chat error:', error)
      return {
        success: false,
        message: '',
        error: error instanceof Error ? error.message : '未知错误',
      }
    }
  }

  /**
   * 生成带上下文的提示词
   */
  buildContextPrompt(context: ChatContext): string {
    const parts: string[] = ['以下是手语识别的结果:']

    if (context.recognitionResult) {
      parts.push(`识别文本: ${context.recognitionResult}`)
    }

    if (context.glossSequence && context.glossSequence.length > 0) {
      parts.push(`Gloss序列: ${context.glossSequence.join(' ')}`)
    }

    if (context.baselineText) {
      parts.push(`基础翻译: ${context.baselineText}`)
    }

    parts.push('\n请基于以上识别结果回答用户的问题。')
    return parts.join('\n')
  }
}

export default new LLMChatService()
