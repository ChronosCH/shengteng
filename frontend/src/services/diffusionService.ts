/**
 * Diffusion Sign Language Production Service
 * 基于 Diffusion 模型的文本到手语生成服务
 */

export interface DiffusionConfig {
  emotion: EmotionType;
  speed: SigningSpeed;
  numInferenceSteps: number;
  guidanceScale: number;
  seed?: number;
}

export enum EmotionType {
  NEUTRAL = 'neutral',
  HAPPY = 'happy',
  SAD = 'sad',
  ANGRY = 'angry',
  SURPRISED = 'surprised',
  EXCITED = 'excited'
}

export enum SigningSpeed {
  SLOW = 'slow',
  NORMAL = 'normal',
  FAST = 'fast'
}

export interface SignSequence {
  keypoints: number[][][]; // [frame][keypoint][x,y,z]
  timestamps: number[];
  confidence: number;
  emotion: EmotionType;
  speed: SigningSpeed;
  text: string;
  duration: number;
  numFrames: number;
}

export interface DiffusionGenerationRequest {
  text: string;
  emotion?: string;
  speed?: string;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number;
}

export interface DiffusionGenerationResponse {
  success: boolean;
  message: string;
  data?: SignSequence;
}

export interface DiffusionStats {
  total_requests: number;
  successful_generations: number;
  average_generation_time: number;
  cache_hits: number;
  cache_size: number;
  is_loaded: boolean;
  device_type: string;
}

class DiffusionService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';
  }

  /**
   * 生成手语序列
   */
  async generateSignSequence(
    text: string,
    config: Partial<DiffusionConfig> = {}
  ): Promise<SignSequence> {
    const request: DiffusionGenerationRequest = {
      text,
      emotion: config.emotion || EmotionType.NEUTRAL,
      speed: config.speed || SigningSpeed.NORMAL,
      num_inference_steps: config.numInferenceSteps || 20,
      guidance_scale: config.guidanceScale || 7.5,
      seed: config.seed
    };

    try {
      const response = await fetch(`${this.baseUrl}/api/diffusion/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: DiffusionGenerationResponse = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      if (!result.data) {
        throw new Error('No data returned from server');
      }

      return result.data;
    } catch (error) {
      console.error('Failed to generate sign sequence:', error);
      throw error;
    }
  }

  /**
   * 获取服务统计信息
   */
  async getStats(): Promise<DiffusionStats> {
    try {
      const response = await fetch(`${this.baseUrl}/api/diffusion/stats`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message || 'Failed to get stats');
      }

      return result.data;
    } catch (error) {
      console.error('Failed to get diffusion stats:', error);
      throw error;
    }
  }

  /**
   * 清空缓存
   */
  async clearCache(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/diffusion/clear-cache`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message || 'Failed to clear cache');
      }
    } catch (error) {
      console.error('Failed to clear diffusion cache:', error);
      throw error;
    }
  }

  /**
   * 批量生成多个文本的手语序列
   */
  async generateBatch(
    texts: string[],
    config: Partial<DiffusionConfig> = {}
  ): Promise<SignSequence[]> {
    const promises = texts.map(text => this.generateSignSequence(text, config));
    return Promise.all(promises);
  }

  /**
   * 预设情绪配置
   */
  getEmotionPresets(): Record<EmotionType, Partial<DiffusionConfig>> {
    return {
      [EmotionType.NEUTRAL]: {
        emotion: EmotionType.NEUTRAL,
        speed: SigningSpeed.NORMAL,
        guidanceScale: 7.5
      },
      [EmotionType.HAPPY]: {
        emotion: EmotionType.HAPPY,
        speed: SigningSpeed.NORMAL,
        guidanceScale: 8.0
      },
      [EmotionType.SAD]: {
        emotion: EmotionType.SAD,
        speed: SigningSpeed.SLOW,
        guidanceScale: 7.0
      },
      [EmotionType.ANGRY]: {
        emotion: EmotionType.ANGRY,
        speed: SigningSpeed.FAST,
        guidanceScale: 8.5
      },
      [EmotionType.SURPRISED]: {
        emotion: EmotionType.SURPRISED,
        speed: SigningSpeed.FAST,
        guidanceScale: 8.0
      },
      [EmotionType.EXCITED]: {
        emotion: EmotionType.EXCITED,
        speed: SigningSpeed.FAST,
        guidanceScale: 9.0
      }
    };
  }

  /**
   * 获取情绪显示名称
   */
  getEmotionDisplayName(emotion: EmotionType): string {
    const names = {
      [EmotionType.NEUTRAL]: '中性',
      [EmotionType.HAPPY]: '开心',
      [EmotionType.SAD]: '难过',
      [EmotionType.ANGRY]: '生气',
      [EmotionType.SURPRISED]: '惊讶',
      [EmotionType.EXCITED]: '兴奋'
    };
    return names[emotion];
  }

  /**
   * 获取速度显示名称
   */
  getSpeedDisplayName(speed: SigningSpeed): string {
    const names = {
      [SigningSpeed.SLOW]: '慢速',
      [SigningSpeed.NORMAL]: '正常',
      [SigningSpeed.FAST]: '快速'
    };
    return names[speed];
  }
}

// 导出单例实例
export const diffusionService = new DiffusionService();
export default diffusionService;
