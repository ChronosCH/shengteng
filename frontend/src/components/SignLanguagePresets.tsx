/**
 * 手语手型预设 - 常用手语动作的关键点配置
 */

// 手部关键点索引映射 (从DetailedHandModel复制)
const HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20
}

export interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility?: number
}

// 手语数字预设
export const SIGN_NUMBERS: { [key: string]: HandKeypoint[] } = {
  '0': [
    // 手腕
    { x: 0.5, y: 0.5, z: 0 },
    // 拇指 - 弯曲
    { x: 0.45, y: 0.45, z: 0.05 },
    { x: 0.42, y: 0.42, z: 0.08 },
    { x: 0.40, y: 0.40, z: 0.10 },
    { x: 0.38, y: 0.38, z: 0.12 },
    // 食指 - 弯曲形成圆圈
    { x: 0.48, y: 0.35, z: 0.02 },
    { x: 0.46, y: 0.25, z: 0.04 },
    { x: 0.44, y: 0.20, z: 0.06 },
    { x: 0.42, y: 0.18, z: 0.08 },
    // 中指 - 弯曲
    { x: 0.50, y: 0.30, z: 0 },
    { x: 0.50, y: 0.20, z: 0.02 },
    { x: 0.50, y: 0.15, z: 0.04 },
    { x: 0.50, y: 0.12, z: 0.06 },
    // 无名指 - 弯曲
    { x: 0.52, y: 0.35, z: -0.02 },
    { x: 0.54, y: 0.25, z: -0.04 },
    { x: 0.56, y: 0.20, z: -0.06 },
    { x: 0.58, y: 0.18, z: -0.08 },
    // 小指 - 弯曲
    { x: 0.55, y: 0.40, z: -0.05 },
    { x: 0.58, y: 0.32, z: -0.08 },
    { x: 0.60, y: 0.28, z: -0.10 },
    { x: 0.62, y: 0.26, z: -0.12 },
  ],

  '1': [
    // 手腕
    { x: 0.5, y: 0.5, z: 0 },
    // 拇指 - 弯曲
    { x: 0.45, y: 0.45, z: 0.05 },
    { x: 0.42, y: 0.42, z: 0.08 },
    { x: 0.40, y: 0.40, z: 0.10 },
    { x: 0.38, y: 0.38, z: 0.12 },
    // 食指 - 伸直
    { x: 0.48, y: 0.35, z: 0.02 },
    { x: 0.46, y: 0.20, z: 0.04 },
    { x: 0.45, y: 0.10, z: 0.06 },
    { x: 0.44, y: 0.05, z: 0.08 },
    // 中指 - 弯曲
    { x: 0.50, y: 0.30, z: 0 },
    { x: 0.50, y: 0.20, z: 0.02 },
    { x: 0.50, y: 0.15, z: 0.04 },
    { x: 0.50, y: 0.12, z: 0.06 },
    // 无名指 - 弯曲
    { x: 0.52, y: 0.35, z: -0.02 },
    { x: 0.54, y: 0.25, z: -0.04 },
    { x: 0.56, y: 0.20, z: -0.06 },
    { x: 0.58, y: 0.18, z: -0.08 },
    // 小指 - 弯曲
    { x: 0.55, y: 0.40, z: -0.05 },
    { x: 0.58, y: 0.32, z: -0.08 },
    { x: 0.60, y: 0.28, z: -0.10 },
    { x: 0.62, y: 0.26, z: -0.12 },
  ],

  '2': [
    // 手腕
    { x: 0.5, y: 0.5, z: 0 },
    // 拇指 - 弯曲
    { x: 0.45, y: 0.45, z: 0.05 },
    { x: 0.42, y: 0.42, z: 0.08 },
    { x: 0.40, y: 0.40, z: 0.10 },
    { x: 0.38, y: 0.38, z: 0.12 },
    // 食指 - 伸直
    { x: 0.48, y: 0.35, z: 0.02 },
    { x: 0.46, y: 0.20, z: 0.04 },
    { x: 0.45, y: 0.10, z: 0.06 },
    { x: 0.44, y: 0.05, z: 0.08 },
    // 中指 - 伸直
    { x: 0.50, y: 0.30, z: 0 },
    { x: 0.50, y: 0.15, z: 0.02 },
    { x: 0.50, y: 0.05, z: 0.04 },
    { x: 0.50, y: 0.00, z: 0.06 },
    // 无名指 - 弯曲
    { x: 0.52, y: 0.35, z: -0.02 },
    { x: 0.54, y: 0.25, z: -0.04 },
    { x: 0.56, y: 0.20, z: -0.06 },
    { x: 0.58, y: 0.18, z: -0.08 },
    // 小指 - 弯曲
    { x: 0.55, y: 0.40, z: -0.05 },
    { x: 0.58, y: 0.32, z: -0.08 },
    { x: 0.60, y: 0.28, z: -0.10 },
    { x: 0.62, y: 0.26, z: -0.12 },
  ],

  '5': [
    // 手腕
    { x: 0.5, y: 0.5, z: 0 },
    // 拇指 - 伸直
    { x: 0.35, y: 0.45, z: 0.05 },
    { x: 0.25, y: 0.42, z: 0.08 },
    { x: 0.18, y: 0.40, z: 0.10 },
    { x: 0.12, y: 0.38, z: 0.12 },
    // 食指 - 伸直
    { x: 0.48, y: 0.35, z: 0.02 },
    { x: 0.46, y: 0.20, z: 0.04 },
    { x: 0.45, y: 0.10, z: 0.06 },
    { x: 0.44, y: 0.05, z: 0.08 },
    // 中指 - 伸直
    { x: 0.50, y: 0.30, z: 0 },
    { x: 0.50, y: 0.15, z: 0.02 },
    { x: 0.50, y: 0.05, z: 0.04 },
    { x: 0.50, y: 0.00, z: 0.06 },
    // 无名指 - 伸直
    { x: 0.52, y: 0.35, z: -0.02 },
    { x: 0.54, y: 0.20, z: -0.04 },
    { x: 0.55, y: 0.10, z: -0.06 },
    { x: 0.56, y: 0.05, z: -0.08 },
    // 小指 - 伸直
    { x: 0.55, y: 0.40, z: -0.05 },
    { x: 0.58, y: 0.25, z: -0.08 },
    { x: 0.60, y: 0.15, z: -0.10 },
    { x: 0.62, y: 0.10, z: -0.12 },
  ]
}

// 常用手语词汇预设
export const SIGN_WORDS: { [key: string]: HandKeypoint[] } = {
  '你好': [
    // 挥手动作的关键点
    { x: 0.5, y: 0.5, z: 0 },
    // 拇指 - 伸直
    { x: 0.35, y: 0.45, z: 0.05 },
    { x: 0.25, y: 0.42, z: 0.08 },
    { x: 0.18, y: 0.40, z: 0.10 },
    { x: 0.12, y: 0.38, z: 0.12 },
    // 其他手指伸直
    { x: 0.48, y: 0.35, z: 0.02 },
    { x: 0.46, y: 0.20, z: 0.04 },
    { x: 0.45, y: 0.10, z: 0.06 },
    { x: 0.44, y: 0.05, z: 0.08 },
    { x: 0.50, y: 0.30, z: 0 },
    { x: 0.50, y: 0.15, z: 0.02 },
    { x: 0.50, y: 0.05, z: 0.04 },
    { x: 0.50, y: 0.00, z: 0.06 },
    { x: 0.52, y: 0.35, z: -0.02 },
    { x: 0.54, y: 0.20, z: -0.04 },
    { x: 0.55, y: 0.10, z: -0.06 },
    { x: 0.56, y: 0.05, z: -0.08 },
    { x: 0.55, y: 0.40, z: -0.05 },
    { x: 0.58, y: 0.25, z: -0.08 },
    { x: 0.60, y: 0.15, z: -0.10 },
    { x: 0.62, y: 0.10, z: -0.12 },
  ],

  '谢谢': [
    // 感谢手势 - 手掌向前
    { x: 0.5, y: 0.5, z: 0 },
    { x: 0.35, y: 0.45, z: 0.05 },
    { x: 0.25, y: 0.42, z: 0.08 },
    { x: 0.18, y: 0.40, z: 0.10 },
    { x: 0.12, y: 0.38, z: 0.12 },
    { x: 0.48, y: 0.35, z: 0.02 },
    { x: 0.46, y: 0.20, z: 0.04 },
    { x: 0.45, y: 0.10, z: 0.06 },
    { x: 0.44, y: 0.05, z: 0.08 },
    { x: 0.50, y: 0.30, z: 0 },
    { x: 0.50, y: 0.15, z: 0.02 },
    { x: 0.50, y: 0.05, z: 0.04 },
    { x: 0.50, y: 0.00, z: 0.06 },
    { x: 0.52, y: 0.35, z: -0.02 },
    { x: 0.54, y: 0.20, z: -0.04 },
    { x: 0.55, y: 0.10, z: -0.06 },
    { x: 0.56, y: 0.05, z: -0.08 },
    { x: 0.55, y: 0.40, z: -0.05 },
    { x: 0.58, y: 0.25, z: -0.08 },
    { x: 0.60, y: 0.15, z: -0.10 },
    { x: 0.62, y: 0.10, z: -0.12 },
  ]
}

// 获取手语预设
export const getSignPreset = (sign: string): HandKeypoint[] | null => {
  // 先检查数字
  if (SIGN_NUMBERS[sign]) {
    return SIGN_NUMBERS[sign]
  }
  
  // 再检查词汇
  if (SIGN_WORDS[sign]) {
    return SIGN_WORDS[sign]
  }
  
  return null
}

// 获取所有可用的手语预设
export const getAvailableSignPresets = (): string[] => {
  return [...Object.keys(SIGN_NUMBERS), ...Object.keys(SIGN_WORDS)]
}

// 创建手语动画序列
export const createSignSequence = (signs: string[], duration: number = 2000): {
  keypoints: HandKeypoint[][]
  timestamps: number[]
  duration: number
} => {
  const keypoints: HandKeypoint[][] = []
  const timestamps: number[] = []
  const signDuration = duration / signs.length
  
  signs.forEach((sign, index) => {
    const preset = getSignPreset(sign)
    if (preset) {
      keypoints.push(preset)
      timestamps.push(index * signDuration)
    }
  })
  
  return {
    keypoints,
    timestamps,
    duration: duration / 1000 // 转换为秒
  }
}

export default {
  SIGN_NUMBERS,
  SIGN_WORDS,
  getSignPreset,
  getAvailableSignPresets,
  createSignSequence
}
