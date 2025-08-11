import * as THREE from 'three'

/**
 * 中国手语动作库 - 专业手语识别与表达
 * 包含常用手语词汇的精确3D关键点数据
 */

export interface SignLanguageGesture {
  id: string
  name: string
  category: 'greeting' | 'daily' | 'emotion' | 'number' | 'alphabet' | 'phrase'
  difficulty: 'easy' | 'medium' | 'hard'
  description: string
  duration: number // 毫秒
  keyframes: SignLanguageKeyframe[]
  handShape: {
    left: HandShape
    right: HandShape
  }
  facialExpression?: FacialExpression
  bodyPose?: BodyPose
}

export interface SignLanguageKeyframe {
  timestamp: number // 相对时间，0-1
  leftHand: HandKeypoint[]
  rightHand: HandKeypoint[]
  leftWrist: THREE.Vector3
  rightWrist: THREE.Vector3
  shoulderPosition?: THREE.Vector3
  headPosition?: THREE.Vector3
  eyeGaze?: THREE.Vector2
}

export interface HandShape {
  thumb: FingerPose
  index: FingerPose
  middle: FingerPose
  ring: FingerPose
  pinky: FingerPose
  orientation: THREE.Euler // 手腕旋转
  position: THREE.Vector3 // 相对于身体的位置
}

export interface FingerPose {
  mcp: number // 掌指关节弯曲度 (0-90度)
  pip: number // 近端指间关节弯曲度 (0-90度)
  dip: number // 远端指间关节弯曲度 (0-90度)
  abduction: number // 手指张开度 (-30到30度)
}

export interface FacialExpression {
  eyebrows: number // -1到1，皱眉到扬眉
  eyes: number // 0到1，睁眼程度
  mouth: number // -1到1，难过到开心
  cheeks: number // 0到1，鼓腮程度
}

export interface BodyPose {
  spine: THREE.Euler
  leftShoulder: THREE.Euler
  rightShoulder: THREE.Euler
  leftElbow: number // 弯曲角度
  rightElbow: number
}

export interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility: number
}

// 专业手语词汇库
export const CHINESE_SIGN_LANGUAGE_LIBRARY: Record<string, SignLanguageGesture> = {
  // === 问候语类 ===
  hello: {
    id: 'hello',
    name: '你好',
    category: 'greeting',
    difficulty: 'easy',
    description: '右手从额头向前挥动，表示问候',
    duration: 2000,
    keyframes: [
      {
        timestamp: 0,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.2, 1.2, 0.1),
      },
      {
        timestamp: 0.3,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.3, 1.4, 0.2),
      },
      {
        timestamp: 0.7,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('wave'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.5, 1.3, 0.4),
      },
      {
        timestamp: 1,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.3, 1.1, 0.2),
      }
    ],
    handShape: {
      left: createRelaxedHand(),
      right: createFlatHand()
    },
    facialExpression: {
      eyebrows: 0.3,
      eyes: 0.8,
      mouth: 0.5,
      cheeks: 0.2
    }
  },

  thank_you: {
    id: 'thank_you',
    name: '谢谢',
    category: 'greeting',
    difficulty: 'easy',
    description: '双手合十向前推，表示感谢',
    duration: 2500,
    keyframes: [
      {
        timestamp: 0,
        leftHand: generateHandKeypoints('flat'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.2, 0.8, 0.1),
        rightWrist: new THREE.Vector3(0.2, 0.8, 0.1),
      },
      {
        timestamp: 0.4,
        leftHand: generateHandKeypoints('prayer'),
        rightHand: generateHandKeypoints('prayer'),
        leftWrist: new THREE.Vector3(-0.05, 0.9, 0.2),
        rightWrist: new THREE.Vector3(0.05, 0.9, 0.2),
      },
      {
        timestamp: 0.8,
        leftHand: generateHandKeypoints('prayer'),
        rightHand: generateHandKeypoints('prayer'),
        leftWrist: new THREE.Vector3(-0.1, 0.8, 0.4),
        rightWrist: new THREE.Vector3(0.1, 0.8, 0.4),
      },
      {
        timestamp: 1,
        leftHand: generateHandKeypoints('flat'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.2, 0.6, 0.2),
        rightWrist: new THREE.Vector3(0.2, 0.6, 0.2),
      }
    ],
    handShape: {
      left: createPrayerHand(),
      right: createPrayerHand()
    },
    facialExpression: {
      eyebrows: 0.2,
      eyes: 0.9,
      mouth: 0.7,
      cheeks: 0.3
    }
  },

  // === 数字类 ===
  one: {
    id: 'one',
    name: '一',
    category: 'number',
    difficulty: 'easy',
    description: '右手食指竖起',
    duration: 1500,
    keyframes: [
      {
        timestamp: 0,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('fist'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.3, 0.8, 0.1),
      },
      {
        timestamp: 0.5,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('one'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.3, 1.0, 0.2),
      },
      {
        timestamp: 1,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('one'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.3, 1.0, 0.2),
      }
    ],
    handShape: {
      left: createRelaxedHand(),
      right: createNumberOneHand()
    }
  },

  // === 日常用语类 ===
  i_love_you: {
    id: 'i_love_you',
    name: '我爱你',
    category: 'emotion',
    difficulty: 'medium',
    description: '右手做"我爱你"手势（拇指、食指、小指伸出）',
    duration: 3000,
    keyframes: [
      {
        timestamp: 0,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('fist'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.3, 0.6, 0.1),
      },
      {
        timestamp: 0.3,
        leftHand: generateHandKeypoints('point_self'),
        rightHand: generateHandKeypoints('fist'),
        leftWrist: new THREE.Vector3(-0.2, 0.4, 0.1),
        rightWrist: new THREE.Vector3(0.3, 0.6, 0.1),
      },
      {
        timestamp: 0.6,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('love'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.2, 0.9, 0.2),
      },
      {
        timestamp: 1,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('ily'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.4, 1.1, 0.3),
      }
    ],
    handShape: {
      left: createRelaxedHand(),
      right: createILYHand()
    },
    facialExpression: {
      eyebrows: 0.4,
      eyes: 0.9,
      mouth: 0.8,
      cheeks: 0.5
    }
  },

  goodbye: {
    id: 'goodbye',
    name: '再见',
    category: 'greeting',
    difficulty: 'easy',
    description: '右手挥动告别',
    duration: 2500,
    keyframes: [
      {
        timestamp: 0,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.4, 1.1, 0.2),
      },
      {
        timestamp: 0.25,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('wave'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.5, 1.2, 0.3),
      },
      {
        timestamp: 0.5,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.4, 1.1, 0.2),
      },
      {
        timestamp: 0.75,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('wave'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.5, 1.2, 0.3),
      },
      {
        timestamp: 1,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.4, 1.1, 0.2),
      }
    ],
    handShape: {
      left: createRelaxedHand(),
      right: createWaveHand()
    }
  },

  // === 复杂手语短语 ===
  good_morning: {
    id: 'good_morning',
    name: '早上好',
    category: 'phrase',
    difficulty: 'medium',
    description: '组合动作：太阳升起 + 问候',
    duration: 4000,
    keyframes: [
      // 太阳升起动作
      {
        timestamp: 0,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('circle'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.2, 0.5, 0.1),
      },
      {
        timestamp: 0.3,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('circle'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.3, 1.0, 0.2),
      },
      // 转换到问候
      {
        timestamp: 0.6,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('flat'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.2, 1.2, 0.1),
      },
      {
        timestamp: 1,
        leftHand: generateHandKeypoints('rest'),
        rightHand: generateHandKeypoints('wave'),
        leftWrist: new THREE.Vector3(-0.3, 0.2, 0),
        rightWrist: new THREE.Vector3(0.5, 1.3, 0.4),
      }
    ],
    handShape: {
      left: createRelaxedHand(),
      right: createCircleHand()
    },
    facialExpression: {
      eyebrows: 0.5,
      eyes: 0.9,
      mouth: 0.7,
      cheeks: 0.4
    }
  }
}

// === 手型生成函数 ===

function generateHandKeypoints(handType: string): HandKeypoint[] {
  const baseKeypoints: HandKeypoint[] = Array(21).fill(null).map(() => ({
    x: 0, y: 0, z: 0, visibility: 1
  }))

  switch (handType) {
    case 'rest':
      return generateRestHand()
    case 'flat':
      return generateFlatHand()
    case 'fist':
      return generateFistHand()
    case 'one':
      return generateNumberOne()
    case 'wave':
      return generateWaveHand()
    case 'prayer':
      return generatePrayerHand()
    case 'point_self':
      return generatePointSelfHand()
    case 'love':
      return generateLoveHand()
    case 'ily':
      return generateILYHand()
    case 'circle':
      return generateCircleHand()
    default:
      return generateRestHand()
  }
}

function generateRestHand(): HandKeypoint[] {
  return [
    // 手腕
    { x: 0.5, y: 0.5, z: 0, visibility: 1 },
    
    // 拇指链 - 自然弯曲
    { x: 0.4, y: 0.45, z: 0.02, visibility: 1 },
    { x: 0.35, y: 0.42, z: 0.04, visibility: 1 },
    { x: 0.32, y: 0.40, z: 0.06, visibility: 1 },
    { x: 0.29, y: 0.38, z: 0.08, visibility: 1 },
    
    // 食指链 - 轻微弯曲
    { x: 0.45, y: 0.35, z: 0.01, visibility: 1 },
    { x: 0.43, y: 0.25, z: 0.02, visibility: 1 },
    { x: 0.42, y: 0.18, z: 0.03, visibility: 1 },
    { x: 0.41, y: 0.12, z: 0.04, visibility: 1 },
    
    // 中指链
    { x: 0.50, y: 0.33, z: 0, visibility: 1 },
    { x: 0.49, y: 0.20, z: 0.01, visibility: 1 },
    { x: 0.48, y: 0.10, z: 0.02, visibility: 1 },
    { x: 0.47, y: 0.03, z: 0.03, visibility: 1 },
    
    // 无名指链
    { x: 0.55, y: 0.35, z: -0.01, visibility: 1 },
    { x: 0.56, y: 0.23, z: 0, visibility: 1 },
    { x: 0.57, y: 0.14, z: 0.01, visibility: 1 },
    { x: 0.58, y: 0.08, z: 0.02, visibility: 1 },
    
    // 小指链
    { x: 0.60, y: 0.38, z: -0.02, visibility: 1 },
    { x: 0.62, y: 0.28, z: -0.01, visibility: 1 },
    { x: 0.63, y: 0.20, z: 0, visibility: 1 },
    { x: 0.64, y: 0.15, z: 0.01, visibility: 1 },
  ]
}

function generateFlatHand(): HandKeypoint[] {
  return [
    // 手腕
    { x: 0.5, y: 0.5, z: 0, visibility: 1 },
    
    // 拇指链 - 张开
    { x: 0.38, y: 0.45, z: 0.02, visibility: 1 },
    { x: 0.32, y: 0.40, z: 0.04, visibility: 1 },
    { x: 0.28, y: 0.35, z: 0.06, visibility: 1 },
    { x: 0.25, y: 0.30, z: 0.08, visibility: 1 },
    
    // 食指链 - 伸直
    { x: 0.45, y: 0.35, z: 0.01, visibility: 1 },
    { x: 0.42, y: 0.20, z: 0.02, visibility: 1 },
    { x: 0.40, y: 0.08, z: 0.03, visibility: 1 },
    { x: 0.38, y: 0.02, z: 0.04, visibility: 1 },
    
    // 中指链 - 伸直
    { x: 0.50, y: 0.33, z: 0, visibility: 1 },
    { x: 0.49, y: 0.15, z: 0.01, visibility: 1 },
    { x: 0.48, y: 0.02, z: 0.02, visibility: 1 },
    { x: 0.47, y: -0.08, z: 0.03, visibility: 1 },
    
    // 无名指链 - 伸直
    { x: 0.55, y: 0.35, z: -0.01, visibility: 1 },
    { x: 0.57, y: 0.18, z: 0, visibility: 1 },
    { x: 0.58, y: 0.05, z: 0.01, visibility: 1 },
    { x: 0.59, y: -0.05, z: 0.02, visibility: 1 },
    
    // 小指链 - 伸直
    { x: 0.60, y: 0.38, z: -0.02, visibility: 1 },
    { x: 0.63, y: 0.22, z: -0.01, visibility: 1 },
    { x: 0.65, y: 0.10, z: 0, visibility: 1 },
    { x: 0.67, y: 0.02, z: 0.01, visibility: 1 },
  ]
}

function generateNumberOne(): HandKeypoint[] {
  const keypoints = generateFistHand()
  
  // 只有食指伸直
  keypoints[5] = { x: 0.45, y: 0.35, z: 0.01, visibility: 1 }
  keypoints[6] = { x: 0.42, y: 0.20, z: 0.02, visibility: 1 }
  keypoints[7] = { x: 0.40, y: 0.08, z: 0.03, visibility: 1 }
  keypoints[8] = { x: 0.38, y: 0.02, z: 0.04, visibility: 1 }
  
  return keypoints
}

function generateFistHand(): HandKeypoint[] {
  return [
    // 手腕
    { x: 0.5, y: 0.5, z: 0, visibility: 1 },
    
    // 拇指链 - 包握
    { x: 0.42, y: 0.48, z: 0.02, visibility: 1 },
    { x: 0.40, y: 0.50, z: 0.04, visibility: 1 },
    { x: 0.38, y: 0.52, z: 0.06, visibility: 1 },
    { x: 0.36, y: 0.54, z: 0.08, visibility: 1 },
    
    // 食指链 - 弯曲握拳
    { x: 0.45, y: 0.35, z: 0.01, visibility: 1 },
    { x: 0.47, y: 0.42, z: 0.02, visibility: 1 },
    { x: 0.48, y: 0.48, z: 0.03, visibility: 1 },
    { x: 0.49, y: 0.52, z: 0.04, visibility: 1 },
    
    // 中指链 - 弯曲握拳
    { x: 0.50, y: 0.33, z: 0, visibility: 1 },
    { x: 0.52, y: 0.42, z: 0.01, visibility: 1 },
    { x: 0.53, y: 0.49, z: 0.02, visibility: 1 },
    { x: 0.54, y: 0.54, z: 0.03, visibility: 1 },
    
    // 无名指链 - 弯曲握拳
    { x: 0.55, y: 0.35, z: -0.01, visibility: 1 },
    { x: 0.57, y: 0.43, z: 0, visibility: 1 },
    { x: 0.58, y: 0.50, z: 0.01, visibility: 1 },
    { x: 0.59, y: 0.55, z: 0.02, visibility: 1 },
    
    // 小指链 - 弯曲握拳
    { x: 0.60, y: 0.38, z: -0.02, visibility: 1 },
    { x: 0.62, y: 0.45, z: -0.01, visibility: 1 },
    { x: 0.63, y: 0.51, z: 0, visibility: 1 },
    { x: 0.64, y: 0.56, z: 0.01, visibility: 1 },
  ]
}

function generateILYHand(): HandKeypoint[] {
  const keypoints = generateFistHand()
  
  // "我爱你"手势：拇指、食指、小指伸出
  // 拇指伸出
  keypoints[1] = { x: 0.38, y: 0.45, z: 0.02, visibility: 1 }
  keypoints[2] = { x: 0.32, y: 0.40, z: 0.04, visibility: 1 }
  keypoints[3] = { x: 0.28, y: 0.35, z: 0.06, visibility: 1 }
  keypoints[4] = { x: 0.25, y: 0.30, z: 0.08, visibility: 1 }
  
  // 食指伸出
  keypoints[5] = { x: 0.45, y: 0.35, z: 0.01, visibility: 1 }
  keypoints[6] = { x: 0.42, y: 0.20, z: 0.02, visibility: 1 }
  keypoints[7] = { x: 0.40, y: 0.08, z: 0.03, visibility: 1 }
  keypoints[8] = { x: 0.38, y: 0.02, z: 0.04, visibility: 1 }
  
  // 小指伸出
  keypoints[17] = { x: 0.60, y: 0.38, z: -0.02, visibility: 1 }
  keypoints[18] = { x: 0.63, y: 0.22, z: -0.01, visibility: 1 }
  keypoints[19] = { x: 0.65, y: 0.10, z: 0, visibility: 1 }
  keypoints[20] = { x: 0.67, y: 0.02, z: 0.01, visibility: 1 }
  
  return keypoints
}

// === 手型创建辅助函数 ===

function createRelaxedHand(): HandShape {
  return {
    thumb: { mcp: 20, pip: 15, dip: 10, abduction: 5 },
    index: { mcp: 30, pip: 25, dip: 20, abduction: 0 },
    middle: { mcp: 35, pip: 30, dip: 25, abduction: 0 },
    ring: { mcp: 40, pip: 35, dip: 30, abduction: 0 },
    pinky: { mcp: 45, pip: 40, dip: 35, abduction: -5 },
    orientation: new THREE.Euler(0, 0, 0),
    position: new THREE.Vector3(0, 0, 0)
  }
}

function createFlatHand(): HandShape {
  return {
    thumb: { mcp: 5, pip: 0, dip: 0, abduction: 20 },
    index: { mcp: 0, pip: 0, dip: 0, abduction: 5 },
    middle: { mcp: 0, pip: 0, dip: 0, abduction: 0 },
    ring: { mcp: 0, pip: 0, dip: 0, abduction: 0 },
    pinky: { mcp: 0, pip: 0, dip: 0, abduction: -5 },
    orientation: new THREE.Euler(0, 0, 0),
    position: new THREE.Vector3(0, 0, 0)
  }
}

function createNumberOneHand(): HandShape {
  return {
    thumb: { mcp: 60, pip: 60, dip: 45, abduction: -10 },
    index: { mcp: 0, pip: 0, dip: 0, abduction: 5 },
    middle: { mcp: 80, pip: 75, dip: 70, abduction: 0 },
    ring: { mcp: 85, pip: 80, dip: 75, abduction: 0 },
    pinky: { mcp: 90, pip: 85, dip: 80, abduction: -5 },
    orientation: new THREE.Euler(0, 0, 0),
    position: new THREE.Vector3(0, 0, 0)
  }
}

function createILYHand(): HandShape {
  return {
    thumb: { mcp: 5, pip: 0, dip: 0, abduction: 25 },
    index: { mcp: 0, pip: 0, dip: 0, abduction: 8 },
    middle: { mcp: 80, pip: 75, dip: 70, abduction: 0 },
    ring: { mcp: 85, pip: 80, dip: 75, abduction: 0 },
    pinky: { mcp: 0, pip: 0, dip: 0, abduction: -15 },
    orientation: new THREE.Euler(0, 0, 0),
    position: new THREE.Vector3(0, 0, 0)
  }
}

function createPrayerHand(): HandShape {
  return {
    thumb: { mcp: 10, pip: 5, dip: 0, abduction: 0 },
    index: { mcp: 5, pip: 0, dip: 0, abduction: 0 },
    middle: { mcp: 5, pip: 0, dip: 0, abduction: 0 },
    ring: { mcp: 5, pip: 0, dip: 0, abduction: 0 },
    pinky: { mcp: 10, pip: 5, dip: 0, abduction: 0 },
    orientation: new THREE.Euler(0, 0, 0),
    position: new THREE.Vector3(0, 0, 0)
  }
}

function createWaveHand(): HandShape {
  return {
    thumb: { mcp: 10, pip: 5, dip: 0, abduction: 15 },
    index: { mcp: 0, pip: 0, dip: 0, abduction: 8 },
    middle: { mcp: 0, pip: 0, dip: 0, abduction: 0 },
    ring: { mcp: 0, pip: 0, dip: 0, abduction: 0 },
    pinky: { mcp: 0, pip: 0, dip: 0, abduction: -8 },
    orientation: new THREE.Euler(0, 0, Math.PI / 6),
    position: new THREE.Vector3(0, 0, 0)
  }
}

function createCircleHand(): HandShape {
  return {
    thumb: { mcp: 30, pip: 25, dip: 20, abduction: 0 },
    index: { mcp: 40, pip: 35, dip: 30, abduction: 5 },
    middle: { mcp: 20, pip: 15, dip: 10, abduction: 0 },
    ring: { mcp: 20, pip: 15, dip: 10, abduction: 0 },
    pinky: { mcp: 25, pip: 20, dip: 15, abduction: -5 },
    orientation: new THREE.Euler(0, 0, 0),
    position: new THREE.Vector3(0, 0, 0)
  }
}

// === 辅助函数 ===
function generateWaveHand(): HandKeypoint[] { return generateFlatHand() }
function generatePrayerHand(): HandKeypoint[] { return generateFlatHand() }
function generatePointSelfHand(): HandKeypoint[] { return generateNumberOne() }
function generateLoveHand(): HandKeypoint[] { return generateFlatHand() }
function generateCircleHand(): HandKeypoint[] { return generateFlatHand() }

// === 手语序列播放器 ===
export class SignLanguagePlayer {
  private currentGesture: SignLanguageGesture | null = null
  private startTime: number = 0
  private isPlaying: boolean = false
  private onFrameUpdate?: (frame: SignLanguageKeyframe) => void

  constructor(onFrameUpdate?: (frame: SignLanguageKeyframe) => void) {
    this.onFrameUpdate = onFrameUpdate
  }

  play(gestureId: string) {
    const gesture = CHINESE_SIGN_LANGUAGE_LIBRARY[gestureId]
    if (!gesture) {
      console.warn(`手语动作 "${gestureId}" 未找到`)
      return
    }

    this.currentGesture = gesture
    this.startTime = Date.now()
    this.isPlaying = true
    this.animate()
  }

  stop() {
    this.isPlaying = false
    this.currentGesture = null
  }

  private animate() {
    if (!this.isPlaying || !this.currentGesture) return

    const elapsed = Date.now() - this.startTime
    const progress = Math.min(elapsed / this.currentGesture.duration, 1)

    // 插值计算当前帧
    const currentFrame = this.interpolateFrame(progress)
    
    if (this.onFrameUpdate) {
      this.onFrameUpdate(currentFrame)
    }

    if (progress < 1) {
      requestAnimationFrame(() => this.animate())
    } else {
      this.isPlaying = false
    }
  }

  private interpolateFrame(progress: number): SignLanguageKeyframe {
    if (!this.currentGesture) throw new Error('No gesture loaded')

    const keyframes = this.currentGesture.keyframes
    
    // 找到当前时间对应的关键帧区间
    let prevFrame = keyframes[0]
    let nextFrame = keyframes[keyframes.length - 1]
    
    for (let i = 0; i < keyframes.length - 1; i++) {
      if (progress >= keyframes[i].timestamp && progress <= keyframes[i + 1].timestamp) {
        prevFrame = keyframes[i]
        nextFrame = keyframes[i + 1]
        break
      }
    }

    // 计算插值因子
    const segmentProgress = (progress - prevFrame.timestamp) / (nextFrame.timestamp - prevFrame.timestamp)
    
    // 插值计算
    return {
      timestamp: progress,
      leftHand: this.interpolateHandKeypoints(prevFrame.leftHand, nextFrame.leftHand, segmentProgress),
      rightHand: this.interpolateHandKeypoints(prevFrame.rightHand, nextFrame.rightHand, segmentProgress),
      leftWrist: prevFrame.leftWrist.clone().lerp(nextFrame.leftWrist, segmentProgress),
      rightWrist: prevFrame.rightWrist.clone().lerp(nextFrame.rightWrist, segmentProgress),
    }
  }

  private interpolateHandKeypoints(prev: HandKeypoint[], next: HandKeypoint[], factor: number): HandKeypoint[] {
    return prev.map((prevKp, index) => {
      const nextKp = next[index]
      return {
        x: prevKp.x + (nextKp.x - prevKp.x) * factor,
        y: prevKp.y + (nextKp.y - prevKp.y) * factor,
        z: prevKp.z + (nextKp.z - prevKp.z) * factor,
        visibility: Math.min(prevKp.visibility, nextKp.visibility)
      }
    })
  }
}

export default CHINESE_SIGN_LANGUAGE_LIBRARY