/**
 * 专业手语动作库 - 基于中国手语国家标准
 * 包含标准化的手语动作数据和动画序列
 */

import * as THREE from 'three'

// 专业手语关键点定义
export interface ProfessionalSignLanguageKeypoint {
  x: number  // 归一化坐标 [0-1]
  y: number  // 归一化坐标 [0-1] 
  z: number  // 深度坐标 [-1, 1]
  visibility: number  // 可见度 [0-1]
  confidence: number  // 置信度 [0-1]
}

// 手语动作帧定义
export interface ProfessionalSignLanguageFrame {
  timestamp: number  // 相对时间 [0-1]
  leftHand: ProfessionalSignLanguageKeypoint[]
  rightHand: ProfessionalSignLanguageKeypoint[]
  leftWrist: THREE.Vector3
  rightWrist: THREE.Vector3
  shoulderWidth: number
  bodyPose?: THREE.Euler
  headPose?: THREE.Euler
  eyeGaze?: THREE.Vector2
  facialExpression?: {
    eyebrows: number
    eyes: number
    mouth: number
    emotion: 'neutral' | 'happy' | 'serious' | 'questioning'
  }
}

// 专业手语词汇定义
export interface ProfessionalSignLanguageGesture {
  id: string
  name: string
  category: 'greeting' | 'daily' | 'emotion' | 'number' | 'alphabet' | 'phrase' | 'question' | 'family'
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert'
  description: string
  instruction: string  // 动作要领
  commonMistakes: string[]  // 常见错误
  duration: number  // 毫秒
  frames: ProfessionalSignLanguageFrame[]
  metadata: {
    region: 'national' | 'beijing' | 'shanghai' | 'guangzhou'  // 方言区域
    frequency: 'high' | 'medium' | 'low'  // 使用频率
    context: string[]  // 使用场景
    relatedGestures: string[]  // 相关手语
  }
}

// 生成专业手势关键点
const generateProfessionalHandPose = (
  poseType: 'rest' | 'open' | 'fist' | 'point' | 'ok' | 'love' | 'peace' | 'wave' | 'pinch' | 'grab' | 'pray',
  handedness: 'left' | 'right' = 'right',
  intensity: number = 1.0
): ProfessionalSignLanguageKeypoint[] => {
  
  const baseConfidence = 0.95
  const baseVisibility = 1.0
  
  switch (poseType) {
    case 'rest':
      // 自然放松状态 - 符合解剖学
      return [
        // 手腕
        { x: 0.5, y: 0.5, z: 0, visibility: baseVisibility, confidence: baseConfidence },
        
        // 拇指链 - 自然弯曲
        { x: 0.45, y: 0.48, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.42, y: 0.46, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.39, y: 0.44, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.36, y: 0.42, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 食指链 - 自然伸展
        { x: 0.47, y: 0.35, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.46, y: 0.25, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.45, y: 0.18, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.44, y: 0.12, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        
        // 中指链 - 自然伸展
        { x: 0.5, y: 0.32, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.2, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.1, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.02, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        
        // 无名指链 - 轻微弯曲
        { x: 0.53, y: 0.35, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.54, y: 0.25, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.55, y: 0.18, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.56, y: 0.12, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        
        // 小指链 - 轻微弯曲
        { x: 0.56, y: 0.38, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.58, y: 0.3, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.6, y: 0.25, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.62, y: 0.2, z: 0.04, visibility: baseVisibility, confidence: baseConfidence }
      ].map(kp => handedness === 'left' ? { ...kp, x: 1 - kp.x } : kp)
      
    case 'open':
      // 张开手掌 - 标准手语起始姿势
      return [
        // 手腕
        { x: 0.5, y: 0.6, z: 0, visibility: baseVisibility, confidence: baseConfidence },
        
        // 拇指链 - 完全张开
        { x: 0.4, y: 0.55, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.35, y: 0.5, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.3, y: 0.45, z: 0.09, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.25, y: 0.4, z: 0.12, visibility: baseVisibility, confidence: baseConfidence },
        
        // 食指链 - 完全伸直
        { x: 0.45, y: 0.4, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.42, y: 0.25, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.4, y: 0.1, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.38, y: -0.05, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 中指链 - 完全伸直
        { x: 0.5, y: 0.38, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.2, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.02, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: -0.15, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 无名指链 - 完全伸直
        { x: 0.55, y: 0.4, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.58, y: 0.25, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.6, y: 0.1, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.62, y: -0.05, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 小指链 - 完全伸直
        { x: 0.6, y: 0.42, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.65, y: 0.3, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.7, y: 0.18, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.75, y: 0.05, z: 0.08, visibility: baseVisibility, confidence: baseConfidence }
      ].map(kp => handedness === 'left' ? { ...kp, x: 1 - kp.x } : kp)
      
    case 'point':
      // 指向手势 - 食指伸直，其他弯曲
      return [
        // 手腕
        { x: 0.5, y: 0.6, z: 0, visibility: baseVisibility, confidence: baseConfidence },
        
        // 拇指链 - 弯曲
        { x: 0.45, y: 0.55, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.43, y: 0.53, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.41, y: 0.51, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.39, y: 0.49, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 食指链 - 完全伸直
        { x: 0.48, y: 0.4, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.47, y: 0.2, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.46, y: 0.0, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.45, y: -0.2, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        
        // 中指链 - 弯曲
        { x: 0.51, y: 0.42, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.53, y: 0.35, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.55, y: 0.3, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.57, y: 0.25, z: 0.07, visibility: baseVisibility, confidence: baseConfidence },
        
        // 无名指链 - 弯曲
        { x: 0.54, y: 0.44, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.57, y: 0.38, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.6, y: 0.33, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.63, y: 0.28, z: 0.07, visibility: baseVisibility, confidence: baseConfidence },
        
        // 小指链 - 弯曲
        { x: 0.57, y: 0.46, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.61, y: 0.42, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.65, y: 0.38, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.69, y: 0.34, z: 0.07, visibility: baseVisibility, confidence: baseConfidence }
      ].map(kp => handedness === 'left' ? { ...kp, x: 1 - kp.x } : kp)
      
    case 'ok':
      // OK手势 - 拇指食指圆圈，其他伸直
      return [
        // 手腕
        { x: 0.5, y: 0.6, z: 0, visibility: baseVisibility, confidence: baseConfidence },
        
        // 拇指链 - 与食指形成圆圈
        { x: 0.46, y: 0.5, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.44, y: 0.45, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.43, y: 0.4, z: 0.07, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.45, y: 0.35, z: 0.09, visibility: baseVisibility, confidence: baseConfidence },
        
        // 食指链 - 与拇指形成圆圈
        { x: 0.48, y: 0.45, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.47, y: 0.38, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.46, y: 0.33, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.45, y: 0.3, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 中指链 - 伸直
        { x: 0.5, y: 0.4, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.2, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.0, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: -0.2, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        
        // 无名指链 - 伸直
        { x: 0.52, y: 0.42, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.54, y: 0.22, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.56, y: 0.02, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.58, y: -0.18, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        
        // 小指链 - 伸直
        { x: 0.55, y: 0.44, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.6, y: 0.26, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.65, y: 0.08, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.7, y: -0.1, z: 0.04, visibility: baseVisibility, confidence: baseConfidence }
      ].map(kp => handedness === 'left' ? { ...kp, x: 1 - kp.x } : kp)
      
    case 'love':
      // 爱心手势 - 食指中指伸直交叉
      return [
        // 手腕
        { x: 0.5, y: 0.6, z: 0, visibility: baseVisibility, confidence: baseConfidence },
        
        // 拇指链 - 弯曲
        { x: 0.45, y: 0.55, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.43, y: 0.52, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.41, y: 0.49, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.39, y: 0.46, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 食指链 - 向内弯曲形成心形
        { x: 0.47, y: 0.42, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.45, y: 0.3, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.44, y: 0.18, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.48, y: 0.1, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 中指链 - 向内弯曲形成心形
        { x: 0.53, y: 0.42, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.55, y: 0.3, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.56, y: 0.18, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.52, y: 0.1, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 无名指链 - 弯曲
        { x: 0.54, y: 0.44, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.57, y: 0.38, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.6, y: 0.33, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.63, y: 0.28, z: 0.07, visibility: baseVisibility, confidence: baseConfidence },
        
        // 小指链 - 弯曲
        { x: 0.57, y: 0.46, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.61, y: 0.42, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.65, y: 0.38, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.69, y: 0.34, z: 0.07, visibility: baseVisibility, confidence: baseConfidence }
      ].map(kp => handedness === 'left' ? { ...kp, x: 1 - kp.x } : kp)
      
    case 'fist':
      // 握拳手势 - 所有手指弯曲
      return [
        // 手腕
        { x: 0.5, y: 0.6, z: 0, visibility: baseVisibility, confidence: baseConfidence },
        
        // 拇指链 - 弯曲握拳
        { x: 0.47, y: 0.55, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.45, y: 0.52, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.43, y: 0.49, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.41, y: 0.46, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 食指链 - 弯曲
        { x: 0.48, y: 0.45, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.4, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.52, y: 0.37, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.54, y: 0.35, z: 0.07, visibility: baseVisibility, confidence: baseConfidence },
        
        // 中指链 - 弯曲
        { x: 0.5, y: 0.47, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.42, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.39, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.37, z: 0.07, visibility: baseVisibility, confidence: baseConfidence },
        
        // 无名指链 - 弯曲
        { x: 0.52, y: 0.45, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.4, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.48, y: 0.37, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.46, y: 0.35, z: 0.07, visibility: baseVisibility, confidence: baseConfidence },
        
        // 小指链 - 弯曲
        { x: 0.53, y: 0.43, z: 0.01, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.49, y: 0.38, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.45, y: 0.35, z: 0.05, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.41, y: 0.33, z: 0.07, visibility: baseVisibility, confidence: baseConfidence }
      ].map(kp => handedness === 'left' ? { ...kp, x: 1 - kp.x } : kp)
      
    case 'wave':
      // 挥手手势 - 手掌张开，适合挥动
      return [
        // 手腕
        { x: 0.5, y: 0.6, z: 0, visibility: baseVisibility, confidence: baseConfidence },
        
        // 拇指链 - 张开
        { x: 0.4, y: 0.55, z: 0.03, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.35, y: 0.5, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.3, y: 0.45, z: 0.09, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.25, y: 0.4, z: 0.12, visibility: baseVisibility, confidence: baseConfidence },
        
        // 食指链 - 伸直
        { x: 0.45, y: 0.4, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.42, y: 0.25, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.4, y: 0.1, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.38, y: -0.05, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 中指链 - 伸直
        { x: 0.5, y: 0.38, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.2, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: 0.02, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.5, y: -0.15, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 无名指链 - 伸直
        { x: 0.55, y: 0.4, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.58, y: 0.25, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.6, y: 0.1, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.62, y: -0.05, z: 0.08, visibility: baseVisibility, confidence: baseConfidence },
        
        // 小指链 - 伸直
        { x: 0.6, y: 0.42, z: 0.02, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.65, y: 0.3, z: 0.04, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.7, y: 0.18, z: 0.06, visibility: baseVisibility, confidence: baseConfidence },
        { x: 0.75, y: 0.05, z: 0.08, visibility: baseVisibility, confidence: baseConfidence }
      ].map(kp => handedness === 'left' ? { ...kp, x: 1 - kp.x } : kp)
      
    case 'pinch':
    case 'grab':
    case 'pray':
      // 这些手势暂时使用rest姿态
      return generateProfessionalHandPose('rest', handedness, intensity)
      
    default:
      return generateProfessionalHandPose('rest', handedness, intensity)
      return generateProfessionalHandPose('rest', handedness, intensity)
  }
}

// 专业手语词汇库
export const PROFESSIONAL_SIGN_LANGUAGE_LIBRARY: Record<string, ProfessionalSignLanguageGesture> = {
  // === 问候语类 ===
  hello: {
    id: 'hello',
    name: '你好',
    category: 'greeting',
    difficulty: 'beginner',
    description: '右手平举至额头前方，五指并拢，手掌向外，轻微向前推动表示问候',
    instruction: '1. 右手举至额头高度\n2. 五指自然并拢\n3. 手掌面向对方\n4. 轻柔向前推动',
    commonMistakes: ['手掌角度错误', '动作过于僵硬', '位置过高或过低'],
    duration: 2500,
    frames: [
      {
        timestamp: 0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('rest', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 0.3, 0),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.2,
          eyes: 0.8,
          mouth: 0.6,
          emotion: 'happy'
        }
      },
      {
        timestamp: 0.3,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('open', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.15, 1.5, 0.1),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0.1, 0, 0),
        facialExpression: {
          eyebrows: 0.4,
          eyes: 0.9,
          mouth: 0.8,
          emotion: 'happy'
        }
      },
      {
        timestamp: 0.7,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('open', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.3, 1.6, 0.3),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0.05, 0),
        headPose: new THREE.Euler(0.1, 0.1, 0),
        facialExpression: {
          eyebrows: 0.6,
          eyes: 1.0,
          mouth: 1.0,
          emotion: 'happy'
        }
      },
      {
        timestamp: 1.0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('open', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 1.5, 0.1),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.2,
          eyes: 0.8,
          mouth: 0.6,
          emotion: 'happy'
        }
      }
    ],
    metadata: {
      region: 'national',
      frequency: 'high',
      context: ['日常问候', '初次见面', '正式场合'],
      relatedGestures: ['goodbye', 'nice_to_meet_you']
    }
  },

  thank_you: {
    id: 'thank_you',
    name: '谢谢',
    category: 'daily',
    difficulty: 'beginner',
    description: '右手握拳，拇指伸出，向胸前做轻微点头动作表示感谢',
    instruction: '1. 右手握拳举至胸前\n2. 拇指向上伸直\n3. 配合轻微点头\n4. 表情真诚',
    commonMistakes: ['拇指角度不对', '位置过高', '表情僵硬'],
    duration: 2000,
    frames: [
      {
        timestamp: 0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('rest', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 0.3, 0),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.3,
          eyes: 0.7,
          mouth: 0.5,
          emotion: 'neutral'
        }
      },
      {
        timestamp: 0.4,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('fist', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.1, 1.2, 0.2),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0.15, 0, 0),
        facialExpression: {
          eyebrows: 0.5,
          eyes: 0.9,
          mouth: 0.8,
          emotion: 'happy'
        }
      },
      {
        timestamp: 0.8,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('fist', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.05, 1.1, 0.25),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0.05, 0, 0),
        headPose: new THREE.Euler(0.25, 0, 0),
        facialExpression: {
          eyebrows: 0.6,
          eyes: 1.0,
          mouth: 1.0,
          emotion: 'happy'
        }
      },
      {
        timestamp: 1.0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('rest', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 0.3, 0),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.3,
          eyes: 0.8,
          mouth: 0.6,
          emotion: 'happy'
        }
      }
    ],
    metadata: {
      region: 'national',
      frequency: 'high',
      context: ['感谢他人', '接受帮助', '礼貌回应'],
      relatedGestures: ['hello', 'you_are_welcome']
    }
  },

  i_love_you: {
    id: 'i_love_you',
    name: '我爱你',
    category: 'emotion',
    difficulty: 'intermediate',
    description: '右手做出爱心手势，双手配合表达深切的爱意',
    instruction: '1. 右手食指中指交叉成心形\n2. 左手辅助表达\n3. 表情温暖真诚\n4. 动作轻柔',
    commonMistakes: ['手指位置不准确', '表情过于夸张', '动作过快'],
    duration: 3000,
    frames: [
      {
        timestamp: 0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('rest', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 0.3, 0),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.2,
          eyes: 0.6,
          mouth: 0.4,
          emotion: 'neutral'
        }
      },
      {
        timestamp: 0.3,
        leftHand: generateProfessionalHandPose('open', 'left'),
        rightHand: generateProfessionalHandPose('love', 'right'),
        leftWrist: new THREE.Vector3(-0.15, 1.0, 0.1),
        rightWrist: new THREE.Vector3(0.15, 1.3, 0.2),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0.1, 0, 0),
        facialExpression: {
          eyebrows: 0.4,
          eyes: 0.9,
          mouth: 0.8,
          emotion: 'happy'
        }
      },
      {
        timestamp: 0.7,
        leftHand: generateProfessionalHandPose('open', 'left'),
        rightHand: generateProfessionalHandPose('love', 'right'),
        leftWrist: new THREE.Vector3(-0.05, 1.2, 0.3),
        rightWrist: new THREE.Vector3(0.05, 1.4, 0.4),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0.05, 0, 0),
        headPose: new THREE.Euler(0.15, 0, 0),
        facialExpression: {
          eyebrows: 0.6,
          eyes: 1.0,
          mouth: 1.0,
          emotion: 'happy'
        }
      },
      {
        timestamp: 1.0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('rest', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 0.3, 0),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.3,
          eyes: 0.8,
          mouth: 0.7,
          emotion: 'happy'
        }
      }
    ],
    metadata: {
      region: 'national',
      frequency: 'medium',
      context: ['表达爱意', '亲密关系', '情感表达'],
      relatedGestures: ['heart', 'kiss', 'hug']
    }
  },

  goodbye: {
    id: 'goodbye',
    name: '再见',
    category: 'greeting',
    difficulty: 'beginner',
    description: '右手举起，手掌向外，左右摆动表示告别',
    instruction: '1. 右手举至肩膀高度\n2. 手掌面向对方\n3. 轻柔左右摆动\n4. 配合微笑',
    commonMistakes: ['摆动过于剧烈', '手掌方向错误', '位置不合适'],
    duration: 3000,
    frames: [
      {
        timestamp: 0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('rest', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 0.3, 0),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.2,
          eyes: 0.7,
          mouth: 0.5,
          emotion: 'neutral'
        }
      },
      {
        timestamp: 0.2,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('wave', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.3, 1.4, 0.2),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0.1, 0),
        headPose: new THREE.Euler(0, 0.1, 0),
        facialExpression: {
          eyebrows: 0.4,
          eyes: 0.9,
          mouth: 0.8,
          emotion: 'happy'
        }
      },
      {
        timestamp: 0.5,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('wave', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.1, 1.4, 0.2),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, -0.1, 0),
        headPose: new THREE.Euler(0, -0.1, 0),
        facialExpression: {
          eyebrows: 0.5,
          eyes: 1.0,
          mouth: 1.0,
          emotion: 'happy'
        }
      },
      {
        timestamp: 0.8,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('wave', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.4, 1.4, 0.2),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0.1, 0),
        headPose: new THREE.Euler(0, 0.1, 0),
        facialExpression: {
          eyebrows: 0.4,
          eyes: 0.9,
          mouth: 0.9,
          emotion: 'happy'
        }
      },
      {
        timestamp: 1.0,
        leftHand: generateProfessionalHandPose('rest', 'left'),
        rightHand: generateProfessionalHandPose('rest', 'right'),
        leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
        rightWrist: new THREE.Vector3(0.25, 0.3, 0),
        shoulderWidth: 0.4,
        bodyPose: new THREE.Euler(0, 0, 0),
        headPose: new THREE.Euler(0, 0, 0),
        facialExpression: {
          eyebrows: 0.3,
          eyes: 0.8,
          mouth: 0.7,
          emotion: 'happy'
        }
      }
    ],
    metadata: {
      region: 'national',
      frequency: 'high',
      context: ['告别离开', '结束对话', '礼貌道别'],
      relatedGestures: ['hello', 'see_you_later']
    }
  }
}

// 专业手语播放器
export class ProfessionalSignLanguagePlayer {
  private currentGesture: ProfessionalSignLanguageGesture | null = null
  private startTime: number = 0
  private isPlaying: boolean = false
  private onFrameUpdate?: (frame: ProfessionalSignLanguageFrame) => void
  private animationId?: number

  constructor(onFrameUpdate?: (frame: ProfessionalSignLanguageFrame) => void) {
    this.onFrameUpdate = onFrameUpdate
  }

  play(gestureId: string) {
    const gesture = PROFESSIONAL_SIGN_LANGUAGE_LIBRARY[gestureId]
    if (!gesture) {
      console.warn(`手语动作不存在: ${gestureId}`)
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
    if (this.animationId) {
      cancelAnimationFrame(this.animationId)
    }
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
      this.animationId = requestAnimationFrame(() => this.animate())
    } else {
      this.isPlaying = false
    }
  }

  private interpolateFrame(progress: number): ProfessionalSignLanguageFrame {
    if (!this.currentGesture || this.currentGesture.frames.length === 0) {
      return this.getDefaultFrame()
    }

    const frames = this.currentGesture.frames
    
    // 找到当前进度对应的帧
    for (let i = 0; i < frames.length - 1; i++) {
      const currentFrameTime = frames[i].timestamp
      const nextFrameTime = frames[i + 1].timestamp
      
      if (progress >= currentFrameTime && progress <= nextFrameTime) {
        const localProgress = (progress - currentFrameTime) / (nextFrameTime - currentFrameTime)
        return this.interpolateBetweenFrames(frames[i], frames[i + 1], localProgress)
      }
    }
    
    return frames[frames.length - 1]
  }

  private interpolateBetweenFrames(
    frame1: ProfessionalSignLanguageFrame, 
    frame2: ProfessionalSignLanguageFrame, 
    factor: number
  ): ProfessionalSignLanguageFrame {
    return {
      timestamp: frame1.timestamp + (frame2.timestamp - frame1.timestamp) * factor,
      leftHand: this.interpolateHandKeypoints(frame1.leftHand, frame2.leftHand, factor),
      rightHand: this.interpolateHandKeypoints(frame1.rightHand, frame2.rightHand, factor),
      leftWrist: frame1.leftWrist.clone().lerp(frame2.leftWrist, factor),
      rightWrist: frame1.rightWrist.clone().lerp(frame2.rightWrist, factor),
      shoulderWidth: frame1.shoulderWidth + (frame2.shoulderWidth - frame1.shoulderWidth) * factor,
      bodyPose: frame1.bodyPose && frame2.bodyPose ? 
        new THREE.Euler(
          frame1.bodyPose.x + (frame2.bodyPose.x - frame1.bodyPose.x) * factor,
          frame1.bodyPose.y + (frame2.bodyPose.y - frame1.bodyPose.y) * factor,
          frame1.bodyPose.z + (frame2.bodyPose.z - frame1.bodyPose.z) * factor
        ) : undefined,
      headPose: frame1.headPose && frame2.headPose ? 
        new THREE.Euler(
          frame1.headPose.x + (frame2.headPose.x - frame1.headPose.x) * factor,
          frame1.headPose.y + (frame2.headPose.y - frame1.headPose.y) * factor,
          frame1.headPose.z + (frame2.headPose.z - frame1.headPose.z) * factor
        ) : undefined,
      eyeGaze: frame1.eyeGaze && frame2.eyeGaze ? 
        frame1.eyeGaze.clone().lerp(frame2.eyeGaze, factor) : undefined,
      facialExpression: frame1.facialExpression && frame2.facialExpression ? {
        eyebrows: frame1.facialExpression.eyebrows + (frame2.facialExpression.eyebrows - frame1.facialExpression.eyebrows) * factor,
        eyes: frame1.facialExpression.eyes + (frame2.facialExpression.eyes - frame1.facialExpression.eyes) * factor,
        mouth: frame1.facialExpression.mouth + (frame2.facialExpression.mouth - frame1.facialExpression.mouth) * factor,
        emotion: factor < 0.5 ? frame1.facialExpression.emotion : frame2.facialExpression.emotion
      } : undefined
    }
  }

  private interpolateHandKeypoints(
    hand1: ProfessionalSignLanguageKeypoint[], 
    hand2: ProfessionalSignLanguageKeypoint[], 
    factor: number
  ): ProfessionalSignLanguageKeypoint[] {
    if (hand1.length !== hand2.length) {
      return factor < 0.5 ? hand1 : hand2
    }

    return hand1.map((kp1, index) => {
      const kp2 = hand2[index]
      return {
        x: kp1.x + (kp2.x - kp1.x) * factor,
        y: kp1.y + (kp2.y - kp1.y) * factor,
        z: kp1.z + (kp2.z - kp1.z) * factor,
        visibility: kp1.visibility + (kp2.visibility - kp1.visibility) * factor,
        confidence: kp1.confidence + (kp2.confidence - kp1.confidence) * factor
      }
    })
  }

  private getDefaultFrame(): ProfessionalSignLanguageFrame {
    return {
      timestamp: 0,
      leftHand: generateProfessionalHandPose('rest', 'left'),
      rightHand: generateProfessionalHandPose('rest', 'right'),
      leftWrist: new THREE.Vector3(-0.25, 0.3, 0),
      rightWrist: new THREE.Vector3(0.25, 0.3, 0),
      shoulderWidth: 0.4,
      bodyPose: new THREE.Euler(0, 0, 0),
      headPose: new THREE.Euler(0, 0, 0),
      facialExpression: {
        eyebrows: 0.2,
        eyes: 0.7,
        mouth: 0.5,
        emotion: 'neutral'
      }
    }
  }
}

export default PROFESSIONAL_SIGN_LANGUAGE_LIBRARY
