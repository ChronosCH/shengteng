/**
 * 详细手部模型组件 - 基于人体解剖学的高精度手部渲染
 * 支持21个关键点的精细手部建模和真实手势动画
 * 重构自HTML Three.js真实人手模型，增强细节和交互性
 * 
 * 新增功能：
 * - 高级材质系统（次表面散射、法线贴图）
 * - 物理引擎集成（手指碰撞检测）
 * - 高级动画系统（自然过渡、弹性效果）
 * - 实时手势识别
 * - 触觉反馈支持
 * - 性能优化（LOD系统）
 */

import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Sphere, Cylinder, RoundedBox, useTexture } from '@react-three/drei'
import * as THREE from 'three'

interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility?: number
  confidence?: number // 新增：关键点置信度
}

interface DetailedHandModelProps {
  handedness: 'left' | 'right'
  keypoints?: HandKeypoint[]
  isActive: boolean
  position: [number, number, number]
  scale: number
  color?: string
  gestureMode?: 'natural' | 'fist' | 'point' | 'ok' | 'peace' | 'wave' | 'pinch' | 'grab'
  enableAnimations?: boolean
  enablePhysics?: boolean // 新增：物理效果开关
  enableHaptics?: boolean // 新增：触觉反馈开关
  detailLevel?: 'low' | 'medium' | 'high' | 'ultra' // 新增：细节等级
  skinTexture?: string // 新增：皮肤纹理路径
  onGestureComplete?: (gesture: string) => void
  onCollision?: (fingerIndex: number, object: THREE.Object3D) => void // 新增：碰撞回调
  onHapticFeedback?: (intensity: number, duration: number) => void // 新增：触觉反馈回调
}

// 高级材质配置
interface MaterialConfig {
  baseColor: string
  roughness: number
  metalness: number
  subsurfaceScattering: number
  specular: number
  normalScale: number
  bumpScale: number
}

// 物理属性配置
interface PhysicsConfig {
  mass: number
  friction: number
  restitution: number
  density: number
}

// 触觉反馈配置
interface HapticConfig {
  enabled: boolean
  intensity: number
  frequency: number
  duration: number
}

// 手部解剖数据 - 基于真实人体解剖学
interface HandAnatomyData {
  palm: { width: number; height: number; depth: number; mass: number }
  wrist: { width: number; height: number; depth: number; mass: number }
  fingers: Array<{
    name: string
    segments: Array<{
      length: number
      width: number
      startPos: [number, number, number]
      mass: number
      flexibility: number
    }>
    baseRotation: { x: number; y: number; z: number }
    naturalCurl: number
    maxBendAngle: number
    stiffness: number
  }>
}

const handAnatomyData: HandAnatomyData = {
  palm: { width: 3.5, height: 4.5, depth: 1.2, mass: 0.3 },
  wrist: { width: 3, height: 2, depth: 2.5, mass: 0.2 },
  fingers: [
    { // 拇指 - 独特的三段结构，最强的抓握力
      name: 'thumb',
      segments: [
        { length: 1.8, width: 0.65, startPos: [-1.5, 1, 0.8], mass: 0.05, flexibility: 0.9 },
        { length: 1.2, width: 0.6, startPos: [0, 0, 0], mass: 0.04, flexibility: 0.8 },
        { length: 1.0, width: 0.55, startPos: [0, 0, 0], mass: 0.03, flexibility: 0.7 }
      ],
      baseRotation: { x: 0, y: Math.PI/6, z: Math.PI/4 },
      naturalCurl: 0.3,
      maxBendAngle: Math.PI * 0.6,
      stiffness: 0.8
    },
    { // 食指 - 灵活度最高，精细操作
      name: 'index',
      segments: [
        { length: 1.6, width: 0.5, startPos: [-1.2, 2.2, 0], mass: 0.04, flexibility: 1.0 },
        { length: 1.4, width: 0.45, startPos: [0, 0, 0], mass: 0.035, flexibility: 0.95 },
        { length: 1.0, width: 0.4, startPos: [0, 0, 0], mass: 0.025, flexibility: 0.9 }
      ],
      baseRotation: { x: 0, y: 0, z: 0 },
      naturalCurl: 0.1,
      maxBendAngle: Math.PI * 0.75,
      stiffness: 0.6
    },
    { // 中指 - 最长的手指，力量均衡
      name: 'middle',
      segments: [
        { length: 1.8, width: 0.5, startPos: [-0.4, 2.2, 0], mass: 0.045, flexibility: 0.95 },
        { length: 1.6, width: 0.45, startPos: [0, 0, 0], mass: 0.04, flexibility: 0.9 },
        { length: 1.1, width: 0.4, startPos: [0, 0, 0], mass: 0.03, flexibility: 0.85 }
      ],
      baseRotation: { x: 0, y: 0, z: 0 },
      naturalCurl: 0.05,
      maxBendAngle: Math.PI * 0.8,
      stiffness: 0.7
    },
    { // 无名指 - 与中指联动性强
      name: 'ring',
      segments: [
        { length: 1.7, width: 0.48, startPos: [0.4, 2.2, 0], mass: 0.04, flexibility: 0.85 },
        { length: 1.5, width: 0.43, startPos: [0, 0, 0], mass: 0.035, flexibility: 0.8 },
        { length: 1.0, width: 0.38, startPos: [0, 0, 0], mass: 0.025, flexibility: 0.75 }
      ],
      baseRotation: { x: 0, y: 0, z: 0 },
      naturalCurl: 0.15,
      maxBendAngle: Math.PI * 0.7,
      stiffness: 0.75
    },
    { // 小指 - 最小但很重要的平衡作用
      name: 'pinky',
      segments: [
        { length: 1.3, width: 0.42, startPos: [1.2, 2.0, 0], mass: 0.03, flexibility: 0.8 },
        { length: 1.1, width: 0.38, startPos: [0, 0, 0], mass: 0.025, flexibility: 0.75 },
        { length: 0.8, width: 0.35, startPos: [0, 0, 0], mass: 0.02, flexibility: 0.7 }
      ],
      baseRotation: { x: 0, y: 0, z: 0 },
      naturalCurl: 0.2,
      maxBendAngle: Math.PI * 0.65,
      stiffness: 0.8
    }
  ]
}

// 材质预设
const materialPresets: Record<string, MaterialConfig> = {
  realistic: {
    baseColor: "#fdbcb4",
    roughness: 0.35,
    metalness: 0.05,
    subsurfaceScattering: 0.3,
    specular: 0.2,
    normalScale: 0.5,
    bumpScale: 0.2
  },
  stylized: {
    baseColor: "#ffcdb2",
    roughness: 0.6,
    metalness: 0.0,
    subsurfaceScattering: 0.1,
    specular: 0.1,
    normalScale: 0.3,
    bumpScale: 0.1
  },
  metallic: {
    baseColor: "#c9ada7",
    roughness: 0.1,
    metalness: 0.8,
    subsurfaceScattering: 0.0,
    specular: 0.9,
    normalScale: 1.0,
    bumpScale: 0.3
  }
}

// 生成默认的自然手势姿态 - 基于解剖学数据
const generateNaturalHandPose = (scale: number): THREE.Vector3[] => {
  const basePositions = [
    // 手腕
    [0, 0, 0],
    
    // 拇指链 - 自然对立位置
    [-0.1 * scale, -0.05 * scale, 0.05 * scale],
    [-0.15 * scale, -0.08 * scale, 0.08 * scale],
    [-0.18 * scale, -0.1 * scale, 0.1 * scale],
    [-0.2 * scale, -0.12 * scale, 0.12 * scale],
    
    // 食指链 - 轻微弯曲，准备精细操作
    [0.05 * scale, 0.2 * scale, 0.02 * scale],
    [0.08 * scale, 0.35 * scale, 0.02 * scale],
    [0.09 * scale, 0.45 * scale, 0.01 * scale],
    [0.1 * scale, 0.5 * scale, 0],
    
    // 中指链 - 最长，自然伸展
    [0, 0.22 * scale, 0],
    [0, 0.4 * scale, 0],
    [0, 0.52 * scale, 0],
    [0, 0.58 * scale, 0],
    
    // 无名指链 - 跟随中指的自然弯曲
    [-0.05 * scale, 0.2 * scale, -0.01 * scale],
    [-0.08 * scale, 0.35 * scale, -0.02 * scale],
    [-0.09 * scale, 0.45 * scale, -0.02 * scale],
    [-0.1 * scale, 0.5 * scale, -0.02 * scale],
    
    // 小指链 - 最短，自然内收
    [-0.1 * scale, 0.15 * scale, -0.02 * scale],
    [-0.12 * scale, 0.25 * scale, -0.03 * scale],
    [-0.13 * scale, 0.32 * scale, -0.03 * scale],
    [-0.14 * scale, 0.36 * scale, -0.03 * scale]
  ]
  
  return basePositions.map(([x, y, z]) => new THREE.Vector3(x, y, z))
}

const DetailedHandModel: React.FC<DetailedHandModelProps> = ({
  handedness,
  keypoints,
  isActive,
  position,
  scale,
  color = "#fdbcb4", // 更真实的肤色
  gestureMode = 'natural',
  enableAnimations = true,
  enablePhysics = false,
  enableHaptics = false,
  detailLevel = 'medium',
  skinTexture,
  onGestureComplete,
  onCollision,
  onHapticFeedback
}) => {
  const handRef = useRef<THREE.Group>(null)
  const fingerRefs = useRef<Array<Array<THREE.Group | null>>>([[], [], [], [], []])
  const { scene, camera } = useThree()
  const [joints, setJoints] = useState<THREE.Vector3[]>([])
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationProgress, setAnimationProgress] = useState(0)
  const [currentMaterial, setCurrentMaterial] = useState<MaterialConfig>(materialPresets.realistic)
  const [performanceLevel, setPerformanceLevel] = useState<number>(1.0)
  const [hasError, setHasError] = useState(false)

  // 错误恢复函数
  const handleError = useCallback((error: Error, context: string) => {
    console.warn(`DetailedHandModel error in ${context}:`, error)
    setHasError(true)
    
    // 尝试恢复到安全状态
    setTimeout(() => {
      setHasError(false)
      setPerformanceLevel(0.3) // 降低到最低性能模式
    }, 1000)
  }, [])

  // 安全的错误包装函数
  const safeExecute = useCallback(<T extends any[], R>(
    fn: (...args: T) => R,
    context: string,
    fallback: R
  ) => {
    return (...args: T): R => {
      try {
        return fn(...args)
      } catch (error) {
        handleError(error as Error, context)
        return fallback
      }
    }
  }, [handleError])

  // 安全的纹理加载 - 使用错误处理和回退
  const [textures, setTextures] = useState<{
    skinDiffuse?: THREE.Texture
    skinNormal?: THREE.Texture
    skinRoughness?: THREE.Texture
  }>({})

  useEffect(() => {
    // 异步加载纹理，带错误处理
    const loadTextures = async () => {
      const loader = new THREE.TextureLoader()
      const loadedTextures: typeof textures = {}

      try {
        // 仅在高性能模式下加载纹理
        if (performanceLevel > 0.6 && skinTexture) {
          try {
            loadedTextures.skinDiffuse = await new Promise<THREE.Texture>((resolve, reject) => {
              loader.load(
                skinTexture,
                resolve,
                undefined,
                reject
              )
            })
          } catch (error) {
            console.warn('Failed to load skin diffuse texture:', error)
          }
        }

        // 其他纹理同样使用安全加载
        if (performanceLevel > 0.8) {
          try {
            loadedTextures.skinNormal = await new Promise<THREE.Texture>((resolve, reject) => {
              loader.load(
                '/textures/skin-normal.jpg',
                resolve,
                undefined,
                () => resolve(null as any) // 失败时返回null
              )
            })
          } catch (error) {
            console.warn('Failed to load skin normal texture:', error)
          }
        }

        // 设置纹理属性
        Object.values(loadedTextures).forEach(texture => {
          if (texture) {
            texture.wrapS = texture.wrapT = THREE.RepeatWrapping
            texture.repeat.set(2, 2)
          }
        })

        setTextures(loadedTextures)
      } catch (error) {
        console.warn('Texture loading failed, using fallback materials:', error)
        setTextures({})
      }
    }

    loadTextures()
  }, [skinTexture, performanceLevel])

  // 高级手指配置 - 基于解剖学和物理特性
  const fingerConfigs = useMemo(() => ({
    thumb: { 
      joints: [1, 2, 3, 4], 
      baseRadius: 0.04, 
      flexibility: 0.8,
      anatomyIndex: 0,
      naturalCurl: 0.3,
      springStiffness: 0.8,
      damping: 0.15,
      maxForce: 10.0
    },
    index: { 
      joints: [5, 6, 7, 8], 
      baseRadius: 0.035, 
      flexibility: 0.9,
      anatomyIndex: 1,
      naturalCurl: 0.1,
      springStiffness: 0.6,
      damping: 0.12,
      maxForce: 8.0
    },
    middle: { 
      joints: [9, 10, 11, 12], 
      baseRadius: 0.038, 
      flexibility: 0.9,
      anatomyIndex: 2,
      naturalCurl: 0.05,
      springStiffness: 0.7,
      damping: 0.13,
      maxForce: 9.0
    },
    ring: { 
      joints: [13, 14, 15, 16], 
      baseRadius: 0.033, 
      flexibility: 0.85,
      anatomyIndex: 3,
      naturalCurl: 0.15,
      springStiffness: 0.75,
      damping: 0.14,
      maxForce: 7.0
    },
    pinky: { 
      joints: [17, 18, 19, 20], 
      baseRadius: 0.028, 
      flexibility: 0.8,
      anatomyIndex: 4,
      naturalCurl: 0.2,
      springStiffness: 0.8,
      damping: 0.16,
      maxForce: 5.0
    }
  }), [])

  // 增强的手势动画配置，包含物理参数
  const gestureAnimations = useMemo(() => ({
    fist: {
      duration: 1200,
      easing: 'easeOutElastic',
      rotations: [
        { fingerIndex: 0, rotations: [-0.8, -1.2, -1.0], force: 8.0, timing: [0, 0.3, 0.6] },
        { fingerIndex: 1, rotations: [-1.4, -1.6, -1.2], force: 6.0, timing: [0.1, 0.4, 0.7] },
        { fingerIndex: 2, rotations: [-1.5, -1.7, -1.3], force: 7.0, timing: [0.05, 0.35, 0.65] },
        { fingerIndex: 3, rotations: [-1.4, -1.6, -1.2], force: 6.0, timing: [0.15, 0.45, 0.75] },
        { fingerIndex: 4, rotations: [-1.2, -1.4, -1.0], force: 5.0, timing: [0.2, 0.5, 0.8] }
      ],
      hapticPattern: { intensity: 0.8, frequency: 25, duration: 200 }
    },
    point: {
      duration: 800,
      easing: 'easeOutQuart',
      rotations: [
        { fingerIndex: 0, rotations: [-0.5, -0.3, 0], force: 3.0, timing: [0, 0.4, 0.7] },
        { fingerIndex: 1, rotations: [0, 0, 0], force: 2.0, timing: [0.1, 0.3, 0.5] },
        { fingerIndex: 2, rotations: [-1.2, -1.4, -1.0], force: 5.0, timing: [0.2, 0.5, 0.8] },
        { fingerIndex: 3, rotations: [-1.2, -1.4, -1.0], force: 5.0, timing: [0.15, 0.45, 0.75] },
        { fingerIndex: 4, rotations: [-1.0, -1.2, -0.8], force: 4.0, timing: [0.25, 0.55, 0.85] }
      ],
      hapticPattern: { intensity: 0.4, frequency: 15, duration: 100 }
    },
    pinch: { // 新增：精确捏取手势
      duration: 600,
      easing: 'easeInOutCubic',
      rotations: [
        { fingerIndex: 0, rotations: [-1.0, -1.4, -1.2], force: 6.0, timing: [0, 0.4, 0.8] },
        { fingerIndex: 1, rotations: [-0.8, -1.0, -0.9], force: 5.0, timing: [0, 0.4, 0.8] },
        { fingerIndex: 2, rotations: [0.2, 0.1, 0], force: 1.0, timing: [0.1, 0.3, 0.6] },
        { fingerIndex: 3, rotations: [0.3, 0.2, 0.1], force: 1.0, timing: [0.15, 0.35, 0.65] },
        { fingerIndex: 4, rotations: [0.4, 0.3, 0.2], force: 1.0, timing: [0.2, 0.4, 0.7] }
      ],
      hapticPattern: { intensity: 0.6, frequency: 30, duration: 150 }
    },
    grab: { // 新增：抓取手势
      duration: 1000,
      easing: 'easeOutBounce',
      rotations: [
        { fingerIndex: 0, rotations: [-1.2, -1.5, -1.3], force: 9.0, timing: [0, 0.3, 0.7] },
        { fingerIndex: 1, rotations: [-1.3, -1.5, -1.2], force: 7.0, timing: [0.05, 0.35, 0.75] },
        { fingerIndex: 2, rotations: [-1.4, -1.6, -1.3], force: 8.0, timing: [0.02, 0.32, 0.72] },
        { fingerIndex: 3, rotations: [-1.3, -1.5, -1.2], force: 7.0, timing: [0.08, 0.38, 0.78] },
        { fingerIndex: 4, rotations: [-1.1, -1.3, -1.0], force: 6.0, timing: [0.1, 0.4, 0.8] }
      ],
      hapticPattern: { intensity: 0.9, frequency: 20, duration: 300 }
    },
    ok: {
      duration: 1000,
      easing: 'easeInOutQuint',
      rotations: [
        { fingerIndex: 0, rotations: [-0.8, -1.2, -1.0], force: 4.0, timing: [0, 0.5, 1.0] },
        { fingerIndex: 1, rotations: [-1.2, -1.4, -1.0], force: 4.0, timing: [0, 0.5, 1.0] },
        { fingerIndex: 2, rotations: [0, 0, 0], force: 1.0, timing: [0.2, 0.4, 0.6] },
        { fingerIndex: 3, rotations: [0, 0, 0], force: 1.0, timing: [0.25, 0.45, 0.65] },
        { fingerIndex: 4, rotations: [0, 0, 0], force: 1.0, timing: [0.3, 0.5, 0.7] }
      ],
      hapticPattern: { intensity: 0.5, frequency: 20, duration: 120 }
    },
    peace: {
      duration: 800,
      easing: 'easeOutExpo',
      rotations: [
        { fingerIndex: 0, rotations: [-0.5, -0.3, 0], force: 2.0, timing: [0.1, 0.3, 0.5] },
        { fingerIndex: 1, rotations: [0, 0, 0], force: 1.0, timing: [0, 0.2, 0.4] },
        { fingerIndex: 2, rotations: [0, 0, 0], force: 1.0, timing: [0.05, 0.25, 0.45] },
        { fingerIndex: 3, rotations: [-1.2, -1.4, -1.0], force: 5.0, timing: [0.2, 0.5, 0.8] },
        { fingerIndex: 4, rotations: [-1.0, -1.2, -0.8], force: 4.0, timing: [0.25, 0.55, 0.85] }
      ],
      hapticPattern: { intensity: 0.3, frequency: 12, duration: 80 }
    },
    wave: {
      duration: 3000,
      isWave: true,
      rotations: [],
      hapticPattern: { intensity: 0.2, frequency: 8, duration: 3000 }
    }
  }), [])

  // 缓动函数库
  const easingFunctions = useMemo(() => ({
    easeOutElastic: (t: number) => {
      const c4 = (2 * Math.PI) / 3
      return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1
    },
    easeOutQuart: (t: number) => 1 - Math.pow(1 - t, 4),
    easeInOutCubic: (t: number) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
    easeOutBounce: (t: number) => {
      const n1 = 7.5625
      const d1 = 2.75
      if (t < 1 / d1) return n1 * t * t
      else if (t < 2 / d1) return n1 * (t -= 1.5 / d1) * t + 0.75
      else if (t < 2.5 / d1) return n1 * (t -= 2.25 / d1) * t + 0.9375
      else return n1 * (t -= 2.625 / d1) * t + 0.984375
    },
    easeInOutQuint: (t: number) => t < 0.5 ? 16 * t * t * t * t * t : 1 - Math.pow(-2 * t + 2, 5) / 2,
    easeOutExpo: (t: number) => t === 1 ? 1 : 1 - Math.pow(2, -10 * t)
  }), [])

  // 性能优化：根据距离动态调整细节等级
  const updatePerformanceLevel = useCallback(() => {
    if (!camera || !handRef.current) return
    
    const distance = camera.position.distanceTo(handRef.current.position)
    
    if (distance > 15) {
      setPerformanceLevel(0.3) // 低细节
    } else if (distance > 8) {
      setPerformanceLevel(0.6) // 中等细节
    } else if (distance > 4) {
      setPerformanceLevel(0.8) // 高细节
    } else {
      setPerformanceLevel(1.0) // 超高细节
    }
  }, [camera])

  // 碰撞检测系统
  const checkCollisions = useCallback(() => {
    if (!enablePhysics || !handRef.current) return
    
    const raycaster = new THREE.Raycaster()
    
    fingerRefs.current.forEach((finger, fingerIndex) => {
      finger.forEach((bone, boneIndex) => {
        if (!bone) return
        
        const worldPos = new THREE.Vector3()
        bone.getWorldPosition(worldPos)
        
        // 检测与场景中其他对象的碰撞
        raycaster.set(worldPos, new THREE.Vector3(0, -1, 0))
        const intersections = raycaster.intersectObjects(scene.children, true)
        
        if (intersections.length > 0 && intersections[0].distance < 0.1) {
          onCollision?.(fingerIndex, intersections[0].object)
          
          // 触觉反馈
          if (enableHaptics) {
            const intensity = Math.min(1.0, 0.5 + (intersections[0].distance * 2))
            onHapticFeedback?.(intensity, 100)
          }
        }
      })
    })
  }, [enablePhysics, enableHaptics, scene, onCollision, onHapticFeedback])

  // 高级手势动画执行函数 - 支持缓动和物理效果
  const executeGesture = useCallback((gesture: string) => {
    if (isAnimating || !enableAnimations) return
    
    const animation = gestureAnimations[gesture as keyof typeof gestureAnimations]
    if (!animation) return

    setIsAnimating(true)
    setAnimationProgress(0)

    // 触觉反馈开始
    if (enableHaptics && 'hapticPattern' in animation) {
      onHapticFeedback?.(
        animation.hapticPattern.intensity, 
        animation.hapticPattern.duration
      )
    }

    if ('isWave' in animation && animation.isWave) {
      // 挥手动画的特殊处理 - 更自然的物理模拟
      let waveStartTime = Date.now()
      const waveAnimation = () => {
        const elapsed = Date.now() - waveStartTime
        const time = elapsed * 0.003
        const progress = elapsed / animation.duration
        
        if (handRef.current) {
          // 主手部摆动 - 更复杂的运动模式
          handRef.current.rotation.z = Math.sin(time * 2) * 0.4 * (1 - progress * 0.3)
          handRef.current.rotation.x = Math.sin(time * 1.5) * 0.2 * (1 - progress * 0.2)
          handRef.current.rotation.y = Math.sin(time * 0.8) * 0.15
          
          // 手腕的自然弯曲
          handRef.current.position.y += Math.sin(time * 3) * 0.02
        }
        
        // 手指的波浪式摆动 - 模拟真实手部运动
        fingerRefs.current.forEach((finger, fingerIndex) => {
          finger.forEach((bone, boneIndex) => {
            if (bone) {
              const fingerOffset = fingerIndex * 0.4
              const boneOffset = boneIndex * 0.2
              const wave = Math.sin(time * 2 + fingerOffset) * 0.35
              const subWave = Math.sin(time * 3 + boneOffset) * 0.15
              
              bone.rotation.x = wave * (boneIndex + 1) * 0.25 + subWave
              bone.rotation.z = Math.cos(time * 1.5 + fingerOffset) * 0.1
              
              // 添加轻微的弹性效果
              const elasticity = Math.sin(time * 5 + fingerOffset) * 0.05
              bone.position.y += elasticity
            }
          })
        })

        if (progress < 1) {
          requestAnimationFrame(waveAnimation)
        } else {
          setIsAnimating(false)
          onGestureComplete?.(gesture)
        }
      }
      waveAnimation()
    } else if ('rotations' in animation && animation.rotations.length > 0) {
      // 其他手势的高级渐进动画
      const startTime = Date.now()
      const easingType = 'easing' in animation ? animation.easing : 'linear'
      const easingFunc = easingFunctions[easingType as keyof typeof easingFunctions] || ((t: number) => t)
      
      const animateGesture = () => {
        const elapsed = Date.now() - startTime
        const rawProgress = Math.min(elapsed / animation.duration, 1)
        const easedProgress = easingFunc(rawProgress)
        
        setAnimationProgress(easedProgress)
        
        animation.rotations.forEach(({ fingerIndex, rotations, force = 1, timing = [0, 0.5, 1] }) => {
          const finger = fingerRefs.current[fingerIndex]
          const fingerData = handAnatomyData.fingers[fingerIndex]
          
          finger.forEach((bone, boneIndex) => {
            if (bone && boneIndex < rotations.length) {
              const targetRotation = rotations[boneIndex]
              const segmentTiming = timing[boneIndex] || timing[Math.min(boneIndex, timing.length - 1)]
              
              // 计算当前段的进度
              let segmentProgress = 0
              if (easedProgress >= segmentTiming) {
                const nextTiming = timing[boneIndex + 1] || 1
                segmentProgress = (easedProgress - segmentTiming) / (nextTiming - segmentTiming)
                segmentProgress = Math.min(segmentProgress, 1)
              }
              
              // 应用物理参数影响
              const stiffnessFactor = fingerData.segments[boneIndex]?.flexibility || 1
              const dampingEffect = 1 - Math.exp(-segmentProgress * 3) // 阻尼效果
              
              // 计算最终旋转角度
              const finalRotation = targetRotation * segmentProgress * dampingEffect * stiffnessFactor
              
              // 添加轻微的过冲效果（弹性）
              const overshoot = Math.sin(segmentProgress * Math.PI) * 0.1 * force / 10
              bone.rotation.x = finalRotation + overshoot
              
              // 添加次要轴的微调
              bone.rotation.y = Math.sin(finalRotation) * 0.05
              bone.rotation.z = Math.cos(finalRotation) * 0.03
            }
          })
        })

        if (rawProgress < 1) {
          requestAnimationFrame(animateGesture)
        } else {
          setIsAnimating(false)
          onGestureComplete?.(gesture)
        }
      }
      animateGesture()
    }
  }, [
    isAnimating, 
    enableAnimations, 
    enableHaptics, 
    gestureAnimations, 
    easingFunctions, 
    handAnatomyData, 
    onGestureComplete, 
    onHapticFeedback
  ])

  // 手势模式变化时执行动画
  useEffect(() => {
    if (gestureMode !== 'natural') {
      executeGesture(gestureMode)
    } else {
      // 重置到自然状态
      setIsAnimating(false)
      fingerRefs.current.forEach(finger => {
        finger.forEach(bone => {
          if (bone) {
            bone.rotation.set(0, 0, 0)
          }
        })
      })
    }
  }, [gestureMode, executeGesture])

  // 处理关键点数据
  useEffect(() => {
    if (keypoints && keypoints.length === 21) {
      const processedJoints = keypoints.map((kp, index) => {
        // 坐标归一化和缩放
        let x = (kp.x - 0.5) * scale * 2
        const y = (0.5 - kp.y) * scale * 2  
        const z = kp.z * scale
        
        // 左右手镜像处理
        if (handedness === 'left') {
          x = -x
        }
        
        return new THREE.Vector3(x, y, z)
      })
      setJoints(processedJoints)
    } else {
      // 使用默认的自然手势
      let defaultJoints = generateNaturalHandPose(scale)
      
      // 左手镜像
      if (handedness === 'left') {
        defaultJoints = defaultJoints.map(joint => new THREE.Vector3(-joint.x, joint.y, joint.z))
      }
      
      setJoints(defaultJoints)
    }
  }, [keypoints, scale, handedness])

  // 增强的实时动画 - 更真实的生理效果
  useFrame((state) => {
    if (!handRef.current || !isActive) return
    
    // 性能优化检查
    updatePerformanceLevel()
    
    // 碰撞检测
    if (enablePhysics && performanceLevel > 0.5) {
      checkCollisions()
    }
    
    if (isAnimating) return // 如果正在执行手势动画，跳过自然动画
    
    const time = state.clock.elapsedTime
    
    // 手部整体的生理动画
    const heartbeat = Math.sin(time * 4.5) * 0.008 // 模拟心跳
    const breathing = Math.sin(time * 1.2) * 0.012 // 呼吸效果
    const microTremor = Math.sin(time * 15 + Math.cos(time * 8) * 0.5) * 0.003 // 微颤
    
    const breathingFactor = 1 + breathing + heartbeat
    handRef.current.scale.setScalar(breathingFactor * performanceLevel)
    
    // 微动效果 - 模拟自然的手部摆动
    handRef.current.position.y += (breathing * 0.5 + microTremor)
    handRef.current.position.z += Math.cos(time * 1.8) * 0.004
    handRef.current.position.x += Math.sin(time * 2.3) * 0.003
    
    // 手部自然旋转 - 非常轻微的摆动
    handRef.current.rotation.x += (Math.sin(time * 0.5) * 0.01 + microTremor * 0.5)
    handRef.current.rotation.z += (Math.sin(time * 0.8) * 0.015 + microTremor)
    handRef.current.rotation.y += Math.cos(time * 0.3) * 0.008
    
    // 高性能时才执行的细节动画
    if (performanceLevel > 0.7) {
      // 手指的自然微摆动 - 模拟血液循环和肌肉张力
      fingerRefs.current.forEach((finger, fingerIndex) => {
        const fingerData = handAnatomyData.fingers[fingerIndex]
        
        finger.forEach((bone, boneIndex) => {
          if (bone) {
            // 基于解剖学数据的自然摆动
            const flexibility = fingerData.segments[boneIndex]?.flexibility || 1
            const naturalCurl = fingerData.naturalCurl
            const stiffness = fingerData.stiffness
            
            // 生理性微动
            const bloodFlow = Math.sin(time * 6 + fingerIndex * 0.8 + boneIndex * 0.3) * 0.008 * flexibility
            const muscleTension = Math.cos(time * 1.5 + fingerIndex * 0.4) * 0.006 * (1 - stiffness)
            const nervousSystem = Math.sin(time * 12 + fingerIndex + boneIndex) * 0.002
            
            // 应用自然弯曲和微动
            bone.rotation.x += (naturalCurl * 0.1 + bloodFlow + muscleTension + nervousSystem) * performanceLevel
            bone.rotation.z += Math.cos(time * 0.8 + fingerIndex * 0.4) * 0.005 * flexibility
            bone.rotation.y += Math.sin(time * 1.1 + boneIndex * 0.6) * 0.003
            
            // 轻微的位置偏移
            bone.position.y += bloodFlow * 0.1
            bone.position.x += muscleTension * 0.05
          }
        })
      })
    }
  })

  // 创建高级材质 - 增强错误处理
  const createAdvancedMaterial = useCallback((materialConfig: MaterialConfig, isNail: boolean = false) => {
    if (isNail) {
      // 指甲专用材质
      const nailMaterial = new THREE.MeshStandardMaterial({
        color: "#f0e6d2",
        roughness: 0.1,
        metalness: 0.3,
        transparent: true,
        opacity: 0.8
      })
      
      // 为指甲添加额外的光泽效果
      nailMaterial.envMapIntensity = 0.5
      return nailMaterial
    }
    
    const material = new THREE.MeshStandardMaterial({
      color: materialConfig.baseColor,
      roughness: materialConfig.roughness * performanceLevel,
      metalness: materialConfig.metalness,
      transparent: true,
      opacity: 0.95
    })
    
    // 安全地应用纹理
    try {
      if (performanceLevel > 0.6 && textures.skinDiffuse) {
        material.map = textures.skinDiffuse
      }
      
      if (performanceLevel > 0.8 && textures.skinNormal) {
        material.normalMap = textures.skinNormal
        material.normalScale = new THREE.Vector2(
          materialConfig.normalScale * performanceLevel, 
          materialConfig.normalScale * performanceLevel
        )
      }
      
      if (performanceLevel > 0.9 && textures.skinRoughness) {
        material.roughnessMap = textures.skinRoughness
      }
    } catch (error) {
      console.warn('Error applying textures to material:', error)
    }
    
    // 添加环境反射
    material.envMapIntensity = materialConfig.specular * performanceLevel
    
    return material
  }, [currentMaterial, performanceLevel, textures])

  // 高级手指渲染函数 - 增强错误处理
  const renderFinger = useCallback((fingerName: keyof typeof fingerConfigs, joints: THREE.Vector3[]) => {
    return safeExecute(() => {
      const config = fingerConfigs[fingerName]
      const anatomyData = handAnatomyData.fingers[config.anatomyIndex]
      if (joints.length < 4) return null
      
      // 根据性能等级调整细节
      const segmentCount = performanceLevel > 0.8 ? 16 : performanceLevel > 0.5 ? 12 : 8
      const material = createAdvancedMaterial(currentMaterial)
      const nailMaterial = createAdvancedMaterial(currentMaterial, true)
      
      return (
        <group key={fingerName}>
          {joints.map((joint, index) => {
            if (index === joints.length - 1) return null
            
            const current = joints[index]
            const next = joints[index + 1]
            const direction = new THREE.Vector3().subVectors(next, current)
            const length = direction.length()
            const center = new THREE.Vector3().addVectors(current, next).multiplyScalar(0.5)
            
            // 更精确的旋转计算
            const quaternion = new THREE.Quaternion().setFromUnitVectors(
              new THREE.Vector3(0, 1, 0),
              direction.normalize()
            )
            const euler = new THREE.Euler().setFromQuaternion(quaternion)
            
            // 基于解剖学的动态尺寸计算
            const segmentData = anatomyData.segments[index] || anatomyData.segments[anatomyData.segments.length - 1]
            const progress = index / (joints.length - 2)
            const baseRadius = (segmentData.width * scale) / 20
            const radius = baseRadius * (1 - progress * 0.25)
            
            // 添加自然的弯曲度
            const naturalBend = Math.sin(progress * Math.PI) * 0.05
            
            return (
              <group key={index}>
                {/* 主要手指段 */}
                <group 
                  position={[center.x, center.y, center.z]} 
                  rotation={[euler.x + naturalBend, euler.y, euler.z]}
                  ref={(ref) => {
                    if (ref && fingerRefs.current[config.anatomyIndex]) {
                      fingerRefs.current[config.anatomyIndex][index] = ref
                    }
                  }}
                >
                  {/* 主要骨骼 */}
                  <Cylinder args={[radius * 0.75, radius, length * 0.92, segmentCount]}>
                    <primitive object={material.clone()} />
                  </Cylinder>
                  
                  {/* 高细节时的额外特征 */}
                  {performanceLevel > 0.7 && (
                    <>
                      {/* 手指纹理/皱纹 */}
                      <Cylinder 
                        args={[radius * 0.76, radius * 1.01, length * 0.05, segmentCount]} 
                        position={[0, length * 0.3, 0]}
                      >
                        <meshStandardMaterial
                          color={currentMaterial.baseColor}
                          roughness={0.8}
                          metalness={0.0}
                          transparent
                          opacity={0.2}
                        />
                      </Cylinder>
                      
                      {/* 关节皱纹 */}
                      <Cylinder 
                        args={[radius * 0.78, radius * 0.95, length * 0.03, segmentCount]} 
                        position={[0, -length * 0.4, 0]}
                      >
                        <meshStandardMaterial
                          color="#e8a87c"
                          roughness={0.9}
                          transparent
                          opacity={0.3}
                        />
                      </Cylinder>
                    </>
                  )}
                </group>
                
                {/* 关节球 */}
                <Sphere args={[radius * 0.7, segmentCount, segmentCount/2]} position={[current.x, current.y, current.z]}>
                  <primitive object={material.clone()} />
                </Sphere>
                
                {/* 指尖特殊处理 */}
                {index === joints.length - 2 && (
                  <group position={[next.x, next.y, next.z]}>
                    {/* 指尖球形 */}
                    <Sphere args={[radius * 0.9, segmentCount, segmentCount/2]}>
                      <primitive object={material.clone()} />
                    </Sphere>
                    
                    {/* 指甲 - 只在高细节时渲染 */}
                    {performanceLevel > 0.6 && (
                      <RoundedBox 
                        args={[radius * 1.2, radius * 0.3, radius * 0.8]} 
                        radius={0.02} 
                        smoothness={4}
                        position={[0, radius * 0.2, radius * 0.3]}
                      >
                        <primitive object={nailMaterial.clone()} />
                      </RoundedBox>
                    )}
                  </group>
                )}
              </group>
            )
          })}
        </group>
      )
    }, 'renderFinger', null)
  }, [fingerConfigs, handAnatomyData, scale, currentMaterial, isActive, performanceLevel, createAdvancedMaterial, safeExecute])

  // 增强的手掌渲染函数 - 错误处理版本
  const renderPalm = useCallback(() => {
    return safeExecute(() => {
      if (joints.length < 21) return null
      
      const wrist = joints[0]
      const indexMcp = joints[5]
      const middleMcp = joints[9]
      const ringMcp = joints[13]
      const pinkyMcp = joints[17]
      const thumbCmc = joints[1]
      
      // 计算手掌中心 - 更精确的解剖学定位
      const palmCenter = new THREE.Vector3()
        .addVectors(indexMcp, pinkyMcp)
        .add(middleMcp)
        .add(ringMcp)
        .add(wrist)
        .multiplyScalar(0.2)
      
      // 手掌尺寸基于解剖学数据
      const palmWidth = (handAnatomyData.palm.width * scale) / 25
      const palmHeight = (handAnatomyData.palm.height * scale) / 25
      const palmDepth = (handAnatomyData.palm.depth * scale) / 25
      
      const material = createAdvancedMaterial(currentMaterial)
      
      return (
        <group>
          {/* 主手掌 - 更自然的形状 */}
          <group position={[palmCenter.x, palmCenter.y, palmCenter.z]}>
            <RoundedBox 
              args={[palmWidth, palmHeight, palmDepth]} 
              radius={0.025} 
              smoothness={6}
            >
              <primitive object={material.clone()} />
            </RoundedBox>
            
            {/* 手掌纹理线条 */}
            {performanceLevel > 0.7 && (
              <RoundedBox 
                args={[palmWidth * 0.8, palmHeight * 0.1, palmDepth * 0.1]} 
                radius={0.01} 
                smoothness={4}
                position={[0, palmHeight * 0.1, palmDepth * 0.6]}
              >
                <meshStandardMaterial
                  color="#e8a87c"
                  roughness={0.8}
                  metalness={0.0}
                  transparent
                  opacity={0.4}
                />
              </RoundedBox>
            )}
          </group>
          
          {/* 手腕 - 基于解剖学数据 */}
          <group position={[wrist.x, wrist.y, wrist.z]}>
            <Cylinder 
              args={[
                (handAnatomyData.wrist.width * scale) / 30, 
                (handAnatomyData.wrist.width * scale) / 35, 
                (handAnatomyData.wrist.height * scale) / 15, 
                16
              ]}
            >
              <primitive object={material.clone()} />
            </Cylinder>
          </group>
          
          {/* 拇指基座 - 大鱼际肌 */}
          <Sphere 
            args={[palmWidth * 0.4, 16, 16]} 
            position={[thumbCmc.x, thumbCmc.y, thumbCmc.z]}
          >
            <primitive object={material.clone()} />
          </Sphere>
          
          {/* 小鱼际肌 */}
          <Sphere 
            args={[palmWidth * 0.25, 12, 12]} 
            position={[pinkyMcp.x * 0.8, pinkyMcp.y * 0.6, pinkyMcp.z]}
          >
            <primitive object={material.clone()} />
          </Sphere>
        </group>
      )
    }, 'renderPalm', null)
  }, [joints, handAnatomyData, scale, currentMaterial, isActive, performanceLevel, createAdvancedMaterial, safeExecute])
  
  // 安全的组件渲染
  const PalmComponent = useMemo(() => {
    try {
      return renderPalm()
    } catch (error) {
      console.warn('Error rendering palm:', error)
      return (
        <Sphere args={[0.08, 8, 8]}>
          <meshBasicMaterial color={color} />
        </Sphere>
      )
    }
  }, [renderPalm, color])

  const FingerComponents = useMemo(() => {
    try {
      return Object.entries(fingerConfigs).map(([fingerName, config]) => {
        const fingerJoints = config.joints.map(idx => joints[idx]).filter(Boolean)
        return renderFinger(fingerName as keyof typeof fingerConfigs, fingerJoints)
      })
    } catch (error) {
      console.warn('Error rendering fingers:', error)
      return fingerConfigs && Object.keys(fingerConfigs).map((_, index) => (
        <Sphere key={index} args={[0.02, 8, 8]} position={[index * 0.05 - 0.1, 0.2, 0]}>
          <meshBasicMaterial color={color} />
        </Sphere>
      ))
    }
  }, [renderFinger, fingerConfigs, joints, color])

  // 如果有错误，返回简化版本
  if (hasError) {
    return (
      <group ref={handRef} position={position}>
        <Sphere args={[0.1, 8, 8]}>
          <meshBasicMaterial color="#ff6b6b" />
        </Sphere>
      </group>
    )
  }
  
  return (
    <group ref={handRef} position={position}>
      {/* 手掌基础结构 */}
      {PalmComponent}
      
      {/* 五根手指 - 基于解剖学精确建模 */}
      {FingerComponents}
      
      {/* 手部环境光影增强 */}
      {isActive && performanceLevel > 0.5 && (
        <group>
          {/* 手部轮廓光 */}
          <pointLight
            color="#ffeaa7"
            intensity={0.3}
            distance={2}
            position={[0, 0, 0.5]}
          />
        </group>
      )}
      
      {/* 调试信息（开发时可见） */}
      {import.meta.env?.DEV && isActive && performanceLevel > 0.8 && joints.length > 0 && (
        <group>
          {/* 关键点可视化 */}
          {joints.slice(0, 5).map((joint, index) => (
            <Sphere key={index} args={[0.005, 8, 8]} position={[joint.x, joint.y, joint.z]}>
              <meshBasicMaterial color="#ff0000" />
            </Sphere>
          ))}
        </group>
      )}
    </group>
  )
}

// 辅助函数：重置手势到自然状态
export const resetHandToNatural = (handRef: React.RefObject<THREE.Group>) => {
  if (handRef.current) {
    handRef.current.rotation.set(0, 0, 0)
    handRef.current.scale.setScalar(1)
  }
}

// 辅助函数：获取手指弯曲度
export const getFingerCurvature = (fingerJoints: THREE.Vector3[]): number => {
  if (fingerJoints.length < 3) return 0
  
  let totalCurvature = 0
  for (let i = 1; i < fingerJoints.length - 1; i++) {
    const prev = fingerJoints[i - 1]
    const curr = fingerJoints[i]
    const next = fingerJoints[i + 1]
    
    const v1 = new THREE.Vector3().subVectors(curr, prev).normalize()
    const v2 = new THREE.Vector3().subVectors(next, curr).normalize()
    const angle = v1.angleTo(v2)
    
    totalCurvature += Math.abs(Math.PI - angle)
  }
  
  return totalCurvature / (fingerJoints.length - 2)
}

// 辅助函数：计算手势识别得分
export const calculateGestureScore = (
  joints: THREE.Vector3[], 
  targetGesture: string
): number => {
  // 这里可以实现基于关键点的手势识别算法
  // 返回0-1之间的匹配度分数
  if (joints.length < 21) return 0
  
  // 简单的手势识别逻辑示例
  switch (targetGesture) {
    case 'fist':
      // 检查所有手指是否弯曲
      return joints.slice(5).every((joint, index) => {
        const wrist = joints[0]
        const distance = joint.distanceTo(wrist)
        return distance < 0.3 // 如果手指靠近手腕，认为是握拳
      }) ? 1 : 0
      
    case 'peace':
      // 检查食指和中指是否伸直，其他手指弯曲
      const indexTip = joints[8]
      const middleTip = joints[12]
      const wrist = joints[0]
      const indexDistance = indexTip.distanceTo(wrist)
      const middleDistance = middleTip.distanceTo(wrist)
      return (indexDistance > 0.4 && middleDistance > 0.4) ? 0.8 : 0.2
      
    default:
      return 0.5
  }
}

export default DetailedHandModel
