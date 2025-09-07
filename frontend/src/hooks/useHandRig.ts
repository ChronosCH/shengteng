import { useRef, useEffect } from 'react'
import * as THREE from 'three'

// 手部骨骼映射配置
const HAND_BONE_MAPPING = {
  left: {
    wrist: 'LeftHand',
    thumb: ['LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3'],
    index: ['LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3'],
    middle: ['LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3'],
    ring: ['LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3'],
    pinky: ['LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']
  },
  right: {
    wrist: 'RightHand',
    thumb: ['RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3'],
    index: ['RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3'],
    middle: ['RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'],
    ring: ['RightHandRing1', 'RightHandRing2', 'RightHandRing3'],
    pinky: ['RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3']
  }
}

// MediaPipe手部关键点索引
const MEDIAPIPE_HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20
}

// 改进的平滑函数
const smoothHandKeypoints = (prev: number[][] | undefined, current: number[][], alpha: number = 0.3): number[][] => {
  if (!prev || prev.length !== current.length) {
    return current
  }

  return current.map((point, i) => {
    const prevPoint = prev[i]
    return [
      prevPoint[0] * (1 - alpha) + point[0] * alpha,
      prevPoint[1] * (1 - alpha) + point[1] * alpha,
      prevPoint[2] * (1 - alpha) + point[2] * alpha
    ]
  })
}

export interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility?: number
}

interface UseHandRigParams {
  skeleton: THREE.Skeleton | null
  leftHand?: HandKeypoint[]
  rightHand?: HandKeypoint[]
  scale?: number
}

export function useHandRig({ skeleton, leftHand, rightHand, scale = 1 }: UseHandRigParams) {
  const prevLeft = useRef<number[][]>()
  const prevRight = useRef<number[][]>()
  const lastUpdateTime = useRef<number>(0)

  useEffect(() => {
    if (!skeleton) return

    try {
      const now = Date.now()
      // 限制更新频率，避免过度计算
      if (now - lastUpdateTime.current < 16) { // ~60fps
        return
      }
      lastUpdateTime.current = now

      // 处理左手
      if (leftHand && leftHand.length === 21) {
        const arr = leftHand.map(p => [p.x, p.y, p.z])
        const smooth = smoothHandKeypoints(prevLeft.current, arr, 0.3)
        prevLeft.current = smooth
        applyHandToSkeleton(skeleton, smooth, 'left', scale)
      }

      // 处理右手
      if (rightHand && rightHand.length === 21) {
        const arr = rightHand.map(p => [p.x, p.y, p.z])
        const smooth = smoothHandKeypoints(prevRight.current, arr, 0.3)
        prevRight.current = smooth
        applyHandToSkeleton(skeleton, smooth, 'right', scale)
      }
    } catch (error) {
      console.warn('Hand rig update error:', error)
    }
  }, [skeleton, leftHand, rightHand, scale])
}

function applyHandToSkeleton(skeleton: THREE.Skeleton, keypoints: number[][], side: 'left' | 'right', scale: number) {
  try {
    if (!skeleton || !skeleton.bones) {
      return
    }

    const mapping = HAND_BONE_MAPPING[side]
    const bones = skeleton.bones

    // 查找骨骼的辅助函数
    const findBone = (name: string) => bones.find(bone =>
      bone.name.toLowerCase().includes(name.toLowerCase())
    )

    // 计算手指弯曲角度
    const calculateFingerBend = (fingerKeypoints: number[][], fingerName: keyof typeof mapping) => {
      if (fingerName === 'wrist' || !Array.isArray(mapping[fingerName])) return 0

      try {
        const joints = mapping[fingerName] as string[]
        if (fingerKeypoints.length < 4) return 0

        // 计算手指的弯曲程度
        const tip = fingerKeypoints[fingerKeypoints.length - 1]
        const base = fingerKeypoints[0]

        // 简单的弯曲计算：基于tip和base的距离
        const distance = Math.sqrt(
          Math.pow(tip[0] - base[0], 2) +
          Math.pow(tip[1] - base[1], 2) +
          Math.pow(tip[2] - base[2], 2)
        )

        // 归一化弯曲角度 (0-1)
        return Math.max(0, Math.min(1, (0.1 - distance) / 0.05))
      } catch (error) {
        return 0
      }
    }

    // 应用手指动画
    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const

    fingers.forEach(fingerName => {
      try {
        const fingerBones = mapping[fingerName] as string[]
        if (!Array.isArray(fingerBones)) return

        // 获取对应的关键点
        let fingerKeypoints: number[][]
        switch (fingerName) {
          case 'thumb':
            fingerKeypoints = keypoints.slice(1, 5) // 拇指关键点
            break
          case 'index':
            fingerKeypoints = keypoints.slice(5, 9) // 食指关键点
            break
          case 'middle':
            fingerKeypoints = keypoints.slice(9, 13) // 中指关键点
            break
          case 'ring':
            fingerKeypoints = keypoints.slice(13, 17) // 无名指关键点
            break
          case 'pinky':
            fingerKeypoints = keypoints.slice(17, 21) // 小指关键点
            break
          default:
            return
        }

        const bendAmount = calculateFingerBend(fingerKeypoints, fingerName)

        // 应用到骨骼
        fingerBones.forEach((boneName, index) => {
          const bone = findBone(boneName)
          if (bone) {
            // 根据弯曲程度调整骨骼旋转
            const rotationAmount = bendAmount * Math.PI * 0.3 * (index + 1) / fingerBones.length
            bone.rotation.z = side === 'left' ? -rotationAmount : rotationAmount
          }
        })
      } catch (error) {
        console.warn(`Error applying ${fingerName} animation:`, error)
      }
    })

    // 更新骨骼矩阵
    skeleton.bones.forEach(bone => {
      bone.updateMatrixWorld(true)
    })

  } catch (error) {
    console.warn('Error in applyHandToSkeleton:', error)
  }
}
