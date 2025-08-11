/**
 * 详细手部模型组件 - 修复版
 * 解决纹理加载和渲染错误问题
 */

import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sphere, Cylinder, RoundedBox, Text } from '@react-three/drei'
import * as THREE from 'three'

interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility?: number
}

interface DetailedHandModelProps {
  handedness: 'left' | 'right'
  keypoints?: HandKeypoint[]
  isActive: boolean
  position: [number, number, number]
  scale: number
  color?: string
  gestureMode?: 'natural' | 'fist' | 'point' | 'ok' | 'peace' | 'wave'
  enableAnimations?: boolean
  showBones?: boolean
  showLabels?: boolean
  enablePhysics?: boolean
  detailLevel?: number
  onGestureComplete?: (gesture: string) => void
  onHandTracked?: (data: any) => void
}

// 手指配置
const FINGER_CONFIG = {
  thumb: { 
    joints: [1, 2, 3, 4], 
    name: '拇指',
    baseRadius: 0.04,
    segments: ['近节', '中节', '远节', '指尖']
  },
  index: { 
    joints: [5, 6, 7, 8], 
    name: '食指',
    baseRadius: 0.035,
    segments: ['掌指关节', '近节', '中节', '远节']
  },
  middle: { 
    joints: [9, 10, 11, 12], 
    name: '中指',
    baseRadius: 0.038,
    segments: ['掌指关节', '近节', '中节', '远节']
  },
  ring: { 
    joints: [13, 14, 15, 16], 
    name: '无名指',
    baseRadius: 0.033,
    segments: ['掌指关节', '近节', '中节', '远节']
  },
  pinky: { 
    joints: [17, 18, 19, 20], 
    name: '小指',
    baseRadius: 0.028,
    segments: ['掌指关节', '近节', '中节', '远节']
  }
}

// 生成默认的自然手势姿态
const generateNaturalHandPose = (scale: number): THREE.Vector3[] => {
  const basePositions = [
    // 手腕 (0)
    [0, 0, 0],
    
    // 拇指链 (1-4)
    [-0.1 * scale, -0.05 * scale, 0.05 * scale],  // 拇指掌指关节
    [-0.15 * scale, -0.08 * scale, 0.08 * scale], // 拇指近节
    [-0.18 * scale, -0.1 * scale, 0.1 * scale],   // 拇指中节
    [-0.2 * scale, -0.12 * scale, 0.12 * scale],  // 拇指远节
    
    // 食指链 (5-8)
    [0.05 * scale, 0.2 * scale, 0.02 * scale],    // 食指掌指关节
    [0.08 * scale, 0.35 * scale, 0.02 * scale],   // 食指近节
    [0.09 * scale, 0.45 * scale, 0.01 * scale],   // 食指中节
    [0.1 * scale, 0.5 * scale, 0],                // 食指远节
    
    // 中指链 (9-12)
    [0, 0.22 * scale, 0],                         // 中指掌指关节
    [0, 0.4 * scale, 0],                          // 中指近节
    [0, 0.52 * scale, 0],                         // 中指中节
    [0, 0.58 * scale, 0],                         // 中指远节
    
    // 无名指链 (13-16)
    [-0.05 * scale, 0.2 * scale, -0.01 * scale],  // 无名指掌指关节
    [-0.08 * scale, 0.35 * scale, -0.02 * scale], // 无名指近节
    [-0.09 * scale, 0.45 * scale, -0.02 * scale], // 无名指中节
    [-0.1 * scale, 0.5 * scale, -0.02 * scale],   // 无名指远节
    
    // 小指链 (17-20)
    [-0.1 * scale, 0.15 * scale, -0.02 * scale],  // 小指掌指关节
    [-0.12 * scale, 0.25 * scale, -0.03 * scale], // 小指近节
    [-0.13 * scale, 0.32 * scale, -0.03 * scale], // 小指中节
    [-0.14 * scale, 0.36 * scale, -0.03 * scale]  // 小指远节
  ]
  
  return basePositions.map(([x, y, z]) => new THREE.Vector3(x, y, z))
}

// 手势预设
const GESTURE_PRESETS = {
  natural: generateNaturalHandPose,
  fist: (scale: number) => {
    const joints = generateNaturalHandPose(scale)
    // 弯曲所有手指形成拳头
    const bendFactor = 0.6
    for (let i = 5; i <= 20; i++) {
      if (joints[i]) {
        joints[i].y *= bendFactor
        joints[i].z += 0.02 * scale
      }
    }
    return joints
  },
  point: (scale: number) => {
    const joints = generateNaturalHandPose(scale)
    // 只保持食指伸直，其他手指弯曲
    for (let i = 9; i <= 20; i++) {
      if (joints[i]) {
        joints[i].y *= 0.4
        joints[i].z += 0.03 * scale
      }
    }
    return joints
  },
  ok: (scale: number) => {
    const joints = generateNaturalHandPose(scale)
    // 拇指和食指形成圆圈
    joints[4].x = joints[8].x - 0.05 * scale
    joints[4].y = joints[8].y - 0.05 * scale
    return joints
  },
  peace: (scale: number) => {
    const joints = generateNaturalHandPose(scale)
    // 食指和中指伸直成V字形
    for (let i = 13; i <= 20; i++) {
      if (joints[i]) {
        joints[i].y *= 0.4
        joints[i].z += 0.03 * scale
      }
    }
    return joints
  },
  wave: (scale: number) => {
    return generateNaturalHandPose(scale) // 基础姿态，动画时会添加波浪效果
  }
}

const DetailedHandModel: React.FC<DetailedHandModelProps> = ({
  handedness,
  keypoints,
  isActive,
  position,
  scale,
  color = "#fdbcb4",
  gestureMode = 'natural',
  enableAnimations = true,
  showBones = false,
  showLabels = false,
  enablePhysics = false,
  detailLevel = 1,
  onGestureComplete,
  onHandTracked
}) => {
  const handRef = useRef<THREE.Group>(null)
  const [joints, setJoints] = useState<THREE.Vector3[]>([])
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentGesture, setCurrentGesture] = useState<string>(gestureMode)

  // 材质管理
  const handMaterials = useMemo(() => {
    const createMaterial = (baseColor: string, roughness = 0.4, metalness = 0.1) => {
      return new THREE.MeshStandardMaterial({
        color: new THREE.Color(baseColor),
        roughness,
        metalness,
        transparent: true,
        opacity: 0.95
      })
    }

    return {
      palm: createMaterial(isActive ? color : "#ffb3a7", 0.5, 0.1),
      finger: createMaterial(isActive ? color : "#ffb3a7", 0.4, 0.05),
      joint: createMaterial(isActive ? "#ff9999" : "#ffcccc", 0.3, 0.2),
      bone: createMaterial("#ffffff", 0.8, 0.0),
      wrist: createMaterial(isActive ? color : "#ffb3a7", 0.6, 0.1)
    }
  }, [isActive, color])

  // 处理关键点数据
  useEffect(() => {
    try {
      let newJoints: THREE.Vector3[]

      if (keypoints && keypoints.length === 21) {
        // 使用真实关键点数据
        newJoints = keypoints.map((kp, index) => {
          let x = (kp.x - 0.5) * scale * 2
          const y = (0.5 - kp.y) * scale * 2  
          const z = kp.z * scale
          
          if (handedness === 'left') {
            x = -x
          }
          
          return new THREE.Vector3(x, y, z)
        })
      } else {
        // 使用手势预设
        const gestureFunction = GESTURE_PRESETS[gestureMode as keyof typeof GESTURE_PRESETS] || GESTURE_PRESETS.natural
        newJoints = gestureFunction(scale)
        
        // 左手镜像
        if (handedness === 'left') {
          newJoints = newJoints.map(joint => new THREE.Vector3(-joint.x, joint.y, joint.z))
        }
      }
      
      setJoints(newJoints)
      
      // 通知手部追踪
      if (onHandTracked) {
        onHandTracked({
          handedness,
          joints: newJoints,
          gesture: gestureMode,
          confidence: keypoints ? 0.9 : 0.7
        })
      }
    } catch (error) {
      console.warn('Error processing keypoints:', error)
      setJoints(generateNaturalHandPose(scale))
    }
  }, [keypoints, scale, handedness, gestureMode, onHandTracked])

  // 动画循环
  useFrame((state) => {
    if (!handRef.current || !isActive) return
    
    const time = state.clock.elapsedTime
    
    if (enableAnimations && !isAnimating) {
      // 自然呼吸效果
      const breathingFactor = 1 + Math.sin(time * 1.2) * 0.01
      handRef.current.scale.setScalar(breathingFactor)
      
      // 微小的自然颤动
      handRef.current.rotation.x = Math.sin(time * 0.5) * 0.01
      handRef.current.rotation.z = Math.sin(time * 0.8) * 0.02
      
      // 波浪手势特殊动画
      if (gestureMode === 'wave') {
        handRef.current.rotation.z = Math.sin(time * 3) * 0.3
      }
    }
  })

  // 渲染手指
  const renderFinger = useCallback((fingerName: keyof typeof FINGER_CONFIG) => {
    const config = FINGER_CONFIG[fingerName]
    const fingerJoints = config.joints.map(idx => joints[idx]).filter(Boolean)
    
    if (fingerJoints.length < 4) return null
    
    return (
      <group key={fingerName}>
        {fingerJoints.map((joint, index) => {
          if (index === fingerJoints.length - 1) return null
          
          const current = fingerJoints[index]
          const next = fingerJoints[index + 1]
          const direction = new THREE.Vector3().subVectors(next, current)
          const length = direction.length()
          const center = new THREE.Vector3().addVectors(current, next).multiplyScalar(0.5)
          
          // 计算旋转
          const quaternion = new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.normalize()
          )
          const euler = new THREE.Euler().setFromQuaternion(quaternion)
          
          // 渐变半径
          const progress = index / (fingerJoints.length - 2)
          const radius = config.baseRadius * (1 - progress * 0.3) * scale
          
          return (
            <group key={index}>
              {/* 手指段 */}
              <group 
                position={[center.x, center.y, center.z]} 
                rotation={[euler.x, euler.y, euler.z]}
              >
                <Cylinder args={[radius * 0.8, radius, length * 0.9, 12]}>
                  <primitive object={handMaterials.finger.clone()} />
                </Cylinder>
              </group>
              
              {/* 关节 */}
              <Sphere 
                args={[radius * 0.6, 12, 12]} 
                position={[current.x, current.y, current.z]}
              >
                <primitive object={handMaterials.joint.clone()} />
              </Sphere>
              
              {/* 指尖 */}
              {index === fingerJoints.length - 2 && (
                <Sphere 
                  args={[radius * 0.9, 12, 12]} 
                  position={[next.x, next.y, next.z]}
                >
                  <primitive object={handMaterials.finger.clone()} />
                </Sphere>
              )}
              
              {/* 骨骼可视化 */}
              {showBones && (
                <group
                  position={[center.x, center.y, center.z]} 
                  rotation={[euler.x, euler.y, euler.z]}
                >
                  <Cylinder args={[0.002, 0.002, length, 6]}>
                    <primitive object={handMaterials.bone.clone()} />
                  </Cylinder>
                </group>
              )}
              
              {/* 标签 */}
              {showLabels && index === 0 && (
                <Text
                  position={[current.x, current.y + 0.05, current.z]}
                  fontSize={0.02}
                  color="#333333"
                  anchorX="center"
                  anchorY="middle"
                >
                  {config.name}
                </Text>
              )}
            </group>
          )
        })}
      </group>
    )
  }, [joints, handMaterials, scale, showBones, showLabels])

  // 渲染手掌
  const renderPalm = useCallback(() => {
    if (joints.length < 21) return null
    
    try {
      const wrist = joints[0]
      const indexMcp = joints[5]
      const middleMcp = joints[9]
      const ringMcp = joints[13]
      const pinkyMcp = joints[17]
      
      if (!wrist || !indexMcp || !middleMcp || !ringMcp || !pinkyMcp) return null
      
      const palmCenter = new THREE.Vector3()
        .addVectors(indexMcp, pinkyMcp)
        .add(middleMcp)
        .add(ringMcp)
        .multiplyScalar(0.25)
      
      return (
        <group>
          {/* 主手掌 */}
          <group position={[palmCenter.x, palmCenter.y, palmCenter.z]}>
            <RoundedBox args={[0.15 * scale, 0.18 * scale, 0.06 * scale]} radius={0.02} smoothness={4}>
              <primitive object={handMaterials.palm.clone()} />
            </RoundedBox>
          </group>
          
          {/* 手腕 */}
          <Sphere args={[0.08 * scale, 16, 16]} position={[wrist.x, wrist.y, wrist.z]}>
            <primitive object={handMaterials.wrist.clone()} />
          </Sphere>
          
          {/* 手掌连接线（骨骼） */}
          {showBones && (
            <>
              {[indexMcp, middleMcp, ringMcp, pinkyMcp].map((mcp, idx) => (
                <group key={idx}>
                  <Cylinder 
                    args={[0.002, 0.002, palmCenter.distanceTo(mcp), 6]}
                    position={[
                      (palmCenter.x + mcp.x) / 2,
                      (palmCenter.y + mcp.y) / 2,
                      (palmCenter.z + mcp.z) / 2
                    ]}
                  >
                    <primitive object={handMaterials.bone.clone()} />
                  </Cylinder>
                </group>
              ))}
            </>
          )}
          
          {/* 手掌标签 */}
          {showLabels && (
            <Text
              position={[palmCenter.x, palmCenter.y - 0.1, palmCenter.z]}
              fontSize={0.03}
              color="#333333"
              anchorX="center"
              anchorY="middle"
            >
              {handedness === 'left' ? '左手' : '右手'}
            </Text>
          )}
        </group>
      )
    } catch (error) {
      console.warn('Error rendering palm:', error)
      return (
        <Sphere args={[0.1 * scale]} position={position}>
          <meshBasicMaterial color={color} />
        </Sphere>
      )
    }
  }, [joints, handMaterials, scale, showBones, showLabels, handedness, position, color])

  // 手势切换动画
  const performGestureAnimation = useCallback(async (targetGesture: string) => {
    if (isAnimating || !handRef.current) return
    
    setIsAnimating(true)
    
    try {
      const gestureFunction = GESTURE_PRESETS[targetGesture as keyof typeof GESTURE_PRESETS]
      if (!gestureFunction) return
      
      const targetJoints = gestureFunction(scale)
      const startJoints = [...joints]
      const duration = 1000 // 1秒动画
      const startTime = Date.now()
      
      const animate = () => {
        const elapsed = Date.now() - startTime
        const progress = Math.min(elapsed / duration, 1)
        const easeProgress = 1 - Math.pow(1 - progress, 3) // ease-out cubic
        
        const interpolatedJoints = startJoints.map((start, index) => {
          const target = targetJoints[index]
          if (!target) return start
          
          return new THREE.Vector3().lerpVectors(start, target, easeProgress)
        })
        
        setJoints(interpolatedJoints)
        
        if (progress < 1) {
          requestAnimationFrame(animate)
        } else {
          setCurrentGesture(targetGesture)
          setIsAnimating(false)
          onGestureComplete?.(targetGesture)
        }
      }
      
      animate()
    } catch (error) {
      console.warn('Error performing gesture animation:', error)
      setIsAnimating(false)
    }
  }, [isAnimating, joints, scale, onGestureComplete])

  // 监听手势模式变化
  useEffect(() => {
    if (gestureMode !== currentGesture && enableAnimations) {
      performGestureAnimation(gestureMode)
    }
  }, [gestureMode, currentGesture, enableAnimations, performGestureAnimation])

  // 如果没有关键点数据，返回简化版本
  if (joints.length === 0) {
    return (
      <group ref={handRef} position={position}>
        <Sphere args={[0.1 * scale, 8, 8]}>
          <meshBasicMaterial color={color} transparent opacity={0.5} />
        </Sphere>
        {showLabels && (
          <Text
            position={[0, 0.15 * scale, 0]}
            fontSize={0.03}
            color="#666666"
            anchorX="center"
            anchorY="middle"
          >
            加载中...
          </Text>
        )}
      </group>
    )
  }
  
  return (
    <group ref={handRef} position={position}>
      {/* 手掌 */}
      {renderPalm()}
      
      {/* 手指 */}
      {Object.keys(FINGER_CONFIG).map(fingerName => 
        renderFinger(fingerName as keyof typeof FINGER_CONFIG)
      )}
      
      {/* 调试信息 */}
      {showLabels && (
        <Text
          position={[0, -0.2 * scale, 0]}
          fontSize={0.02}
          color="#666666"
          anchorX="center"
          anchorY="middle"
        >
          {`${handedness} | ${currentGesture} | ${joints.length}点`}
        </Text>
      )}
    </group>
  )
}

export default DetailedHandModel
