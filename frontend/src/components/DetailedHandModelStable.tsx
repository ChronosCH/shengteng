/**
 * 简化但稳定的手部模型组件
 * 解决纹理加载和渲染错误问题
 */

import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sphere, Cylinder, RoundedBox } from '@react-three/drei'
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
  onGestureComplete?: (gesture: string) => void
}

// 生成默认的自然手势姿态
const generateNaturalHandPose = (scale: number): THREE.Vector3[] => {
  const basePositions = [
    // 手腕
    [0, 0, 0],
    
    // 拇指链
    [-0.1 * scale, -0.05 * scale, 0.05 * scale],
    [-0.15 * scale, -0.08 * scale, 0.08 * scale],
    [-0.18 * scale, -0.1 * scale, 0.1 * scale],
    [-0.2 * scale, -0.12 * scale, 0.12 * scale],
    
    // 食指链
    [0.05 * scale, 0.2 * scale, 0.02 * scale],
    [0.08 * scale, 0.35 * scale, 0.02 * scale],
    [0.09 * scale, 0.45 * scale, 0.01 * scale],
    [0.1 * scale, 0.5 * scale, 0],
    
    // 中指链
    [0, 0.22 * scale, 0],
    [0, 0.4 * scale, 0],
    [0, 0.52 * scale, 0],
    [0, 0.58 * scale, 0],
    
    // 无名指链
    [-0.05 * scale, 0.2 * scale, -0.01 * scale],
    [-0.08 * scale, 0.35 * scale, -0.02 * scale],
    [-0.09 * scale, 0.45 * scale, -0.02 * scale],
    [-0.1 * scale, 0.5 * scale, -0.02 * scale],
    
    // 小指链
    [-0.1 * scale, 0.15 * scale, -0.02 * scale],
    [-0.12 * scale, 0.25 * scale, -0.03 * scale],
    [-0.13 * scale, 0.32 * scale, -0.03 * scale],
    [-0.14 * scale, 0.36 * scale, -0.03 * scale]
  ]
  
  return basePositions.map(([x, y, z]) => new THREE.Vector3(x, y, z))
}

const DetailedHandModelStable: React.FC<DetailedHandModelProps> = ({
  handedness,
  keypoints,
  isActive,
  position,
  scale,
  color = "#fdbcb4",
  gestureMode = 'natural',
  enableAnimations = true,
  onGestureComplete
}) => {
  const handRef = useRef<THREE.Group>(null)
  const [joints, setJoints] = useState<THREE.Vector3[]>([])
  const [isAnimating, setIsAnimating] = useState(false)

  // 手指配置
  const fingerConfigs = useMemo(() => ({
    thumb: { joints: [1, 2, 3, 4], baseRadius: 0.04 },
    index: { joints: [5, 6, 7, 8], baseRadius: 0.035 },
    middle: { joints: [9, 10, 11, 12], baseRadius: 0.038 },
    ring: { joints: [13, 14, 15, 16], baseRadius: 0.033 },
    pinky: { joints: [17, 18, 19, 20], baseRadius: 0.028 }
  }), [])

  // 处理关键点数据
  useEffect(() => {
    try {
      if (keypoints && keypoints.length === 21) {
        const processedJoints = keypoints.map((kp, index) => {
          let x = (kp.x - 0.5) * scale * 2
          const y = (0.5 - kp.y) * scale * 2  
          const z = kp.z * scale
          
          if (handedness === 'left') {
            x = -x
          }
          
          return new THREE.Vector3(x, y, z)
        })
        setJoints(processedJoints)
      } else {
        let defaultJoints = generateNaturalHandPose(scale)
        
        if (handedness === 'left') {
          defaultJoints = defaultJoints.map(joint => new THREE.Vector3(-joint.x, joint.y, joint.z))
        }
        
        setJoints(defaultJoints)
      }
    } catch (error) {
      console.warn('Error processing keypoints:', error)
      setJoints(generateNaturalHandPose(scale))
    }
  }, [keypoints, scale, handedness])

  // 简化的动画
  useFrame((state) => {
    if (!handRef.current || !isActive || isAnimating) return
    
    const time = state.clock.elapsedTime
    const breathingFactor = 1 + Math.sin(time * 1.2) * 0.01
    
    handRef.current.scale.setScalar(breathingFactor)
    handRef.current.rotation.x = Math.sin(time * 0.5) * 0.01
    handRef.current.rotation.z = Math.sin(time * 0.8) * 0.02
  })

  // 基础材质
  const baseMaterial = useMemo(() => (
    <meshStandardMaterial
      color={isActive ? color : "#ffb3a7"}
      roughness={0.4}
      metalness={0.1}
      transparent
      opacity={0.95}
    />
  ), [isActive, color])

  // 渲染手指
  const renderFinger = useCallback((fingerName: keyof typeof fingerConfigs) => {
    const config = fingerConfigs[fingerName]
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
          
          const quaternion = new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.normalize()
          )
          const euler = new THREE.Euler().setFromQuaternion(quaternion)
          
          const progress = index / (fingerJoints.length - 2)
          const radius = config.baseRadius * (1 - progress * 0.3)
          
          return (
            <group key={index}>
              {/* 手指段 */}
              <group position={[center.x, center.y, center.z]} rotation={[euler.x, euler.y, euler.z]}>
                <Cylinder args={[radius * 0.8, radius, length * 0.9, 12]}>
                  {baseMaterial}
                </Cylinder>
              </group>
              
              {/* 关节 */}
              <Sphere args={[radius * 0.6, 12, 12]} position={[current.x, current.y, current.z]}>
                {baseMaterial}
              </Sphere>
              
              {/* 指尖 */}
              {index === fingerJoints.length - 2 && (
                <Sphere args={[radius * 0.9, 12, 12]} position={[next.x, next.y, next.z]}>
                  {baseMaterial}
                </Sphere>
              )}
            </group>
          )
        })}
      </group>
    )
  }, [fingerConfigs, joints, baseMaterial])

  // 渲染手掌
  const renderPalm = useCallback(() => {
    if (joints.length < 21) return null
    
    const wrist = joints[0]
    const indexMcp = joints[5]
    const middleMcp = joints[9]
    const ringMcp = joints[13]
    const pinkyMcp = joints[17]
    
    const palmCenter = new THREE.Vector3()
      .addVectors(indexMcp, pinkyMcp)
      .add(middleMcp)
      .add(ringMcp)
      .multiplyScalar(0.25)
    
    return (
      <group>
        {/* 主手掌 */}
        <group position={[palmCenter.x, palmCenter.y, palmCenter.z]}>
          <RoundedBox args={[0.15, 0.18, 0.06]} radius={0.02} smoothness={4}>
            {baseMaterial}
          </RoundedBox>
        </group>
        
        {/* 手腕 */}
        <Sphere args={[0.08, 12, 12]} position={[wrist.x, wrist.y, wrist.z]}>
          {baseMaterial}
        </Sphere>
      </group>
    )
  }, [joints, baseMaterial])

  // 如果没有关键点数据，返回简化版本
  if (joints.length === 0) {
    return (
      <group ref={handRef} position={position}>
        <Sphere args={[0.1, 8, 8]}>
          <meshBasicMaterial color={color} />
        </Sphere>
      </group>
    )
  }
  
  return (
    <group ref={handRef} position={position}>
      {/* 手掌 */}
      {renderPalm()}
      
      {/* 手指 */}
      {Object.keys(fingerConfigs).map(fingerName => 
        renderFinger(fingerName as keyof typeof fingerConfigs)
      )}
    </group>
  )
}

export default DetailedHandModelStable
