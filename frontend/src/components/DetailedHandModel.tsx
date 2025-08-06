/**
 * 精细手部模型组件 - 完全重构版本
 * 使用更真实的人类手部建模，告别"鸡爪"外观
 */

import React, { useRef, useEffect, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sphere, Cylinder, Box, RoundedBox } from '@react-three/drei'
import * as THREE from 'three'

// MediaPipe手部关键点索引映射
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
} as const

interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility?: number
}

interface DetailedHandModelProps {
  handedness: 'left' | 'right'
  keypoints?: HandKeypoint[]
  isActive?: boolean
  color?: string
  position?: [number, number, number]
  scale?: number
}

// 真实的手指组件 - 使用胶囊形状
const RealisticFinger: React.FC<{
  segments: THREE.Vector3[]
  fingerType: 'thumb' | 'index' | 'middle' | 'ring' | 'pinky'
  isActive: boolean
  baseColor: string
}> = ({ segments, fingerType, isActive, baseColor }) => {
  const fingerRef = useRef<THREE.Group>(null)
  
  // 根据手指类型获取参数
  const getFingerParams = (type: string) => {
    switch (type) {
      case 'thumb':
        return { baseRadius: 0.035, tipRadius: 0.025, segments: 16 }
      case 'index':
        return { baseRadius: 0.032, tipRadius: 0.020, segments: 16 }
      case 'middle':
        return { baseRadius: 0.035, tipRadius: 0.022, segments: 16 }
      case 'ring':
        return { baseRadius: 0.030, tipRadius: 0.018, segments: 16 }
      case 'pinky':
        return { baseRadius: 0.025, tipRadius: 0.015, segments: 16 }
      default:
        return { baseRadius: 0.030, tipRadius: 0.020, segments: 16 }
    }
  }
  
  const params = getFingerParams(fingerType)
  
  if (segments.length < 2) return null
  
  return (
    <group ref={fingerRef}>
      {segments.map((segment, index) => {
        if (index === segments.length - 1) return null
        
        const start = segments[index]
        const end = segments[index + 1]
        const direction = new THREE.Vector3().subVectors(end, start)
        const length = direction.length()
        const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5)
        
        // 计算半径 - 从根部到指尖逐渐变细
        const progress = index / (segments.length - 2)
        const currentRadius = params.baseRadius * (1 - progress * 0.4)
        const nextRadius = params.baseRadius * (1 - (progress + 0.2) * 0.4)
        
        // 计算旋转角度使圆柱体指向正确方向
        const axis = new THREE.Vector3(0, 1, 0)
        const quaternion = new THREE.Quaternion().setFromUnitVectors(axis, direction.normalize())
        const euler = new THREE.Euler().setFromQuaternion(quaternion)
        
        return (
          <group key={index}>
            {/* 手指节段 - 使用锥形圆柱体模拟胶囊 */}
            <group position={[midpoint.x, midpoint.y, midpoint.z]} rotation={[euler.x, euler.y, euler.z]}>
              {/* 主圆柱体 */}
              <Cylinder args={[nextRadius, currentRadius, length * 0.8, 12]}>
                <meshStandardMaterial
                  color={baseColor}
                  roughness={0.6}
                  metalness={0.1}
                  emissive={isActive ? baseColor : '#000000'}
                  emissiveIntensity={isActive ? 0.1 : 0}
                />
              </Cylinder>
              
              {/* 两端的半球体创造胶囊效果 */}
              <Sphere args={[currentRadius, 8, 8]} position={[0, -length * 0.4, 0]}>
                <meshStandardMaterial
                  color={baseColor}
                  roughness={0.6}
                  metalness={0.1}
                />
              </Sphere>
              <Sphere args={[nextRadius, 8, 8]} position={[0, length * 0.4, 0]}>
                <meshStandardMaterial
                  color={baseColor}
                  roughness={0.6}
                  metalness={0.1}
                />
              </Sphere>
            </group>
            
            {/* 关节球 - 更小更自然 */}
            <Sphere args={[currentRadius * 0.7, 12, 12]} position={[start.x, start.y, start.z]}>
              <meshStandardMaterial
                color={isActive ? '#ffb3ba' : baseColor}
                roughness={0.5}
                metalness={0.1}
                emissive={isActive ? '#ff6b6b' : '#000000'}
                emissiveIntensity={isActive ? 0.1 : 0}
              />
            </Sphere>
            
            {/* 指尖特殊处理 */}
            {index === segments.length - 2 && (
              <group position={[end.x, end.y, end.z]}>
                {/* 指尖 - 椭圆形更真实 */}
                <mesh scale={[1, 1, 1.2]}>
                  <sphereGeometry args={[nextRadius, 12, 12]} />
                  <meshStandardMaterial
                    color={baseColor}
                    roughness={0.5}
                    metalness={0.1}
                    emissive={isActive ? '#ff8a80' : '#000000'}
                    emissiveIntensity={isActive ? 0.2 : 0}
                  />
                </mesh>
                
                {/* 指甲 - 更真实的形状和位置 */}
                <RoundedBox 
                  args={[nextRadius * 1.1, nextRadius * 0.5, nextRadius * 0.2]} 
                  position={[0, 0, nextRadius * 0.8]}
                  radius={0.003}
                >
                  <meshStandardMaterial
                    color="#f5f5dc"
                    roughness={0.2}
                    metalness={0.3}
                    transparent
                    opacity={0.9}
                  />
                </RoundedBox>
                
                {/* 指纹纹理效果 */}
                <mesh position={[0, 0, nextRadius * 0.3]} scale={[0.8, 0.8, 0.3]}>
                  <sphereGeometry args={[nextRadius, 8, 8]} />
                  <meshStandardMaterial
                    color="#f4c2a1"
                    roughness={0.9}
                    metalness={0.0}
                    transparent
                    opacity={0.3}
                  />
                </mesh>
              </group>
            )}
          </group>
        )
      })}
    </group>
  )
}

// 真实的手掌组件
const RealisticPalm: React.FC<{
  keypoints: THREE.Vector3[]
  color: string
  isActive: boolean
  handedness: 'left' | 'right'
}> = ({ keypoints, color, isActive, handedness }) => {
  const palmRef = useRef<THREE.Group>(null)
  
  if (keypoints.length < 21) return null
  
  // 计算手掌关键位置
  const wrist = keypoints[HAND_LANDMARKS.WRIST]
  const indexMcp = keypoints[HAND_LANDMARKS.INDEX_MCP]
  const middleMcp = keypoints[HAND_LANDMARKS.MIDDLE_MCP]
  const ringMcp = keypoints[HAND_LANDMARKS.RING_MCP]
  const pinkyMcp = keypoints[HAND_LANDMARKS.PINKY_MCP]
  const thumbCmc = keypoints[HAND_LANDMARKS.THUMB_CMC]
  
  // 手掌中心
  const palmCenter = new THREE.Vector3()
    .addVectors(wrist, indexMcp)
    .add(middleMcp)
    .add(pinkyMcp)
    .divideScalar(4)
  
  return (
    <group ref={palmRef}>
      {/* 手掌主体 - 使用椭圆形状更贴合手掌 */}
      <group position={[palmCenter.x, palmCenter.y, palmCenter.z]}>
        {/* 手掌底层 - 椭圆形 */}
        <mesh scale={[1.2, 1, 0.8]}>
          <sphereGeometry args={[0.12, 16, 12]} />
          <meshStandardMaterial
            color={color}
            roughness={0.7}
            metalness={0.05}
            emissive={isActive ? color : '#000000'}
            emissiveIntensity={isActive ? 0.05 : 0}
          />
        </mesh>
        
        {/* 手掌表层 - 添加皮肤纹理感 */}
        <mesh position={[0, 0, 0.02]} scale={[1.15, 0.95, 0.7]}>
          <sphereGeometry args={[0.12, 12, 10]} />
          <meshStandardMaterial
            color="#f4c2a1"
            roughness={0.85}
            metalness={0.0}
            transparent
            opacity={0.4}
          />
        </mesh>
        
        {/* 手掌纹路 - 生命线 */}
        <mesh position={[-0.05, -0.03, 0.04]} scale={[0.8, 0.05, 0.1]}>
          <sphereGeometry args={[0.08, 8, 6]} />
          <meshStandardMaterial
            color="#e8b4a0"
            roughness={0.9}
            metalness={0.0}
            transparent
            opacity={0.6}
          />
        </mesh>
      </group>
      
      {/* 大鱼际肌 - 拇指根部的肌肉突起 */}
      <group position={[thumbCmc.x - 0.02, thumbCmc.y - 0.05, thumbCmc.z + 0.01]}>
        <mesh scale={[1.3, 1, 0.8]}>
          <sphereGeometry args={[0.04, 12, 10]} />
          <meshStandardMaterial
            color={color}
            roughness={0.8}
            metalness={0.0}
          />
        </mesh>
        
        {/* 大鱼际肌的高光部分 */}
        <mesh position={[0, 0, 0.015]} scale={[1.1, 0.8, 0.6]}>
          <sphereGeometry args={[0.035, 8, 8]} />
          <meshStandardMaterial
            color="#ffcc80"
            roughness={0.9}
            metalness={0.0}
            transparent
            opacity={0.5}
          />
        </mesh>
      </group>
      
      {/* 小鱼际肌 - 小指一侧的肌肉 */}
      <group position={[pinkyMcp.x + 0.015, pinkyMcp.y - 0.04, pinkyMcp.z]}>
        <mesh scale={[0.8, 1.2, 0.7]}>
          <sphereGeometry args={[0.025, 10, 8]} />
          <meshStandardMaterial
            color={color}
            roughness={0.8}
            metalness={0.0}
          />
        </mesh>
      </group>
      
      {/* 掌骨连接 - 更自然的骨骼结构 */}
      {[
        { mcp: indexMcp, name: 'index' },
        { mcp: middleMcp, name: 'middle' },
        { mcp: ringMcp, name: 'ring' },
        { mcp: pinkyMcp, name: 'pinky' }
      ].map((finger, index) => {
        const direction = new THREE.Vector3().subVectors(finger.mcp, wrist)
        const length = direction.length()
        const midpoint = new THREE.Vector3().addVectors(wrist, finger.mcp).multiplyScalar(0.5)
        
        // 根据手指位置调整掌骨粗细
        const baseRadius = index === 1 ? 0.018 : 0.015 // 中指掌骨最粗
        const tipRadius = index === 3 ? 0.010 : 0.012 // 小指最细
        
        return (
          <group key={finger.name}>
            {/* 掌骨主体 */}
            <mesh position={[midpoint.x, midpoint.y, midpoint.z]}>
              <cylinderGeometry args={[tipRadius, baseRadius, length * 0.8, 10]} />
              <meshStandardMaterial
                color={color}
                roughness={0.6}
                metalness={0.1}
                transparent
                opacity={0.7}
              />
            </mesh>
            
            {/* MCP关节 - 指关节突起 */}
            <Sphere args={[tipRadius * 1.5, 10, 10]} position={[finger.mcp.x, finger.mcp.y, finger.mcp.z]}>
              <meshStandardMaterial
                color={isActive ? '#ffb3ba' : color}
                roughness={0.5}
                metalness={0.1}
              />
            </Sphere>
          </group>
        )
      })}
      
      {/* 手腕连接部分 - 更精细的腕关节 */}
      <group position={[wrist.x, wrist.y - 0.02, wrist.z]}>
        {/* 主腕骨 */}
        <mesh>
          <cylinderGeometry args={[0.045, 0.05, 0.06, 16]} />
          <meshStandardMaterial
            color={color}
            roughness={0.5}
            metalness={0.1}
          />
        </mesh>
        
        {/* 腕关节细节 */}
        <mesh position={[0, -0.04, 0]}>
          <cylinderGeometry args={[0.04, 0.045, 0.03, 12]} />
          <meshStandardMaterial
            color="#f0a868"
            roughness={0.6}
            metalness={0.05}
          />
        </mesh>
        
        {/* 腕部肌腱突起 */}
        {[-0.02, 0.02].map((offset, index) => (
          <mesh key={index} position={[offset, 0, 0.025]} scale={[0.7, 1, 0.5]}>
            <sphereGeometry args={[0.015, 8, 6]} />
            <meshStandardMaterial
              color={color}
              roughness={0.7}
              metalness={0.0}
              transparent
              opacity={0.8}
            />
          </mesh>
        ))}
      </group>
      
      {/* 手掌纹路细节 */}
      <group position={[palmCenter.x, palmCenter.y, palmCenter.z + 0.035]}>
        {/* 生命线 */}
        <mesh position={[-0.04, -0.02, 0]} rotation={[0, 0, Math.PI / 6]} scale={[0.06, 0.003, 0.8]}>
          <cylinderGeometry args={[0.001, 0.002, 0.08, 6]} />
          <meshStandardMaterial
            color="#d4a574"
            roughness={0.9}
            metalness={0.0}
            transparent
            opacity={0.7}
          />
        </mesh>
        
        {/* 智慧线 */}
        <mesh position={[0, 0.01, 0]} rotation={[0, 0, -Math.PI / 12]} scale={[0.05, 0.003, 0.6]}>
          <cylinderGeometry args={[0.001, 0.002, 0.06, 6]} />
          <meshStandardMaterial
            color="#d4a574"
            roughness={0.9}
            metalness={0.0}
            transparent
            opacity={0.6}
          />
        </mesh>
        
        {/* 感情线 */}
        <mesh position={[0.02, 0.04, 0]} rotation={[0, 0, -Math.PI / 8]} scale={[0.04, 0.003, 0.5]}>
          <cylinderGeometry args={[0.001, 0.0015, 0.05, 6]} />
          <meshStandardMaterial
            color="#d4a574"
            roughness={0.9}
            metalness={0.0}
            transparent
            opacity={0.5}
          />
        </mesh>
      </group>
    </group>
  )
}

// 主手部模型组件
const DetailedHandModel: React.FC<DetailedHandModelProps> = ({
  handedness,
  keypoints,
  isActive = false,
  color = '#fdbcb4',
  position = [0, 0, 0],
  scale = 1
}) => {
  const handRef = useRef<THREE.Group>(null)
  const [normalizedKeypoints, setNormalizedKeypoints] = useState<THREE.Vector3[]>([])
  
  // 处理关键点数据
  useEffect(() => {
    if (keypoints && keypoints.length === 21) {
      const normalized = keypoints.map(kp => {
        const x = (kp.x - 0.5) * 2 * scale
        const y = (0.5 - kp.y) * 2 * scale
        const z = kp.z * scale
        return new THREE.Vector3(x, y, z)
      })
      setNormalizedKeypoints(normalized)
    } else {
      setNormalizedKeypoints(getRealisticHandPose(scale))
    }
  }, [keypoints, scale])
  
  // 自然的呼吸动画
  useFrame((state) => {
    if (handRef.current && isActive) {
      const breathe = Math.sin(state.clock.elapsedTime * 1.2) * 0.01 + 1
      handRef.current.scale.setScalar(breathe)
      
      const sway = Math.sin(state.clock.elapsedTime * 0.6) * 0.015
      handRef.current.rotation.z = sway
    }
  })
  
  // 提取手指路径
  const getFingerPath = (landmarks: number[]): THREE.Vector3[] => {
    return landmarks.map(idx => normalizedKeypoints[idx]).filter(Boolean)
  }
  
  if (normalizedKeypoints.length === 0) return null
  
  return (
    <group ref={handRef} position={position}>
      {/* 手掌 */}
      <RealisticPalm
        keypoints={normalizedKeypoints}
        color={color}
        isActive={isActive}
        handedness={handedness}
      />
      
      {/* 拇指 */}
      <RealisticFinger
        segments={getFingerPath([
          HAND_LANDMARKS.THUMB_CMC,
          HAND_LANDMARKS.THUMB_MCP,
          HAND_LANDMARKS.THUMB_IP,
          HAND_LANDMARKS.THUMB_TIP
        ])}
        fingerType="thumb"
        isActive={isActive}
        baseColor={color}
      />
      
      {/* 食指 */}
      <RealisticFinger
        segments={getFingerPath([
          HAND_LANDMARKS.INDEX_MCP,
          HAND_LANDMARKS.INDEX_PIP,
          HAND_LANDMARKS.INDEX_DIP,
          HAND_LANDMARKS.INDEX_TIP
        ])}
        fingerType="index"
        isActive={isActive}
        baseColor={color}
      />
      
      {/* 中指 */}
      <RealisticFinger
        segments={getFingerPath([
          HAND_LANDMARKS.MIDDLE_MCP,
          HAND_LANDMARKS.MIDDLE_PIP,
          HAND_LANDMARKS.MIDDLE_DIP,
          HAND_LANDMARKS.MIDDLE_TIP
        ])}
        fingerType="middle"
        isActive={isActive}
        baseColor={color}
      />
      
      {/* 无名指 */}
      <RealisticFinger
        segments={getFingerPath([
          HAND_LANDMARKS.RING_MCP,
          HAND_LANDMARKS.RING_PIP,
          HAND_LANDMARKS.RING_DIP,
          HAND_LANDMARKS.RING_TIP
        ])}
        fingerType="ring"
        isActive={isActive}
        baseColor={color}
      />
      
      {/* 小指 */}
      <RealisticFinger
        segments={getFingerPath([
          HAND_LANDMARKS.PINKY_MCP,
          HAND_LANDMARKS.PINKY_PIP,
          HAND_LANDMARKS.PINKY_DIP,
          HAND_LANDMARKS.PINKY_TIP
        ])}
        fingerType="pinky"
        isActive={isActive}
        baseColor={color}
      />
    </group>
  )
}

// 更真实的默认手势
const getRealisticHandPose = (scale: number): THREE.Vector3[] => {
  const pose = [
    // 手腕
    new THREE.Vector3(0, 0, 0),
    
    // 拇指链 - 更自然的角度
    new THREE.Vector3(-0.20, -0.12, 0.08),
    new THREE.Vector3(-0.28, -0.08, 0.12),
    new THREE.Vector3(-0.34, -0.04, 0.16),
    new THREE.Vector3(-0.38, 0, 0.18),
    
    // 食指链 - 轻微弯曲
    new THREE.Vector3(-0.06, 0.25, 0.04),
    new THREE.Vector3(-0.05, 0.38, 0.06),
    new THREE.Vector3(-0.04, 0.48, 0.07),
    new THREE.Vector3(-0.03, 0.56, 0.08),
    
    // 中指链 - 最长，微弯
    new THREE.Vector3(0.02, 0.28, 0.02),
    new THREE.Vector3(0.03, 0.44, 0.03),
    new THREE.Vector3(0.04, 0.58, 0.04),
    new THREE.Vector3(0.05, 0.68, 0.05),
    
    // 无名指链 - 跟随中指
    new THREE.Vector3(0.10, 0.26, 0),
    new THREE.Vector3(0.11, 0.40, 0.01),
    new THREE.Vector3(0.12, 0.52, 0.02),
    new THREE.Vector3(0.13, 0.62, 0.03),
    
    // 小指链 - 最短，更弯曲
    new THREE.Vector3(0.18, 0.20, -0.02),
    new THREE.Vector3(0.19, 0.30, -0.01),
    new THREE.Vector3(0.20, 0.38, 0),
    new THREE.Vector3(0.21, 0.44, 0.01),
  ]
  
  return pose.map(point => point.multiplyScalar(scale))
}

export default DetailedHandModel
export { HAND_LANDMARKS }
