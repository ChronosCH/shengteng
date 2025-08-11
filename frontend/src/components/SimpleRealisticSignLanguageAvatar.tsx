/**
 * 简化版高质量3D Avatar组件 - 修复渲染错误
 * 特点：
 * 1. 简化的人体建模，避免渲染错误
 * 2. 优化的手部结构
 * 3. 稳定的渲染性能
 */

import React, { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { 
  OrbitControls, 
  Text, 
  Sphere, 
  Cylinder,
  Environment,
  ContactShadows
} from '@react-three/drei'
import * as THREE from 'three'

interface SimpleRealisticAvatarProps {
  signText: string
  isPerforming: boolean
  leftHandKeypoints?: Array<{x: number, y: number, z: number, visibility?: number}>
  rightHandKeypoints?: Array<{x: number, y: number, z: number, visibility?: number}>
  showBones?: boolean
  realisticMode?: boolean
  animationSpeed?: number
  onAvatarReady?: (avatar: THREE.Object3D) => void
  onSignComplete?: (signText: string) => void
}

// 简化的人体建模组件
const SimpleHumanBody: React.FC<{
  isPerforming: boolean
  realisticMode: boolean
}> = ({ isPerforming, realisticMode }) => {
  const bodyRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (!bodyRef.current) return
    
    const time = state.clock.elapsedTime
    
    // 自然的呼吸动画
    const breathingScale = 1 + Math.sin(time * 0.8) * 0.02
    bodyRef.current.scale.y = breathingScale
    
    // 轻微的身体摆动
    if (isPerforming) {
      bodyRef.current.rotation.z = Math.sin(time * 0.6) * 0.03
      bodyRef.current.position.x = Math.sin(time * 0.4) * 0.01
    }
  })
  
  const skinColor = realisticMode ? "#fdbcb4" : "#ffcdb2"
  const clothingColor = realisticMode ? "#e8d5c4" : "#f0e6d6"
  
  return (
    <group ref={bodyRef}>
      {/* 头部 */}
      <group position={[0, 1.65, 0]}>
        <Sphere args={[0.12, 24, 16]}>
          <meshStandardMaterial
            color={skinColor}
            roughness={0.6}
            metalness={0.0}
          />
        </Sphere>
        
        {/* 简化的面部特征 */}
        <group position={[0, 0, 0.1]}>
          {/* 眼睛 */}
          <Sphere args={[0.008, 8, 6]} position={[-0.03, 0.02, 0.02]}>
            <meshStandardMaterial color="#000000" />
          </Sphere>
          <Sphere args={[0.008, 8, 6]} position={[0.03, 0.02, 0.02]}>
            <meshStandardMaterial color="#000000" />
          </Sphere>
          
          {/* 嘴巴 */}
          <mesh position={[0, -0.04, 0.02]}>
            <boxGeometry args={[0.025, 0.005, 0.008]} />
            <meshStandardMaterial color="#d4776b" roughness={0.4} />
          </mesh>
        </group>
        
        {/* 头发 */}
        <Sphere args={[0.13, 16, 12]} position={[0, 0.08, -0.02]}>
          <meshStandardMaterial
            color="#4a4a4a"
            roughness={0.8}
          />
        </Sphere>
      </group>
      
      {/* 颈部 */}
      <Cylinder args={[0.04, 0.05, 0.15, 12]} position={[0, 1.45, 0]}>
        <meshStandardMaterial
          color={skinColor}
          roughness={0.6}
        />
      </Cylinder>
      
      {/* 躯干 */}
      <group position={[0, 1.0, 0]}>
        {/* 胸部 */}
        <mesh position={[0, 0.2, 0]}>
          <boxGeometry args={[0.25, 0.35, 0.15]} />
          <meshStandardMaterial
            color={clothingColor}
            roughness={0.7}
          />
        </mesh>
        
        {/* 腰部 */}
        <mesh position={[0, -0.1, 0]}>
          <boxGeometry args={[0.22, 0.25, 0.12]} />
          <meshStandardMaterial
            color={clothingColor}
            roughness={0.7}
          />
        </mesh>
      </group>
      
      {/* 肩膀 */}
      <Sphere args={[0.06, 12, 8]} position={[-0.15, 1.25, 0]}>
        <meshStandardMaterial color={skinColor} roughness={0.6} />
      </Sphere>
      <Sphere args={[0.06, 12, 8]} position={[0.15, 1.25, 0]}>
        <meshStandardMaterial color={skinColor} roughness={0.6} />
      </Sphere>
      
      {/* 手臂 */}
      <group position={[-0.15, 1.1, 0]}>
        {/* 上臂 */}
        <Cylinder args={[0.035, 0.04, 0.3, 12]} position={[0, -0.15, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.6} />
        </Cylinder>
        
        {/* 肘部 */}
        <Sphere args={[0.04, 12, 8]} position={[0, -0.32, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.7} />
        </Sphere>
        
        {/* 前臂 */}
        <Cylinder args={[0.03, 0.035, 0.3, 12]} position={[0, -0.47, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.6} />
        </Cylinder>
      </group>
      
      <group position={[0.15, 1.1, 0]}>
        {/* 上臂 */}
        <Cylinder args={[0.035, 0.04, 0.3, 12]} position={[0, -0.15, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.6} />
        </Cylinder>
        
        {/* 肘部 */}
        <Sphere args={[0.04, 12, 8]} position={[0, -0.32, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.7} />
        </Sphere>
        
        {/* 前臂 */}
        <Cylinder args={[0.03, 0.035, 0.3, 12]} position={[0, -0.47, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.6} />
        </Cylinder>
      </group>
    </group>
  )
}

// 简化的手部组件
const SimpleHand: React.FC<{
  handedness: 'left' | 'right'
  keypoints?: Array<{x: number, y: number, z: number, visibility?: number}>
  isActive: boolean
  showBones: boolean
  position: [number, number, number]
  scale: number
  realisticMode: boolean
}> = ({ handedness, keypoints, isActive, showBones, position, scale, realisticMode }) => {
  const handRef = useRef<THREE.Group>(null)
  const [joints, setJoints] = useState<THREE.Vector3[]>([])
  
  // 生成简化的手势数据
  useEffect(() => {
    if (keypoints && keypoints.length === 21) {
      const processedJoints = keypoints.map((kp) => {
        const x = (kp.x - 0.5) * scale * 1.5
        const y = (0.5 - kp.y) * scale * 1.5
        const z = kp.z * scale * 0.8
        return new THREE.Vector3(x, y, z)
      })
      setJoints(processedJoints)
    } else {
      // 生成默认手势
      setJoints(generateSimpleHandPose(scale, handedness))
    }
  }, [keypoints, scale, handedness])
  
  // 手部动画
  useFrame((state) => {
    if (!handRef.current || !isActive) return
    
    const time = state.clock.elapsedTime
    const breathingFactor = 1 + Math.sin(time * 1.2) * 0.01
    const naturalSway = Math.sin(time * 0.8) * 0.02
    
    handRef.current.scale.setScalar(breathingFactor)
    handRef.current.rotation.z = naturalSway * (handedness === 'left' ? -1 : 1)
  })
  
  if (joints.length === 0) return null
  
  const skinColor = realisticMode ? "#fdbcb4" : "#ffcdb2"
  
  return (
    <group ref={handRef} position={position}>
      {/* 手掌 */}
      <mesh position={[joints[0]?.x || 0, joints[0]?.y + 0.05 || 0, joints[0]?.z || 0]}>
        <boxGeometry args={[0.08, 0.1, 0.04]} />
        <meshStandardMaterial
          color={skinColor}
          roughness={0.6}
          metalness={0.02}
        />
      </mesh>
      
      {/* 简化的手指 */}
      {joints.slice(1, 21).map((joint, index) => (
        <Sphere key={index} args={[0.015, 8, 6]} position={[joint.x, joint.y, joint.z]}>
          <meshStandardMaterial
            color={skinColor}
            roughness={0.5}
          />
        </Sphere>
      ))}
      
      {/* 手腕 */}
      <Cylinder args={[0.035, 0.04, 0.08, 12]} position={[joints[0]?.x || 0, joints[0]?.y || 0, joints[0]?.z || 0]}>
        <meshStandardMaterial
          color={skinColor}
          roughness={0.6}
        />
      </Cylinder>
      
      {/* 显示骨骼（调试用） */}
      {showBones && joints.map((joint, index) => (
        <mesh key={`bone-${index}`} position={[joint.x, joint.y, joint.z]}>
          <sphereGeometry args={[0.005, 8, 6]} />
          <meshBasicMaterial color="#ff0000" />
        </mesh>
      ))}
    </group>
  )
}

// 生成简单的手势姿态
const generateSimpleHandPose = (scale: number, handedness: 'left' | 'right'): THREE.Vector3[] => {
  const basePositions = [
    // 手腕 (0)
    [0, 0, 0],
    
    // 拇指 (1-4)
    [-0.02, 0.02, 0.02],
    [-0.035, 0.035, 0.035],
    [-0.045, 0.045, 0.045],
    [-0.055, 0.055, 0.055],
    
    // 食指 (5-8)
    [-0.02, 0.06, 0.01],
    [-0.02, 0.095, 0.015],
    [-0.02, 0.125, 0.02],
    [-0.02, 0.145, 0.025],
    
    // 中指 (9-12)
    [0, 0.065, 0.01],
    [0, 0.105, 0.015],
    [0, 0.14, 0.02],
    [0, 0.165, 0.025],
    
    // 无名指 (13-16)
    [0.02, 0.06, 0.01],
    [0.02, 0.095, 0.015],
    [0.02, 0.125, 0.02],
    [0.02, 0.145, 0.025],
    
    // 小指 (17-20)
    [0.035, 0.055, 0.008],
    [0.035, 0.08, 0.012],
    [0.035, 0.1, 0.016],
    [0.035, 0.115, 0.02]
  ]
  
  return basePositions.map(pos => {
    const mirrored = handedness === 'left' ? [-pos[0], pos[1], pos[2]] : pos
    return new THREE.Vector3(
      mirrored[0] * scale,
      mirrored[1] * scale,
      mirrored[2] * scale
    )
  })
}

// 主要组件
const SimpleRealisticSignLanguageAvatar: React.FC<SimpleRealisticAvatarProps> = ({
  signText,
  isPerforming,
  leftHandKeypoints,
  rightHandKeypoints,
  showBones = false,
  realisticMode = true,
  animationSpeed = 1.0,
  onAvatarReady,
  onSignComplete
}) => {
  const avatarRef = useRef<THREE.Group>(null)
  
  useEffect(() => {
    if (avatarRef.current && onAvatarReady) {
      onAvatarReady(avatarRef.current)
    }
  }, [onAvatarReady])
  
  return (
    <Canvas
      shadows
      camera={{ position: [0, 1.5, 6], fov: 45 }}
      style={{ width: '100%', height: '100%' }}
      gl={{ 
        antialias: true, 
        alpha: true,
        powerPreference: "high-performance"
      }}
      dpr={[1, 1.5]}
    >
      {/* 环境设置 */}
      <Environment preset="city" />
      <ContactShadows 
        position={[0, -1.5, 0]} 
        opacity={0.3} 
        scale={10} 
        blur={2} 
        far={3} 
      />
      
      {/* 简化的照明 */}
      <ambientLight intensity={0.4} />
      <directionalLight 
        position={[5, 8, 5]} 
        intensity={0.8} 
        castShadow
      />
      <pointLight position={[-3, 3, -3]} intensity={0.2} color="#ffd4c4" />
      
      {/* Avatar主体 */}
      <group ref={avatarRef} position={[0, 0, 0]}>
        {/* 身体 */}
        <SimpleHumanBody 
          isPerforming={isPerforming}
          realisticMode={realisticMode}
        />
        
        {/* 左手 */}
        <SimpleHand
          handedness="left"
          keypoints={leftHandKeypoints}
          isActive={isPerforming}
          showBones={showBones}
          position={[-0.15, 0.48, 0.1]}
          scale={0.8}
          realisticMode={realisticMode}
        />
        
        {/* 右手 */}
        <SimpleHand
          handedness="right"
          keypoints={rightHandKeypoints}
          isActive={isPerforming}
          showBones={showBones}
          position={[0.15, 0.48, 0.1]}
          scale={0.8}
          realisticMode={realisticMode}
        />
        
        {/* 显示当前手语文本 - 使用默认字体 */}
        {signText && (
          <Text
            position={[0, 2.2, 0]}
            fontSize={0.15}
            color={realisticMode ? "#2c5aa0" : "#ff6b6b"}
            anchorX="center"
            anchorY="middle"
          >
            {signText}
          </Text>
        )}
      </group>
      
      {/* 相机控制 */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxPolarAngle={Math.PI * 0.75}
        minDistance={3}
        maxDistance={15}
        target={[0, 1, 0]}
      />
    </Canvas>
  )
}

export default SimpleRealisticSignLanguageAvatar
export type { SimpleRealisticAvatarProps }
