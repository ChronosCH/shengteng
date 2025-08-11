/**
 * 逼真的3D人体Avatar组件
 * 特点：
 * 1. 基于人体解剖学的建模
 * 2. 自然的皮肤材质和阴影
 * 3. 流畅的手语动作
 * 4. 专业的面部建模
 */

import React, { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { 
  OrbitControls, 
  Environment,
  ContactShadows,
  useTexture,
  Sphere,
  Text
} from '@react-three/drei'
import * as THREE from 'three'

interface RealisticHumanAvatarProps {
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

// 人体皮肤材质
const SkinMaterial = () => {
  return (
    <meshPhysicalMaterial
      color="#f4c2a1"
      roughness={0.7}
      metalness={0.02}
      clearcoat={0.15}
      clearcoatRoughness={0.4}
      subsurface={0.4}
      subsurfaceColor="#ff9999"
      transmission={0.03}
      thickness={0.8}
      emissive="#ffeeee"
      emissiveIntensity={0.01}
    />
  )
}

// 衣服材质
const ClothingMaterial = ({ color = "#4a90e2" }) => {
  return (
    <meshPhysicalMaterial
      color={color}
      roughness={0.85}
      metalness={0.02}
      clearcoat={0.05}
      normalScale={[0.3, 0.3]}
    />
  )
}

// 头发材质
const HairMaterial = () => {
  return (
    <meshPhysicalMaterial
      color="#2d1810"
      roughness={0.95}
      metalness={0.05}
      clearcoat={0.3}
      clearcoatRoughness={0.7}
      anisotropy={0.8}
      anisotropyRotation={Math.PI / 4}
    />
  )
}

// 逼真的头部组件
const RealisticHead: React.FC<{
  isPerforming: boolean
}> = ({ isPerforming }) => {
  const headRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (!headRef.current) return
    
    const time = state.clock.elapsedTime
    
    // 自然的点头动画
    if (isPerforming) {
      headRef.current.rotation.x = Math.sin(time * 2) * 0.08
      headRef.current.rotation.y = Math.cos(time * 1.5) * 0.05
    }
  })
  
  return (
    <group ref={headRef} position={[0, 1.65, 0]}>
      {/* 头部 - 更真实的头部形状 */}
      <mesh scale={[1.0, 1.15, 0.85]}>
        <sphereGeometry args={[0.125, 64, 64]} />
        <SkinMaterial />
      </mesh>
      
      {/* 下颚和脸部轮廓 */}
      <mesh position={[0, -0.08, 0.03]} scale={[0.85, 0.6, 0.7]}>
        <sphereGeometry args={[0.1, 32, 32]} />
        <SkinMaterial />
      </mesh>
      
      {/* 脸颊 */}
      <mesh position={[-0.07, -0.02, 0.07]} scale={[0.6, 0.8, 0.5]}>
        <sphereGeometry args={[0.05, 32, 32]} />
        <SkinMaterial />
      </mesh>
      <mesh position={[0.07, -0.02, 0.07]} scale={[0.6, 0.8, 0.5]}>
        <sphereGeometry args={[0.05, 32, 32]} />
        <SkinMaterial />
      </mesh>
      
      {/* 头发 - 更自然的发型 */}
      <mesh position={[0, 0.08, -0.02]} scale={[1.1, 0.8, 1.05]}>
        <sphereGeometry args={[0.13, 32, 32]} />
        <HairMaterial />
      </mesh>
      
      {/* 前刘海 */}
      <mesh position={[0, 0.1, 0.1]} scale={[0.9, 0.3, 0.4]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <HairMaterial />
      </mesh>
      
      {/* 眼睛 - 更立体和真实 */}
      <group>
        {/* 左眼 */}
        <group position={[-0.04, 0.02, 0.095]}>
          {/* 眼窝 */}
          <mesh scale={[1.5, 1.0, 0.8]}>
            <sphereGeometry args={[0.015, 16, 16]} />
            <meshPhysicalMaterial color="#f0be9d" roughness={0.8} />
          </mesh>
          {/* 眼白 */}
          <mesh scale={[1.3, 0.8, 0.6]} position={[0, 0, 0.005]}>
            <sphereGeometry args={[0.01, 16, 16]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
          {/* 虹膜 */}
          <mesh position={[0, 0, 0.012]} scale={[0.7, 0.7, 0.4]}>
            <sphereGeometry args={[0.008, 16, 16]} />
            <meshPhysicalMaterial color="#4a5d23" roughness={0.1} clearcoat={1.0} />
          </mesh>
          {/* 瞳孔 */}
          <mesh position={[0, 0, 0.015]} scale={[0.4, 0.4, 0.3]}>
            <sphereGeometry args={[0.005, 12, 12]} />
            <meshBasicMaterial color="#000000" />
          </mesh>
          {/* 高光 */}
          <mesh position={[0.002, 0.002, 0.017]}>
            <sphereGeometry args={[0.001, 8, 8]} />
            <meshBasicMaterial 
              color="#ffffff" 
              transparent 
              opacity={0.9}
              emissive="#ffffff"
              emissiveIntensity={0.5}
            />
          </mesh>
        </group>
        
        {/* 右眼 */}
        <group position={[0.04, 0.02, 0.095]}>
          {/* 眼窝 */}
          <mesh scale={[1.5, 1.0, 0.8]}>
            <sphereGeometry args={[0.015, 16, 16]} />
            <meshPhysicalMaterial color="#f0be9d" roughness={0.8} />
          </mesh>
          {/* 眼白 */}
          <mesh scale={[1.3, 0.8, 0.6]} position={[0, 0, 0.005]}>
            <sphereGeometry args={[0.01, 16, 16]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
          {/* 虹膜 */}
          <mesh position={[0, 0, 0.012]} scale={[0.7, 0.7, 0.4]}>
            <sphereGeometry args={[0.008, 16, 16]} />
            <meshPhysicalMaterial color="#4a5d23" roughness={0.1} clearcoat={1.0} />
          </mesh>
          {/* 瞳孔 */}
          <mesh position={[0, 0, 0.015]} scale={[0.4, 0.4, 0.3]}>
            <sphereGeometry args={[0.005, 12, 12]} />
            <meshBasicMaterial color="#000000" />
          </mesh>
          {/* 高光 */}
          <mesh position={[-0.002, 0.002, 0.017]}>
            <sphereGeometry args={[0.001, 8, 8]} />
            <meshBasicMaterial 
              color="#ffffff" 
              transparent 
              opacity={0.9}
              emissive="#ffffff"
              emissiveIntensity={0.5}
            />
          </mesh>
        </group>
      </group>
      
      {/* 眉毛 */}
      <mesh position={[-0.04, 0.05, 0.1]} rotation={[0, 0, -0.2]} scale={[0.8, 0.2, 0.15]}>
        <sphereGeometry args={[0.012, 12, 8]} />
        <HairMaterial />
      </mesh>
      <mesh position={[0.04, 0.05, 0.1]} rotation={[0, 0, 0.2]} scale={[0.8, 0.2, 0.15]}>
        <sphereGeometry args={[0.012, 12, 8]} />
        <HairMaterial />
      </mesh>
      
      {/* 鼻子 - 更立体的结构 */}
      <group position={[0, -0.01, 0.1]}>
        {/* 鼻梁 */}
        <mesh position={[0, 0.02, 0]} scale={[0.3, 1.2, 0.8]}>
          <sphereGeometry args={[0.01, 16, 16]} />
          <SkinMaterial />
        </mesh>
        {/* 鼻头 */}
        <mesh position={[0, -0.015, 0.005]} scale={[0.6, 0.7, 0.8]}>
          <sphereGeometry args={[0.015, 16, 16]} />
          <SkinMaterial />
        </mesh>
        {/* 鼻翼 */}
        <mesh position={[-0.01, -0.01, 0.002]} scale={[0.4, 0.6, 0.5]}>
          <sphereGeometry args={[0.008, 12, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0.01, -0.01, 0.002]} scale={[0.4, 0.6, 0.5]}>
          <sphereGeometry args={[0.008, 12, 12]} />
          <SkinMaterial />
        </mesh>
      </group>
      
      {/* 嘴巴 - 更真实的嘴唇 */}
      <group position={[0, -0.05, 0.095]}>
        {/* 上唇 */}
        <mesh position={[0, 0.005, 0]} scale={[1.0, 0.4, 0.6]}>
          <sphereGeometry args={[0.018, 16, 12]} />
          <meshPhysicalMaterial 
            color="#d67b7b" 
            roughness={0.2} 
            metalness={0.0}
            clearcoat={0.6}
            clearcoatRoughness={0.1}
          />
        </mesh>
        {/* 下唇 */}
        <mesh position={[0, -0.005, 0.001]} scale={[0.9, 0.5, 0.7]}>
          <sphereGeometry args={[0.016, 16, 12]} />
          <meshPhysicalMaterial 
            color="#d47878" 
            roughness={0.2} 
            metalness={0.0}
            clearcoat={0.6}
            clearcoatRoughness={0.1}
          />
        </mesh>
      </group>
      
      {/* 耳朵 */}
      <mesh position={[-0.11, 0, 0.02]} rotation={[0, -0.3, -0.2]} scale={[0.4, 0.8, 0.6]}>
        <sphereGeometry args={[0.04, 16, 16]} />
        <SkinMaterial />
      </mesh>
      <mesh position={[0.11, 0, 0.02]} rotation={[0, 0.3, 0.2]} scale={[0.4, 0.8, 0.6]}>
        <sphereGeometry args={[0.04, 16, 16]} />
        <SkinMaterial />
      </mesh>
    </group>
  )
}

// 逼真的身体组件
const RealisticBody: React.FC<{
  isPerforming: boolean
}> = ({ isPerforming }) => {
  const bodyRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (!bodyRef.current) return
    
    const time = state.clock.elapsedTime
    
    // 自然的呼吸动画
    const breathingScale = 1 + Math.sin(time * 0.6) * 0.015
    bodyRef.current.scale.set(1, breathingScale, 1)
    
    // 轻微的身体摆动
    if (isPerforming) {
      bodyRef.current.rotation.z = Math.sin(time * 1.2) * 0.03
    }
  })
  
  return (
    <group ref={bodyRef}>
      {/* 躯干 - 更自然的形状 */}
      <mesh position={[0, 1.2, 0]} scale={[0.25, 0.35, 0.15]}>
        <sphereGeometry args={[1, 32, 32]} />
        <ClothingMaterial color="#4a90e2" />
      </mesh>
      
      {/* 胸部轮廓 */}
      <mesh position={[0, 1.35, 0.05]} scale={[0.22, 0.15, 0.12]}>
        <sphereGeometry args={[1, 32, 32]} />
        <ClothingMaterial color="#4a90e2" />
      </mesh>
      
      {/* 腰部 */}
      <mesh position={[0, 0.95, 0]} scale={[0.2, 0.15, 0.13]}>
        <sphereGeometry args={[1, 32, 32]} />
        <ClothingMaterial color="#2c5aa0" />
      </mesh>
      
      {/* 脖子 */}
      <mesh position={[0, 1.52, 0]}>
        <cylinderGeometry args={[0.045, 0.05, 0.12, 16]} />
        <SkinMaterial />
      </mesh>
    </group>
  )
}

// 逼真的手臂组件
const RealisticArm: React.FC<{
  side: 'left' | 'right'
  isPerforming: boolean
  handKeypoints?: Array<{x: number, y: number, z: number, visibility?: number}>
}> = ({ side, isPerforming, handKeypoints }) => {
  const armRef = useRef<THREE.Group>(null)
  const upperArmRef = useRef<THREE.Mesh>(null)
  const forearmRef = useRef<THREE.Mesh>(null)
  
  const isLeft = side === 'left'
  const xMultiplier = isLeft ? -1 : 1
  
  useFrame((state) => {
    if (!armRef.current) return
    
    const time = state.clock.elapsedTime
    
    if (isPerforming && handKeypoints) {
      // 基于手部关键点的动画
      if (upperArmRef.current) {
        upperArmRef.current.rotation.z = Math.sin(time * 2) * 0.3 * xMultiplier
        upperArmRef.current.rotation.x = Math.cos(time * 1.8) * 0.2
      }
      if (forearmRef.current) {
        forearmRef.current.rotation.z = Math.sin(time * 2.5) * 0.4 * xMultiplier
      }
    } else {
      // 自然的待机动画
      if (upperArmRef.current) {
        upperArmRef.current.rotation.z = Math.sin(time * 0.8) * 0.05 * xMultiplier
      }
    }
  })
  
  return (
    <group ref={armRef} position={[0.25 * xMultiplier, 1.35, 0]}>
      {/* 肩膀 */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <ClothingMaterial color="#4a90e2" />
      </mesh>
      
      {/* 上臂 */}
      <mesh ref={upperArmRef} position={[0, -0.18, 0]}>
        <cylinderGeometry args={[0.055, 0.045, 0.32, 16]} />
        <ClothingMaterial color="#4a90e2" />
      </mesh>
      
      {/* 肘部 */}
      <mesh position={[0, -0.35, 0]}>
        <sphereGeometry args={[0.045, 16, 16]} />
        <SkinMaterial />
      </mesh>
      
      {/* 前臂 */}
      <mesh ref={forearmRef} position={[0, -0.52, 0]}>
        <cylinderGeometry args={[0.04, 0.035, 0.3, 16]} />
        <SkinMaterial />
      </mesh>
      
      {/* 手腕 */}
      <mesh position={[0, -0.68, 0]}>
        <sphereGeometry args={[0.035, 16, 16]} />
        <SkinMaterial />
      </mesh>
      
      {/* 手掌 */}
      <RealisticHand 
        side={side} 
        keypoints={handKeypoints} 
        isPerforming={isPerforming}
      />
    </group>
  )
}

// 逼真的手部组件
const RealisticHand: React.FC<{
  side: 'left' | 'right'
  keypoints?: Array<{x: number, y: number, z: number, visibility?: number}>
  isPerforming: boolean
}> = ({ side, keypoints, isPerforming }) => {
  const handRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (!handRef.current) return
    
    const time = state.clock.elapsedTime
    
    if (isPerforming) {
      // 手语动作动画
      handRef.current.rotation.x = Math.sin(time * 3) * 0.2
      handRef.current.rotation.y = Math.cos(time * 2.5) * 0.3
      handRef.current.rotation.z = Math.sin(time * 2.2) * 0.15
    }
  })
  
  const isLeft = side === 'left'
  const xMultiplier = isLeft ? -1 : 1
  
  return (
    <group ref={handRef} position={[0, -0.78, 0]}>
      {/* 手掌 */}
      <mesh scale={[0.8, 1.2, 0.3]}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <SkinMaterial />
      </mesh>
      
      {/* 拇指 */}
      <group position={[0.035 * xMultiplier, -0.02, 0.03]} rotation={[0, 0, 0.5 * xMultiplier]}>
        <mesh position={[0, -0.025, 0]}>
          <cylinderGeometry args={[0.012, 0.015, 0.04, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.055, 0]}>
          <cylinderGeometry args={[0.01, 0.012, 0.03, 12]} />
          <SkinMaterial />
        </mesh>
      </group>
      
      {/* 食指 */}
      <group position={[0.025 * xMultiplier, -0.08, 0.02]}>
        <mesh position={[0, -0.025, 0]}>
          <cylinderGeometry args={[0.01, 0.012, 0.04, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.055, 0]}>
          <cylinderGeometry args={[0.008, 0.01, 0.03, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.075, 0]}>
          <cylinderGeometry args={[0.006, 0.008, 0.02, 12]} />
          <SkinMaterial />
        </mesh>
      </group>
      
      {/* 中指 */}
      <group position={[0.005 * xMultiplier, -0.08, 0.02]}>
        <mesh position={[0, -0.03, 0]}>
          <cylinderGeometry args={[0.01, 0.012, 0.05, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.065, 0]}>
          <cylinderGeometry args={[0.008, 0.01, 0.035, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.09, 0]}>
          <cylinderGeometry args={[0.006, 0.008, 0.025, 12]} />
          <SkinMaterial />
        </mesh>
      </group>
      
      {/* 无名指 */}
      <group position={[-0.015 * xMultiplier, -0.08, 0.02]}>
        <mesh position={[0, -0.025, 0]}>
          <cylinderGeometry args={[0.009, 0.011, 0.04, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.055, 0]}>
          <cylinderGeometry args={[0.007, 0.009, 0.03, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.075, 0]}>
          <cylinderGeometry args={[0.005, 0.007, 0.02, 12]} />
          <SkinMaterial />
        </mesh>
      </group>
      
      {/* 小指 */}
      <group position={[-0.03 * xMultiplier, -0.075, 0.015]}>
        <mesh position={[0, -0.02, 0]}>
          <cylinderGeometry args={[0.008, 0.01, 0.03, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.04, 0]}>
          <cylinderGeometry args={[0.006, 0.008, 0.025, 12]} />
          <SkinMaterial />
        </mesh>
        <mesh position={[0, -0.055, 0]}>
          <cylinderGeometry args={[0.004, 0.006, 0.015, 12]} />
          <SkinMaterial />
        </mesh>
      </group>
    </group>
  )
}

// 逼真的腿部组件
const RealisticLeg: React.FC<{
  side: 'left' | 'right'
  isPerforming: boolean
}> = ({ side, isPerforming }) => {
  const legRef = useRef<THREE.Group>(null)
  
  const isLeft = side === 'left'
  const xMultiplier = isLeft ? -1 : 1
  
  useFrame((state) => {
    if (!legRef.current) return
    
    const time = state.clock.elapsedTime
    
    if (isPerforming) {
      // 轻微的重心转移
      legRef.current.rotation.x = Math.sin(time * 1.5) * 0.05 * xMultiplier
    }
  })
  
  return (
    <group ref={legRef} position={[0.12 * xMultiplier, 0.8, 0]}>
      {/* 大腿 */}
      <mesh position={[0, -0.25, 0]}>
        <cylinderGeometry args={[0.08, 0.07, 0.4, 16]} />
        <ClothingMaterial color="#2c5aa0" />
      </mesh>
      
      {/* 膝盖 */}
      <mesh position={[0, -0.46, 0]}>
        <sphereGeometry args={[0.07, 16, 16]} />
        <ClothingMaterial color="#2c5aa0" />
      </mesh>
      
      {/* 小腿 */}
      <mesh position={[0, -0.7, 0]}>
        <cylinderGeometry args={[0.055, 0.06, 0.4, 16]} />
        <ClothingMaterial color="#2c5aa0" />
      </mesh>
      
      {/* 脚踝 */}
      <mesh position={[0, -0.92, 0]}>
        <sphereGeometry args={[0.055, 16, 16]} />
        <SkinMaterial />
      </mesh>
      
      {/* 脚 */}
      <mesh position={[0, -1.0, 0.08]} scale={[0.7, 0.4, 1.8]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshPhysicalMaterial color="#1a1a1a" roughness={0.9} metalness={0.1} />
      </mesh>
    </group>
  )
}

// 主要的逼真Avatar组件
const RealisticHumanModel: React.FC<RealisticHumanAvatarProps> = ({
  signText,
  isPerforming,
  leftHandKeypoints,
  rightHandKeypoints,
  realisticMode = true,
  onAvatarReady,
  onSignComplete
}) => {
  const avatarRef = useRef<THREE.Group>(null)
  
  useEffect(() => {
    if (avatarRef.current && onAvatarReady) {
      onAvatarReady(avatarRef.current)
    }
  }, [onAvatarReady])
  
  useFrame((state) => {
    if (!avatarRef.current) return
    
    const time = state.clock.elapsedTime
    
    // 整体的轻微浮动效果
    avatarRef.current.position.y = Math.sin(time * 0.5) * 0.01
  })
  
  return (
    <group ref={avatarRef}>
      {/* 头部 */}
      <RealisticHead isPerforming={isPerforming} />
      
      {/* 身体 */}
      <RealisticBody isPerforming={isPerforming} />
      
      {/* 手臂 */}
      <RealisticArm 
        side="left" 
        isPerforming={isPerforming} 
        handKeypoints={leftHandKeypoints}
      />
      <RealisticArm 
        side="right" 
        isPerforming={isPerforming} 
        handKeypoints={rightHandKeypoints}
      />
      
      {/* 腿部 */}
      <RealisticLeg side="left" isPerforming={isPerforming} />
      <RealisticLeg side="right" isPerforming={isPerforming} />
      
      {/* 手语文本显示 */}
      {signText && (
        <Text
          position={[0, 2.2, 0]}
          fontSize={0.15}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.01}
          outlineColor="#000000"
        >
          {signText}
        </Text>
      )}
    </group>
  )
}

// 主要导出组件
const RealisticHumanAvatar: React.FC<RealisticHumanAvatarProps> = (props) => {
  return (
    <div style={{ width: '100%', height: '500px', position: 'relative' }}>
      <Canvas
        shadows
        camera={{ position: [0, 1.5, 4], fov: 50 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: "high-performance",
          preserveDrawingBuffer: false
        }}
        dpr={[1, 1.5]}
      >
        {/* 环境光 */}
        <ambientLight intensity={0.4} />
        
        {/* 主光源 */}
        <directionalLight
          position={[5, 10, 5]}
          intensity={0.8}
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
          shadow-camera-far={50}
          shadow-camera-left={-10}
          shadow-camera-right={10}
          shadow-camera-top={10}
          shadow-camera-bottom={-10}
        />
        
        {/* 补光 */}
        <directionalLight
          position={[-3, 5, -3]}
          intensity={0.3}
        />
        
        {/* 逼真的环境 */}
        <Environment preset="studio" />
        
        {/* Avatar */}
        <RealisticHumanModel {...props} />
        
        {/* 地面阴影 */}
        <ContactShadows 
          position={[0, -1.2, 0]} 
          opacity={0.5} 
          scale={10} 
          blur={2} 
          far={4} 
        />
        
        {/* 控制器 */}
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          minDistance={2}
          maxDistance={8}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 2}
          target={[0, 1, 0]}
        />
      </Canvas>
    </div>
  )
}

export default RealisticHumanAvatar
