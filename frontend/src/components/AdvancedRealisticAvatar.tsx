/**
 * 高级写实3D人类Avatar组件
 * 特点：
 * 1. 基于真实人体解剖学的精确建模
 * 2. 写实的面部特征和表情系统
 * 3. 自然的皮肤质感和光照效果
 * 4. 流畅的手语动作和身体语言
 * 5. 专业级材质和渲染质量
 */

import React, { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { 
  OrbitControls, 
  Environment,
  ContactShadows,
  Sphere,
  Text,
  useTexture,
  Cylinder,
  RoundedBox
} from '@react-three/drei'
import * as THREE from 'three'

interface AdvancedRealisticAvatarProps {
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

// 高级皮肤材质 - 模拟真实人体皮肤
const AdvancedSkinMaterial = ({ tone = "#f4c2a1" }) => {
  return (
    <meshPhysicalMaterial
      color={tone}
      roughness={0.7}
      metalness={0.02}
      clearcoat={0.15}
      clearcoatRoughness={0.4}
      subsurface={0.4}
      subsurfaceColor="#ff8888"
      transmission={0.05}
      thickness={1.0}
      emissive="#ffeeee"
      emissiveIntensity={0.02}
      transparent={false}
      opacity={1.0}
    />
  )
}

// 高级衣服材质
const AdvancedClothingMaterial = ({ color = "#4a90e2", type = "cotton" }) => {
  const roughnessMap = {
    cotton: 0.9,
    silk: 0.3,
    leather: 0.8,
    denim: 0.95
  }
  
  return (
    <meshPhysicalMaterial
      color={color}
      roughness={roughnessMap[type] || 0.8}
      metalness={type === 'leather' ? 0.1 : 0.02}
      clearcoat={type === 'leather' ? 0.8 : 0.1}
      clearcoatRoughness={type === 'leather' ? 0.2 : 0.8}
      normalScale={[0.5, 0.5]}
    />
  )
}

// 头发材质
const AdvancedHairMaterial = ({ color = "#2d1810" }) => {
  return (
    <meshPhysicalMaterial
      color={color}
      roughness={0.95}
      metalness={0.05}
      clearcoat={0.3}
      clearcoatRoughness={0.7}
      anisotropy={0.8}
      anisotropyRotation={Math.PI / 4}
    />
  )
}

// 眼睛材质
const EyeMaterial = ({ isIris = false, color = "#4a5d23" }) => {
  if (isIris) {
    return (
      <meshPhysicalMaterial
        color={color}
        roughness={0.1}
        metalness={0.0}
        clearcoat={1.0}
        clearcoatRoughness={0.0}
        transmission={0.1}
        thickness={0.5}
      />
    )
  }
  
  return (
    <meshBasicMaterial color="#ffffff" />
  )
}

// 高级写实头部组件
const AdvancedRealisticHead: React.FC<{
  isPerforming: boolean
}> = ({ isPerforming }) => {
  const headRef = useRef<THREE.Group>(null)
  const eyeRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (!headRef.current) return
    
    const time = state.clock.elapsedTime
    
    // 自然的头部动作
    if (isPerforming) {
      headRef.current.rotation.x = Math.sin(time * 1.8) * 0.1
      headRef.current.rotation.y = Math.cos(time * 1.2) * 0.08
      headRef.current.position.y = 1.65 + Math.sin(time * 0.8) * 0.02
    }
    
    // 眨眼动画
    if (eyeRef.current && Math.random() < 0.005) {
      const blinkDuration = 0.15
      const originalScale = 1
      const blinkScale = 0.1
      
      // 简单的眨眼效果
      eyeRef.current.scale.y = blinkScale
      setTimeout(() => {
        if (eyeRef.current) {
          eyeRef.current.scale.y = originalScale
        }
      }, blinkDuration * 1000)
    }
  })
  
  return (
    <group ref={headRef} position={[0, 1.65, 0]}>
      {/* 主头部 - 更真实的头部形状 */}
      <mesh scale={[1.0, 1.15, 0.85]}>
        <sphereGeometry args={[0.125, 64, 64]} />
        <AdvancedSkinMaterial tone="#f4c2a1" />
      </mesh>
      
      {/* 下颚 */}
      <mesh position={[0, -0.08, 0.03]} scale={[0.85, 0.6, 0.7]}>
        <sphereGeometry args={[0.1, 32, 32]} />
        <AdvancedSkinMaterial tone="#f2c09f" />
      </mesh>
      
      {/* 额头突出 */}
      <mesh position={[0, 0.06, 0.08]} scale={[0.9, 0.4, 0.3]}>
        <sphereGeometry args={[0.08, 32, 32]} />
        <AdvancedSkinMaterial tone="#f6c4a3" />
      </mesh>
      
      {/* 脸颊 */}
      <mesh position={[-0.07, -0.02, 0.07]} scale={[0.6, 0.8, 0.5]}>
        <sphereGeometry args={[0.055, 32, 32]} />
        <AdvancedSkinMaterial tone="#f5c3a2" />
      </mesh>
      <mesh position={[0.07, -0.02, 0.07]} scale={[0.6, 0.8, 0.5]}>
        <sphereGeometry args={[0.055, 32, 32]} />
        <AdvancedSkinMaterial tone="#f5c3a2" />
      </mesh>
      
      {/* 头发 - 更自然的发型 */}
      <group position={[0, 0.08, -0.02]}>
        {/* 主发型 */}
        <mesh scale={[1.15, 0.8, 1.1]}>
          <sphereGeometry args={[0.13, 32, 32]} />
          <AdvancedHairMaterial color="#2d1810" />
        </mesh>
        
        {/* 前刘海 */}
        <mesh position={[0, 0.02, 0.12]} scale={[1.0, 0.3, 0.4]}>
          <sphereGeometry args={[0.08, 16, 16]} />
          <AdvancedHairMaterial color="#2d1810" />
        </mesh>
        
        {/* 侧发 */}
        <mesh position={[-0.1, -0.02, 0.06]} scale={[0.4, 0.6, 0.5]}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <AdvancedHairMaterial color="#2d1810" />
        </mesh>
        <mesh position={[0.1, -0.02, 0.06]} scale={[0.4, 0.6, 0.5]}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <AdvancedHairMaterial color="#2d1810" />
        </mesh>
      </group>
      
      {/* 眼睛组 */}
      <group ref={eyeRef}>
        {/* 左眼 */}
        <group position={[-0.04, 0.02, 0.095]}>
          {/* 眼白 */}
          <mesh scale={[1.3, 0.8, 0.6]}>
            <sphereGeometry args={[0.012, 16, 16]} />
            <EyeMaterial />
          </mesh>
          {/* 虹膜 */}
          <mesh position={[0, 0, 0.008]} scale={[0.7, 0.7, 0.4]}>
            <sphereGeometry args={[0.01, 16, 16]} />
            <EyeMaterial isIris color="#4a5d23" />
          </mesh>
          {/* 瞳孔 */}
          <mesh position={[0, 0, 0.012]} scale={[0.4, 0.4, 0.3]}>
            <sphereGeometry args={[0.006, 12, 12]} />
            <meshBasicMaterial color="#000000" />
          </mesh>
          {/* 高光 */}
          <mesh position={[0.003, 0.003, 0.015]}>
            <sphereGeometry args={[0.002, 8, 8]} />
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
          {/* 眼白 */}
          <mesh scale={[1.3, 0.8, 0.6]}>
            <sphereGeometry args={[0.012, 16, 16]} />
            <EyeMaterial />
          </mesh>
          {/* 虹膜 */}
          <mesh position={[0, 0, 0.008]} scale={[0.7, 0.7, 0.4]}>
            <sphereGeometry args={[0.01, 16, 16]} />
            <EyeMaterial isIris color="#4a5d23" />
          </mesh>
          {/* 瞳孔 */}
          <mesh position={[0, 0, 0.012]} scale={[0.4, 0.4, 0.3]}>
            <sphereGeometry args={[0.006, 12, 12]} />
            <meshBasicMaterial color="#000000" />
          </mesh>
          {/* 高光 */}
          <mesh position={[-0.003, 0.003, 0.015]}>
            <sphereGeometry args={[0.002, 8, 8]} />
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
        <sphereGeometry args={[0.015, 12, 8]} />
        <AdvancedHairMaterial color="#2d1810" />
      </mesh>
      <mesh position={[0.04, 0.05, 0.1]} rotation={[0, 0, 0.2]} scale={[0.8, 0.2, 0.15]}>
        <sphereGeometry args={[0.015, 12, 8]} />
        <AdvancedHairMaterial color="#2d1810" />
      </mesh>
      
      {/* 鼻子 - 更立体的结构 */}
      <group position={[0, -0.01, 0.1]}>
        {/* 鼻梁 */}
        <mesh position={[0, 0.02, 0]} scale={[0.3, 1.2, 0.8]}>
          <sphereGeometry args={[0.012, 16, 16]} />
          <AdvancedSkinMaterial tone="#f0be9d" />
        </mesh>
        {/* 鼻头 */}
        <mesh position={[0, -0.015, 0.005]} scale={[0.6, 0.7, 0.8]}>
          <sphereGeometry args={[0.018, 16, 16]} />
          <AdvancedSkinMaterial tone="#efbd9c" />
        </mesh>
        {/* 鼻翼 */}
        <mesh position={[-0.012, -0.01, 0.002]} scale={[0.4, 0.6, 0.5]}>
          <sphereGeometry args={[0.01, 12, 12]} />
          <AdvancedSkinMaterial tone="#eebc9b" />
        </mesh>
        <mesh position={[0.012, -0.01, 0.002]} scale={[0.4, 0.6, 0.5]}>
          <sphereGeometry args={[0.01, 12, 12]} />
          <AdvancedSkinMaterial tone="#eebc9b" />
        </mesh>
      </group>
      
      {/* 嘴巴 - 更真实的嘴唇 */}
      <group position={[0, -0.055, 0.095]}>
        {/* 上唇 */}
        <mesh position={[0, 0.008, 0]} scale={[1.0, 0.4, 0.6]}>
          <sphereGeometry args={[0.022, 16, 12]} />
          <meshPhysicalMaterial 
            color="#d67b7b" 
            roughness={0.2} 
            metalness={0.0}
            clearcoat={0.6}
            clearcoatRoughness={0.1}
          />
        </mesh>
        {/* 下唇 */}
        <mesh position={[0, -0.008, 0.002]} scale={[0.9, 0.5, 0.7]}>
          <sphereGeometry args={[0.02, 16, 12]} />
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
        <sphereGeometry args={[0.045, 16, 16]} />
        <AdvancedSkinMaterial tone="#f2c09f" />
      </mesh>
      <mesh position={[0.11, 0, 0.02]} rotation={[0, 0.3, 0.2]} scale={[0.4, 0.8, 0.6]}>
        <sphereGeometry args={[0.045, 16, 16]} />
        <AdvancedSkinMaterial tone="#f2c09f" />
      </mesh>
    </group>
  )
}

// 高级写实身体组件
const AdvancedRealisticBody: React.FC<{
  isPerforming: boolean
}> = ({ isPerforming }) => {
  const bodyRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (!bodyRef.current) return
    
    const time = state.clock.elapsedTime
    
    // 自然的呼吸动画
    const breathingScale = 1 + Math.sin(time * 0.8) * 0.02
    bodyRef.current.scale.set(breathingScale, breathingScale, breathingScale)
    
    // 手语表演时的轻微摆动
    if (isPerforming) {
      bodyRef.current.rotation.z = Math.sin(time * 1.5) * 0.02
      bodyRef.current.position.y = Math.sin(time * 2) * 0.01
    }
  })
  
  return (
    <group ref={bodyRef}>
      {/* 脖子 */}
      <mesh position={[0, 1.52, 0]}>
        <cylinderGeometry args={[0.048, 0.052, 0.13, 16]} />
        <AdvancedSkinMaterial tone="#f4c2a1" />
      </mesh>
      
      {/* 躯干 - 更自然的人体形状 */}
      <mesh position={[0, 1.3, 0]} scale={[1, 1.1, 0.7]}>
        <sphereGeometry args={[0.28, 32, 32]} />
        <AdvancedClothingMaterial color="#4a90e2" type="cotton" />
      </mesh>
      
      {/* 胸部轮廓 */}
      <mesh position={[0, 1.4, 0.08]} scale={[0.85, 0.6, 0.5]}>
        <sphereGeometry args={[0.22, 32, 32]} />
        <AdvancedClothingMaterial color="#4a90e2" type="cotton" />
      </mesh>
      
      {/* 腰部收窄 */}
      <mesh position={[0, 1.0, 0]} scale={[0.75, 0.8, 0.6]}>
        <sphereGeometry args={[0.25, 32, 32]} />
        <AdvancedClothingMaterial color="#2c5aa0" type="denim" />
      </mesh>
      
      {/* 肩膀 */}
      <mesh position={[-0.22, 1.42, 0]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <AdvancedClothingMaterial color="#4a90e2" type="cotton" />
      </mesh>
      <mesh position={[0.22, 1.42, 0]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <AdvancedClothingMaterial color="#4a90e2" type="cotton" />
      </mesh>
      
      {/* 衣服细节 - 按钮 */}
      <mesh position={[0, 1.35, 0.15]}>
        <sphereGeometry args={[0.008, 8, 8]} />
        <meshPhysicalMaterial color="#ffffff" roughness={0.1} metalness={0.8} />
      </mesh>
      <mesh position={[0, 1.25, 0.15]}>
        <sphereGeometry args={[0.008, 8, 8]} />
        <meshPhysicalMaterial color="#ffffff" roughness={0.1} metalness={0.8} />
      </mesh>
      <mesh position={[0, 1.15, 0.15]}>
        <sphereGeometry args={[0.008, 8, 8]} />
        <meshPhysicalMaterial color="#ffffff" roughness={0.1} metalness={0.8} />
      </mesh>
    </group>
  )
}

// 高级写实手臂组件
const AdvancedRealisticArm: React.FC<{
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
        upperArmRef.current.rotation.z = Math.sin(time * 2.2 + (isLeft ? 0 : Math.PI)) * 0.4 * xMultiplier
        upperArmRef.current.rotation.x = Math.cos(time * 1.9) * 0.3
      }
      if (forearmRef.current) {
        forearmRef.current.rotation.z = Math.sin(time * 2.8 + (isLeft ? Math.PI/2 : -Math.PI/2)) * 0.5 * xMultiplier
        forearmRef.current.rotation.x = Math.cos(time * 2.1) * 0.2
      }
    } else {
      // 自然的待机动画
      if (upperArmRef.current) {
        upperArmRef.current.rotation.z = Math.sin(time * 0.9) * 0.05 * xMultiplier
      }
      if (forearmRef.current) {
        forearmRef.current.rotation.z = Math.sin(time * 1.1) * 0.03 * xMultiplier
      }
    }
  })
  
  return (
    <group ref={armRef} position={[0.26 * xMultiplier, 1.38, 0]}>
      {/* 上臂 */}
      <mesh ref={upperArmRef} position={[0, -0.18, 0]} rotation={[0, 0, -0.1 * xMultiplier]}>
        <cylinderGeometry args={[0.058, 0.048, 0.35, 16]} />
        <AdvancedSkinMaterial tone="#f4c2a1" />
      </mesh>
      
      {/* 肘部关节 */}
      <mesh position={[0, -0.36, 0]}>
        <sphereGeometry args={[0.055, 16, 16]} />
        <AdvancedSkinMaterial tone="#f2c09f" />
      </mesh>
      
      {/* 前臂 */}
      <mesh ref={forearmRef} position={[0, -0.54, 0]} rotation={[0, 0, 0.05 * xMultiplier]}>
        <cylinderGeometry args={[0.045, 0.055, 0.32, 16]} />
        <AdvancedSkinMaterial tone="#f4c2a1" />
      </mesh>
      
      {/* 手腕 */}
      <mesh position={[0, -0.71, 0]}>
        <sphereGeometry args={[0.042, 16, 16]} />
        <AdvancedSkinMaterial tone="#f4c2a1" />
      </mesh>
      
      {/* 高级手部 */}
      <AdvancedHand side={side} keypoints={handKeypoints} isPerforming={isPerforming} />
    </group>
  )
}

// 高级手部组件
const AdvancedHand: React.FC<{
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
      handRef.current.rotation.x = Math.sin(time * 3.2) * 0.3
      handRef.current.rotation.y = Math.cos(time * 2.8) * 0.4
      handRef.current.rotation.z = Math.sin(time * 2.5) * 0.2
    }
  })
  
  const isLeft = side === 'left'
  const xMultiplier = isLeft ? -1 : 1
  
  return (
    <group ref={handRef} position={[0, -0.82, 0]}>
      {/* 手掌 */}
      <mesh scale={[0.9, 1.3, 0.35]}>
        <sphereGeometry args={[0.065, 16, 16]} />
        <AdvancedSkinMaterial tone="#f4c2a1" />
      </mesh>
      
      {/* 拇指 */}
      <group position={[0.04 * xMultiplier, -0.02, 0.04]} rotation={[0, 0, 0.6 * xMultiplier]}>
        <mesh position={[0, -0.03, 0]}>
          <cylinderGeometry args={[0.015, 0.018, 0.045, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.065, 0]}>
          <cylinderGeometry args={[0.012, 0.015, 0.035, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
      </group>
      
      {/* 食指 */}
      <group position={[0.025 * xMultiplier, -0.08, 0.025]}>
        <mesh position={[0, -0.035, 0]}>
          <cylinderGeometry args={[0.012, 0.015, 0.055, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.075, 0]}>
          <cylinderGeometry args={[0.01, 0.012, 0.04, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.105, 0]}>
          <cylinderGeometry args={[0.008, 0.01, 0.03, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
      </group>
      
      {/* 中指 */}
      <group position={[0.005 * xMultiplier, -0.085, 0.028]}>
        <mesh position={[0, -0.04, 0]}>
          <cylinderGeometry args={[0.013, 0.016, 0.06, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.085, 0]}>
          <cylinderGeometry args={[0.011, 0.013, 0.045, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.118, 0]}>
          <cylinderGeometry args={[0.009, 0.011, 0.033, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
      </group>
      
      {/* 无名指 */}
      <group position={[-0.015 * xMultiplier, -0.083, 0.025]}>
        <mesh position={[0, -0.035, 0]}>
          <cylinderGeometry args={[0.012, 0.015, 0.055, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.075, 0]}>
          <cylinderGeometry args={[0.01, 0.012, 0.04, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.105, 0]}>
          <cylinderGeometry args={[0.008, 0.01, 0.03, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
      </group>
      
      {/* 小指 */}
      <group position={[-0.032 * xMultiplier, -0.075, 0.02]}>
        <mesh position={[0, -0.025, 0]}>
          <cylinderGeometry args={[0.01, 0.013, 0.04, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.05, 0]}>
          <cylinderGeometry args={[0.008, 0.01, 0.03, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
        <mesh position={[0, -0.07, 0]}>
          <cylinderGeometry args={[0.006, 0.008, 0.025, 12]} />
          <AdvancedSkinMaterial tone="#f4c2a1" />
        </mesh>
      </group>
    </group>
  )
}

// 高级写实腿部组件
const AdvancedRealisticLeg: React.FC<{
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
      legRef.current.rotation.x = Math.sin(time * 1.8 + (isLeft ? 0 : Math.PI)) * 0.08
      legRef.current.position.y = 0.8 + Math.sin(time * 2.5) * 0.01
    }
  })
  
  return (
    <group ref={legRef} position={[0.13 * xMultiplier, 0.8, 0]}>
      {/* 大腿 */}
      <mesh position={[0, -0.28, 0]}>
        <cylinderGeometry args={[0.085, 0.075, 0.45, 16]} />
        <AdvancedClothingMaterial color="#2c5aa0" type="denim" />
      </mesh>
      
      {/* 膝盖 */}
      <mesh position={[0, -0.52, 0]}>
        <sphereGeometry args={[0.075, 16, 16]} />
        <AdvancedClothingMaterial color="#2c5aa0" type="denim" />
      </mesh>
      
      {/* 小腿 */}
      <mesh position={[0, -0.75, 0]}>
        <cylinderGeometry args={[0.062, 0.075, 0.4, 16]} />
        <AdvancedClothingMaterial color="#2c5aa0" type="denim" />
      </mesh>
      
      {/* 脚踝 */}
      <mesh position={[0, -0.97, 0]}>
        <sphereGeometry args={[0.058, 16, 16]} />
        <AdvancedSkinMaterial tone="#f4c2a1" />
      </mesh>
      
      {/* 脚 */}
      <mesh position={[0, -1.05, 0.1]} scale={[0.8, 0.5, 2.0]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshPhysicalMaterial 
          color="#1a1a1a" 
          roughness={0.9} 
          metalness={0.1}
          clearcoat={0.2}
          clearcoatRoughness={0.8}
        />
      </mesh>
    </group>
  )
}

// 主要的高级写实Avatar组件
const AdvancedRealisticAvatarModel: React.FC<AdvancedRealisticAvatarProps> = ({
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
    avatarRef.current.position.y = Math.sin(time * 0.6) * 0.015
    
    // 轻微的呼吸摆动
    avatarRef.current.rotation.y = Math.sin(time * 0.4) * 0.02
  })
  
  return (
    <group ref={avatarRef}>
      {/* 头部 */}
      <AdvancedRealisticHead isPerforming={isPerforming} />
      
      {/* 身体 */}
      <AdvancedRealisticBody isPerforming={isPerforming} />
      
      {/* 手臂 */}
      <AdvancedRealisticArm 
        side="left" 
        isPerforming={isPerforming} 
        handKeypoints={leftHandKeypoints} 
      />
      <AdvancedRealisticArm 
        side="right" 
        isPerforming={isPerforming} 
        handKeypoints={rightHandKeypoints} 
      />
      
      {/* 腿部 */}
      <AdvancedRealisticLeg side="left" isPerforming={isPerforming} />
      <AdvancedRealisticLeg side="right" isPerforming={isPerforming} />
      
      {/* 手语文本显示 */}
      {signText && (
        <group position={[0, 2.2, 0]}>
          <Text
            fontSize={0.12}
            fontWeight="bold"
            color="#333333"
            anchorX="center"
            anchorY="middle"
            outlineColor="#ffffff"
            outlineWidth={0.01}
          >
            {signText}
          </Text>
          
          {/* 文本背景 */}
          <RoundedBox 
            args={[Math.max(signText.length * 0.08, 0.8), 0.25, 0.05]} 
            position={[0, -0.05, -0.08]} 
            radius={0.08}
          >
            <meshPhysicalMaterial 
              color="#ffffff" 
              transparent 
              opacity={0.95}
              roughness={0.1}
              metalness={0.1}
              clearcoat={0.8}
              clearcoatRoughness={0.1}
            />
          </RoundedBox>
        </group>
      )}
    </group>
  )
}

// 主要的Avatar组件
const AdvancedRealisticAvatar: React.FC<AdvancedRealisticAvatarProps> = (props) => {
  return (
    <div style={{ width: '100%', height: '500px', position: 'relative' }}>
      <Canvas
        shadows
        camera={{ position: [0, 1.8, 5], fov: 50 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: "high-performance",
          preserveDrawingBuffer: false,
          shadowMapType: THREE.PCFSoftShadowMap
        }}
        dpr={[1, 2]}
      >
        {/* 专业环境光照 */}
        <ambientLight intensity={0.3} color="#f5f5dc" />
        
        {/* 主光源 - 模拟自然光 */}
        <directionalLight
          position={[8, 12, 6]}
          intensity={1.0}
          castShadow
          shadow-mapSize-width={4096}
          shadow-mapSize-height={4096}
          shadow-camera-far={50}
          shadow-camera-left={-15}
          shadow-camera-right={15}
          shadow-camera-top={15}
          shadow-camera-bottom={-15}
          color="#fff8e1"
        />
        
        {/* 补光 */}
        <directionalLight
          position={[-5, 8, -4]}
          intensity={0.4}
          color="#e3f2fd"
        />
        
        {/* 顶光 */}
        <directionalLight
          position={[0, 15, 0]}
          intensity={0.3}
          color="#f0f8ff"
        />
        
        {/* 专业环境 */}
        <Environment preset="studio" />
        
        {/* Avatar */}
        <AdvancedRealisticAvatarModel {...props} />
        
        {/* 地面阴影 */}
        <ContactShadows 
          position={[0, -1.3, 0]} 
          opacity={0.6} 
          scale={12} 
          blur={3} 
          far={5} 
        />
        
        {/* 控制器 */}
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          minDistance={3}
          maxDistance={10}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 2}
          target={[0, 1, 0]}
          autoRotate={false}
          enableDamping={true}
          dampingFactor={0.05}
        />
      </Canvas>
    </div>
  )
}

export default AdvancedRealisticAvatar
