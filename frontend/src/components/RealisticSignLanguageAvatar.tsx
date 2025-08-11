/**
 * 高质量3D Avatar组件 - 专业手语演示系统
 * 特点：
 * 1. 写实的人体建模
 * 2. 专业的手部解剖结构
 * 3. 流畅的手语动作
 * 4. 高质量的渲染效果
 */

import React, { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { 
  OrbitControls, 
  Text, 
  Box, 
  Sphere, 
  Cylinder, 
  RoundedBox,
  Environment,
  ContactShadows,
  useTexture,
  MeshDistortMaterial,
  MeshTransmissionMaterial,
  Float
} from '@react-three/drei'
import * as THREE from 'three'

interface RealisticAvatarProps {
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

// 高质量人体建模组件
const RealisticHumanBody: React.FC<{
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
  
  return (
    <group ref={bodyRef}>
      {/* 头部 - 更加写实 */}
      <group position={[0, 1.65, 0]}>
        {/* 头颅 */}
        <mesh>
          <sphereGeometry args={[0.12, 32, 24]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
            metalness={0.0}
          />
        </mesh>
        
        {/* 面部特征 */}
        <group position={[0, 0, 0.1]}>
          {/* 眼睛 */}
          <mesh position={[-0.03, 0.02, 0.02]}>
            <sphereGeometry args={[0.008, 16, 12]} />
            <meshStandardMaterial color="#000000" />
          </mesh>
          <mesh position={[0.03, 0.02, 0.02]}>
            <sphereGeometry args={[0.008, 16, 12]} />
            <meshStandardMaterial color="#000000" />
          </mesh>
          
          {/* 鼻子 */}
          <mesh position={[0, -0.01, 0.03]}>
            <coneGeometry args={[0.008, 0.02, 8]} />
            <meshStandardMaterial
              color={realisticMode ? "#f4a992" : "#ffb8a2"}
              roughness={0.7}
            />
          </mesh>
          
          {/* 嘴巴 */}
          <mesh position={[0, -0.04, 0.02]}>
            <boxGeometry args={[0.025, 0.005, 0.008]} />
            <meshStandardMaterial color="#d4776b" roughness={0.4} />
          </mesh>
        </group>
        
        {/* 头发 */}
        <mesh position={[0, 0.08, -0.02]}>
          <sphereGeometry args={[0.13, 24, 16]} />
          <meshStandardMaterial
            color="#4a4a4a"
            roughness={0.8}
            metalness={0.1}
          />
        </mesh>
      </group>
      
      {/* 颈部 */}
      <group position={[0, 1.45, 0]}>
        <mesh>
          <cylinderGeometry args={[0.04, 0.05, 0.15, 16]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
      </group>
      
      {/* 躯干 - 更加自然的形状 */}
      <group position={[0, 1.0, 0]}>
        {/* 胸部 */}
        <mesh position={[0, 0.2, 0]}>
          <boxGeometry args={[0.25, 0.35, 0.15]} />
          <meshStandardMaterial
            color={realisticMode ? "#e8d5c4" : "#f0e6d6"}
            roughness={0.7}
          />
        </mesh>
        
        {/* 腰部 */}
        <mesh position={[0, -0.1, 0]}>
          <boxGeometry args={[0.22, 0.25, 0.12]} />
          <meshStandardMaterial
            color={realisticMode ? "#e8d5c4" : "#f0e6d6"}
            roughness={0.7}
          />
        </mesh>
      </group>
      
      {/* 肩膀 */}
      <group position={[0, 1.25, 0]}>
        <mesh position={[-0.15, 0, 0]}>
          <sphereGeometry args={[0.06, 16, 12]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
        <mesh position={[0.15, 0, 0]}>
          <sphereGeometry args={[0.06, 16, 12]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
      </group>
      
      {/* 手臂 - 更加自然的形状 */}
      <group position={[-0.15, 1.1, 0]}>
        {/* 上臂 */}
        <mesh position={[0, -0.15, 0]}>
          <cylinderGeometry args={[0.035, 0.04, 0.3, 16]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
        
        {/* 肘部 */}
        <mesh position={[0, -0.32, 0]}>
          <sphereGeometry args={[0.04, 16, 12]} />
          <meshStandardMaterial
            color={realisticMode ? "#f4a992" : "#ffb8a2"}
            roughness={0.7}
          />
        </mesh>
        
        {/* 前臂 */}
        <mesh position={[0, -0.47, 0]}>
          <cylinderGeometry args={[0.03, 0.035, 0.3, 16]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
      </group>
      
      <group position={[0.15, 1.1, 0]}>
        {/* 上臂 */}
        <mesh position={[0, -0.15, 0]}>
          <cylinderGeometry args={[0.035, 0.04, 0.3, 16]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
        
        {/* 肘部 */}
        <mesh position={[0, -0.32, 0]}>
          <sphereGeometry args={[0.04, 16, 12]} />
          <meshStandardMaterial
            color={realisticMode ? "#f4a992" : "#ffb8a2"}
            roughness={0.7}
          />
        </mesh>
        
        {/* 前臂 */}
        <mesh position={[0, -0.47, 0]}>
          <cylinderGeometry args={[0.03, 0.035, 0.3, 16]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
      </group>
    </group>
  )
}

// 专业手部组件 - 基于真实解剖结构
const ProfessionalHand: React.FC<{
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
  
  // 专业手指配置 - 基于人体解剖学
  const fingerConfigs = useMemo(() => ({
    thumb: { 
      joints: [1, 2, 3, 4], 
      baseRadius: 0.028,
      segments: [
        { length: 0.05, radius: 0.028 },
        { length: 0.04, radius: 0.025 },
        { length: 0.035, radius: 0.022 }
      ],
      baseRotation: [0, 0, Math.PI / 6]
    },
    index: { 
      joints: [5, 6, 7, 8], 
      baseRadius: 0.022,
      segments: [
        { length: 0.055, radius: 0.022 },
        { length: 0.045, radius: 0.02 },
        { length: 0.035, radius: 0.018 }
      ],
      baseRotation: [0, 0, 0]
    },
    middle: { 
      joints: [9, 10, 11, 12], 
      baseRadius: 0.024,
      segments: [
        { length: 0.06, radius: 0.024 },
        { length: 0.05, radius: 0.022 },
        { length: 0.04, radius: 0.02 }
      ],
      baseRotation: [0, 0, 0]
    },
    ring: { 
      joints: [13, 14, 15, 16], 
      baseRadius: 0.021,
      segments: [
        { length: 0.055, radius: 0.021 },
        { length: 0.045, radius: 0.019 },
        { length: 0.035, radius: 0.017 }
      ],
      baseRotation: [0, 0, 0]
    },
    pinky: { 
      joints: [17, 18, 19, 20], 
      baseRadius: 0.018,
      segments: [
        { length: 0.045, radius: 0.018 },
        { length: 0.035, radius: 0.016 },
        { length: 0.028, radius: 0.014 }
      ],
      baseRotation: [0, 0, -Math.PI / 12]
    }
  }), [])
  
  // 生成自然手势或处理关键点数据
  useEffect(() => {
    if (keypoints && keypoints.length === 21) {
      const processedJoints = keypoints.map((kp, index) => {
        const x = (kp.x - 0.5) * scale * 1.5
        const y = (0.5 - kp.y) * scale * 1.5
        const z = kp.z * scale * 0.8
        return new THREE.Vector3(x, y, z)
      })
      setJoints(processedJoints)
    } else {
      // 生成自然的手势姿态
      setJoints(generateNaturalHandPose(scale, handedness))
    }
  }, [keypoints, scale, handedness])
  
  // 手部动画
  useFrame((state) => {
    if (!handRef.current || !isActive) return
    
    const time = state.clock.elapsedTime
    
    // 细微的手部动作
    const breathingFactor = 1 + Math.sin(time * 1.2) * 0.01
    const naturalSway = Math.sin(time * 0.8) * 0.02
    
    handRef.current.scale.setScalar(breathingFactor)
    handRef.current.rotation.z = naturalSway * (handedness === 'left' ? -1 : 1)
    handRef.current.position.y += Math.sin(time * 1.5) * 0.005
  })
  
  const renderProfessionalFinger = (
    fingerName: keyof typeof fingerConfigs, 
    fingerJoints: THREE.Vector3[]
  ) => {
    const config = fingerConfigs[fingerName]
    if (fingerJoints.length < 4) return null
    
    return (
      <group key={fingerName}>
        {config.segments.map((segment, index) => {
          if (index >= fingerJoints.length - 1) return null
          
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
          
          return (
            <group key={index}>
              {/* 手指段 - 更自然的形状 */}
              <group position={[center.x, center.y, center.z]} rotation={[euler.x, euler.y, euler.z]}>
                <mesh>
                  <cylinderGeometry args={[
                    segment.radius * 0.9, 
                    segment.radius, 
                    length, 
                    16
                  ]} />
                  <meshStandardMaterial
                    color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
                    roughness={0.6}
                    metalness={0.05}
                  />
                </mesh>
              </group>
              
              {/* 关节 */}
              <mesh position={[current.x, current.y, current.z]}>
                <sphereGeometry args={[segment.radius * 0.7, 12, 8]} />
                <meshStandardMaterial
                  color={realisticMode ? "#f4a992" : "#ffb8a2"}
                  roughness={0.7}
                  metalness={0.02}
                />
              </mesh>
              
              {/* 指尖 */}
              {index === config.segments.length - 1 && (
                <group position={[next.x, next.y, next.z]}>
                  <mesh>
                    <sphereGeometry args={[segment.radius * 0.8, 16, 12]} />
                    <meshStandardMaterial
                      color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
                      roughness={0.4}
                      metalness={0.1}
                    />
                  </mesh>
                  
                  {/* 指甲 */}
                  <mesh position={[0, 0, segment.radius * 0.9]}>
                    <boxGeometry args={[
                      segment.radius * 1.2, 
                      segment.radius * 0.6, 
                      segment.radius * 0.2
                    ]} />
                    <meshStandardMaterial
                      color="#f8f8ff"
                      roughness={0.1}
                      metalness={0.3}
                      transparent
                      opacity={0.9}
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
  
  if (joints.length === 0) return null
  
  return (
    <group ref={handRef} position={position}>
      {/* 专业手掌 */}
      <ProfessionalPalm 
        joints={joints}
        isActive={isActive}
        handedness={handedness}
        showBones={showBones}
        realisticMode={realisticMode}
      />
      
      {/* 专业手指 */}
      {Object.entries(fingerConfigs).map(([fingerName, config]) => {
        const fingerJoints = config.joints.map(idx => joints[idx]).filter(Boolean)
        return renderProfessionalFinger(fingerName as keyof typeof fingerConfigs, fingerJoints)
      })}
      
      {/* 手腕 */}
      <group position={[joints[0]?.x || 0, joints[0]?.y || 0, joints[0]?.z || 0]}>
        <mesh>
          <cylinderGeometry args={[0.035, 0.04, 0.08, 16]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
          />
        </mesh>
      </group>
    </group>
  )
}

// 专业手掌组件
const ProfessionalPalm: React.FC<{
  joints: THREE.Vector3[]
  isActive: boolean
  handedness: 'left' | 'right'
  showBones: boolean
  realisticMode: boolean
}> = ({ joints, isActive, handedness, showBones, realisticMode }) => {
  if (joints.length < 21) return null
  
  const wrist = joints[0]
  const indexMcp = joints[5]
  const middleMcp = joints[9]
  const ringMcp = joints[13]
  const pinkyMcp = joints[17]
  
  // 计算手掌几何
  const palmCenter = new THREE.Vector3()
    .addVectors(indexMcp, pinkyMcp)
    .add(wrist)
    .divideScalar(3)
  
  return (
    <group>
      {/* 手掌主体 - 解剖学准确的形状 */}
      <group position={[palmCenter.x, palmCenter.y, palmCenter.z]}>
        {/* 手掌基础形状 */}
        <mesh scale={[1.2, 1.0, 0.4]}>
          <boxGeometry args={[0.08, 0.1, 0.06]} />
          <meshStandardMaterial
            color={realisticMode ? "#fdbcb4" : "#ffcdb2"}
            roughness={0.6}
            metalness={0.02}
          />
        </mesh>
        
        {/* 手掌纹理细节 */}
        <mesh position={[0, 0, 0.025]} scale={[1.15, 0.95, 0.3]}>
          <boxGeometry args={[0.08, 0.1, 0.04]} />
          <meshStandardMaterial
            color={realisticMode ? "#f4a992" : "#ffb8a2"}
            roughness={0.8}
            metalness={0.0}
            transparent
            opacity={0.6}
          />
        </mesh>
        
        {/* 掌心凹陷 */}
        <mesh position={[0, -0.01, 0.02]} scale={[0.8, 0.7, 0.5]}>
          <sphereGeometry args={[0.02, 16, 12]} />
          <meshStandardMaterial
            color={realisticMode ? "#e89b8c" : "#ff9999"}
            roughness={0.9}
            metalness={0.0}
            transparent
            opacity={0.3}
          />
        </mesh>
      </group>
      
      {/* 手掌连接部分 */}
      {[indexMcp, middleMcp, ringMcp, pinkyMcp].map((mcp, index) => (
        <mesh key={index} position={[mcp.x, mcp.y, mcp.z]}>
          <sphereGeometry args={[0.015, 12, 8]} />
          <meshStandardMaterial
            color={realisticMode ? "#f4a992" : "#ffb8a2"}
            roughness={0.7}
          />
        </mesh>
      ))}
    </group>
  )
}

// 生成自然手势姿态
const generateNaturalHandPose = (scale: number, handedness: 'left' | 'right'): THREE.Vector3[] => {
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
const RealisticSignLanguageAvatar: React.FC<RealisticAvatarProps> = ({
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
        powerPreference: "high-performance",
        shadowMapType: THREE.PCFSoftShadowMap
      }}
      dpr={[1, 2]}
    >
      {/* 环境设置 */}
      <Environment preset="apartment" />
      <ContactShadows 
        position={[0, -1.5, 0]} 
        opacity={0.3} 
        scale={15} 
        blur={2.5} 
        far={3} 
      />
      
      {/* 专业照明 */}
      <ambientLight intensity={0.4} />
      <directionalLight 
        position={[5, 8, 5]} 
        intensity={1.0} 
        castShadow
        shadow-mapSize-width={4096}
        shadow-mapSize-height={4096}
        shadow-camera-far={50}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
      />
      <pointLight position={[-3, 3, -3]} intensity={0.3} color="#ffd4c4" />
      <pointLight position={[3, 2, 3]} intensity={0.2} color="#c4d4ff" />
      
      {/* Avatar主体 */}
      <group ref={avatarRef} position={[0, 0, 0]}>
        {/* 身体 */}
        <RealisticHumanBody 
          isPerforming={isPerforming}
          realisticMode={realisticMode}
        />
        
        {/* 左手 */}
        <ProfessionalHand
          handedness="left"
          keypoints={leftHandKeypoints}
          isActive={isPerforming}
          showBones={showBones}
          position={[-0.15, 0.48, 0.1]}
          scale={0.8}
          realisticMode={realisticMode}
        />
        
        {/* 右手 */}
        <ProfessionalHand
          handedness="right"
          keypoints={rightHandKeypoints}
          isActive={isPerforming}
          showBones={showBones}
          position={[0.15, 0.48, 0.1]}
          scale={0.8}
          realisticMode={realisticMode}
        />
        
        {/* 显示当前手语文本 */}
        {signText && (
          <Float speed={1} rotationIntensity={0.1} floatIntensity={0.2}>
            <Text
              position={[0, 2.2, 0]}
              fontSize={0.15}
              color={realisticMode ? "#2c5aa0" : "#ff6b6b"}
              anchorX="center"
              anchorY="middle"
              font="/fonts/SimHei.woff"
            >
              {signText}
            </Text>
          </Float>
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
        autoRotate={false}
        autoRotateSpeed={0.5}
      />
      
      {/* 背景 */}
      <fog attach="fog" args={['#f0f8ff', 15, 30]} />
    </Canvas>
  )
}

export default RealisticSignLanguageAvatar
export type { RealisticAvatarProps }
