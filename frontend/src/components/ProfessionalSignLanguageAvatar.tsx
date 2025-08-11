/**
 * 专业手语Avatar组件 - 针对手语识别优化
 * 特点：
 * 1. 高精度手部建模
 * 2. 专业手语动作库
 * 3. 实时手语表达
 * 4. 写实渲染效果
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
  ContactShadows
} from '@react-three/drei'
import * as THREE from 'three'

// 专业手语关键点映射
const SIGN_LANGUAGE_LANDMARKS = {
  // MediaPipe Holistic 543个关键点中的手语相关点
  LEFT_HAND_START: 468,
  LEFT_HAND_END: 488,
  RIGHT_HAND_START: 489,
  RIGHT_HAND_END: 509,
  FACE_START: 0,
  FACE_END: 467,
  POSE_START: 489,
  POSE_END: 532
} as const

interface SignLanguageKeypoint {
  x: number
  y: number
  z: number
  visibility: number
}

interface SignLanguageAvatarProps {
  // 手语相关属性
  signText: string
  isPerforming: boolean
  signKeypoints?: SignLanguageKeypoint[]
  leftHandKeypoints?: SignLanguageKeypoint[]
  rightHandKeypoints?: SignLanguageKeypoint[]
  faceKeypoints?: SignLanguageKeypoint[]
  poseKeypoints?: SignLanguageKeypoint[]
  
  // 显示设置
  showBones?: boolean
  showWireframe?: boolean
  realisticMode?: boolean
  animationSpeed?: number
  
  // 回调函数
  onAvatarReady?: (avatar: THREE.Object3D) => void
  onSignComplete?: (signText: string) => void
}

// 专业手部组件 - 针对手语优化
const ProfessionalHandModel: React.FC<{
  handedness: 'left' | 'right'
  keypoints?: SignLanguageKeypoint[]
  isActive: boolean
  showBones: boolean
  position: [number, number, number]
  scale: number
}> = ({ handedness, keypoints, isActive, showBones, position, scale }) => {
  const handRef = useRef<THREE.Group>(null)
  const [joints, setJoints] = useState<THREE.Vector3[]>([])
  
  // 手语专用手指配置
  const fingerConfigs = useMemo(() => ({
    thumb: { 
      joints: [1, 2, 3, 4], 
      baseRadius: 0.04, 
      flexibility: 0.8,
      range: { pitch: [-30, 90], yaw: [-60, 60], roll: [-45, 45] }
    },
    index: { 
      joints: [5, 6, 7, 8], 
      baseRadius: 0.035, 
      flexibility: 0.9,
      range: { pitch: [-10, 90], yaw: [-25, 25], roll: [-20, 20] }
    },
    middle: { 
      joints: [9, 10, 11, 12], 
      baseRadius: 0.038, 
      flexibility: 0.9,
      range: { pitch: [-10, 90], yaw: [-20, 20], roll: [-15, 15] }
    },
    ring: { 
      joints: [13, 14, 15, 16], 
      baseRadius: 0.033, 
      flexibility: 0.85,
      range: { pitch: [-10, 90], yaw: [-20, 20], roll: [-15, 15] }
    },
    pinky: { 
      joints: [17, 18, 19, 20], 
      baseRadius: 0.028, 
      flexibility: 0.8,
      range: { pitch: [-10, 90], yaw: [-30, 30], roll: [-20, 20] }
    }
  }), [])
  
  // 处理MediaPipe关键点数据
  useEffect(() => {
    if (keypoints && keypoints.length === 21) {
      const processedJoints = keypoints.map((kp, index) => {
        // 坐标归一化和缩放
        const x = (kp.x - 0.5) * scale * 2
        const y = (0.5 - kp.y) * scale * 2  
        const z = kp.z * scale
        
        return new THREE.Vector3(x, y, z)
      })
      setJoints(processedJoints)
    } else {
      // 使用默认的自然手势
      setJoints(generateNaturalHandPose(scale))
    }
  }, [keypoints, scale])
  
  // 实时手语动画
  useFrame((state) => {
    if (!handRef.current || !isActive) return
    
    // 手语表达时的微妙动作
    const time = state.clock.elapsedTime
    const breathingFactor = 1 + Math.sin(time * 1.5) * 0.02
    const microMovement = Math.sin(time * 2.3) * 0.01
    
    handRef.current.scale.setScalar(breathingFactor)
    handRef.current.position.y += microMovement
    
    // 手指的自然摆动
    handRef.current.rotation.z = Math.sin(time * 0.8) * 0.05
  })
  
  const renderFinger = (fingerName: keyof typeof fingerConfigs, joints: THREE.Vector3[]) => {
    const config = fingerConfigs[fingerName]
    if (joints.length < 4) return null
    
    return (
      <group key={fingerName}>
        {joints.map((joint, index) => {
          if (index === joints.length - 1) return null
          
          const current = joints[index]
          const next = joints[index + 1]
          const direction = new THREE.Vector3().subVectors(next, current)
          const length = direction.length()
          const center = new THREE.Vector3().addVectors(current, next).multiplyScalar(0.5)
          
          // 计算旋转
          const quaternion = new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.normalize()
          )
          const euler = new THREE.Euler().setFromQuaternion(quaternion)
          
          // 动态半径计算
          const progress = index / (joints.length - 2)
          const radius = config.baseRadius * (1 - progress * 0.3)
          
          return (
            <group key={index}>
              {/* 手指段 */}
              <group position={[center.x, center.y, center.z]} rotation={[euler.x, euler.y, euler.z]}>
                <Cylinder args={[radius * 0.8, radius, length * 0.9, 16]}>
                  <meshStandardMaterial
                    color={isActive ? "#ffd4c4" : "#ffb3a7"}
                    roughness={0.4}
                    metalness={0.1}
                  />
                </Cylinder>
              </group>
              
              {/* 关节 */}
              <Sphere args={[radius * 0.6, 16, 16]} position={[current.x, current.y, current.z]}>
                <meshStandardMaterial
                  color={isActive ? "#ff9999" : "#ffb3a7"}
                  roughness={0.3}
                  metalness={0.2}
                />
              </Sphere>
              
              {/* 指尖特效 */}
              {index === joints.length - 2 && (
                <group position={[next.x, next.y, next.z]}>
                  {/* 指尖 */}
                  <Sphere args={[radius * 0.9, 20, 20]}>
                    <meshStandardMaterial
                      color={isActive ? "#ffb3a7" : "#ff9999"}
                      roughness={0.2}
                      metalness={0.3}
                    />
                  </Sphere>
                  
                  {/* 指甲 */}
                  <RoundedBox 
                    args={[radius * 1.2, radius * 0.6, radius * 0.3]}
                    position={[0, 0, radius * 0.8]}
                    radius={0.005}
                  >
                    <meshStandardMaterial
                      color="#f8f8ff"
                      roughness={0.1}
                      metalness={0.8}
                      transparent
                      opacity={0.9}
                    />
                  </RoundedBox>
                </group>
              )}
              
              {/* 骨骼显示（调试用） */}
              {showBones && (
                <mesh position={[center.x, center.y, center.z]} rotation={[euler.x, euler.y, euler.z]}>
                  <boxGeometry args={[0.005, length, 0.005]} />
                  <meshBasicMaterial color="#ff0000" />
                </mesh>
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
      {/* 手掌 */}
      <ProfessionalPalm 
        joints={joints}
        isActive={isActive}
        handedness={handedness}
        showBones={showBones}
      />
      
      {/* 手指 */}
      {Object.entries(fingerConfigs).map(([fingerName, config]) => {
        const fingerJoints = config.joints.map(idx => joints[idx]).filter(Boolean)
        return renderFinger(fingerName as keyof typeof fingerConfigs, fingerJoints)
      })}
    </group>
  )
}

// 专业手掌组件
const ProfessionalPalm: React.FC<{
  joints: THREE.Vector3[]
  isActive: boolean
  handedness: 'left' | 'right'
  showBones: boolean
}> = ({ joints, isActive, handedness, showBones }) => {
  if (joints.length < 21) return null
  
  const wrist = joints[0]
  const indexMcp = joints[5]
  const middleMcp = joints[9]
  const ringMcp = joints[13]
  const pinkyMcp = joints[17]
  const thumbCmc = joints[1]
  
  // 计算手掌中心和方向
  const palmCenter = new THREE.Vector3()
    .addVectors(indexMcp, pinkyMcp)
    .add(wrist)
    .divideScalar(3)
  
  return (
    <group>
      {/* 手掌主体 - 更解剖学准确 */}
      <group position={[palmCenter.x, palmCenter.y, palmCenter.z]}>
        {/* 手掌底座 */}
        <mesh scale={[1.4, 1.1, 0.6]}>
          <sphereGeometry args={[0.08, 24, 16]} />
          <meshStandardMaterial
            color={isActive ? "#ffd4c4" : "#ffb3a7"}
            roughness={0.6}
            metalness={0.05}
          />
        </mesh>
        
        {/* 手掌表面纹理 */}
        <mesh position={[0, 0, 0.02]} scale={[1.35, 1.05, 0.5]}>
          <sphereGeometry args={[0.08, 16, 12]} />
          <meshStandardMaterial
            color="#f4c2a1"
            roughness={0.8}
            metalness={0.0}
            transparent
            opacity={0.6}
          />
        </mesh>
      </group>
      
      {/* 大鱼际肌（拇指侧肌肉） */}
      <group position={[thumbCmc.x - 0.03, thumbCmc.y - 0.04, thumbCmc.z]}>
        <mesh scale={[1.5, 1.2, 0.8]}>
          <sphereGeometry args={[0.035, 16, 12]} />
          <meshStandardMaterial
            color={isActive ? "#ffcc99" : "#ffb3a7"}
            roughness={0.7}
            metalness={0.0}
          />
        </mesh>
      </group>
      
      {/* 小鱼际肌（小指侧肌肉） */}
      <group position={[pinkyMcp.x + 0.02, pinkyMcp.y - 0.03, pinkyMcp.z]}>
        <mesh scale={[0.9, 1.3, 0.7]}>
          <sphereGeometry args={[0.025, 12, 10]} />
          <meshStandardMaterial
            color={isActive ? "#ffcc99" : "#ffb3a7"}
            roughness={0.7}
            metalness={0.0}
          />
        </mesh>
      </group>
      
      {/* 掌骨连接 */}
      {[
        { mcp: indexMcp, name: 'index', width: 0.015 },
        { mcp: middleMcp, name: 'middle', width: 0.018 },
        { mcp: ringMcp, name: 'ring', width: 0.016 },
        { mcp: pinkyMcp, name: 'pinky', width: 0.012 }
      ].map((metacarpal) => {
        const direction = new THREE.Vector3().subVectors(metacarpal.mcp, wrist)
        const length = direction.length()
        const center = new THREE.Vector3().addVectors(wrist, metacarpal.mcp).multiplyScalar(0.5)
        
        return (
          <group key={metacarpal.name}>
            {/* 掌骨 */}
            <mesh position={[center.x, center.y, center.z]}>
              <cylinderGeometry args={[metacarpal.width * 0.8, metacarpal.width, length * 0.7, 12]} />
              <meshStandardMaterial
                color={isActive ? "#ffd4c4" : "#ffb3a7"}
                roughness={0.5}
                metalness={0.1}
                transparent
                opacity={0.8}
              />
            </mesh>
            
            {/* MCP关节 */}
            <Sphere args={[metacarpal.width * 1.2, 12, 12]} position={[metacarpal.mcp.x, metacarpal.mcp.y, metacarpal.mcp.z]}>
              <meshStandardMaterial
                color={isActive ? "#ff9999" : "#ffb3a7"}
                roughness={0.4}
                metalness={0.2}
              />
            </Sphere>
          </group>
        )
      })}
      
      {/* 手腕连接 */}
      <group position={[wrist.x, wrist.y - 0.03, wrist.z]}>
        <Cylinder args={[0.04, 0.05, 0.08, 20]}>
          <meshStandardMaterial
            color={isActive ? "#ffd4c4" : "#ffb3a7"}
            roughness={0.5}
            metalness={0.1}
          />
        </Cylinder>
      </group>
    </group>
  )
}

// 生成自然的手势姿态
const generateNaturalHandPose = (scale: number): THREE.Vector3[] => {
  // 基于人体工程学的自然手势
  return [
    // 手腕 (0)
    new THREE.Vector3(0, 0, 0).multiplyScalar(scale),
    
    // 拇指链 (1-4) - 自然的对握位置
    new THREE.Vector3(-0.25, -0.15, 0.10).multiplyScalar(scale),
    new THREE.Vector3(-0.35, -0.10, 0.15).multiplyScalar(scale),
    new THREE.Vector3(-0.42, -0.05, 0.20).multiplyScalar(scale),
    new THREE.Vector3(-0.47, 0, 0.23).multiplyScalar(scale),
    
    // 食指链 (5-8) - 略微弯曲
    new THREE.Vector3(-0.08, 0.30, 0.05).multiplyScalar(scale),
    new THREE.Vector3(-0.06, 0.45, 0.08).multiplyScalar(scale),
    new THREE.Vector3(-0.04, 0.58, 0.10).multiplyScalar(scale),
    new THREE.Vector3(-0.02, 0.68, 0.12).multiplyScalar(scale),
    
    // 中指链 (9-12) - 最长手指
    new THREE.Vector3(0.02, 0.32, 0.03).multiplyScalar(scale),
    new THREE.Vector3(0.04, 0.50, 0.05).multiplyScalar(scale),
    new THREE.Vector3(0.06, 0.66, 0.07).multiplyScalar(scale),
    new THREE.Vector3(0.08, 0.78, 0.09).multiplyScalar(scale),
    
    // 无名指链 (13-16) - 跟随中指
    new THREE.Vector3(0.12, 0.30, 0.01).multiplyScalar(scale),
    new THREE.Vector3(0.14, 0.46, 0.03).multiplyScalar(scale),
    new THREE.Vector3(0.16, 0.60, 0.05).multiplyScalar(scale),
    new THREE.Vector3(0.18, 0.71, 0.07).multiplyScalar(scale),
    
    // 小指链 (17-20) - 最小，更弯曲
    new THREE.Vector3(0.22, 0.25, -0.02).multiplyScalar(scale),
    new THREE.Vector3(0.24, 0.36, 0).multiplyScalar(scale),
    new THREE.Vector3(0.26, 0.45, 0.02).multiplyScalar(scale),
    new THREE.Vector3(0.28, 0.52, 0.04).multiplyScalar(scale),
  ]
}

// 主Avatar组件
const SignLanguageAvatar: React.FC<SignLanguageAvatarProps> = ({
  signText,
  isPerforming,
  leftHandKeypoints,
  rightHandKeypoints,
  showBones = false,
  showWireframe = false,
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
    >
      {/* 环境设置 */}
      <Environment preset="studio" />
      <ContactShadows 
        position={[0, -2, 0]} 
        opacity={0.4} 
        scale={10} 
        blur={2} 
        far={4} 
      />
      
      {/* 照明 */}
      <ambientLight intensity={0.3} />
      <directionalLight 
        position={[5, 5, 5]} 
        intensity={0.8} 
        castShadow
        shadow-mapSize-width={4096}
        shadow-mapSize-height={4096}
      />
      <pointLight position={[-5, 2, -5]} intensity={0.4} color="#ffd4c4" />
      
      {/* Avatar主体 */}
      <group ref={avatarRef}>
        {/* 头部 - 简化但专业 */}
        <group position={[0, 1.6, 0]}>
          <Sphere args={[0.35, 32, 32]}>
            <meshStandardMaterial
              color={realisticMode ? "#ffd4c4" : "#ffb3a7"}
              roughness={0.4}
              metalness={0.1}
            />
          </Sphere>
          
          {/* 面部特征 */}
          <Sphere args={[0.04, 16, 16]} position={[-0.12, 0.1, 0.3]}>
            <meshStandardMaterial color="#2c3e50" />
          </Sphere>
          <Sphere args={[0.04, 16, 16]} position={[0.12, 0.1, 0.3]}>
            <meshStandardMaterial color="#2c3e50" />
          </Sphere>
          
          {/* 嘴部 */}
          <RoundedBox args={[0.15, 0.04, 0.02]} position={[0, -0.08, 0.32]} radius={0.02}>
            <meshStandardMaterial color="#e74c3c" />
          </RoundedBox>
        </group>
        
        {/* 身体 */}
        <group position={[0, 0.3, 0]}>
          <RoundedBox args={[0.6, 1.2, 0.3]} radius={0.08}>
            <meshStandardMaterial
              color={realisticMode ? "#4a90e2" : "#5ca3f5"}
              roughness={0.6}
              metalness={0.1}
            />
          </RoundedBox>
        </group>
        
        {/* 左臂 */}
        <group position={[-0.45, 0.8, 0]}>
          <RoundedBox args={[0.12, 0.5, 0.12]} radius={0.03}>
            <meshStandardMaterial
              color={realisticMode ? "#ffd4c4" : "#ffb3a7"}
              roughness={0.4}
              metalness={0.1}
            />
          </RoundedBox>
        </group>
        
        {/* 左前臂 */}
        <group position={[-0.45, 0.1, 0]}>
          <RoundedBox args={[0.10, 0.4, 0.10]} radius={0.02}>
            <meshStandardMaterial
              color={realisticMode ? "#ffd4c4" : "#ffb3a7"}
              roughness={0.4}
              metalness={0.1}
            />
          </RoundedBox>
        </group>
        
        {/* 右臂 */}
        <group position={[0.45, 0.8, 0]}>
          <RoundedBox args={[0.12, 0.5, 0.12]} radius={0.03}>
            <meshStandardMaterial
              color={realisticMode ? "#ffd4c4" : "#ffb3a7"}
              roughness={0.4}
              metalness={0.1}
            />
          </RoundedBox>
        </group>
        
        {/* 右前臂 */}
        <group position={[0.45, 0.1, 0]}>
          <RoundedBox args={[0.10, 0.4, 0.10]} radius={0.02}>
            <meshStandardMaterial
              color={realisticMode ? "#ffd4c4" : "#ffb3a7"}
              roughness={0.4}
              metalness={0.1}
            />
          </RoundedBox>
        </group>
        
        {/* 专业手部模型 */}
        <ProfessionalHandModel
          handedness="left"
          keypoints={leftHandKeypoints}
          isActive={isPerforming}
          showBones={showBones}
          position={[-0.45, -0.25, 0]}
          scale={0.3}
        />
        
        <ProfessionalHandModel
          handedness="right"
          keypoints={rightHandKeypoints}
          isActive={isPerforming}
          showBones={showBones}
          position={[0.45, -0.25, 0]}
          scale={0.3}
        />
        
        {/* 手语文本显示 */}
        {signText && (
          <group position={[0, 2.5, 0]}>
            <Text
              fontSize={0.2}
              color="#2c3e50"
              anchorX="center"
              anchorY="middle"
              maxWidth={3}
            >
              {signText}
            </Text>
            
            {/* 文本背景 */}
            <RoundedBox 
              args={[signText.length * 0.12 + 0.8, 0.4, 0.05]} 
              position={[0, 0, -0.1]} 
              radius={0.08}
            >
              <meshStandardMaterial
                color="#ffffff"
                transparent
                opacity={0.9}
                roughness={0.2}
                metalness={0.1}
              />
            </RoundedBox>
          </group>
        )}
      </group>
      
      {/* 相机控制 */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxPolarAngle={Math.PI * 0.7}
        minDistance={2}
        maxDistance={12}
        target={[0, 0.8, 0]}
      />
    </Canvas>
  )
}

export default SignLanguageAvatar
export type { SignLanguageAvatarProps, SignLanguageKeypoint }