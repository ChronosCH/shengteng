/**
 * Three.js 3D Avatar组件 - 使用React Three Fiber
 * 支持精细手部模型和手语动作
 */

import React, { useRef, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Box, Sphere, Cylinder, RoundedBox } from '@react-three/drei'
import * as THREE from 'three'
import DetailedHandModel from './DetailedHandModel'

interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility?: number
}

interface AvatarMeshProps {
  text: string
  isActive: boolean
  animationType: string
  signSequence?: {
    keypoints: number[][][]
    timestamps: number[]
    duration: number
  }
  leftHandKeypoints?: HandKeypoint[]
  rightHandKeypoints?: HandKeypoint[]
  onMeshReady?: (mesh: THREE.Object3D) => void
}

// 3D Avatar网格组件
const AvatarMesh: React.FC<AvatarMeshProps> = ({
  text,
  isActive,
  animationType,
  signSequence,
  leftHandKeypoints,
  rightHandKeypoints,
  onMeshReady
}) => {
  const meshRef = useRef<THREE.Group>(null)
  const [hovered, setHovered] = useState(false)
  const [animationTime, setAnimationTime] = useState(0)
  const [isPlayingSequence, setIsPlayingSequence] = useState(false)

  // 处理手语序列播放
  useEffect(() => {
    if (signSequence && signSequence.keypoints.length > 0) {
      setIsPlayingSequence(true)
      setAnimationTime(0)

      // 播放完成后重置
      const timer = setTimeout(() => {
        setIsPlayingSequence(false)
      }, signSequence.duration * 1000)

      return () => clearTimeout(timer)
    }
  }, [signSequence])

  // 网格就绪回调
  useEffect(() => {
    if (meshRef.current && onMeshReady) {
      onMeshReady(meshRef.current)
    }
  }, [onMeshReady])

  // 应用手语关键点到Avatar
  const applySignKeypoints = (group: THREE.Group, keypoints: number[][]) => {
    // 这里简化处理，实际应该根据MediaPipe关键点映射到3D模型
    if (keypoints.length < 543) return

    // 左手关键点 (468-488)
    const leftHandPoints = keypoints.slice(468, 489)
    // 右手关键点 (489-509)
    const rightHandPoints = keypoints.slice(489, 510)

    // 应用手部动作（简化版本）
    if (leftHandPoints.length > 0 && rightHandPoints.length > 0) {
      const leftHandCenter = leftHandPoints[0] // 手腕位置
      const rightHandCenter = rightHandPoints[0]

      // 调整Avatar手臂位置
      group.children.forEach((child) => {
        if (child.userData.type === 'leftArm') {
          child.position.set(leftHandCenter[0] * 2, leftHandCenter[1] * 2, leftHandCenter[2] * 2)
        } else if (child.userData.type === 'rightArm') {
          child.position.set(rightHandCenter[0] * 2, rightHandCenter[1] * 2, rightHandCenter[2] * 2)
        }
      })
    }
  }

  // 动画循环
  useFrame((state, delta) => {
    if (!meshRef.current) return

    // 如果正在播放手语序列
    if (isPlayingSequence && signSequence) {
      setAnimationTime(prev => prev + delta)

      // 根据时间找到对应的关键点帧
      const frameIndex = Math.floor((animationTime / signSequence.duration) * signSequence.keypoints.length)

      if (frameIndex < signSequence.keypoints.length) {
        const currentFrame = signSequence.keypoints[frameIndex]

        // 应用手语关键点到Avatar
        applySignKeypoints(meshRef.current, currentFrame)
      }
    } else {
      // 基础旋转动画
      if (isActive) {
        meshRef.current.rotation.y += delta * 0.2
      }

      // 根据动画类型调整姿态
      switch (animationType) {
        case '挥手':
          meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 3) * 0.1
          break
        case '鞠躬':
          meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 2) * 0.2 - 0.1
          break
        case '告别':
          meshRef.current.rotation.y += delta * 0.5
          break
        case '手语表达':
          meshRef.current.position.y = Math.sin(state.clock.elapsedTime * 2) * 0.1
          break
        default:
          // 待机状态的轻微浮动
          meshRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.05
      }
    }

    // 悬停效果
    if (hovered) {
      meshRef.current.scale.setScalar(1.1)
    } else if (!isPlayingSequence) {
      meshRef.current.scale.setScalar(1)
    }
  })

  return (
    <group 
      ref={meshRef}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* 优化的头部 - 更人性化 */}
      <group position={[0, 1.5, 0]}>
        {/* 主头部 */}
        <Sphere args={[0.7, 32, 32]}>
          <meshStandardMaterial 
            color={isActive ? "#ffcc80" : "#ffab91"} 
            roughness={0.6}
            metalness={0.1}
          />
        </Sphere>
        
        {/* 面部特征 */}
        {/* 眼睛 */}
        <Sphere args={[0.08, 16, 16]} position={[-0.18, 0.15, 0.6]}>
          <meshStandardMaterial color="#2e2e2e" />
        </Sphere>
        <Sphere args={[0.08, 16, 16]} position={[0.18, 0.15, 0.6]}>
          <meshStandardMaterial color="#2e2e2e" />
        </Sphere>
        
        {/* 眼球高光 */}
        <Sphere args={[0.02, 8, 8]} position={[-0.16, 0.17, 0.65]}>
          <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.3} />
        </Sphere>
        <Sphere args={[0.02, 8, 8]} position={[0.2, 0.17, 0.65]}>
          <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.3} />
        </Sphere>
        
        {/* 鼻子 */}
        <Sphere args={[0.03, 8, 8]} position={[0, 0.05, 0.65]}>
          <meshStandardMaterial color={isActive ? "#ffb74d" : "#ff8a65"} />
        </Sphere>
        
        {/* 嘴巴 - 更立体 */}
        <RoundedBox args={[0.25, 0.06, 0.08]} position={[0, -0.1, 0.6]} radius={0.03}>
          <meshStandardMaterial 
            color={isActive ? "#f48fb1" : "#e57373"} 
          />
        </RoundedBox>
      </group>

      {/* 优化的身体 - 更自然的形状 */}
      <group position={[0, 0, 0]}>
        {/* 躯干 */}
        <RoundedBox args={[1.0, 1.8, 0.5]} radius={0.1}>
          <meshStandardMaterial 
            color={isActive ? "#81c784" : "#a5d6a7"} 
            roughness={0.4}
            metalness={0.1}
          />
        </RoundedBox>
        
        {/* 胸部细节 */}
        <RoundedBox args={[0.8, 0.6, 0.3]} position={[0, 0.4, 0.1]} radius={0.05}>
          <meshStandardMaterial 
            color={isActive ? "#66bb6a" : "#81c784"} 
            roughness={0.5}
          />
        </RoundedBox>
      </group>

      {/* 优化的左臂 - 更自然的比例 */}
      <group position={[-0.7, 0.6, 0]}>
        {/* 上臂 */}
        <RoundedBox args={[0.2, 0.7, 0.18]} radius={0.05}>
          <meshStandardMaterial
            color={isActive ? "#ffcc80" : "#ffab91"}
            roughness={0.4}
            metalness={0.1}
          />
        </RoundedBox>
        
        {/* 肩膀关节 */}
        <Sphere args={[0.12, 12, 12]} position={[0, 0.25, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffb74d" : "#ff8a65"}
            roughness={0.3}
          />
        </Sphere>
      </group>

      {/* 左前臂 */}
      <group position={[-0.7, -0.15, 0]}>
        <RoundedBox args={[0.16, 0.6, 0.15]} radius={0.04}>
          <meshStandardMaterial
            color={isActive ? "#ffcc80" : "#ffab91"}
            roughness={0.4}
            metalness={0.1}
          />
        </RoundedBox>
        
        {/* 肘关节 */}
        <Sphere args={[0.1, 12, 12]} position={[0, 0.2, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffb74d" : "#ff8a65"}
            roughness={0.3}
          />
        </Sphere>
      </group>

      {/* 优化的左手 */}
      <DetailedHandModel
        handedness="left"
        keypoints={leftHandKeypoints}
        isActive={isActive}
        position={[-0.7, -0.6, 0]}
        scale={0.25}
        color={isActive ? "#ffcc80" : "#ffab91"}
      />

      {/* 优化的右臂 - 对称设计 */}
      <group position={[0.7, 0.6, 0]}>
        <RoundedBox args={[0.2, 0.7, 0.18]} radius={0.05}>
          <meshStandardMaterial
            color={isActive ? "#ffcc80" : "#ffab91"}
            roughness={0.4}
            metalness={0.1}
          />
        </RoundedBox>
        
        <Sphere args={[0.12, 12, 12]} position={[0, 0.25, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffb74d" : "#ff8a65"}
            roughness={0.3}
          />
        </Sphere>
      </group>

      {/* 右前臂 */}
      <group position={[0.7, -0.15, 0]}>
        <RoundedBox args={[0.16, 0.6, 0.15]} radius={0.04}>
          <meshStandardMaterial
            color={isActive ? "#ffcc80" : "#ffab91"}
            roughness={0.4}
            metalness={0.1}
          />
        </RoundedBox>
        
        <Sphere args={[0.1, 12, 12]} position={[0, 0.2, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffb74d" : "#ff8a65"}
            roughness={0.3}
          />
        </Sphere>
      </group>

      {/* 优化的右手 */}
      <DetailedHandModel
        handedness="right"
        keypoints={rightHandKeypoints}
        isActive={isActive}
        position={[0.7, -0.6, 0]}
        scale={0.25}
        color={isActive ? "#ffcc80" : "#ffab91"}
      />

      {/* 优化的左腿 */}
      <group position={[-0.3, -1.4, 0]}>
        {/* 大腿 */}
        <RoundedBox args={[0.25, 0.8, 0.22]} radius={0.05}>
          <meshStandardMaterial 
            color={isActive ? "#5c6bc0" : "#7986cb"} 
            roughness={0.5}
          />
        </RoundedBox>
        
        {/* 膝盖 */}
        <Sphere args={[0.08, 12, 12]} position={[0, -0.45, 0]}>
          <meshStandardMaterial
            color={isActive ? "#3f51b5" : "#5c6bc0"}
            roughness={0.3}
          />
        </Sphere>
        
        {/* 小腿 */}
        <RoundedBox args={[0.2, 0.7, 0.18]} position={[0, -0.8, 0]} radius={0.04}>
          <meshStandardMaterial 
            color={isActive ? "#5c6bc0" : "#7986cb"} 
            roughness={0.5}
          />
        </RoundedBox>
        
        {/* 脚 */}
        <RoundedBox args={[0.15, 0.08, 0.3]} position={[0, -1.2, 0.1]} radius={0.03}>
          <meshStandardMaterial 
            color={isActive ? "#424242" : "#616161"} 
            roughness={0.7}
          />
        </RoundedBox>
      </group>

      {/* 优化的右腿 */}
      <group position={[0.3, -1.4, 0]}>
        <RoundedBox args={[0.25, 0.8, 0.22]} radius={0.05}>
          <meshStandardMaterial 
            color={isActive ? "#5c6bc0" : "#7986cb"} 
            roughness={0.5}
          />
        </RoundedBox>
        
        <Sphere args={[0.08, 12, 12]} position={[0, -0.45, 0]}>
          <meshStandardMaterial
            color={isActive ? "#3f51b5" : "#5c6bc0"}
            roughness={0.3}
          />
        </Sphere>
        
        <RoundedBox args={[0.2, 0.7, 0.18]} position={[0, -0.8, 0]} radius={0.04}>
          <meshStandardMaterial 
            color={isActive ? "#5c6bc0" : "#7986cb"} 
            roughness={0.5}
          />
        </RoundedBox>
        
        <RoundedBox args={[0.15, 0.08, 0.3]} position={[0, -1.2, 0.1]} radius={0.03}>
          <meshStandardMaterial 
            color={isActive ? "#424242" : "#616161"} 
            roughness={0.7}
          />
        </RoundedBox>
      </group>

      {/* 文本气泡 - 优化样式 */}
      {text && (
        <group position={[0, 3, 0]}>
          <Text
            fontSize={0.25}
            color="#2e2e2e"
            anchorX="center"
            anchorY="middle"
            maxWidth={4}
            font="/fonts/inter.woff"
          >
            {text}
          </Text>
          {/* 气泡背景 - 更圆润 */}
          <RoundedBox args={[text.length * 0.15 + 1, 0.6, 0.08]} position={[0, 0, -0.1]} radius={0.1}>
            <meshStandardMaterial 
              color="#ffffff" 
              transparent 
              opacity={0.95}
              roughness={0.2}
              metalness={0.1}
            />
          </RoundedBox>
          
          {/* 气泡尾巴 */}
          <group position={[0, -0.4, -0.05]} rotation={[0, 0, Math.PI / 4]}>
            <RoundedBox args={[0.15, 0.15, 0.06]} radius={0.03}>
              <meshStandardMaterial 
                color="#ffffff" 
                transparent 
                opacity={0.95}
              />
            </RoundedBox>
          </group>
        </group>
      )}
    </group>
  )
}

// 场景环境组件
const SceneEnvironment: React.FC = () => {
  return (
    <>
      {/* 环境光 */}
      <ambientLight intensity={0.4} />
      
      {/* 主光源 */}
      <directionalLight 
        position={[10, 10, 5]} 
        intensity={1}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      
      {/* 补光 */}
      <pointLight position={[-10, -10, -5]} intensity={0.3} />
      
      {/* 地面 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -3, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#e0e0e0" />
      </mesh>
    </>
  )
}

// 相机控制组件
const CameraController: React.FC = () => {
  const { camera } = useThree()
  
  useEffect(() => {
    camera.position.set(0, 2, 8)
    camera.lookAt(0, 0, 0)
  }, [camera])

  return null
}

interface ThreeAvatarProps {
  text: string
  isActive: boolean
  animationType: string
  signSequence?: {
    keypoints: number[][][]
    timestamps: number[]
    duration: number
  }
  leftHandKeypoints?: HandKeypoint[]
  rightHandKeypoints?: HandKeypoint[]
  onAvatarMeshReady?: (mesh: THREE.Object3D) => void
}

const ThreeAvatar: React.FC<ThreeAvatarProps> = ({
  text,
  isActive,
  animationType,
  signSequence,
  leftHandKeypoints,
  rightHandKeypoints,
  onAvatarMeshReady
}) => {
  return (
    <Canvas
      shadows
      camera={{ position: [0, 2, 8], fov: 50 }}
      style={{ width: '100%', height: '100%' }}
    >
      <CameraController />
      <SceneEnvironment />
      
      <AvatarMesh
        text={text}
        isActive={isActive}
        animationType={animationType}
        signSequence={signSequence}
        leftHandKeypoints={leftHandKeypoints}
        rightHandKeypoints={rightHandKeypoints}
        onMeshReady={onAvatarMeshReady}
      />
      
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxPolarAngle={Math.PI / 2}
        minDistance={3}
        maxDistance={15}
      />
    </Canvas>
  )
}

export default ThreeAvatar
