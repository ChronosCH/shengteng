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
      {/* 更真实的头部 */}
      <group position={[0, 1.6, 0]}>
        {/* 主头部 - 椭圆形更自然 */}
        <mesh scale={[1, 1.1, 0.8]}>
          <sphereGeometry args={[0.65, 32, 32]} />
          <meshStandardMaterial 
            color={isActive ? "#ffd1a9" : "#ffb380"} 
            roughness={0.7}
            metalness={0.1}
            normalScale={[0.3, 0.3]}
          />
        </mesh>
        
        {/* 面部细节 */}
        {/* 更真实的眼睛 */}
        <group position={[-0.22, 0.12, 0.55]}>
          {/* 眼白 */}
          <mesh scale={[1.2, 0.8, 0.6]}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color="#ffffff" roughness={0.1} />
          </mesh>
          {/* 虹膜 */}
          <mesh position={[0, 0, 0.05]} scale={[0.7, 0.7, 0.3]}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color="#4a5d23" roughness={0.3} />
          </mesh>
          {/* 瞳孔 */}
          <mesh position={[0, 0, 0.08]} scale={[0.4, 0.4, 0.2]}>
            <sphereGeometry args={[0.08, 8, 8]} />
            <meshStandardMaterial color="#000000" />
          </mesh>
          {/* 高光 */}
          <mesh position={[0.02, 0.02, 0.09]}>
            <sphereGeometry args={[0.015, 8, 8]} />
            <meshStandardMaterial 
              color="#ffffff" 
              emissive="#ffffff" 
              emissiveIntensity={0.5}
              transparent
              opacity={0.9}
            />
          </mesh>
        </group>
        
        <group position={[0.22, 0.12, 0.55]}>
          <mesh scale={[1.2, 0.8, 0.6]}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color="#ffffff" roughness={0.1} />
          </mesh>
          <mesh position={[0, 0, 0.05]} scale={[0.7, 0.7, 0.3]}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color="#4a5d23" roughness={0.3} />
          </mesh>
          <mesh position={[0, 0, 0.08]} scale={[0.4, 0.4, 0.2]}>
            <sphereGeometry args={[0.08, 8, 8]} />
            <meshStandardMaterial color="#000000" />
          </mesh>
          <mesh position={[-0.02, 0.02, 0.09]}>
            <sphereGeometry args={[0.015, 8, 8]} />
            <meshStandardMaterial 
              color="#ffffff" 
              emissive="#ffffff" 
              emissiveIntensity={0.5}
              transparent
              opacity={0.9}
            />
          </mesh>
        </group>
        
        {/* 眉毛 */}
        <RoundedBox args={[0.25, 0.03, 0.02]} position={[-0.22, 0.25, 0.6]} radius={0.01}>
          <meshStandardMaterial color="#8b4513" roughness={0.9} />
        </RoundedBox>
        <RoundedBox args={[0.25, 0.03, 0.02]} position={[0.22, 0.25, 0.6]} radius={0.01}>
          <meshStandardMaterial color="#8b4513" roughness={0.9} />
        </RoundedBox>
        
        {/* 更立体的鼻子 */}
        <group position={[0, 0.02, 0.6]}>
          <mesh scale={[0.6, 1.2, 1.5]}>
            <sphereGeometry args={[0.04, 12, 12]} />
            <meshStandardMaterial color={isActive ? "#ffc999" : "#ff9966"} roughness={0.6} />
          </mesh>
          {/* 鼻孔 */}
          <mesh position={[-0.015, -0.02, 0.03]} scale={[0.3, 0.5, 0.8]}>
            <sphereGeometry args={[0.01, 6, 6]} />
            <meshStandardMaterial color="#cc6633" roughness={0.9} />
          </mesh>
          <mesh position={[0.015, -0.02, 0.03]} scale={[0.3, 0.5, 0.8]}>
            <sphereGeometry args={[0.01, 6, 6]} />
            <meshStandardMaterial color="#cc6633" roughness={0.9} />
          </mesh>
        </group>
        
        {/* 更真实的嘴巴 */}
        <group position={[0, -0.15, 0.55]}>
          {/* 嘴唇上部 */}
          <RoundedBox args={[0.22, 0.03, 0.05]} radius={0.015}>
            <meshStandardMaterial 
              color={isActive ? "#ff6b9d" : "#e57373"} 
              roughness={0.3}
              metalness={0.1}
            />
          </RoundedBox>
          {/* 嘴唇下部 */}
          <RoundedBox args={[0.26, 0.04, 0.06]} position={[0, -0.02, 0]} radius={0.02}>
            <meshStandardMaterial 
              color={isActive ? "#ff5a8a" : "#d85577"} 
              roughness={0.3}
              metalness={0.1}
            />
          </RoundedBox>
        </group>
        
        {/* 耳朵 */}
        <mesh position={[-0.6, 0, 0.1]} rotation={[0, -0.3, 0]} scale={[0.5, 0.8, 0.6]}>
          <sphereGeometry args={[0.12, 12, 12]} />
          <meshStandardMaterial color={isActive ? "#ffd1a9" : "#ffb380"} roughness={0.7} />
        </mesh>
        <mesh position={[0.6, 0, 0.1]} rotation={[0, 0.3, 0]} scale={[0.5, 0.8, 0.6]}>
          <sphereGeometry args={[0.12, 12, 12]} />
          <meshStandardMaterial color={isActive ? "#ffd1a9" : "#ffb380"} roughness={0.7} />
        </mesh>

        {/* 头发 */}
        <mesh position={[0, 0.3, -0.1]} scale={[1.1, 0.4, 1.2]}>
          <sphereGeometry args={[0.7, 16, 16]} />
          <meshStandardMaterial 
            color="#2d1810" 
            roughness={0.9}
            metalness={0.1}
          />
        </mesh>
        
        {/* 脖子 */}
        <Cylinder args={[0.25, 0.3, 0.4, 16]} position={[0, -0.5, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffd1a9" : "#ffb380"}
            roughness={0.6}
            metalness={0.1}
          />
        </Cylinder>
      </group>

      {/* 更自然的身体 */}
      <group position={[0, 0, 0]}>
        {/* 胸部 - 梯形更自然 */}
        <mesh scale={[1, 1.2, 0.6]}>
          <cylinderGeometry args={[0.6, 0.8, 1.5, 16]} />
          <meshStandardMaterial 
            color={isActive ? "#4fc3f7" : "#81c784"} 
            roughness={0.5}
            metalness={0.1}
          />
        </mesh>
        
        {/* 衣服细节 */}
        <RoundedBox args={[1.4, 1.3, 0.4]} position={[0, 0.1, 0.15]} radius={0.05}>
          <meshStandardMaterial 
            color={isActive ? "#42a5f5" : "#66bb6a"} 
            roughness={0.6}
          />
        </RoundedBox>
        
        {/* 衣服按钮 */}
        <Sphere args={[0.03, 8, 8]} position={[0, 0.4, 0.36]}>
          <meshStandardMaterial color="#ffffff" roughness={0.2} metalness={0.8} />
        </Sphere>
        <Sphere args={[0.03, 8, 8]} position={[0, 0.1, 0.36]}>
          <meshStandardMaterial color="#ffffff" roughness={0.2} metalness={0.8} />
        </Sphere>
        <Sphere args={[0.03, 8, 8]} position={[0, -0.2, 0.36]}>
          <meshStandardMaterial color="#ffffff" roughness={0.2} metalness={0.8} />
        </Sphere>
        
        {/* 腰部 */}
        <Cylinder args={[0.5, 0.6, 0.3, 16]} position={[0, -0.9, 0]}>
          <meshStandardMaterial 
            color={isActive ? "#1976d2" : "#5c6bc0"} 
            roughness={0.7}
          />
        </Cylinder>
      </group>

      {/* 改进的左臂 - 更自然的形状和连接 */}
      <group position={[-0.85, 0.4, 0]} rotation={[0, 0, -0.2]}>
        {/* 肩膀 */}
        <Sphere args={[0.18, 16, 16]} position={[0, 0.3, 0]}>
          <meshStandardMaterial
            color={isActive ? "#4fc3f7" : "#81c784"}
            roughness={0.5}
          />
        </Sphere>
        
        {/* 上臂 */}
        <Cylinder args={[0.12, 0.15, 0.8, 16]} position={[0, -0.1, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffd1a9" : "#ffb380"}
            roughness={0.6}
            metalness={0.1}
          />
        </Cylinder>
        
        {/* 肘关节 */}
        <Sphere args={[0.14, 16, 16]} position={[0, -0.5, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffc999" : "#ff9966"}
            roughness={0.5}
          />
        </Sphere>
      </group>

      {/* 左前臂 */}
      <group position={[-0.85, -0.35, 0]} rotation={[0, 0, -0.1]}>
        <Cylinder args={[0.1, 0.12, 0.7, 16]}>
          <meshStandardMaterial
            color={isActive ? "#ffd1a9" : "#ffb380"}
            roughness={0.6}
            metalness={0.1}
          />
        </Cylinder>
      </group>

      {/* 优化的左手 - 调用高质量手部模型 */}
      <DetailedHandModel
        handedness="left"
        keypoints={leftHandKeypoints}
        isActive={isActive}
        position={[-0.85, -0.85, 0]}
        scale={0.35}
        color={isActive ? "#ffd1a9" : "#ffb380"}
      />

      {/* 改进的右臂 - 对称设计 */}
      <group position={[0.85, 0.4, 0]} rotation={[0, 0, 0.2]}>
        <Sphere args={[0.18, 16, 16]} position={[0, 0.3, 0]}>
          <meshStandardMaterial
            color={isActive ? "#4fc3f7" : "#81c784"}
            roughness={0.5}
          />
        </Sphere>
        
        <Cylinder args={[0.12, 0.15, 0.8, 16]} position={[0, -0.1, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffd1a9" : "#ffb380"}
            roughness={0.6}
            metalness={0.1}
          />
        </Cylinder>
        
        <Sphere args={[0.14, 16, 16]} position={[0, -0.5, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffc999" : "#ff9966"}
            roughness={0.5}
          />
        </Sphere>
      </group>

      {/* 右前臂 */}
      <group position={[0.85, -0.35, 0]} rotation={[0, 0, 0.1]}>
        <Cylinder args={[0.1, 0.12, 0.7, 16]}>
          <meshStandardMaterial
            color={isActive ? "#ffd1a9" : "#ffb380"}
            roughness={0.6}
            metalness={0.1}
          />
        </Cylinder>
      </group>

      {/* 优化的右手 */}
      <DetailedHandModel
        handedness="right"
        keypoints={rightHandKeypoints}
        isActive={isActive}
        position={[0.85, -0.85, 0]}
        scale={0.35}
        color={isActive ? "#ffd1a9" : "#ffb380"}
      />

      {/* 改进的左腿 */}
      <group position={[-0.25, -1.5, 0]}>
        {/* 大腿 */}
        <Cylinder args={[0.18, 0.22, 0.9, 16]} position={[0, -0.1, 0]}>
          <meshStandardMaterial 
            color={isActive ? "#1976d2" : "#5c6bc0"} 
            roughness={0.7}
          />
        </Cylinder>
        
        {/* 膝盖 */}
        <Sphere args={[0.12, 16, 16]} position={[0, -0.55, 0]}>
          <meshStandardMaterial
            color={isActive ? "#1565c0" : "#3f51b5"}
            roughness={0.6}
          />
        </Sphere>
        
        {/* 小腿 */}
        <Cylinder args={[0.14, 0.18, 0.8, 16]} position={[0, -0.95, 0]}>
          <meshStandardMaterial 
            color={isActive ? "#1976d2" : "#5c6bc0"} 
            roughness={0.7}
          />
        </Cylinder>
        
        {/* 脚踝 */}
        <Sphere args={[0.08, 12, 12]} position={[0, -1.35, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffd1a9" : "#ffb380"}
            roughness={0.6}
          />
        </Sphere>
        
        {/* 脚 - 更真实的形状 */}
        <group position={[0, -1.45, 0.12]}>
          <RoundedBox args={[0.2, 0.1, 0.35]} radius={0.03}>
            <meshStandardMaterial 
              color={isActive ? "#2e2e2e" : "#424242"} 
              roughness={0.8}
            />
          </RoundedBox>
          {/* 鞋带 */}
          <RoundedBox args={[0.15, 0.02, 0.25]} position={[0, 0.06, 0]} radius={0.01}>
            <meshStandardMaterial color="#ffffff" roughness={0.3} />
          </RoundedBox>
        </group>
      </group>

      {/* 改进的右腿 */}
      <group position={[0.25, -1.5, 0]}>
        <Cylinder args={[0.18, 0.22, 0.9, 16]} position={[0, -0.1, 0]}>
          <meshStandardMaterial 
            color={isActive ? "#1976d2" : "#5c6bc0"} 
            roughness={0.7}
          />
        </Cylinder>
        
        <Sphere args={[0.12, 16, 16]} position={[0, -0.55, 0]}>
          <meshStandardMaterial
            color={isActive ? "#1565c0" : "#3f51b5"}
            roughness={0.6}
          />
        </Sphere>
        
        <Cylinder args={[0.14, 0.18, 0.8, 16]} position={[0, -0.95, 0]}>
          <meshStandardMaterial 
            color={isActive ? "#1976d2" : "#5c6bc0"} 
            roughness={0.7}
          />
        </Cylinder>
        
        <Sphere args={[0.08, 12, 12]} position={[0, -1.35, 0]}>
          <meshStandardMaterial
            color={isActive ? "#ffd1a9" : "#ffb380"}
            roughness={0.6}
          />
        </Sphere>
        
        <group position={[0, -1.45, 0.12]}>
          <RoundedBox args={[0.2, 0.1, 0.35]} radius={0.03}>
            <meshStandardMaterial 
              color={isActive ? "#2e2e2e" : "#424242"} 
              roughness={0.8}
            />
          </RoundedBox>
          <RoundedBox args={[0.15, 0.02, 0.25]} position={[0, 0.06, 0]} radius={0.01}>
            <meshStandardMaterial color="#ffffff" roughness={0.3} />
          </RoundedBox>
        </group>
      </group>

      {/* 改进的文本气泡 */}
      {text && (
        <group position={[0, 3.2, 0]}>
          <Text
            fontSize={0.3}
            color="#2e2e2e"
            anchorX="center"
            anchorY="middle"
            maxWidth={5}
            font="/fonts/inter.woff"
            outlineWidth={0.01}
            outlineColor="#ffffff"
          >
            {text}
          </Text>
          {/* 气泡背景 - 更立体 */}
          <RoundedBox 
            args={[Math.max(text.length * 0.18, 1.5), 0.8, 0.12]} 
            position={[0, 0, -0.15]} 
            radius={0.15}
          >
            <meshStandardMaterial 
              color="#ffffff" 
              transparent 
              opacity={0.95}
              roughness={0.1}
              metalness={0.1}
              emissive="#f0f8ff"
              emissiveIntensity={0.1}
            />
          </RoundedBox>
          
          {/* 气泡阴影 */}
          <RoundedBox 
            args={[Math.max(text.length * 0.18, 1.5) + 0.1, 0.8, 0.08]} 
            position={[0.05, -0.05, -0.2]} 
            radius={0.15}
          >
            <meshStandardMaterial 
              color="#000000" 
              transparent 
              opacity={0.1}
              roughness={1}
            />
          </RoundedBox>
          
          {/* 气泡尾巴 */}
          <group position={[0, -0.5, -0.1]} rotation={[0, 0, Math.PI / 4]}>
            <RoundedBox args={[0.2, 0.2, 0.08]} radius={0.05}>
              <meshStandardMaterial 
                color="#ffffff" 
                transparent 
                opacity={0.95}
                roughness={0.1}
                metalness={0.1}
              />
            </RoundedBox>
          </group>
        </group>
      )}
    </group>
  )
}

// 优化的场景环境组件
const SceneEnvironment: React.FC = () => {
  return (
    <>
      {/* 环境光 - 更柔和 */}
      <ambientLight intensity={0.6} color="#f0f8ff" />
      
      {/* 主光源 - 模拟自然光 */}
      <directionalLight 
        position={[10, 15, 8]} 
        intensity={1.2}
        color="#fff8e1"
        castShadow
        shadow-mapSize-width={4096}
        shadow-mapSize-height={4096}
        shadow-camera-far={50}
        shadow-camera-left={-20}
        shadow-camera-right={20}
        shadow-camera-top={20}
        shadow-camera-bottom={-20}
      />
      
      {/* 补光 - 填充阴影 */}
      <pointLight position={[-8, 8, 6]} intensity={0.4} color="#e3f2fd" />
      <pointLight position={[8, -5, -8]} intensity={0.3} color="#fff3e0" />
      
      {/* 背景光环 */}
      <spotLight
        position={[0, 20, 0]}
        angle={Math.PI / 3}
        penumbra={0.5}
        intensity={0.5}
        color="#f8bbd9"
        castShadow
      />
      
      {/* 优化的地面 - 更美观 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -3.2, 0]} receiveShadow>
        <circleGeometry args={[25, 64]} />
        <meshStandardMaterial 
          color="#f5f5f5" 
          roughness={0.8}
          metalness={0.1}
          transparent
          opacity={0.9}
        />
      </mesh>
      
      {/* 地面装饰圆环 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -3.18, 0]}>
        <ringGeometry args={[8, 10, 32]} />
        <meshStandardMaterial 
          color="#e8f5e8" 
          transparent
          opacity={0.6}
          roughness={0.9}
        />
      </mesh>
      
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -3.16, 0]}>
        <ringGeometry args={[12, 14, 32]} />
        <meshStandardMaterial 
          color="#fff3e0" 
          transparent
          opacity={0.4}
          roughness={0.9}
        />
      </mesh>

      {/* 背景装饰球体 - 营造空间感 */}
      <group>
        <mesh position={[-15, 8, -10]}>
          <sphereGeometry args={[2, 16, 16]} />
          <meshStandardMaterial 
            color="#ffcdd2" 
            transparent
            opacity={0.3}
            roughness={0.8}
          />
        </mesh>
        
        <mesh position={[18, 6, -15]}>
          <sphereGeometry args={[1.5, 16, 16]} />
          <meshStandardMaterial 
            color="#c8e6c9" 
            transparent
            opacity={0.25}
            roughness={0.8}
          />
        </mesh>
        
        <mesh position={[0, 12, -20]}>
          <sphereGeometry args={[3, 16, 16]} />
          <meshStandardMaterial 
            color="#e1f5fe" 
            transparent
            opacity={0.2}
            roughness={0.9}
          />
        </mesh>
      </group>

      {/* 环境雾效 */}
      <fog attach="fog" args={['#f0f8ff', 25, 60]} />
    </>
  )
}

// 优化的相机控制组件
const CameraController: React.FC = () => {
  const { camera } = useThree()
  
  useEffect(() => {
    // 设置更好的初始视角
    camera.position.set(0, 1, 12)
    camera.lookAt(0, 0.5, 0)
    camera.updateProjectionMatrix()
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
      camera={{ position: [0, 1, 12], fov: 45, near: 0.1, far: 100 }}
      style={{ width: '100%', height: '100%' }}
      gl={{ 
        antialias: true, 
        alpha: true,
        powerPreference: "high-performance",
        shadowMapType: THREE.PCFSoftShadowMap
      }}
      dpr={[1, 2]}
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
        enableDamping={true}
        dampingFactor={0.05}
        maxPolarAngle={Math.PI * 0.8}
        minPolarAngle={Math.PI * 0.1}
        minDistance={5}
        maxDistance={25}
        target={[0, 0.5, 0]}
        autoRotate={false}
        autoRotateSpeed={0.5}
      />
    </Canvas>
  )
}

export default ThreeAvatar
