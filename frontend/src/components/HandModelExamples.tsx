/**
 * DetailedHandModel 快速使用指南
 */

import React from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
import DetailedHandModel from './DetailedHandModel'

// 基础使用示例
export const BasicHandExample = () => {
  return (
    <Canvas camera={{ position: [0, 0, 8] }}>
      <Environment preset="studio" />
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 8]} intensity={1.2} castShadow />
      
      <DetailedHandModel
        handedness="right"
        isActive={true}
        position={[0, 0, 0]}
        scale={1.0}
        gestureMode="natural"
        enableAnimations={true}
      />
      
      <OrbitControls />
    </Canvas>
  )
}

// 高级功能示例
export const AdvancedHandExample = () => {
  const handleGestureComplete = (gesture: string) => {
    console.log(`手势完成: ${gesture}`)
  }
  
  const handleCollision = (fingerIndex: number, object: THREE.Object3D) => {
    console.log(`手指 ${fingerIndex} 碰撞检测`)
  }
  
  const handleHapticFeedback = (intensity: number, duration: number) => {
    // 触觉反馈处理
    if (navigator.vibrate) {
      navigator.vibrate(duration)
    }
  }

  return (
    <Canvas camera={{ position: [0, 0, 8] }} shadows>
      <Environment preset="studio" />
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 8]} intensity={1.2} castShadow />
      
      <DetailedHandModel
        handedness="right"
        isActive={true}
        position={[0, 0, 0]}
        scale={1.2}
        color="#fdbcb4"
        gestureMode="fist"
        enableAnimations={true}
        enablePhysics={true}
        enableHaptics={true}
        detailLevel="ultra"
        skinTexture="/textures/realistic-skin.jpg"
        onGestureComplete={handleGestureComplete}
        onCollision={handleCollision}
        onHapticFeedback={handleHapticFeedback}
      />
      
      <OrbitControls />
    </Canvas>
  )
}

// 双手对比示例
export const DualHandExample = () => {
  return (
    <Canvas camera={{ position: [0, 0, 10] }}>
      <Environment preset="studio" />
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 8]} intensity={1.2} castShadow />
      
      {/* 左手 - 和平手势 */}
      <DetailedHandModel
        handedness="left"
        isActive={true}
        position={[-3, 0, 0]}
        scale={1.0}
        gestureMode="peace"
        enableAnimations={true}
        detailLevel="high"
      />
      
      {/* 右手 - OK手势 */}
      <DetailedHandModel
        handedness="right"
        isActive={true}
        position={[3, 0, 0]}
        scale={1.0}
        gestureMode="ok"
        enableAnimations={true}
        detailLevel="high"
      />
      
      <OrbitControls />
    </Canvas>
  )
}

// 性能优化示例
export const PerformanceOptimizedExample = () => {
  return (
    <Canvas camera={{ position: [0, 0, 8] }}>
      <Environment preset="apartment" />
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      
      <DetailedHandModel
        handedness="right"
        isActive={true}
        position={[0, 0, 0]}
        scale={1.0}
        gestureMode="natural"
        enableAnimations={true}
        enablePhysics={false} // 禁用物理计算以提升性能
        enableHaptics={false} // 禁用触觉反馈
        detailLevel="medium"  // 中等细节等级
      />
      
      <OrbitControls />
    </Canvas>
  )
}

export default {
  BasicHandExample,
  AdvancedHandExample,
  DualHandExample,
  PerformanceOptimizedExample
}
