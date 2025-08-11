/**
 * 详细手部模型演示组件
 * 展示重构后的3D数字人手指细节和手势动画
 */

import React, { useState, useCallback } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei'
import DetailedHandModel from './DetailedHandModel'
import './HandModelDemo.css'

interface HandModelDemoProps {
  className?: string
}

const HandModelDemo: React.FC<HandModelDemoProps> = ({ className }) => {
  const [leftGesture, setLeftGesture] = useState<string>('natural')
  const [rightGesture, setRightGesture] = useState<string>('natural')
  const [enableAnimations, setEnableAnimations] = useState(true)
  const [handScale, setHandScale] = useState(1.0)

  const handleGestureComplete = useCallback((gesture: string) => {
    console.log(`手势动画完成: ${gesture}`)
  }, [])

  const gestureButtons = [
    { key: 'natural', label: '自然状态' },
    { key: 'fist', label: '握拳' },
    { key: 'point', label: '指向' },
    { key: 'ok', label: 'OK手势' },
    { key: 'peace', label: '和平手势' },
    { key: 'wave', label: '挥手' }
  ]

  return (
    <div className={`hand-model-demo ${className || ''}`}>
      {/* 控制面板 */}
      <div className="demo-controls">
        <div className="control-section">
          <h3>🖐️ 真实手部模型演示</h3>
          <p>基于人体解剖学的高精度3D手部建模</p>
        </div>

        <div className="control-section">
          <h4>左手手势</h4>
          <div className="gesture-buttons">
            {gestureButtons.map(({ key, label }) => (
              <button
                key={`left-${key}`}
                className={leftGesture === key ? 'active' : ''}
                onClick={() => setLeftGesture(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="control-section">
          <h4>右手手势</h4>
          <div className="gesture-buttons">
            {gestureButtons.map(({ key, label }) => (
              <button
                key={`right-${key}`}
                className={rightGesture === key ? 'active' : ''}
                onClick={() => setRightGesture(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="control-section">
          <h4>参数调节</h4>
          <div className="slider-control">
            <label>手部缩放: {handScale.toFixed(1)}</label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={handScale}
              onChange={(e) => setHandScale(parseFloat(e.target.value))}
            />
          </div>
          
          <div className="checkbox-control">
            <label>
              <input
                type="checkbox"
                checked={enableAnimations}
                onChange={(e) => setEnableAnimations(e.target.checked)}
              />
              启用动画效果
            </label>
          </div>
        </div>

        <div className="control-section">
          <h4>操作说明</h4>
          <ul>
            <li>拖拽旋转视角</li>
            <li>滚轮缩放</li>
            <li>选择不同手势查看动画</li>
            <li>调节参数观察细节变化</li>
          </ul>
        </div>
      </div>

      {/* 3D 场景 */}
      <div className="canvas-container">
        <Canvas
          camera={{ position: [0, 0, 8], fov: 50 }}
          shadows
          style={{ background: 'linear-gradient(135deg, #2c3e50 0%, #3498db 100%)' }}
        >
          {/* 环境设置 */}
          <Environment preset="studio" />
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[10, 10, 8]}
            intensity={1.2}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <pointLight position={[-8, 5, 6]} intensity={0.6} />
          <pointLight position={[5, -2, 8]} intensity={0.5} color="#ffaa88" />

          {/* 左手模型 */}
          <DetailedHandModel
            handedness="left"
            isActive={true}
            position={[-2, 0, 0]}
            scale={handScale}
            color="#fdbcb4"
            gestureMode={leftGesture as any}
            enableAnimations={enableAnimations}
            onGestureComplete={handleGestureComplete}
          />

          {/* 右手模型 */}
          <DetailedHandModel
            handedness="right"
            isActive={true}
            position={[2, 0, 0]}
            scale={handScale}
            color="#fdbcb4"
            gestureMode={rightGesture as any}
            enableAnimations={enableAnimations}
            onGestureComplete={handleGestureComplete}
          />

          {/* 地面阴影 */}
          <ContactShadows
            position={[0, -3, 0]}
            opacity={0.3}
            scale={10}
            blur={2.5}
            far={4}
          />

          {/* 相机控制 */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={3}
            maxDistance={20}
            target={[0, 0, 0]}
          />
        </Canvas>
      </div>
    </div>
  )
}

export default HandModelDemo
