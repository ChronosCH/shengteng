/**
 * 修复版手部模型演示组件
 */

import React, { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
import DetailedHandModelFixed from './DetailedHandModelFixed'
import DetailedHandModelStable from './DetailedHandModelStable'

const HandModelDemoFixed: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<'fixed' | 'stable'>('stable')
  const [gestureMode, setGestureMode] = useState<'natural' | 'fist' | 'point' | 'ok' | 'peace' | 'wave'>('natural')
  const [showBones, setShowBones] = useState(false)
  const [showLabels, setShowLabels] = useState(false)
  const [enableAnimations, setEnableAnimations] = useState(true)
  const [handScale, setHandScale] = useState(1)

  const handleGestureComplete = (gesture: string) => {
    console.log('手势完成:', gesture)
  }

  const handleHandTracked = (data: any) => {
    console.log('手部追踪数据:', data)
  }

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex' }}>
      {/* 控制面板 */}
      <div style={{ 
        width: '300px', 
        padding: '20px', 
        backgroundColor: '#f5f5f5',
        borderRight: '1px solid #ddd',
        overflow: 'auto'
      }}>
        <h3>手部模型控制</h3>
        
        <div style={{ marginBottom: '20px' }}>
          <label>模型版本:</label>
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value as 'fixed' | 'stable')}
            style={{ width: '100%', padding: '5px', marginTop: '5px' }}
          >
            <option value="stable">稳定版 (推荐)</option>
            <option value="fixed">修复版 (功能完整)</option>
          </select>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>手势模式:</label>
          <select 
            value={gestureMode} 
            onChange={(e) => setGestureMode(e.target.value as any)}
            style={{ width: '100%', padding: '5px', marginTop: '5px' }}
          >
            <option value="natural">自然</option>
            <option value="fist">拳头</option>
            <option value="point">指向</option>
            <option value="ok">OK手势</option>
            <option value="peace">胜利手势</option>
            <option value="wave">挥手</option>
          </select>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>
            <input 
              type="checkbox" 
              checked={enableAnimations}
              onChange={(e) => setEnableAnimations(e.target.checked)}
            />
            启用动画
          </label>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>
            <input 
              type="checkbox" 
              checked={showBones}
              onChange={(e) => setShowBones(e.target.checked)}
            />
            显示骨骼
          </label>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>
            <input 
              type="checkbox" 
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
            />
            显示标签
          </label>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>手部缩放: {handScale.toFixed(1)}</label>
          <input 
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            value={handScale}
            onChange={(e) => setHandScale(parseFloat(e.target.value))}
            style={{ width: '100%', marginTop: '5px' }}
          />
        </div>

        <div style={{ 
          padding: '15px', 
          backgroundColor: '#e8f4f8', 
          borderRadius: '5px',
          fontSize: '12px'
        }}>
          <h4>使用说明:</h4>
          <ul style={{ margin: 0, paddingLeft: '20px' }}>
            <li>拖拽旋转视角</li>
            <li>滚轮缩放</li>
            <li>切换手势查看效果</li>
            <li>启用骨骼显示内部结构</li>
            <li>稳定版更可靠，修复版功能更全</li>
          </ul>
        </div>
      </div>

      {/* 3D场景 */}
      <div style={{ flex: 1 }}>
        <Canvas
          camera={{ position: [0, 0, 1], fov: 60 }}
          style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
        >
          <ambientLight intensity={0.4} />
          <directionalLight 
            position={[5, 5, 5]} 
            intensity={0.8}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <pointLight position={[-5, 5, 5]} intensity={0.3} />

          {/* 环境光照 */}
          <Environment preset="studio" />

          {/* 左手 */}
          {selectedModel === 'stable' ? (
            <DetailedHandModelStable
              handedness="left"
              isActive={true}
              position={[-0.3, 0, 0]}
              scale={handScale}
              gestureMode={gestureMode}
              enableAnimations={enableAnimations}
              onGestureComplete={handleGestureComplete}
            />
          ) : (
            <DetailedHandModelFixed
              handedness="left"
              isActive={true}
              position={[-0.3, 0, 0]}
              scale={handScale}
              gestureMode={gestureMode}
              enableAnimations={enableAnimations}
              showBones={showBones}
              showLabels={showLabels}
              onGestureComplete={handleGestureComplete}
              onHandTracked={handleHandTracked}
            />
          )}

          {/* 右手 */}
          {selectedModel === 'stable' ? (
            <DetailedHandModelStable
              handedness="right"
              isActive={true}
              position={[0.3, 0, 0]}
              scale={handScale}
              gestureMode={gestureMode}
              enableAnimations={enableAnimations}
              onGestureComplete={handleGestureComplete}
            />
          ) : (
            <DetailedHandModelFixed
              handedness="right"
              isActive={true}
              position={[0.3, 0, 0]}
              scale={handScale}
              gestureMode={gestureMode}
              enableAnimations={enableAnimations}
              showBones={showBones}
              showLabels={showLabels}
              onGestureComplete={handleGestureComplete}
              onHandTracked={handleHandTracked}
            />
          )}

          <OrbitControls 
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            target={[0, 0, 0]}
            maxDistance={3}
            minDistance={0.5}
          />
        </Canvas>
      </div>
    </div>
  )
}

export default HandModelDemoFixed
