import React, { useRef, useEffect, Suspense, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows, useGLTF } from '@react-three/drei'
import * as THREE from 'three'
import { useHandRig, HandKeypoint } from '../../hooks/useHandRig'

interface ImprovedAvatarProps {
  leftHandKeypoints?: HandKeypoint[]
  rightHandKeypoints?: HandKeypoint[]
  isActive: boolean
  signText?: string
  onReady?: (obj: THREE.Object3D) => void
}

const MODEL_URL = '/models/avatar/girl_avatar.glb'

// 安全的 GLTF 加载组件
const AvatarModelGLTF: React.FC<Omit<ImprovedAvatarProps,'onReady'>> = ({ leftHandKeypoints, rightHandKeypoints, isActive }) => {
  const group = useRef<THREE.Group>(null)
  const skeletonRef = useRef<THREE.Skeleton|null>(null)
  const [scene, setScene] = useState<any>(null)
  const [loadError, setLoadError] = useState(false)

  // 安全加载模型
  useEffect(() => {
    try {
      const gltf = useGLTF(MODEL_URL) as any
      if (gltf?.scene) {
        setScene(gltf.scene)
        setLoadError(false)
      } else {
        setLoadError(true)
      }
    } catch (error) {
      console.warn('GLTF model loading failed:', error)
      setLoadError(true)
    }
  }, [])

  useEffect(()=>{
    if(scene && group.current && !loadError){
      // 清除之前的子对象
      while(group.current.children.length > 0) {
        group.current.remove(group.current.children[0])
      }
      
      // 复制场景避免多实例引用问题
      const inst = scene.clone(true)
      group.current.add(inst)
      
      // 尝试从任一 SkinnedMesh 取 skeleton
      scene.traverse((child: any)=>{
        if(child.isSkinnedMesh && !skeletonRef.current){
          skeletonRef.current = child.skeleton
        }
      })
    }
  },[scene, loadError])

  useHandRig({ skeleton: skeletonRef.current, leftHand: leftHandKeypoints, rightHand: rightHandKeypoints, scale: 1 })

  // Idle 呼吸
  useFrame((state)=>{
    if(!group.current || loadError) return
    try {
      const t = state.clock.elapsedTime
      const breathe = 1 + Math.sin(t*0.8)*0.015
      group.current.scale.y = breathe
      if(isActive){
        group.current.rotation.y = Math.sin(t*0.3)*0.05
      }
    } catch (error) {
      console.warn('Animation frame error:', error)
    }
  })

  if (loadError) {
    return <PlaceholderModel isActive={isActive} leftHandKeypoints={leftHandKeypoints} rightHandKeypoints={rightHandKeypoints} />
  }

  return <group ref={group} position={[0,-1.1,0]} />
}

// 占位模型：当 GLB 未就绪时使用，防止页面报错
const PlaceholderModel: React.FC<{ isActive: boolean } & Pick<ImprovedAvatarProps,'leftHandKeypoints'|'rightHandKeypoints'>> = ({ isActive }) => {
  const group = useRef<THREE.Group>(null)
  useFrame((state)=>{
    if(!group.current) return
    const t = state.clock.elapsedTime
    const breathe = 1 + Math.sin(t*0.8)*0.015
    group.current.scale.y = breathe
    if(isActive){
      group.current.rotation.y = Math.sin(t*0.3)*0.05
    }
  })
  return (
    <group ref={group} position={[0,-1.1,0]}>
      <mesh position={[0,1,0]} castShadow>
        <sphereGeometry args={[0.25, 32, 32]} />
        <meshStandardMaterial color="#E6C7B2" roughness={0.6} metalness={0.1} />
      </mesh>
      <mesh position={[0,0.5,0]} castShadow>
        <boxGeometry args={[0.6, 0.8, 0.3]} />
        <meshStandardMaterial color="#6C7A89" roughness={0.8} />
      </mesh>
      <mesh position={[0,0,0]} receiveShadow>
        <cylinderGeometry args={[0.05, 0.07, 0.5, 12]} />
        <meshStandardMaterial color="#6C7A89" roughness={0.8} />
      </mesh>
    </group>
  )
}

const ImprovedAvatar: React.FC<ImprovedAvatarProps> = (props) => {
  const [modelAvailable, setModelAvailable] = useState(true) // 默认假设可用
  const [webGLSupported, setWebGLSupported] = useState(true)

  useEffect(()=>{
    // 检查WebGL支持
    try {
      const canvas = document.createElement('canvas')
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
      if (!gl) {
        setWebGLSupported(false)
        return
      }
    } catch (error) {
      setWebGLSupported(false)
      return
    }

    // 检查模型文件
    let canceled = false
    fetch(MODEL_URL, { method: 'HEAD' })
      .then(res => { 
        if(!canceled) {
          setModelAvailable(res.ok)
        }
      })
      .catch(() => { 
        if(!canceled) {
          console.warn('Avatar model not available, using placeholder')
          setModelAvailable(false)
        }
      })
    return () => { canceled = true }
  }, [])

  // 如果WebGL不支持，返回简单提示
  if (!webGLSupported) {
    return (
      <div style={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        borderRadius: '8px'
      }}>
        <div style={{ textAlign: 'center', color: '#666' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🤖</div>
          <div>您的浏览器不支持3D显示</div>
          <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>请使用现代浏览器</div>
        </div>
      </div>
    )
  }

  try {
    return (
      <Canvas 
        camera={{ position:[0,1.4,3.2], fov:42 }} 
        shadows 
        dpr={[1,2]} 
        style={{width:'100%',height:'100%'}}
        onCreated={(state) => {
          // 设置渲染器配置
          try {
            const renderer = state.gl as any
            if (renderer.setClearColor) {
              renderer.setClearColor('#f8fafc')
            }
            if (renderer.setPixelRatio) {
              renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
            }
          } catch (error) {
            console.warn('Failed to configure renderer:', error)
          }
        }}
        onError={(error) => {
          console.error('Canvas rendering error:', error)
        }}
      >
        <Suspense fallback={null}>
          <Environment preset="studio" />
          <ContactShadows position={[0,-1.1,0]} opacity={0.4} scale={8} blur={2} far={2.5} />
          <ambientLight intensity={0.4} />
          <directionalLight position={[4,6,4]} intensity={1} castShadow shadow-mapSize-width={2048} shadow-mapSize-height={2048} />
          {modelAvailable 
            ? <AvatarModelGLTF {...props} /> 
            : <PlaceholderModel isActive={props.isActive} leftHandKeypoints={props.leftHandKeypoints} rightHandKeypoints={props.rightHandKeypoints} />}
          <OrbitControls enablePan enableZoom enableRotate target={[0,1.0,0]} maxPolarAngle={Math.PI*0.75} minDistance={1.8} maxDistance={6} />
        </Suspense>
      </Canvas>
    )
  } catch (error) {
    console.error('ImprovedAvatar render error:', error)
    return (
      <div style={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        borderRadius: '8px'
      }}>
        <div style={{ textAlign: 'center', color: '#666' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
          <div>3D渲染出现问题</div>
          <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>请刷新页面重试</div>
        </div>
      </div>
    )
  }
}

export default ImprovedAvatar
