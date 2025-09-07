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

// 改进的安全GLTF加载组件
const AvatarModelGLTF: React.FC<Omit<ImprovedAvatarProps,'onReady'>> = ({ leftHandKeypoints, rightHandKeypoints, isActive }) => {
  const group = useRef<THREE.Group>(null)
  const skeletonRef = useRef<THREE.Skeleton|null>(null)
  const [scene, setScene] = useState<any>(null)
  const [loadError, setLoadError] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // 安全加载模型
  useEffect(() => {
    let mounted = true

    const loadModel = async () => {
      try {
        setIsLoading(true)

        // 检查模型文件是否存在
        const response = await fetch(MODEL_URL, { method: 'HEAD' })
        if (!response.ok) {
          throw new Error('Model file not found')
        }

        // 加载GLTF模型
        const gltf = useGLTF(MODEL_URL) as any

        if (mounted) {
          if (gltf?.scene) {
            setScene(gltf.scene)
            setLoadError(false)
          } else {
            setLoadError(true)
          }
          setIsLoading(false)
        }
      } catch (error) {
        console.warn('GLTF model loading failed:', error)
        if (mounted) {
          setLoadError(true)
          setIsLoading(false)
        }
      }
    }

    loadModel()

    return () => {
      mounted = false
    }
  }, [])

  useEffect(()=>{
    if(scene && group.current && !loadError && !isLoading){
      try {
        // 清除之前的子对象
        while(group.current.children.length > 0) {
          group.current.remove(group.current.children[0])
        }

        // 复制场景避免多实例引用问题
        const inst = scene.clone(true)
        group.current.add(inst)

        // 尝试从任一 SkinnedMesh 取 skeleton
        inst.traverse((child: any)=>{
          if(child.isSkinnedMesh && child.skeleton && !skeletonRef.current){
            skeletonRef.current = child.skeleton
          }
        })
      } catch (error) {
        console.warn('Scene setup error:', error)
        setLoadError(true)
      }
    }
  },[scene, loadError, isLoading])

  // 使用手部绑定（如果可用）
  useHandRig({
    skeleton: skeletonRef.current,
    leftHand: leftHandKeypoints,
    rightHand: rightHandKeypoints,
    scale: 1
  })

  // 改进的动画循环
  useFrame((state)=>{
    if(!group.current || loadError || isLoading) return

    try {
      const t = state.clock.elapsedTime

      // 呼吸动画
      const breathe = 1 + Math.sin(t * 0.8) * 0.01
      group.current.scale.setY(breathe)

      // 活跃状态的轻微摆动
      if(isActive){
        const sway = Math.sin(t * 0.3) * 0.03
        group.current.rotation.y = sway
      } else {
        // 非活跃时回到中性位置
        group.current.rotation.y = THREE.MathUtils.lerp(group.current.rotation.y, 0, 0.05)
      }
    } catch (error) {
      console.warn('Animation frame error:', error)
    }
  })

  // 加载状态
  if (isLoading) {
    return <LoadingModel />
  }

  // 错误状态
  if (loadError) {
    return <PlaceholderModel isActive={isActive} leftHandKeypoints={leftHandKeypoints} rightHandKeypoints={rightHandKeypoints} />
  }

  return <group ref={group} position={[0,-1.1,0]} />
}

// 加载中的模型
const LoadingModel: React.FC = () => {
  const group = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (!group.current) return
    group.current.rotation.y = state.clock.elapsedTime * 0.5
  })

  return (
    <group ref={group} position={[0,-1.1,0]}>
      <mesh position={[0,1,0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#94a3b8" wireframe />
      </mesh>
      <mesh position={[0,0.5,0]}>
        <boxGeometry args={[0.5, 0.7, 0.25]} />
        <meshStandardMaterial color="#94a3b8" wireframe />
      </mesh>
    </group>
  )
}

// 改进的占位模型：当GLB未就绪时使用，提供更好的视觉效果
const PlaceholderModel: React.FC<{ isActive: boolean } & Pick<ImprovedAvatarProps,'leftHandKeypoints'|'rightHandKeypoints'>> = ({ isActive, leftHandKeypoints, rightHandKeypoints }) => {
  const group = useRef<THREE.Group>(null)
  const headRef = useRef<THREE.Mesh>(null)
  const leftArmRef = useRef<THREE.Group>(null)
  const rightArmRef = useRef<THREE.Group>(null)

  useFrame((state)=>{
    if(!group.current) return

    try {
      const t = state.clock.elapsedTime

      // 呼吸动画
      const breathe = 1 + Math.sin(t * 0.8) * 0.01
      group.current.scale.setY(breathe)

      // 头部轻微点头
      if (headRef.current) {
        headRef.current.rotation.x = Math.sin(t * 0.6) * 0.05
      }

      // 活跃状态的身体摆动
      if(isActive){
        const sway = Math.sin(t * 0.3) * 0.03
        group.current.rotation.y = sway

        // 手臂轻微摆动
        if (leftArmRef.current) {
          leftArmRef.current.rotation.z = Math.sin(t * 0.4) * 0.1
        }
        if (rightArmRef.current) {
          rightArmRef.current.rotation.z = -Math.sin(t * 0.4) * 0.1
        }
      } else {
        // 回到中性位置
        group.current.rotation.y = THREE.MathUtils.lerp(group.current.rotation.y, 0, 0.05)
        if (leftArmRef.current) {
          leftArmRef.current.rotation.z = THREE.MathUtils.lerp(leftArmRef.current.rotation.z, 0, 0.05)
        }
        if (rightArmRef.current) {
          rightArmRef.current.rotation.z = THREE.MathUtils.lerp(rightArmRef.current.rotation.z, 0, 0.05)
        }
      }

      // 根据关键点数据调整手臂位置
      if (leftHandKeypoints && leftHandKeypoints.length > 0 && leftArmRef.current) {
        const wrist = leftHandKeypoints[0] // 手腕关键点
        leftArmRef.current.rotation.x = (wrist.y - 0.5) * 0.5
        leftArmRef.current.rotation.y = (wrist.x - 0.5) * 0.3
      }

      if (rightHandKeypoints && rightHandKeypoints.length > 0 && rightArmRef.current) {
        const wrist = rightHandKeypoints[0] // 手腕关键点
        rightArmRef.current.rotation.x = (wrist.y - 0.5) * 0.5
        rightArmRef.current.rotation.y = -(wrist.x - 0.5) * 0.3
      }
    } catch (error) {
      console.warn('Placeholder animation error:', error)
    }
  })

  return (
    <group ref={group} position={[0,-1.1,0]}>
      {/* 头部 */}
      <mesh ref={headRef} position={[0,1.6,0]} castShadow>
        <sphereGeometry args={[0.22, 32, 32]} />
        <meshStandardMaterial color="#F4C2A1" roughness={0.6} metalness={0.1} />
      </mesh>

      {/* 身体 */}
      <mesh position={[0,1,0]} castShadow>
        <boxGeometry args={[0.5, 0.7, 0.25]} />
        <meshStandardMaterial color="#4A90E2" roughness={0.8} />
      </mesh>

      {/* 左臂 */}
      <group ref={leftArmRef} position={[-0.35,1.2,0]}>
        <mesh position={[0,-0.2,0]} castShadow>
          <boxGeometry args={[0.12, 0.4, 0.12]} />
          <meshStandardMaterial color="#F4C2A1" roughness={0.6} />
        </mesh>
        <mesh position={[0,-0.5,0]} castShadow>
          <boxGeometry args={[0.1, 0.3, 0.1]} />
          <meshStandardMaterial color="#F4C2A1" roughness={0.6} />
        </mesh>
        {/* 左手 */}
        <mesh position={[0,-0.7,0]} castShadow>
          <sphereGeometry args={[0.08, 16, 16]} />
          <meshStandardMaterial color="#F4C2A1" roughness={0.6} />
        </mesh>
      </group>

      {/* 右臂 */}
      <group ref={rightArmRef} position={[0.35,1.2,0]}>
        <mesh position={[0,-0.2,0]} castShadow>
          <boxGeometry args={[0.12, 0.4, 0.12]} />
          <meshStandardMaterial color="#F4C2A1" roughness={0.6} />
        </mesh>
        <mesh position={[0,-0.5,0]} castShadow>
          <boxGeometry args={[0.1, 0.3, 0.1]} />
          <meshStandardMaterial color="#F4C2A1" roughness={0.6} />
        </mesh>
        {/* 右手 */}
        <mesh position={[0,-0.7,0]} castShadow>
          <sphereGeometry args={[0.08, 16, 16]} />
          <meshStandardMaterial color="#F4C2A1" roughness={0.6} />
        </mesh>
      </group>

      {/* 腿部 */}
      <mesh position={[0,0.2,0]} castShadow>
        <boxGeometry args={[0.4, 0.6, 0.2]} />
        <meshStandardMaterial color="#2C3E50" roughness={0.8} />
      </mesh>

      {/* 脚部 */}
      <mesh position={[0,-0.2,0]} receiveShadow>
        <boxGeometry args={[0.3, 0.1, 0.4]} />
        <meshStandardMaterial color="#34495E" roughness={0.8} />
      </mesh>
    </group>
  )
}

const ImprovedAvatar: React.FC<ImprovedAvatarProps> = (props) => {
  const [modelAvailable, setModelAvailable] = useState(true)
  const [webGLSupported, setWebGLSupported] = useState(true)
  const [renderError, setRenderError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(()=>{
    let mounted = true

    // 检查WebGL支持
    const checkWebGL = () => {
      try {
        const canvas = document.createElement('canvas')
        const gl = canvas.getContext('webgl2') ||
                   canvas.getContext('webgl') ||
                   canvas.getContext('experimental-webgl')

        if (!gl) {
          if (mounted) setWebGLSupported(false)
          return false
        }

        // 检查WebGL扩展
        const extensions = [
          'OES_texture_float',
          'OES_texture_half_float',
          'WEBGL_depth_texture'
        ]

        extensions.forEach(ext => {
          try {
            (gl as WebGLRenderingContext).getExtension(ext)
          } catch (e) {
            // 扩展不可用，继续
          }
        })

        return true
      } catch (error) {
        console.warn('WebGL check failed:', error)
        if (mounted) setWebGLSupported(false)
        return false
      }
    }

    if (!checkWebGL()) {
      return
    }

    // 检查模型文件可用性
    const checkModel = async () => {
      try {
        const response = await fetch(MODEL_URL, { method: 'HEAD' })
        if (mounted) {
          setModelAvailable(response.ok)
          if (!response.ok) {
            console.warn('Avatar model not available, using placeholder')
          }
        }
      } catch (error) {
        console.warn('Model check failed:', error)
        if (mounted) {
          setModelAvailable(false)
        }
      }
    }

    checkModel()

    return () => {
      mounted = false
    }
  }, [])

  // WebGL不支持的回退UI
  if (!webGLSupported) {
    return (
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        borderRadius: '8px',
        border: '2px dashed #cbd5e0'
      }}>
        <div style={{ textAlign: 'center', color: '#666', padding: '2rem' }}>
          <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🤖</div>
          <div style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            3D显示不可用
          </div>
          <div style={{ fontSize: '0.9rem', color: '#888' }}>
            您的浏览器不支持WebGL
          </div>
          <div style={{ fontSize: '0.8rem', marginTop: '1rem', color: '#999' }}>
            建议使用Chrome、Firefox或Safari浏览器
          </div>
        </div>
      </div>
    )
  }

  // 渲染错误的回退UI
  if (renderError) {
    return (
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #fef5e7 0%, #f6ad55 100%)',
        borderRadius: '8px',
        border: '2px solid #ed8936'
      }}>
        <div style={{ textAlign: 'center', color: '#744210', padding: '2rem' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            3D渲染出现问题
          </div>
          <div style={{ fontSize: '0.8rem', marginBottom: '1rem' }}>
            {renderError}
          </div>
          <button
            onClick={() => setRenderError(null)}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#ed8936',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            重新尝试
          </button>
        </div>
      </div>
    )
  }

  try {
    return (
      <Canvas
        ref={canvasRef}
        camera={{ position:[0,1.4,3.2], fov:42 }}
        shadows
        dpr={[1, Math.min(window.devicePixelRatio, 2)]}
        style={{width:'100%',height:'100%'}}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance"
        }}
        onCreated={(state) => {
          try {
            // 通知父组件Avatar已准备就绪
            if (props.onReady && state.scene) {
              props.onReady(state.scene)
            }
          } catch (error) {
            console.warn('Failed to configure renderer:', error)
          }
        }}
        onError={(error) => {
          console.error('Canvas rendering error:', error)
          setRenderError('3D渲染出现错误')
        }}
      >
        <Suspense fallback={<LoadingModel />}>
          <Environment preset="studio" />
          <ContactShadows
            position={[0,-1.1,0]}
            opacity={0.3}
            scale={8}
            blur={2}
            far={2.5}
          />
          <ambientLight intensity={0.5} />
          <directionalLight
            position={[4,6,4]}
            intensity={0.8}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
            shadow-camera-near={0.1}
            shadow-camera-far={20}
            shadow-camera-left={-5}
            shadow-camera-right={5}
            shadow-camera-top={5}
            shadow-camera-bottom={-5}
          />

          {modelAvailable ? (
            <AvatarModelGLTF {...props} />
          ) : (
            <PlaceholderModel
              isActive={props.isActive}
              leftHandKeypoints={props.leftHandKeypoints}
              rightHandKeypoints={props.rightHandKeypoints}
            />
          )}

          <OrbitControls
            enablePan={false}
            enableZoom={true}
            enableRotate={true}
            target={[0,1.0,0]}
            maxPolarAngle={Math.PI*0.75}
            minDistance={2.0}
            maxDistance={8}
            dampingFactor={0.05}
            enableDamping={true}
          />
        </Suspense>
      </Canvas>
    )
  } catch (error) {
    console.error('ImprovedAvatar render error:', error)
    setRenderError(error instanceof Error ? error.message : '渲染失败')
    return null
  }
}

export default ImprovedAvatar
