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

// 移除直接 preload，避免资源缺失时报错
// useGLTF.preload(MODEL_URL)

// 仅在资源存在时加载 GLTF
const AvatarModelGLTF: React.FC<Omit<ImprovedAvatarProps,'onReady'>> = ({ leftHandKeypoints, rightHandKeypoints, isActive }) => {
  const group = useRef<THREE.Group>(null)
  const { scene } = useGLTF(MODEL_URL) as any
  const skeletonRef = useRef<THREE.Skeleton|null>(null)

  useEffect(()=>{
    if(scene && group.current){
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
  },[scene])

  useHandRig({ skeleton: skeletonRef.current, leftHand: leftHandKeypoints, rightHand: rightHandKeypoints, scale: 1 })

  // Idle 呼吸
  useFrame((state)=>{
    if(!group.current) return
    const t = state.clock.elapsedTime
    const breathe = 1 + Math.sin(t*0.8)*0.015
    group.current.scale.y = breathe
    if(isActive){
      group.current.rotation.y = Math.sin(t*0.3)*0.05
    }
  })

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
  const [modelAvailable, setModelAvailable] = useState(false)

  useEffect(()=>{
    let canceled = false
    fetch(MODEL_URL, { method: 'HEAD' })
      .then(res => { if(!canceled) setModelAvailable(res.ok) })
      .catch(() => { if(!canceled) setModelAvailable(false) })
    return () => { canceled = true }
  }, [])

  return (
    <Canvas camera={{ position:[0,1.4,3.2], fov:42 }} shadows dpr={[1,2]} style={{width:'100%',height:'100%'}}>
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
}

export default ImprovedAvatar
