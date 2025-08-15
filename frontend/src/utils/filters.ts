import * as THREE from 'three'

// 指定平滑系数
export function smoothVector(prev: THREE.Vector3 | undefined, cur: THREE.Vector3, alpha=0.25){
  if(!prev) return cur.clone()
  return prev.clone().lerp(cur, alpha)
}

export function smoothQuaternion(prev: THREE.Quaternion | undefined, cur: THREE.Quaternion, alpha=0.35){
  if(!prev) return cur.clone()
  const out = prev.clone()
  out.slerp(cur, alpha)
  return out
}

// 对 Mediapipe 手21点位置做简单指数平滑
export function smoothHandKeypoints(prev: number[][] | undefined, cur: number[][], alpha=0.3){
  if(!prev) return cur.map(p=>[...p])
  return cur.map((p,i)=>{
    const q = prev[i] || p
    return [
      q[0] + (p[0]-q[0])*alpha,
      q[1] + (p[1]-q[1])*alpha,
      q[2] + (p[2]-q[2])*alpha,
    ]
  })
}
