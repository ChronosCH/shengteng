import { useRef, useEffect } from 'react'
import * as THREE from 'three'

// 简化的默认映射
const DEFAULT_MAPPING = {
  fingers: {}
}

// 默认的平滑函数
const defaultSmoothHandKeypoints = (prev: number[][] | undefined, current: number[][], alpha: number) => current

export interface HandKeypoint { x:number; y:number; z:number; visibility?:number }

interface UseHandRigParams {
  skeleton: THREE.Skeleton | null
  leftHand?: HandKeypoint[]
  rightHand?: HandKeypoint[]
  scale?: number
}

export function useHandRig({ skeleton, leftHand, rightHand, scale=1 }: UseHandRigParams){
  const prevLeft = useRef<number[][]>()
  const prevRight = useRef<number[][]>()

  useEffect(()=>{
    if(!skeleton) return

    try {
      const update = () => {
        if(leftHand && leftHand.length===21){
          const arr = leftHand.map(p=>[p.x,p.y,p.z])
          const smooth = defaultSmoothHandKeypoints(prevLeft.current, arr, 0.35)
          prevLeft.current = smooth
          applyHand(skeleton, smooth, 'left', scale)
        }
        if(rightHand && rightHand.length===21){
          const arr = rightHand.map(p=>[p.x,p.y,p.z])
          const smooth = defaultSmoothHandKeypoints(prevRight.current, arr, 0.35)
          prevRight.current = smooth
          applyHand(skeleton, smooth, 'right', scale)
        }
      }

      update()
    } catch (error) {
      console.warn('Hand rig update error:', error)
    }
  },[skeleton, leftHand, rightHand, scale])
}

function applyHand(skeleton: THREE.Skeleton, kp: number[][], side: 'left'|'right', scale:number){
  try {
    // 使用默认映射，目前为空
    const mapping = DEFAULT_MAPPING

    if (!mapping || !mapping.fingers) {
      return
    }

    // 后续可以添加实际的手部骨骼映射逻辑
    console.log(`Applying ${side} hand rigging with ${kp.length} keypoints`)
    
  } catch (error) {
    console.warn('Error in applyHand:', error)
  }
}
