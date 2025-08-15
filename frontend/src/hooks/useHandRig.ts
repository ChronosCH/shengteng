import { useRef, useEffect } from 'react'
import * as THREE from 'three'
import { LEFT_HAND_BONE_MAPPING, RIGHT_HAND_BONE_MAPPING } from '../mapping/handMap'
import { smoothHandKeypoints } from '../utils/filters'

export interface HandKeypoint { x:number; y:number; z:number; visibility?:number }

interface UseHandRigParams {
  skeleton: THREE.Skeleton | null
  leftHand?: HandKeypoint[]
  rightHand?: HandKeypoint[]
  scale?: number
}

// 假设模型 A-Pose，手掌朝下。后续可添加基准姿势校正矩阵。
export function useHandRig({ skeleton, leftHand, rightHand, scale=1 }: UseHandRigParams){
  const prevLeft = useRef<number[][]>()
  const prevRight = useRef<number[][]>()

  useEffect(()=>{
    if(!skeleton) return

    const update = () => {
      if(leftHand && leftHand.length===21){
        const arr = leftHand.map(p=>[p.x,p.y,p.z])
        const smooth = smoothHandKeypoints(prevLeft.current, arr, 0.35)
        prevLeft.current = smooth
        applyHand(skeleton, smooth, 'left', scale)
      }
      if(rightHand && rightHand.length===21){
        const arr = rightHand.map(p=>[p.x,p.y,p.z])
        const smooth = smoothHandKeypoints(prevRight.current, arr, 0.35)
        prevRight.current = smooth
        applyHand(skeleton, smooth, 'right', scale)
      }
    }

    update()
  },[skeleton, leftHand, rightHand, scale])
}

function applyHand(skeleton: THREE.Skeleton, kp: number[][], side: 'left'|'right', scale:number){
  const mapping = side==='left'? LEFT_HAND_BONE_MAPPING : RIGHT_HAND_BONE_MAPPING

  Object.values(mapping.fingers).forEach(f=>{
    // finger joints: e.g. 4 points -> 3 bone segments
    for(let i=0;i<f.bones.length;i++){
      const bName = f.bones[i]
      const bone = skeleton.bones.find(b=>b.name===bName)
      if(!bone) return
      const jStart = kp[f.joints[i]]
      const jEnd = kp[f.joints[i+1]] || jStart
      if(!jStart || !jEnd) return
      const v = new THREE.Vector3(jEnd[0]-jStart[0], jEnd[1]-jStart[1], jEnd[2]-jStart[2])
      if(v.lengthSq()<1e-6) return
      v.normalize()
      // 基准向量 (模型中假设 Y 轴指向下游关节)
      const base = new THREE.Vector3(0, -1, 0)
      const q = new THREE.Quaternion().setFromUnitVectors(base, v)
      bone.quaternion.slerp(q, 0.6)
    }
  })
}
