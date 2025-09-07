/**
 * 简化的手语虚拟人组件
 * 提供稳定的2D动画效果，作为3D模型的可靠替代方案
 */

import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Typography,
  Paper,
  Fade,
  Zoom,
} from '@mui/material'
import { HandKeypoint } from '../hooks/useHandRig'

interface SimpleSignLanguageAvatarProps {
  text?: string
  isActive: boolean
  leftHandKeypoints?: HandKeypoint[]
  rightHandKeypoints?: HandKeypoint[]
  signSequence?: any
}

const SimpleSignLanguageAvatar: React.FC<SimpleSignLanguageAvatarProps> = ({
  text,
  isActive,
  leftHandKeypoints,
  rightHandKeypoints,
}) => {
  const [currentGesture, setCurrentGesture] = useState<string>('neutral')
  const [animationPhase, setAnimationPhase] = useState<number>(0)
  const animationRef = useRef<number>()

  // 根据关键点数据分析手势
  const analyzeGesture = (leftHand?: HandKeypoint[], rightHand?: HandKeypoint[]) => {
    if (!leftHand && !rightHand) return 'neutral'
    
    // 简单的手势识别逻辑
    if (leftHand && rightHand) {
      // 双手都有数据
      const leftY = leftHand[0]?.y || 0.5
      const rightY = rightHand[0]?.y || 0.5
      
      if (leftY < 0.3 && rightY < 0.3) {
        return 'hands_up'
      } else if (leftY > 0.7 && rightY > 0.7) {
        return 'hands_down'
      } else {
        return 'gesturing'
      }
    } else if (leftHand || rightHand) {
      // 单手手势
      return 'one_hand'
    }
    
    return 'neutral'
  }

  // 动画循环
  useEffect(() => {
    const animate = () => {
      setAnimationPhase(prev => (prev + 0.02) % (Math.PI * 2))
      animationRef.current = requestAnimationFrame(animate)
    }
    
    if (isActive) {
      animationRef.current = requestAnimationFrame(animate)
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isActive])

  // 更新手势
  useEffect(() => {
    const gesture = analyzeGesture(leftHandKeypoints, rightHandKeypoints)
    setCurrentGesture(gesture)
  }, [leftHandKeypoints, rightHandKeypoints])

  // 获取手势对应的emoji和动画
  const getGestureDisplay = () => {
    const breathe = 1 + Math.sin(animationPhase * 0.8) * 0.05
    const sway = isActive ? Math.sin(animationPhase * 0.3) * 2 : 0
    
    switch (currentGesture) {
      case 'hands_up':
        return {
          emoji: '🙋‍♀️',
          transform: `scale(${breathe}) rotate(${sway}deg)`,
          description: '举手'
        }
      case 'hands_down':
        return {
          emoji: '🧘‍♀️',
          transform: `scale(${breathe}) rotate(${sway}deg)`,
          description: '放下'
        }
      case 'gesturing':
        return {
          emoji: '👩‍🏫',
          transform: `scale(${breathe}) rotate(${sway}deg)`,
          description: '手语中'
        }
      case 'one_hand':
        return {
          emoji: '👋',
          transform: `scale(${breathe}) rotate(${sway}deg)`,
          description: '单手手势'
        }
      default:
        return {
          emoji: '👤',
          transform: `scale(${breathe}) rotate(${sway}deg)`,
          description: '待机'
        }
    }
  }

  const gestureDisplay = getGestureDisplay()

  return (
    <Paper
      elevation={3}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        background: isActive 
          ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
          : 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        color: isActive ? 'white' : '#333',
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s ease',
        borderRadius: 2,
      }}
    >
      {/* 背景动画效果 */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: isActive 
            ? 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 70%)'
            : 'none',
          animation: isActive ? 'pulse 3s ease-in-out infinite' : 'none',
        }}
      />
      
      {/* 虚拟人主体 */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        {/* 头像区域 */}
        <Zoom in={true} timeout={500}>
          <Box
            sx={{
              fontSize: '6rem',
              mb: 2,
              transform: gestureDisplay.transform,
              transition: 'transform 0.1s ease',
              filter: isActive ? 'drop-shadow(0 0 20px rgba(255,255,255,0.3))' : 'none',
            }}
          >
            {gestureDisplay.emoji}
          </Box>
        </Zoom>

        {/* 状态指示 */}
        <Typography 
          variant="h6" 
          gutterBottom 
          sx={{ 
            fontWeight: 600,
            opacity: isActive ? 1 : 0.8,
            transition: 'opacity 0.3s ease',
          }}
        >
          手语虚拟人
        </Typography>

        {/* 当前手势描述 */}
        <Typography 
          variant="body2" 
          sx={{ 
            mb: 2,
            opacity: 0.8,
            fontSize: '0.9rem',
          }}
        >
          {gestureDisplay.description}
        </Typography>

        {/* 识别文本显示 */}
        {text && (
          <Fade in={Boolean(text)} timeout={300}>
            <Box
              sx={{
                bgcolor: isActive ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)',
                px: 3,
                py: 1.5,
                borderRadius: 2,
                maxWidth: 250,
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255,255,255,0.2)',
              }}
            >
              <Typography 
                variant="body1" 
                sx={{ 
                  fontWeight: 500,
                  wordBreak: 'break-word',
                }}
              >
                {text}
              </Typography>
            </Box>
          </Fade>
        )}

        {/* 活跃状态指示器 */}
        {isActive && (
          <Box
            sx={{
              position: 'absolute',
              bottom: -20,
              left: '50%',
              transform: 'translateX(-50%)',
              display: 'flex',
              gap: 0.5,
            }}
          >
            {[0, 1, 2].map((i) => (
              <Box
                key={i}
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  bgcolor: 'rgba(255,255,255,0.6)',
                  animation: `bounce 1.4s ease-in-out ${i * 0.16}s infinite both`,
                }}
              />
            ))}
          </Box>
        )}
      </Box>

      {/* CSS动画定义 */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 0.8; }
            50% { opacity: 1; }
          }
          
          @keyframes bounce {
            0%, 80%, 100% { 
              transform: scale(0);
            } 40% { 
              transform: scale(1);
            }
          }
        `}
      </style>
    </Paper>
  )
}

export default SimpleSignLanguageAvatar
