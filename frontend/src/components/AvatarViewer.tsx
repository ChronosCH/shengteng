/**
 * Avatar查看器组件 - 显示3D虚拟人
 */

import React, { useRef, useEffect, useState, Suspense } from 'react'
import {
  Box,
  Typography,
  IconButton,
  Tooltip,
  Paper,
  CircularProgress,
  Grid,
} from '@mui/material'
import {
  Fullscreen,
  FullscreenExit,
  Refresh,
} from '@mui/icons-material'
import DiffusionPanel from './DiffusionPanel'
import WebXRPanel from './WebXRPanel'
import { SignSequence } from '../services/diffusionService'
import { getSignPreset } from './SignLanguagePresets'
import ImprovedAvatar from './avatar/ImprovedAvatar'
import ThreeDErrorBoundary, { SimpleAvatarFallback } from './ThreeDErrorBoundary'
import SimpleSignLanguageAvatar from './SimpleSignLanguageAvatar'

interface HandKeypoint {
  x: number
  y: number
  z: number
  visibility?: number
}

interface AvatarViewerProps {
  text: string
  isActive: boolean
  onAvatarMeshReady?: (mesh: THREE.Object3D) => void
  signSequence?: any
  leftHandKeypoints?: HandKeypoint[]
  rightHandKeypoints?: HandKeypoint[]
}

const AvatarViewer: React.FC<AvatarViewerProps> = ({
  text,
  isActive,
  onAvatarMeshReady,
  signSequence: externalSignSequence,
  leftHandKeypoints,
  rightHandKeypoints,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  // const [avatarStyle, setAvatarStyle] = useState<'cartoon' | 'realistic' | 'improved'>('cartoon')
  const [currentSignSequence, setCurrentSignSequence] = useState<SignSequence | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isXRActive, setIsXRActive] = useState(false)
  const [avatarMesh, setAvatarMesh] = useState<THREE.Object3D | null>(null)
  const [autoHandKeypoints, setAutoHandKeypoints] = useState<{
    left?: HandKeypoint[]
    right?: HandKeypoint[]
  }>({})

  // 模拟3D场景初始化
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 2000)

    return () => clearTimeout(timer)
  }, [])

  // 自动检测文本中的手语预设
  useEffect(() => {
    if (text) {
      // 检查是否包含数字或常见词汇
      const preset = getSignPreset(text.trim())
      if (preset) {
        setAutoHandKeypoints({
          left: preset,
          right: preset // 对于简单手势，双手可以相同
        })
      } else {
        // 检查文本中的单个字符
        const chars = text.split('')
        const firstCharLocal = chars[0]
        const charPreset = getSignPreset(firstCharLocal)
        if (charPreset) {
          setAutoHandKeypoints({
            left: charPreset,
            right: charPreset
          })
        } else {
          setAutoHandKeypoints({})
        }
      }
    } else {
      setAutoHandKeypoints({})
    }
  }, [text])

  // 处理 Diffusion 生成的手语序列
  const handleSequenceGenerated = (sequence: SignSequence) => {
    setCurrentSignSequence(sequence)
    setIsGenerating(false)
  }

  // 处理 WebXR 会话开始
  const handleXRSessionStart = () => {
    setIsXRActive(true)
  }

  // 处理 WebXR 会话结束
  const handleXRSessionEnd = () => {
    setIsXRActive(false)
  }

  // 处理 Avatar 网格就绪
  const handleAvatarMeshReady = (mesh: THREE.Object3D) => {
    setAvatarMesh(mesh)
    onAvatarMeshReady?.(mesh)
  }

  // 使用外部传入的序列或内部生成的序列
  const activeSignSequence = externalSignSequence || currentSignSequence

  // 处理全屏切换
  const handleFullscreenToggle = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  // 监听全屏状态变化
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  // 模拟Avatar动画
  const getAvatarAnimation = () => {
    if (!isActive || !text) {
      return '待机'
    }
    
    // 根据文本内容返回不同的动画状态
    if (text.includes('你好')) return '挥手'
    if (text.includes('谢谢')) return '鞠躬'
    if (text.includes('再见')) return '告别'
    return '手语表达'
  }

  return (
    <Box
      ref={containerRef}
      sx={{
        height: '100%',
        bgcolor: isFullscreen ? 'black' : 'transparent',
      }}
    >
      {/* 简化的控制栏 - 只显示必要信息 */}
      <Box sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 2,
        opacity: isFullscreen ? 0.9 : 1,
        px: 1,
      }}>
        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
          {getAvatarAnimation()}
          {currentSignSequence && (
            <Typography component="span" color="primary" sx={{ ml: 1, fontWeight: 600 }}>
              | {currentSignSequence.text}
            </Typography>
          )}
        </Typography>

        <Box>
          <Tooltip title="重新加载">
            <IconButton
              size="small"
              onClick={() => {
                setIsLoading(true)
                setTimeout(() => setIsLoading(false), 1000)
              }}
            >
              <Refresh />
            </IconButton>
          </Tooltip>

          <Tooltip title={isFullscreen ? "退出全屏" : "全屏显示"}>
            <IconButton size="small" onClick={handleFullscreenToggle}>
              {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* 主要3D Avatar显示区域 - 占据大部分空间 */}
      <Box sx={{
        height: 'calc(100% - 60px)',
        position: 'relative',
        bgcolor: isFullscreen ? 'black' : 'rgba(248, 253, 255, 0.8)',
        borderRadius: isFullscreen ? 0 : 3,
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: isFullscreen ? 'none' : '0 8px 32px rgba(0, 0, 0, 0.1)',
      }}>
        <Suspense fallback={
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress size={60} thickness={4} />
            <Typography variant="body1" color="text.secondary" sx={{ mt: 2, fontWeight: 500 }}>
              正在加载3D模型...
            </Typography>
          </Box>
        }>
          {/* 使用错误边界包装3D组件，提供多层回退方案 */}
          <ThreeDErrorBoundary
            fallback={
              <SimpleSignLanguageAvatar
                text={text}
                isActive={isActive}
                leftHandKeypoints={leftHandKeypoints || autoHandKeypoints.left}
                rightHandKeypoints={rightHandKeypoints || autoHandKeypoints.right}
              />
            }
          >
            <Box sx={{position:'absolute', inset:0}}>
              <ImprovedAvatar
                isActive={isActive}
                leftHandKeypoints={leftHandKeypoints || autoHandKeypoints.left}
                rightHandKeypoints={rightHandKeypoints || autoHandKeypoints.right}
                onReady={handleAvatarMeshReady}
              />
            </Box>
          </ThreeDErrorBoundary>

          {/* 风格指示器改为固定显示 */}
          <Box
            sx={{
              position: 'absolute',
              top: 16,
              right: 16,
              bgcolor: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              px: 2,
              py: 1,
              borderRadius: 2,
              fontSize: '0.8rem',
              fontWeight: 600,
              zIndex: 10,
              backdropFilter: 'blur(10px)',
            }}
          >
            👤 高质量Avatar
          </Box>

          {/* 生成状态指示器 */}
          {isGenerating && (
            <Box
              sx={{
                position: 'absolute',
                bottom: 16,
                left: 16,
                bgcolor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                px: 2,
                py: 1,
                borderRadius: 2,
                fontSize: '0.8rem',
                fontWeight: 600,
                zIndex: 10,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                backdropFilter: 'blur(10px)',
              }}
            >
              <CircularProgress size={18} color="inherit" thickness={4} />
              正在生成手语动画...
            </Box>
          )}

          {/* 活动状态指示器 */}
          {isActive && (
            <Box
              sx={{
                position: 'absolute',
                top: 16,
                left: 16,
                bgcolor: 'rgba(76, 175, 76, 0.9)',
                color: 'white',
                px: 2,
                py: 1,
                borderRadius: 2,
                fontSize: '0.8rem',
                fontWeight: 600,
                zIndex: 10,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                animation: 'pulse 2s infinite',
              }}
            >
              🔴 实时演示
            </Box>
          )}
        </Suspense>
      </Box>

      {/* 控制面板 - 在非全屏时显示在底部 */}
      {!isFullscreen && !isXRActive && (
        <Box sx={{ 
          position: 'absolute', 
          bottom: 0, 
          left: 0, 
          right: 0, 
          p: 2,
          background: 'linear-gradient(to top, rgba(255, 255, 255, 0.95), transparent)',
          backdropFilter: 'blur(10px)',
        }}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <DiffusionPanel
                onSequenceGenerated={handleSequenceGenerated}
                isGenerating={isGenerating}
              />
            </Grid>
            <Grid item xs={6}>
              <WebXRPanel
                onSessionStart={handleXRSessionStart}
                onSessionEnd={handleXRSessionEnd}
                avatarMesh={avatarMesh || undefined}
              />
            </Grid>
          </Grid>
        </Box>
      )}

      {/* 底部提示信息 */}
      {!isFullscreen && (
        <Box sx={{ 
          position: 'absolute', 
          bottom: 8, 
          left: '50%', 
          transform: 'translateX(-50%)',
          zIndex: 5,
        }}>
          <Typography 
            variant="caption" 
            color="text.secondary" 
            sx={{ 
              bgcolor: 'rgba(255, 255, 255, 0.8)', 
              px: 2, 
              py: 0.5, 
              borderRadius: 2,
              backdropFilter: 'blur(10px)',
              fontWeight: 500,
            }}
          >
            💡 点击全屏按钮获得更好的观看体验
          </Typography>
        </Box>
      )}
    </Box>
  )
}

export default AvatarViewer
