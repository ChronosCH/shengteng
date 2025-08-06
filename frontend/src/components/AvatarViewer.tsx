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
  ThreeDRotation,
  Fullscreen,
  FullscreenExit,
  Refresh,
} from '@mui/icons-material'
import ThreeAvatar from './ThreeAvatar'
import DiffusionPanel from './DiffusionPanel'
import WebXRPanel from './WebXRPanel'
import { SignSequence } from '../services/diffusionService'
import { getSignPreset } from './SignLanguagePresets'

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
  const [avatarStyle, setAvatarStyle] = useState<'cartoon' | 'realistic'>('cartoon')
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
        const firstChar = chars[0]
        const charPreset = getSignPreset(firstChar)
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
      <Grid container spacing={2} sx={{ height: '100%' }}>
        {/* 控制面板 */}
        {!isFullscreen && !isXRActive && (
          <>
            <Grid item xs={12} md={3}>
              <DiffusionPanel
                onSequenceGenerated={handleSequenceGenerated}
                isGenerating={isGenerating}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <WebXRPanel
                onSessionStart={handleXRSessionStart}
                onSessionEnd={handleXRSessionEnd}
                avatarMesh={avatarMesh || undefined}
              />
            </Grid>
          </>
        )}

        {/* 3D Avatar 显示区域 */}
        <Grid item xs={12} md={isFullscreen ? 12 : (isXRActive ? 12 : 6)}>
          <Box sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
          }}>
            {/* 控制栏 */}
            <Box sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 1,
              opacity: isFullscreen ? 0.8 : 1,
            }}>
              <Typography variant="subtitle2" color="text.secondary">
                动画: {getAvatarAnimation()}
                {currentSignSequence && (
                  <Typography component="span" color="primary" sx={{ ml: 1 }}>
                    | 播放中: {currentSignSequence.text}
                  </Typography>
                )}
              </Typography>

              <Box>
                <Tooltip title="切换风格">
                  <IconButton
                    size="small"
                    onClick={() => setAvatarStyle(prev => prev === 'cartoon' ? 'realistic' : 'cartoon')}
                  >
                    <ThreeDRotation />
                  </IconButton>
                </Tooltip>

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

            {/* 3D Avatar显示区域 */}
            <Box sx={{
              flex: 1,
              position: 'relative',
              bgcolor: 'grey.100',
              borderRadius: isFullscreen ? 0 : 1,
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Suspense fallback={
                <Box sx={{ textAlign: 'center' }}>
                  <CircularProgress size={40} />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    正在加载3D模型...
                  </Typography>
                </Box>
              }>
                <ThreeAvatar
                  text={text}
                  isActive={isActive}
                  animationType={getAvatarAnimation()}
                  signSequence={activeSignSequence ? {
                    keypoints: activeSignSequence.keypoints,
                    timestamps: activeSignSequence.timestamps,
                    duration: activeSignSequence.duration
                  } : undefined}
                  leftHandKeypoints={leftHandKeypoints || autoHandKeypoints.left}
                  rightHandKeypoints={rightHandKeypoints || autoHandKeypoints.right}
                  onAvatarMeshReady={handleAvatarMeshReady}
                />

                {/* 风格指示器 */}
                <Box
                  sx={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    bgcolor: 'rgba(0, 0, 0, 0.6)',
                    color: 'white',
                    px: 1,
                    py: 0.5,
                    borderRadius: 1,
                    fontSize: '0.75rem',
                    zIndex: 10,
                  }}
                >
                  {avatarStyle === 'cartoon' ? '卡通风格' : '写实风格'}
                </Box>

                {/* 生成状态指示器 */}
                {isGenerating && (
                  <Box
                    sx={{
                      position: 'absolute',
                      bottom: 10,
                      left: 10,
                      bgcolor: 'rgba(0, 0, 0, 0.6)',
                      color: 'white',
                      px: 1,
                      py: 0.5,
                      borderRadius: 1,
                      fontSize: '0.75rem',
                      zIndex: 10,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                    }}
                  >
                    <CircularProgress size={16} color="inherit" />
                    正在生成手语...
                  </Box>
                )}
              </Suspense>
            </Box>

            {/* 底部信息 */}
            {!isFullscreen && (
              <Box sx={{ mt: 1, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  💡 使用左侧面板生成手语动画，点击全屏按钮获得更好的观看体验
                </Typography>
              </Box>
            )}
          </Box>
        </Grid>
      </Grid>
    </Box>
  )
}

export default AvatarViewer
