/**
 * 安全的Fade组件 - 解决MUI Fade的scrollTop错误
 */

import React, { useState, useEffect, useRef } from 'react'
import { Box, BoxProps } from '@mui/material'
import { Fade, FadeProps } from '@mui/material'

interface SafeFadeProps extends Omit<FadeProps, 'children'> {
  children: React.ReactNode
  wrapperProps?: BoxProps
}

const SafeFade: React.FC<SafeFadeProps> = ({ 
  children, 
  wrapperProps,
  in: fadeIn = true,
  timeout = 300,
  ...fadeProps 
}) => {
  const [shouldRender, setShouldRender] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // 延迟一帧确保DOM已经准备好
    const timer = requestAnimationFrame(() => {
      setShouldRender(true)
    })
    return () => cancelAnimationFrame(timer)
  }, [])

  useEffect(() => {
    // 确保容器元素有scrollTop属性
    if (containerRef.current && typeof containerRef.current.scrollTop === 'undefined') {
      Object.defineProperty(containerRef.current, 'scrollTop', {
        value: 0,
        writable: true,
        configurable: true
      })
    }
  }, [shouldRender])

  // 如果还没准备好渲染，直接显示内容而不使用Fade
  if (!shouldRender) {
    return (
      <Box ref={containerRef} {...wrapperProps}>
        {children}
      </Box>
    )
  }

  try {
    return (
      <Box ref={containerRef} {...wrapperProps}>
        <Fade 
          in={fadeIn} 
          timeout={timeout}
          {...fadeProps}
        >
          <div>
            {children}
          </div>
        </Fade>
      </Box>
    )
  } catch (error) {
    // 如果Fade组件出错，直接显示内容
    console.warn('Fade组件渲染失败，使用无动画版本:', error)
    return (
      <Box ref={containerRef} {...wrapperProps}>
        {children}
      </Box>
    )
  }
}

export default SafeFade
