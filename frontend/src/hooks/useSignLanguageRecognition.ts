/**
 * 手语识别钩子
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { websocketService, RecognitionResult } from '../services/websocket'

export interface UseSignLanguageRecognitionReturn {
  isRecognizing: boolean
  currentText: string
  confidence: number
  glossSequence: string[]
  startRecognition: () => Promise<void>
  stopRecognition: () => void
  websocketService: typeof websocketService
  stats: {
    totalRecognitions: number
    averageConfidence: number
    lastUpdateTime: number
  }
}

export function useSignLanguageRecognition(): UseSignLanguageRecognitionReturn {
  const [isRecognizing, setIsRecognizing] = useState(false)
  const [currentText, setCurrentText] = useState('')
  const [confidence, setConfidence] = useState(0)
  const [glossSequence, setGlossSequence] = useState<string[]>([])
  const [stats, setStats] = useState({
    totalRecognitions: 0,
    averageConfidence: 0,
    lastUpdateTime: 0,
  })

  const confidenceHistoryRef = useRef<number[]>([])

  // 处理识别结果
  const handleRecognitionResult = useCallback((result: RecognitionResult) => {
    setCurrentText(result.text)
    setConfidence(result.confidence)
    setGlossSequence(result.glossSequence)

    // 更新统计信息
    confidenceHistoryRef.current.push(result.confidence)
    if (confidenceHistoryRef.current.length > 100) {
      confidenceHistoryRef.current.shift() // 保持最近100个结果
    }

    const averageConfidence = confidenceHistoryRef.current.reduce((a, b) => a + b, 0) / confidenceHistoryRef.current.length

    setStats(prev => ({
      totalRecognitions: prev.totalRecognitions + 1,
      averageConfidence,
      lastUpdateTime: Date.now(),
    }))

    console.log('识别结果:', {
      text: result.text,
      confidence: result.confidence,
      glossSequence: result.glossSequence,
    })
  }, [])

  // 处理WebSocket错误
  const handleWebSocketError = useCallback((error: string) => {
    console.error('WebSocket错误:', error)
    setIsRecognizing(false)
  }, [])

  // 处理WebSocket断开
  const handleWebSocketDisconnect = useCallback(() => {
    console.log('WebSocket连接断开')
    setIsRecognizing(false)
  }, [])

  // 设置WebSocket事件监听器
  useEffect(() => {
    websocketService.on('recognition_result', handleRecognitionResult)
    websocketService.on('error', handleWebSocketError)
    websocketService.on('disconnect', handleWebSocketDisconnect)

    return () => {
      websocketService.off('recognition_result', handleRecognitionResult)
      websocketService.off('error', handleWebSocketError)
      websocketService.off('disconnect', handleWebSocketDisconnect)
    }
  }, [handleRecognitionResult, handleWebSocketError, handleWebSocketDisconnect])

  // 开始识别
  const startRecognition = useCallback(async () => {
    try {
      if (!websocketService.isConnected()) {
        await websocketService.connect()
      }
      
      setIsRecognizing(true)
      setCurrentText('')
      setConfidence(0)
      setGlossSequence([])
      
      console.log('开始手语识别')
    } catch (error) {
      console.error('启动识别失败:', error)
      throw error
    }
  }, [])

  // 停止识别
  const stopRecognition = useCallback(() => {
    setIsRecognizing(false)
    console.log('停止手语识别')
  }, [])

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      if (isRecognizing) {
        stopRecognition()
      }
    }
  }, [isRecognizing, stopRecognition])

  return {
    isRecognizing,
    currentText,
    confidence,
    glossSequence,
    startRecognition,
    stopRecognition,
    websocketService,
    stats,
  }
}
