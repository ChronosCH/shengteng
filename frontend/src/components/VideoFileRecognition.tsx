import React, { useState, useRef, useCallback } from 'react'
import { Box, Button, LinearProgress, Typography, Paper, Stack, Chip, Divider, Table, TableBody, TableCell, TableHead, TableRow } from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import DownloadIcon from '@mui/icons-material/Download'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import signRecognitionService, { RecognitionResultData } from '../services/signRecognitionService'

interface Props {
  onResult?: (r: RecognitionResultData) => void
}

const VideoFileRecognition: React.FC<Props> = ({ onResult }) => {
  const [file, setFile] = useState<File | null>(null)
  const [fileHash, setFileHash] = useState<string>('')
  const [taskId, setTaskId] = useState<string>('')
  const [progress, setProgress] = useState<number>(0)
  const [status, setStatus] = useState<string>('idle')
  const [result, setResult] = useState<RecognitionResultData | null>(null)
  const [error, setError] = useState<string>('')
  const abortRef = useRef<AbortController | null>(null)

  const reset = () => {
    abortRef.current?.abort()
    setFile(null)
    setFileHash('')
    setTaskId('')
    setProgress(0)
    setStatus('idle')
    setResult(null)
    setError('')
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
      setError('')
    }
  }

  const uploadAndStart = useCallback(async () => {
    try {
      if (!file) return
      setStatus('uploading')
      const uploadRes = await signRecognitionService.uploadVideo(file)
      if (!uploadRes.success || !uploadRes.data) throw new Error(uploadRes.message)
      setFileHash(uploadRes.data.file_hash)

      setStatus('starting')
      const tid = await signRecognitionService.startTask(uploadRes.data.file_hash)
      setTaskId(tid)

      setStatus('processing')
      abortRef.current = new AbortController()
      const r = await signRecognitionService.pollResult(tid, 2000, p => setProgress(p), abortRef.current.signal)
      setResult(r)
      setStatus('finished')
      setProgress(1)
      onResult?.(r)
    } catch (e: any) {
      setError(e.message || '处理失败')
      setStatus('error')
    }
  }, [file, onResult])

  const downloadSrt = () => {
    if (!result?.srt_path) return
    const url = signRecognitionService.getSrtDownloadUrl(result.srt_path)
    window.open(url, '_blank')
  }

  return (
    <Paper variant="outlined" sx={{ p: 3, borderRadius: 3 }}>
      <Stack spacing={2}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>离线视频手语识别</Typography>
        <Stack direction="row" spacing={2} alignItems="center">
          <Button variant="contained" component="label" startIcon={<CloudUploadIcon />}>选择视频
            <input hidden type="file" accept="video/*" onChange={handleFileChange} />
          </Button>
          {file && <Chip label={file.name} color="info" />}
          {status === 'uploading' && <Chip label="上传中" color="warning" />}
          {status === 'processing' && <Chip label="处理中" color="info" />}
          {status === 'finished' && <Chip label="已完成" color="success" />}
          {status === 'error' && <Chip label="出错" color="error" />}
        </Stack>
        {progress > 0 && status !== 'finished' && (
          <Box>
            <LinearProgress variant="determinate" value={Math.min(progress * 100, 100)} sx={{ height: 8, borderRadius: 2 }} />
            <Typography variant="caption" color="text.secondary">进度 {(progress * 100).toFixed(1)}%</Typography>
          </Box>
        )}
        <Stack direction="row" spacing={2}>
          <Button disabled={!file || status === 'processing' || status === 'uploading'} variant="contained" startIcon={<PlayArrowIcon />} onClick={uploadAndStart}>开始</Button>
          <Button disabled={status !== 'processing'} color="warning" variant="outlined" startIcon={<RestartAltIcon />} onClick={() => abortRef.current?.abort()}>取消</Button>
          <Button disabled={!result?.srt_path} variant="outlined" startIcon={<DownloadIcon />} onClick={downloadSrt}>字幕</Button>
          <Button disabled={status==='processing'} onClick={reset}>重置</Button>
        </Stack>
        {error && <Typography color="error" variant="body2">{error}</Typography>}
        {result && (
          <Box>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>识别结果</Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>文本: {result.text || '(空)'}</Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>Gloss: {result.gloss_sequence.join(' ')}</Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>总体置信度: {(result.overall_confidence*100).toFixed(2)}%</Typography>
            <Typography variant="caption" color="text.secondary">时长: {result.duration.toFixed(2)}s 帧: {result.frame_count} FPS: {result.fps.toFixed(1)}</Typography>
            <Box sx={{ mt: 2, maxHeight: 240, overflow: 'auto' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>分段</Typography>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600 }}>#</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>时间(s)</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Gloss</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>置信度</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {result.segments.map((seg, i) => (
                    <TableRow key={i} hover>
                      <TableCell>{i+1}</TableCell>
                      <TableCell>{seg.start_time.toFixed(2)} - {seg.end_time.toFixed(2)}</TableCell>
                      <TableCell>{seg.gloss_sequence.join(' ')}</TableCell>
                      <TableCell>{(seg.confidence*100).toFixed(1)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          </Box>
        )}
      </Stack>
    </Paper>
  )
}

export default VideoFileRecognition
