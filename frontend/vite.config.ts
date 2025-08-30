import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const root = new URL('.', import.meta.url).pathname
  const env = loadEnv(mode, root, '')
  const target = env.VITE_API_URL || 'http://localhost:8000'
  return {
    plugins: [react()],
    esbuild: {
      // 忽略 TypeScript 错误
      logOverride: { 'this-is-undefined-in-esm': 'silent' },
      target: 'es2020'
    },
    define: {
      // 忽略类型检查
      'process.env.NODE_ENV': '"development"'
    },
    server: {
      port: 5173,
      host: true,
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          secure: false,
        },
        '/ws': {
          target,
          ws: true,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
    optimizeDeps: {
      include: ['three', '@react-three/fiber', '@react-three/drei'],
    },
  }
})
