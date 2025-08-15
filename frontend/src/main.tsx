import React from 'react'
import ReactDOM from 'react-dom/client'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import App from './App'

// 导入MUI Fade修复
import './utils/muiFadeFix'

// 马卡龙配色主题
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#FFB6C1', // 粉红色
      light: '#FFE4E1',
      dark: '#FF69B4',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#98FB98', // 薄荷绿
      light: '#F0FFF0',
      dark: '#90EE90',
      contrastText: '#2E7D32',
    },
    error: {
      main: '#FFB3BA', // 柔和的红色
      light: '#FFCCCB',
      dark: '#FF6B6B',
    },
    warning: {
      main: '#FFDFBA', // 柔和的橙色
      light: '#FFF2E6',
      dark: '#FFB347',
    },
    info: {
      main: '#BFEFFF', // 天蓝色
      light: '#E6F7FF',
      dark: '#87CEEB',
    },
    success: {
      main: '#98FB98', // 薄荷绿
      light: '#F0FFF0',
      dark: '#90EE90',
    },
    background: {
      default: '#FFFEF7', // 温暖的米白色
      paper: '#FFFFFF',
    },
    text: {
      primary: '#4A4A4A',
      secondary: '#7A7A7A',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
      color: '#4A4A4A',
    },
    h5: {
      fontWeight: 500,
      color: '#4A4A4A',
    },
    h6: {
      fontWeight: 500,
      color: '#4A4A4A',
    },
    body1: {
      fontSize: '0.95rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.85rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 20,
          fontWeight: 500,
          padding: '12px 24px',
          boxShadow: '0 4px 12px rgba(255, 182, 193, 0.3)',
          '&:hover': {
            boxShadow: '0 6px 16px rgba(255, 182, 193, 0.4)',
            transform: 'translateY(-2px)',
          },
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        contained: {
          background: 'linear-gradient(135deg, #FFB6C1 0%, #FF69B4 100%)',
          color: '#FFFFFF',
          '&:hover': {
            background: 'linear-gradient(135deg, #FF69B4 0%, #FF1493 100%)',
          },
        },
        outlined: {
          borderColor: '#FFB6C1',
          color: '#FF69B4',
          '&:hover': {
            borderColor: '#FF69B4',
            backgroundColor: 'rgba(255, 182, 193, 0.1)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(255, 182, 193, 0.2)',
          '&:hover': {
            boxShadow: '0 12px 40px rgba(0, 0, 0, 0.12)',
          },
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(255, 182, 193, 0.2)',
          overflow: 'hidden',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #FFB6C1 0%, #98FB98 100%)',
          boxShadow: '0 4px 20px rgba(255, 182, 193, 0.3)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          '& .MuiTabs-indicator': {
            backgroundColor: '#FF69B4',
            height: 3,
            borderRadius: 2,
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '1rem',
          color: '#7A7A7A',
          '&.Mui-selected': {
            color: '#FF69B4',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          fontWeight: 500,
        },
        filled: {
          background: 'linear-gradient(135deg, #FFB6C1 0%, #98FB98 100%)',
          color: '#FFFFFF',
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          '&:hover': {
            backgroundColor: 'rgba(255, 182, 193, 0.1)',
            transform: 'scale(1.05)',
          },
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        },
      },
    },
  },
})

// 隐藏加载屏幕
const hideLoading = () => {
  const loadingElement = document.getElementById('loading')
  if (loadingElement) {
    loadingElement.style.display = 'none'
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <App />
  </ThemeProvider>,
)

// 应用加载完成后隐藏加载屏幕
setTimeout(hideLoading, 100)
