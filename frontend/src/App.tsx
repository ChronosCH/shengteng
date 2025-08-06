import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import { CssBaseline } from '@mui/material'
import { useState } from 'react'

import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout.tsx'
import HomePage from './pages/HomePage.tsx'
import RecognitionPage from './pages/RecognitionPage.tsx'
import LearningPage from './pages/LearningPage.tsx'
import AvatarPage from './pages/AvatarPage.tsx'
import SettingsPage from './pages/SettingsPage.tsx'
import LabPage from './pages/LabPage.tsx'

function App() {
  const [darkMode, setDarkMode] = useState(false)

  // 马卡龙配色主题
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: darkMode ? '#FF9AA2' : '#FFB3BA', // 柔和粉色
        light: darkMode ? '#FFB3BA' : '#FFD6CC',
        dark: darkMode ? '#FF6B73' : '#FF8A95',
        contrastText: darkMode ? '#2D1B39' : '#4A4A4A',
      },
      secondary: {
        main: darkMode ? '#B5EAD7' : '#C7CEDB', // 薄荷绿/淡紫
        light: darkMode ? '#C7CEDB' : '#E0E4CC',
        dark: darkMode ? '#9BC1BC' : '#A8B5C5',
        contrastText: darkMode ? '#2D1B39' : '#4A4A4A',
      },
      success: {
        main: '#B5EAD7', // 薄荷绿
        light: '#C7F0DB',
        dark: '#9BC1BC',
      },
      info: {
        main: '#C7CEDB', // 淡紫蓝
        light: '#D6DCE5',
        dark: '#A8B5C5',
      },
      warning: {
        main: '#FFDAB9', // 桃色
        light: '#FFE7CC',
        dark: '#FFCC99',
      },
      error: {
        main: '#FFB3BA', // 淡粉红
        light: '#FFD6CC',
        dark: '#FF9AA2',
      },
      background: {
        default: darkMode ? '#1A1625' : '#FEFEFE',
        paper: darkMode ? '#2D1B39' : '#FFFFFF',
      },
      text: {
        primary: darkMode ? '#E8E3F0' : '#4A4A4A',
        secondary: darkMode ? '#B8A9C9' : '#7A7A7A',
      },
    },
    typography: {
      fontFamily: '"Inter", "Microsoft YaHei", "PingFang SC", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontWeight: 700,
        letterSpacing: '-0.02em',
      },
      h2: {
        fontWeight: 600,
        letterSpacing: '-0.01em',
      },
      h3: {
        fontWeight: 600,
      },
      h4: {
        fontWeight: 600,
      },
      h5: {
        fontWeight: 500,
      },
      h6: {
        fontWeight: 500,
      },
      body1: {
        lineHeight: 1.7,
      },
      body2: {
        lineHeight: 1.6,
      },
    },
    shape: {
      borderRadius: 16, // 更圆润的边角
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
            boxShadow: darkMode 
              ? '0 8px 32px rgba(0, 0, 0, 0.3)' 
              : '0 8px 32px rgba(255, 179, 186, 0.15)',
            border: darkMode 
              ? '1px solid rgba(255, 154, 162, 0.2)' 
              : '1px solid rgba(255, 179, 186, 0.1)',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            textTransform: 'none',
            fontWeight: 500,
            padding: '10px 24px',
          },
          contained: {
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 4px 16px rgba(255, 179, 186, 0.3)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 20,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 20,
            fontWeight: 500,
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? 'rgba(45, 27, 57, 0.95)' : 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(20px)',
            borderBottom: darkMode 
              ? '1px solid rgba(255, 154, 162, 0.1)' 
              : '1px solid rgba(255, 179, 186, 0.1)',
            boxShadow: 'none',
          },
        },
      },
    },
  })

  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Layout darkMode={darkMode} onToggleDarkMode={() => setDarkMode(!darkMode)}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/recognition" element={<RecognitionPage />} />
              <Route path="/learning" element={<LearningPage />} />
              <Route path="/avatar" element={<AvatarPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/lab" element={<LabPage />} />
            </Routes>
          </Layout>
        </Router>
      </ThemeProvider>
    </ErrorBoundary>
  )
}

export default App
