import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import AssistantPage from './pages/AssistantPage'
import DocsPage from './pages/DocsPage'
import ExamplesPage from './pages/ExamplesPage'
import LandingPage from './pages/LandingPage'

function Navigation() {
  return (
    <header className="site-header">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          P
        </div>
        <div className="brand-info">
          <strong>Pavement AI</strong>
          <p>Local Configuration Assistant</p>
        </div>
      </div>

      <nav className="site-nav" aria-label="Primary">
        <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Overview
        </NavLink>
        <NavLink to="/app" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Assistant
        </NavLink>
        <NavLink to="/docs" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Documentation
        </NavLink>
        <NavLink to="/examples" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Examples
        </NavLink>
      </nav>
    </header>
  )
}

function RoutedContent() {
  const location = useLocation()
  const reduced = useReducedMotion()

  return (
    <AnimatePresence mode="wait">
      <motion.main
        key={location.pathname}
        className="page-shell"
        initial={reduced ? { opacity: 1 } : { opacity: 0, scale: 0.995 }}
        animate={reduced ? { opacity: 1 } : { opacity: 1, scale: 1 }}
        exit={reduced ? { opacity: 1 } : { opacity: 0, scale: 1.005 }}
        transition={reduced ? { duration: 0 } : { duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
      >
        <Routes location={location}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/app" element={<AssistantPage />} />
          <Route path="/docs" element={<DocsPage />} />
          <Route path="/examples" element={<ExamplesPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </motion.main>
    </AnimatePresence>
  )
}

export default function App() {
  return (
    <div className="site-container">
      <Navigation />
      <RoutedContent />
    </div>
  )
}
