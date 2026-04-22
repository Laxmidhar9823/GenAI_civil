import { useState, useEffect, useLayoutEffect, useCallback, useRef } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'

export const TOUR_STORAGE_KEY = 'pavement_tour_v1'

export function shouldShowTour(): boolean {
  return !sessionStorage.getItem(TOUR_STORAGE_KEY)
}

export function markTourDone(): void {
  sessionStorage.setItem(TOUR_STORAGE_KEY, '1')
}

interface TourStep {
  tourId: string
  title: string
  description: string
  tooltipSide: 'left' | 'right'
  centerTooltipVertically?: boolean
  showScrollHint?: boolean
  scrollTargetIntoView?: boolean
}

const ONBOARDING_STEPS: TourStep[] = [
  {
    tourId: 'params-panel',
    title: 'Configuration Panel',
    description:
      'As you chat, parameters are collected here automatically. Use the ✎ button to manually edit any value. Scroll down inside the panel to see the full list and export options.',
    tooltipSide: 'left',
    showScrollHint: true,
  },
  {
    tourId: 'chat-main',
    title: 'Chat Interface',
    description:
      'This is where you interact with the AI assistant. It guides you through a series of questions to build your pavement configuration step by step.',
    tooltipSide: 'left',
    centerTooltipVertically: true,
  },
  {
    tourId: 'ollama-settings',
    title: 'Model & API Settings',
    description:
      'Configure your Ollama server URL, model name, and optional API key here. Click "Check Connection" to verify everything is working before you start chatting.',
    tooltipSide: 'right',
  },
  {
    tourId: 'reset-session',
    title: 'Reset Session',
    description:
      'Clear all collected parameters and start a fresh conversation from the beginning at any time with this button.',
    tooltipSide: 'right',
    scrollTargetIntoView: true,
  },
]

const EXPORT_STEPS: TourStep[] = [
  {
    tourId: 'enable-export',
    title: 'Enable Export',
    description:
      'All parameters have been collected! Click this button to confirm your configuration and unlock the analysis and export tools below.',
    tooltipSide: 'left',
    centerTooltipVertically: true,
  },
  {
    tourId: 'generate-analysis',
    title: 'Generate Analysis',
    description:
      'Run the full finite element analysis using your configuration. This solves the pavement structural model and produces VTK result files ready for 3D visualization.',
    tooltipSide: 'left',
    centerTooltipVertically: true,
  },
  {
    tourId: 'export-json',
    title: 'Export Final JSON',
    description:
      'Download all parameters as a structured JSON file — perfect for archiving your design or re-running the same analysis later.',
    tooltipSide: 'left',
    centerTooltipVertically: true,
  },
  {
    tourId: 'export-text',
    title: 'Export Text Report',
    description:
      'Generate a human-readable summary of your parameters and results — ideal for reporting, review, or sharing with colleagues.',
    tooltipSide: 'left',
    centerTooltipVertically: true,
  },
]

interface SpotRect {
  left: number
  top: number
  width: number
  height: number
}

const PADDING = 10
const TOOLTIP_W = 296
const TOOLTIP_GAP = 20

function TourOverlay({ steps, onDone }: { steps: TourStep[]; onDone: () => void }) {
  const [step, setStep] = useState(0)
  const [spotRect, setSpotRect] = useState<SpotRect | null>(null)
  const [visible, setVisible] = useState(true)

  const currentStep = steps[step]

  const measureElement = useCallback((tourId: string) => {
    const el = document.querySelector<HTMLElement>(`[data-tour="${tourId}"]`)
    if (!el) return
    const r = el.getBoundingClientRect()
    if (r.width === 0 && r.height === 0) return
    setSpotRect({ left: r.left, top: r.top, width: r.width, height: r.height })
  }, [])

  useLayoutEffect(() => {
    const { tourId, scrollTargetIntoView } = currentStep
    const el = document.querySelector<HTMLElement>(`[data-tour="${tourId}"]`)
    if (!el) return

    if (scrollTargetIntoView) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      const t = setTimeout(() => measureElement(tourId), 360)
      return () => clearTimeout(t)
    }

    measureElement(tourId)
  }, [step, currentStep, measureElement])

  useEffect(() => {
    const onResize = () => measureElement(currentStep.tourId)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [currentStep.tourId, measureElement])

  const handleDone = useCallback(() => {
    setVisible(false)
    setTimeout(() => onDone(), 280)
  }, [onDone])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleDone()
      if (e.key === 'ArrowRight') {
        setStep(s => (s < steps.length - 1 ? s + 1 : s))
        if (step === steps.length - 1) handleDone()
      }
      if (e.key === 'ArrowLeft') setStep(s => Math.max(0, s - 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [handleDone, step, steps.length])

  const goNext = useCallback(() => {
    if (step < steps.length - 1) setStep(s => s + 1)
    else handleDone()
  }, [step, steps.length, handleDone])

  const goPrev = useCallback(() => setStep(s => Math.max(0, s - 1)), [])

  if (!spotRect) return null

  const sl = spotRect.left - PADDING
  const st = spotRect.top - PADDING
  const sw = spotRect.width + PADDING * 2
  const sh = spotRect.height + PADDING * 2

  let tooltipLeft: number
  if (currentStep.tooltipSide === 'left') {
    tooltipLeft = sl - TOOLTIP_W - TOOLTIP_GAP
  } else {
    tooltipLeft = sl + sw + TOOLTIP_GAP
  }
  tooltipLeft = Math.max(12, Math.min(tooltipLeft, window.innerWidth - TOOLTIP_W - 12))

  const tooltipTop = currentStep.centerTooltipVertically
    ? Math.max(12, window.innerHeight / 2 - 130)
    : Math.max(12, Math.min(st + 8, window.innerHeight - 300))

  const scrollHintX = sl + sw / 2 - 16
  const scrollHintY = st + sh - 52

  return createPortal(
    <AnimatePresence>
      {visible && (
        <motion.div
          key="tour-root"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.28 }}
          style={{ position: 'fixed', inset: 0, zIndex: 9000, pointerEvents: 'none' }}
        >
          {/* Modal backdrop */}
          <div
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 9000,
              pointerEvents: 'auto',
              cursor: 'default',
            }}
            onClick={(e) => e.stopPropagation()}
          />

          {/* Spotlight */}
          <motion.div
            animate={{ left: sl, top: st, width: sw, height: sh }}
            transition={{ duration: 0.48, ease: [0.19, 1, 0.22, 1] }}
            style={{
              position: 'fixed',
              left: sl,
              top: st,
              width: sw,
              height: sh,
              borderRadius: 14,
              boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.68)',
              border: '2px solid rgba(0, 113, 227, 0.82)',
              outline: '5px solid rgba(0, 113, 227, 0.14)',
              outlineOffset: '1px',
              zIndex: 9001,
              pointerEvents: 'none',
            }}
          />

          {/* Scroll-down hint arrow */}
          <AnimatePresence>
            {currentStep.showScrollHint && (
              <motion.div
                key={`scroll-hint-${step}`}
                initial={{ opacity: 0, scale: 0.6 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.6 }}
                transition={{ delay: 0.55, duration: 0.3 }}
                style={{
                  position: 'fixed',
                  left: scrollHintX,
                  top: scrollHintY,
                  zIndex: 9003,
                  pointerEvents: 'none',
                }}
              >
                <motion.div
                  animate={{ y: [0, 8, 0] }}
                  transition={{ repeat: Infinity, duration: 1.35, ease: 'easeInOut' }}
                  style={{
                    width: 32,
                    height: 32,
                    background: 'linear-gradient(145deg, #0071e3, #0055b3)',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontSize: 13,
                    boxShadow: '0 4px 16px rgba(0, 113, 227, 0.55)',
                  }}
                >
                  ↓
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Tooltip card */}
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, y: 10, scale: 0.94 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.94 }}
              transition={{ duration: 0.3, ease: [0.19, 1, 0.22, 1] }}
              style={{
                position: 'fixed',
                left: tooltipLeft,
                top: tooltipTop,
                width: TOOLTIP_W,
                zIndex: 9004,
                background: 'rgba(255, 255, 255, 0.97)',
                backdropFilter: 'saturate(200%) blur(24px)',
                WebkitBackdropFilter: 'saturate(200%) blur(24px)',
                border: '1px solid rgba(0, 0, 0, 0.07)',
                borderRadius: 18,
                boxShadow:
                  '0 24px 52px rgba(0, 0, 0, 0.2), inset 0 0 0 1px rgba(255, 255, 255, 0.6)',
                padding: '20px 22px 18px',
                pointerEvents: 'auto',
                userSelect: 'none',
              }}
            >
              {/* Progress dots + skip */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 14,
                }}
              >
                <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                  {steps.map((_, i) => (
                    <motion.div
                      key={i}
                      animate={{
                        width: i === step ? 22 : 6,
                        backgroundColor:
                          i === step ? '#0071e3' : i < step ? '#5ac8fa' : '#d2d2d7',
                      }}
                      transition={{ duration: 0.28 }}
                      style={{ height: 6, borderRadius: 3 }}
                    />
                  ))}
                </div>
                <SkipButton onClick={handleDone} />
              </div>

              <h4
                style={{
                  fontSize: 15.5,
                  fontWeight: 650,
                  color: '#1d1d1f',
                  letterSpacing: '-0.022em',
                  margin: '0 0 7px 0',
                  fontFamily: 'inherit',
                }}
              >
                {currentStep.title}
              </h4>

              <p
                style={{
                  fontSize: 13,
                  lineHeight: 1.62,
                  color: '#515154',
                  margin: '0 0 18px 0',
                  fontFamily: 'inherit',
                }}
              >
                {currentStep.description}
              </p>

              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span
                  style={{
                    fontSize: 11.5,
                    color: '#a1a1a6',
                    marginRight: 'auto',
                    fontVariantNumeric: 'tabular-nums',
                    fontFamily: 'inherit',
                  }}
                >
                  {step + 1} / {steps.length}
                </span>

                {step > 0 && <BackButton onClick={goPrev} />}
                <NextButton
                  onClick={goNext}
                  isLast={step === steps.length - 1}
                />
              </div>
            </motion.div>
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  )
}

export default function OnboardingTour({ onDone }: { onDone: () => void }) {
  const handleDone = useCallback(() => {
    markTourDone()
    onDone()
  }, [onDone])
  return <TourOverlay steps={ONBOARDING_STEPS} onDone={handleDone} />
}

export function ExportTour({ onDone }: { onDone: () => void }) {
  return <TourOverlay steps={EXPORT_STEPS} onDone={onDone} />
}

// ── Small sub-components to keep button hover styles clean ──────────────────

function SkipButton({ onClick }: { onClick: () => void }) {
  const ref = useRef<HTMLButtonElement>(null)
  return (
    <button
      ref={ref}
      onClick={onClick}
      onMouseEnter={() => { if (ref.current) ref.current.style.color = '#1d1d1f' }}
      onMouseLeave={() => { if (ref.current) ref.current.style.color = '#86868b' }}
      style={{
        background: 'none',
        border: 'none',
        fontSize: 11.5,
        color: '#86868b',
        cursor: 'pointer',
        padding: '3px 8px',
        borderRadius: 8,
        fontFamily: 'inherit',
        transition: 'color 0.2s',
      }}
    >
      Skip tour
    </button>
  )
}

function BackButton({ onClick }: { onClick: () => void }) {
  const ref = useRef<HTMLButtonElement>(null)
  return (
    <button
      ref={ref}
      onClick={onClick}
      onMouseEnter={() => {
        if (!ref.current) return
        ref.current.style.borderColor = '#0071e3'
        ref.current.style.color = '#0071e3'
      }}
      onMouseLeave={() => {
        if (!ref.current) return
        ref.current.style.borderColor = '#d2d2d7'
        ref.current.style.color = '#3a3a3c'
      }}
      style={{
        height: 32,
        padding: '0 14px',
        border: '1px solid #d2d2d7',
        borderRadius: 99,
        background: 'transparent',
        color: '#3a3a3c',
        fontSize: 12.5,
        fontWeight: 500,
        cursor: 'pointer',
        fontFamily: 'inherit',
        transition: 'all 0.2s ease',
      }}
    >
      ← Back
    </button>
  )
}

function NextButton({ onClick, isLast }: { onClick: () => void; isLast: boolean }) {
  const ref = useRef<HTMLButtonElement>(null)
  return (
    <button
      ref={ref}
      onClick={onClick}
      onMouseEnter={() => {
        if (!ref.current) return
        ref.current.style.transform = 'translateY(-1px)'
        ref.current.style.boxShadow = '0 4px 16px rgba(0, 113, 227, 0.48)'
      }}
      onMouseLeave={() => {
        if (!ref.current) return
        ref.current.style.transform = 'translateY(0)'
        ref.current.style.boxShadow = '0 2px 10px rgba(0, 113, 227, 0.38)'
      }}
      style={{
        height: 32,
        padding: '0 18px',
        border: 'none',
        borderRadius: 99,
        background: 'linear-gradient(135deg, #0071e3, #0059b8)',
        color: 'white',
        fontSize: 12.5,
        fontWeight: 600,
        cursor: 'pointer',
        fontFamily: 'inherit',
        boxShadow: '0 2px 10px rgba(0, 113, 227, 0.38)',
        transition: 'all 0.2s ease',
      }}
    >
      {isLast ? 'Get Started →' : 'Next →'}
    </button>
  )
}
