import { motion, useReducedMotion } from 'framer-motion'
import type { PropsWithChildren } from 'react'

export default function MotionSection({ children, className }: PropsWithChildren<{ className?: string }>) {
  const reduced = useReducedMotion()

  return (
    <motion.section
      className={className}
      initial={reduced ? { opacity: 1 } : { opacity: 0, y: 18 }}
      whileInView={reduced ? { opacity: 1 } : { opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={
        reduced
          ? { duration: 0 }
          : {
              duration: 0.56,
              ease: [0.22, 1, 0.36, 1],
            }
      }
    >
      {children}
    </motion.section>
  )
}
