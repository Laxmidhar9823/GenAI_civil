import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite + React (TypeScript)
// API base URL is configured via VITE_API_BASE_URL (see src/lib/api.ts)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
})
