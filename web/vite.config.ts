import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// GitHub Pages serves the site from https://<user>.github.io/<repo>/, so every
// asset path must be prefixed with the repo name. Use import.meta.env.BASE_URL
// in code instead of hard-coding this.
const REPO_BASE = '/doodle-live/'

export default defineConfig({
  base: REPO_BASE,
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
