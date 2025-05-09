import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { // Requests to /api/... will be proxied
        target: 'http://localhost:5000', // Your Flask backend
        changeOrigin: true,
        // rewrite: (path) => path.replace(/^\/api/, '') // if your Flask routes don't have /api
      }
    }
  }
})