import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev the UI calls /api/* on its own origin and Vite forwards it to FastAPI,
// so the browser never needs CORS and no backend URL is baked into the bundle.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.API_URL ?? "http://localhost:8000", changeOrigin: true } },
  },
});
