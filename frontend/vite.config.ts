import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL("../web/react-dist", import.meta.url)),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/characters": "http://127.0.0.1:8000",
      "/sessions": "http://127.0.0.1:8000",
      "/settings": "http://127.0.0.1:8000",
      "/images": "http://127.0.0.1:8000",
      "/chat": "http://127.0.0.1:8000",
      "/transcribe": "http://127.0.0.1:8000",
    },
  },
});
