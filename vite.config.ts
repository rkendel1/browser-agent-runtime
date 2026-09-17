import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        demo: resolve(import.meta.dirname, "demo/index.html"),
        benchmark: resolve(import.meta.dirname, "demo/benchmark.html"),
      },
    },
  },
});
