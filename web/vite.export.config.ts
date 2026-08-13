import { defineConfig } from 'vite';

/**
 * Node-targeted bundle for the native Dawn export worker.
 *
 * Vite embeds every `*.wgsl?raw` import from the browser renderer into this
 * bundle. The native binding stays external so Node loads its platform binary
 * at runtime rather than Rollup attempting to copy or translate it.
 */
export default defineConfig({
  build: {
    emptyOutDir: true,
    outDir: 'dist-worker',
    ssr: 'src/export/worker.ts',
    rollupOptions: {
      external: ['webgpu'],
      output: {
        entryFileNames: 'worker.mjs',
      },
    },
  },
});
