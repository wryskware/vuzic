/**
 * The two vizfx modules `node --test` cannot import on its own, and the smallest
 * shim that makes them importable. **Not a test file** — the runner's glob is
 * `tests/*.test.ts`, so this is a helper the two `vizfx-*.test.ts` files share.
 *
 * ## Why a shim is needed at all
 *
 * Every other test in this directory imports src/ directly, because the mapping
 * layer's own modules carry explicit `.ts` extensions and import nothing a
 * bundler has to synthesise. The vizfx substrate has two exceptions, and both
 * are Vite features rather than TypeScript ones:
 *
 *   1. `nebula.ts` imports its two shaders as `'./shaders/warp.wgsl?raw'`. Node
 *      resolves the path fine and then refuses the extension.
 *   2. `vizfx.ts` reaches `render/postfx.ts`, whose imports are extensionless.
 *
 * So the shim is two hooks: `.wgsl` loads as its own text (exactly what `?raw`
 * means), and a relative specifier with no extension is retried as `.ts`.
 * Neither replaces a line of logic — the modules under test are the real ones,
 * unmodified — which is the difference between this and a mock. Without it the
 * θ table, the WGSL prelude and `applyExtras` would all be untestable, and
 * `applyExtras`'s "must never throw" is a contract stated in `mapping/target.ts`.
 *
 * `VizFxSim` is constructible headlessly on purpose rather than by luck: its
 * constructor only sizes typed arrays and builds a `PostFx` (which stores its
 * config and nothing else), and every GPU path is guarded by `this.ready &&
 * this.ctx`. Nothing below calls `init`, `tick` or `render`.
 */
import { registerHooks } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import type { VizFxVisual } from '../src/sim/vizfx/slots.ts';
import type { VizFxConfig } from '../src/sim/vizfx/config.ts';
import type { ModTarget } from '../src/mapping/target.ts';

registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith('.') && !/\.(ts|js|mjs|wgsl)(\?|$)/.test(specifier)) {
      try {
        return next(`${specifier}.ts`, context);
      } catch {
        // not a .ts file after all — fall through to the real resolution error
      }
    }
    return next(specifier, context);
  },
  load(url, context, next) {
    if (!url.includes('.wgsl')) return next(url, context);
    const text = readFileSync(fileURLToPath((url.split('?')[0] ?? url)), 'utf8');
    return {
      format: 'module',
      shortCircuit: true,
      source: `export default ${JSON.stringify(text)};`,
    };
  },
});

/** The visual under test: the θ table, its macros and its two shader sources. */
export const NEBULA: VizFxVisual = (
  (await import('../src/sim/vizfx/nebula/nebula.ts')) as { NEBULA: VizFxVisual }
).NEBULA;

/** The chassis. Only its pure halves are exercised; see the header. */
export const VizFxSim = (
  (await import('../src/sim/vizfx/vizfx.ts')) as {
    VizFxSim: new (visual: VizFxVisual, seed: number, config?: VizFxConfig) => ModTarget & {
      config: VizFxConfig;
      serializeExtras(): Record<string, unknown>;
      applyExtras(raw: Record<string, unknown> | undefined): void;
    };
  }
).VizFxSim;
