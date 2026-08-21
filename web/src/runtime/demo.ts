/**
 * Is this the published demo cut?
 *
 * Not `import.meta.env.PROD` — `npm run build` is production too, and it is the
 * build you want when running the whole thing locally with a server beside it.
 * This is specifically "the build that goes to CloudFront", set by
 * `build:publish` via `--mode demo` and `.env.demo`.
 *
 * Roadmap item 10 defines that shape as static only: no `terrarium-server`, no
 * uploads, no export UI. It is deliberately expressed as two absences rather
 * than a feature flag threaded through the UI, because both surfaces were
 * already optional — the upload half of the track panel exists only when a
 * server answered the probe, and every panel's export folder exists only when
 * `main.ts` hands it a host. So the demo build simply declines to probe and
 * declines to supply the host, and the UI it drives disappears on its own.
 *
 * The alternative — shipping the probe and letting it fail — is what the first
 * deploy did, and over HTTPS it is worse than untidy: the probe reaches for
 * `http://localhost:8765` from an `https://` page, which the browser blocks as
 * mixed content and reports as a console error to every visitor.
 *
 * Optional chaining because this is evaluated at module load and the unit tests
 * run under bare Node, where `import.meta.env` does not exist at all. Vite
 * substitutes the whole object at build time regardless, so the published bundle
 * reads a constant here.
 *
 * That is enough for rollup to drop `probeServer`'s body outright (verified: the
 * demo bundle contains no `AbortSignal.timeout`). The export panel's code does
 * still ship — it is statically imported and only gated where it is *called* —
 * so what the demo build guarantees there is that it never runs, not that it is
 * absent. Removing it too would mean a dynamic import, which is a bundle-size
 * argument and not one this cut needs to have.
 */
export const DEMO_BUILD = import.meta.env?.VITE_DEMO === '1';
