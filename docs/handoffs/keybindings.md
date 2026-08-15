# Handoff — Keybindings mini-thread

Scoped 2026-08-15. A short, self-contained package meant to run in a
**worktree**, in parallel with other threads. Roadmap: Phase 3 item 7,
pulled forward.

Routing note for the orchestrator: implementation worker = Opus,
reasoning effort **low/medium** — this is a small, well-specified change
to already-built UI.

## Objective

Three bindings, all view-layer only:

- **`s`** toggles the settings panel (the Tweakpane workbench).
- **`t`** toggles the timeline + top menu together — the `#transport`
  bar (play / halt / status) and the scrub/timeline strip drawn on the
  `#overlay` canvas. One key, both surfaces: the goal is "nothing but
  the terrarium".
- **Double-click** toggles fullscreen. Bind it on the stage/overlay
  canvases, not `window`, so double-clicks inside the workbench or on
  buttons don't trigger it.

## Where things live

- `web/src/main.ts:1170` — the existing `window` keydown handler. Add
  the new keys here; do not create a second listener. Existing claims:
  `h` (halt), `Space` (play/pause), arrows (seek), and — only while the
  explorer is active — `r`, `Backspace`, `Escape`. Don't collide.
- `web/index.html` — `#stage`, `#overlay`, `#transport`.
- `web/src/ui/workbench.ts` — the workbench builds/mounts the panel;
  find its root element (or add a small show/hide affordance to its API
  if that's cleaner than styling its container from outside).

## Constraints

- The existing handler bails when `ev.target` is an `HTMLInputElement`.
  Letter keys make this guard load-bearing: extend it to cover
  `HTMLTextAreaElement`, `HTMLSelectElement`, and `isContentEditable`,
  so typing a profile name never toggles the UI.
- Visibility state is **session-only** — deliberately not persisted. A
  fresh load always shows the full UI. Do not add it to autosave,
  extras blocks, or profiles.
- Hiding must not tear anything down: the overlay keeps rendering state
  (`hidden` via CSS/class, not destroyed), the transport keeps working
  via keyboard (`Space` still plays while the bar is hidden).
- Fullscreen: use the standard Fullscreen API on `document.body` (or
  the app root) so the canvases *and* any visible UI participate;
  `resize` already flows through the existing `window` resize listener,
  but verify the canvas resizes on `fullscreenchange` too.
- Don't touch simulation, persistence, or panel content. This thread is
  bindings only.

## Verification

- Unit-testable slice: extract the key → action dispatch (including the
  editable-target guard) into a small pure function and test it with
  synthetic events (`node --test`, like the rest of `web/tests/`).
- Manual pass in a real browser for the rest: `s`, `t`, double-click,
  typing `s`/`t` into the profile-name box (must not toggle), `t` then
  `Space` (must still play), fullscreen in/out restores layout.
- Automation caveats, non-negotiable: automated tabs run rAF throttled
  (~4–8 fps), so do not judge motion or fps there; and do not leave
  residue in `localStorage` (`lmt.*` keys) — don't save profiles or
  tweak sliders while verifying. A worktree dev server on its own port
  is its own origin, which isolates you; keep it that way by not
  verifying against the main dev server's port.
- Finish with the full suite: `npm test` in `web/`.

## Return format

Report: what was bound where, the exact guard logic, any API added to
the workbench, test results (full-suite count), and anything about
fullscreen/canvas-resize behavior that surprised you — with reasoning,
not just outcomes.
