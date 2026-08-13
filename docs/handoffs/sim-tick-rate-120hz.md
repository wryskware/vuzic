# Handoff — raise the simulation tick rate to 120 Hz

**Status:** specified, not started.
**Decision:** made by the user on 2026-08-12. Target is a fixed **120 Hz** tick.
**Scope:** `web/src/main.ts`, `web/src/audio/clock.ts`, `web/src/sim/plife/`,
with mandatory *no-regression* compensation in `web/src/sim/physarum/` and
`web/src/sim/vizfx/`.

---

## 1. Objective

Particle life currently produces **60 distinct particle states per second**. On a
120/240 Hz display every state is presented 2–4 times, so 1/2 to 3/4 of rendered
frames show particles that have not moved. Raise the tick rate so the sim
produces **120 states per second** at unchanged wall-clock world speed.

This is a motion-fidelity change. It is **not** a performance change — it costs
roughly 2× the plife sim GPU work per second, and the existing particle governor
will shed population to pay for it. That trade (smoother motion, fewer particles)
is accepted; see §6 for what to measure.

**Explicitly rejected alternatives**, do not substitute one:

- Render-side interpolation between the last two tick states. Cheaper and
  smoother, but the dynamics stay 60 Hz. The user wants real states.
- Capping or reducing the render rate. This was an earlier misreading of the
  request; it is the opposite of the goal.

---

## 2. The finding that shapes this change — read before planning

**Only plife is dt-parameterized. The other two substrates are per-step models,
and a naive global tick-rate change silently breaks both.**

| substrate | time model | effect of doubling tick rate, uncompensated |
|---|---|---|
| **plife** | true dt integration — `force·dt`, `exp(-friction·dt)`, `pos += v·dt`, `dt/tau` (`shaders/step.wgsl:420-431`, `:271-273`) | correct: same world speed, 2× temporal resolution. **This is the target.** |
| **physarum** | **no dt at all.** `moveDist` = distance *per step*, `rotate` = radians *per step*, `decay`/`deposit` per step (`physarum/shaders/common.wgsl:65-68`) | **world runs 2× fast, trails decay 2× fast in wall-clock. Whole preset library invalidated.** |
| **vizfx** | per-step factors by explicit design — its own comment says "every per-step quantity in θ (`zoom`, `decay`, `rotate`) is a *per-step* factor" (`vizfx/vizfx.ts:111-124`) | **same breakage.** Also `Globals.time` is documented as `tick / 60` (`vizfx/shaders/common.wgsl:33`) — a hardcoded rate. |

`SECONDS_PER_TICK` (`main.ts:51`) is app-global and feeds `TimelineSampler`,
`AudioClock`, `ImpulseEngine` and `modulator.update`. Changing it in place drags
physarum and vizfx along with plife. **That is the single biggest hazard in this
task.** A worker who edits the three `1/60` constants and declares victory will
ship two visibly broken substrates.

### Required design: decouple app tick rate from per-substrate step rate

Introduce an app-level `TICK_HZ = 120` (so `SECONDS_PER_TICK = 1/120`) and give
each substrate an explicit **steps-per-tick** factor so its *steps per second* is
preserved unless deliberately changed:

- **plife** — 1 step per tick, `SUBSTEP_DT = 1/120`. Steps/sec 60 → 120. Changed
  on purpose.
- **physarum** — 1 step per **2** ticks. Steps/sec stays 60. Nothing about its
  world changes.
- **vizfx** — 1 step per **2** ticks. Steps/sec stays 60. Nothing changes.

Implement the halving **outside θ**. `config.speed` is "steps per clock tick" and
is part of the preset/θ vector for both substrates — do **not** rewrite saved
`speed` values or the θ ranges to compensate. Put the factor in the substrate's
own tick handling, where `speed` is accumulated (plife's is at `plife.ts:1965-1969`),
so existing presets and autosaves keep their meaning and the explorer's search
space is untouched.

The odd-tick case matters: physarum and vizfx now see ticks on which they run
zero steps. Both already tolerate zero-substep ticks (plife's render path notes
"a frame can be drawn with zero substeps", `plife.ts:2383-2384`) — confirm the
same holds for the other two rather than assuming it.

---

## 3. Changes

### 3.1 The clock

| file:line | from | to |
|---|---|---|
| `main.ts:51` | `SECONDS_PER_TICK = 1/60` | `1/120` (derive from a named `TICK_HZ`) |
| `main.ts:70` | `MAX_FREE_TICKS_PER_FRAME = 4` | `8` — preserves the same *wall-clock* free-run tolerance |
| `audio/clock.ts:63` | `maxTicksPerFrame ?? 8` | `16` — same reason; this is the catch-up cap at `clock.ts:244-250` |
| `audio/clock.ts:29` | `DEFAULT_TICK = 1/60` | `1/120`, or leave and always pass explicitly — state which and why |

These four move together. `maxTicksPerFrame` in particular: leaving it at 8 halves
the stall tolerance and the sim starts discarding time (`clock.ts:250`) at twice
the frame rate it used to.

### 3.2 plife

- `SUBSTEP_DT` (`plife.ts:220`) → `1/120`.
- `TICK_DT` (`plife.ts:229`) → `1/120`. **This duplicate is now load-bearing.**
  Its own comment says "If the app's timestep ever becomes configurable, this is
  the line that has to follow it" — that is now. It drives the population-lane
  EMAs via `updatePopulation` (`plife.ts:1955`). Prefer threading the real value
  through `Sim.tick` over editing a second constant; if you keep it a constant,
  add a test that fails if it and `SECONDS_PER_TICK` disagree.
- Update the comments at `plife.ts:214-229` and `:1981` ("measurably in the 1/60 s
  a substep covers") — they state the old rate as fact.
- `preset.ts:429` and `shaders/step.wgsl:189` both assert stability "at dt = 1/60".
  Halving dt strictly *increases* explicit-Euler stability margin, so this is safe,
  but the comments must be corrected rather than left lying.

### 3.3 physarum and vizfx — no-regression only

Apply the 1-step-per-2-ticks factor. Success is "pixel-comparable to before at
the same seed and track." Any visible change is a bug in this task, not a new
look. `vizfx/shaders/common.wgsl:33`'s `tick / 60` must be re-derived from the
step rate, not the tick rate — check where TS writes that word before assuming
which.

### 3.4 Things that are already rate-parametric — verify, don't rebuild

- `TimelineSampler` takes `secondsPerTick` and interpolates rows
  (`sampler.ts:130-139`). Higher tick rate just samples the analysis more finely.
- `ImpulseEngine` / `EventCursor` take `secondsPerTick` (`impulses.ts:227-233`).
  Note `EventCursor.tickOf` rounds to nearest tick and its comment reasons about
  16.7 ms (`sampler.ts:78-86`) — the behaviour is correct and gets *better*, the
  comment needs updating.
- `renderDtFrames` (`plife.ts:2377-2380`, `physarum.ts:186-187`) measures the gap
  between `render()` calls. Untouched by this change.
- Auto-exposure uses measured seconds against `autoTau` (`render/postfx.ts:350`).
  Untouched.

---

## 4. Accepted consequences — state them, don't design around them

1. **Cost.** plife physics doubles per second. The governor
   (`plife.ts:896-923`) will shed particles. This is the trade the user accepted.
   Do not "fix" it by touching the control law or its defaults (`config.ts:526-527`).
2. **plife presets shift.** Halving dt refines the Euler integration toward the
   true solution; trajectories change. With ~48% of particles pinned at `maxSpeed`
   (`plife.ts:186-193`) the clustering character may move visibly. Expected,
   acceptable, **must be reported with before/after** — the project's deciding
   criterion is that the output looks good, so if a shipped preset looks worse,
   that is a finding for the user, not something to quietly retune.
3. **Determinism changes.** The PCG stream is keyed on tick, so the same
   `(track, seed)` produces a different world. Determinism here is tooling, not
   product (`CLAUDE.md`), so this is fine — but any golden-value test will break
   and must be re-baselined deliberately, not deleted.
4. **60 Hz displays.** They now pay 2× physics for zero visible gain. Note it;
   do not build adaptive rate selection in this pass.

---

## 5. Explicit exclusions

- Do not make `dt` variable or frame-rate-derived. Fixed-timestep is settled
  (`plan.md`); 1/60 → 1/120 keeps it fixed. Anything else is a reversal that
  needs the user's sign-off.
- Do not add render-side interpolation. Considered and rejected above.
- Do not cap or reduce the render rate.
- Do not touch the governor's law, constants, or defaults.
- Do not rewrite θ, preset `speed` values, or explorer search ranges.

---

## 6. Verification

Tests in `web/tests/`, vitest.

**Unit — required.**
- Tick accounting: over 10 s of simulated transport at 120 Hz, plife takes 1200
  steps, physarum and vizfx take 600 each. This is the whole §2 hazard as one
  assertion — write it first.
- `AudioClock.pump` catch-up: a 100 ms stall is absorbed without discarding time
  at the new `maxTicksPerFrame`; a 2 s stall still resyncs via `clock.ts:250`.
- Impulse envelope decay over a fixed wall-clock interval is unchanged between
  60 Hz and 120 Hz ticking (it takes dt, so this should pass — it is the
  regression guard).
- A guard tying `TICK_DT` to `SECONDS_PER_TICK` if `TICK_DT` survives as a
  constant.

**Cost + budget — required, this is the number the decision rests on.** Record
settled `effectiveBudget` (`PlifeStats`, `plife.ts:327-328`) at 60 Hz vs 120 Hz
on the same machine, track, and window size, after ~60 s of settling each.
Report the actual population cost of the change. If it is much worse than ~2×,
something else is wrong and that is a finding.

**Visual — by the author, not by an agent.** Two standing constraints:
automation tabs run rAF at ~4–8 fps, so a subagent cannot judge motion,
transients, or fps readouts — which is precisely what this change is about; and
browser-verifying agents pollute the autosave slot, so any such work needs a
persisted-state contract. An agent may verify equilibrium stills only.

The three things to look at: plife motion smoothness on the 120 Hz+ display (the
point of the change); plife preset character before/after (§4.2); physarum and
vizfx unchanged (§3.3).

---

## 7. Return format

- The diff, plus reasoning for the steps-per-tick placement and how odd ticks are
  handled in physarum/vizfx.
- Test output verbatim, including failures.
- The §6 budget comparison with its conditions.
- Before/after on plife preset character, honestly — including "this one looks
  worse".
- Confirmation that physarum and vizfx are unchanged, and how you established it.
- Any hardcoded-60 site not listed in §8 that you found.

---

## 8. Source map

| what | where |
|---|---|
| `SECONDS_PER_TICK`, free-run cap | `web/src/main.ts:51`, `:70`, `:1400-1408` |
| sampler / impulse / modulator wiring | `web/src/main.ts:255`, `:264`, `:352`, `:1365-1366` |
| tick pump + catch-up cap | `web/src/audio/clock.ts:29`, `:63`, `:232-252` |
| plife `SUBSTEP_DT` / `TICK_DT` | `web/src/sim/plife/plife.ts:214-229` |
| plife substep expansion | `web/src/sim/plife/plife.ts:1965-1969` |
| plife dt use in shader | `web/src/sim/plife/shaders/step.wgsl:271-273`, `:420-431` |
| plife stability claims | `web/src/sim/plife/preset.ts:429`, `shaders/step.wgsl:189` |
| governor law + defaults | `web/src/sim/plife/plife.ts:896-923`, `config.ts:526-527` |
| physarum per-step params | `web/src/sim/physarum/shaders/common.wgsl:65-68` |
| vizfx `STEP_DT` + per-step rationale | `web/src/sim/vizfx/vizfx.ts:111-124` |
| vizfx `time = tick/60` | `web/src/sim/vizfx/shaders/common.wgsl:33-36` |
| sampler / event cursor rate params | `web/src/timeline/sampler.ts:67-86`, `:128-139` |
| impulse engine rate params | `web/src/sim/impulses.ts:223-241`, `:269-272` |
