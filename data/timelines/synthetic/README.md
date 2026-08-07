# Synthetic timeline fixture

A fabricated ~210 s / 120 BPM "track" emitted in the version-2 timeline format
from `docs/plan.md` Decision 2, and the shared contract fixture that both the
web app and the Python analysis pipeline conform to. It has no audio behind it;
its purpose is to let the loader, sampler, mapping layer and simulation be built
and tested before any real analysis exists, and to give the analysis pipeline a
byte-layout target to match. It contains obvious, deliberately legible events —
drums entering at 16 s, bass at 24 s, vocals through the verses and louder in the
choruses, pads dominating the intro and bridge — plus a 64-dim latent trajectory
whose per-label centers make the three choruses land in the same neighbourhood.
`recurTime` / `recurStr` are present but all zero. Regenerate with
`node tools/gen_synthetic_timeline.mjs` from the repo root; the generator is
zero-dependency and seeded, so the output is byte-identical every run.
