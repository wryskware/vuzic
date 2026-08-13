# Agent instructions

Read `CLAUDE.md` and `docs/plan.md` before making changes; their project
orientation and settled decisions apply to Codex as well.

## Indexed code retrieval

For unfamiliar implementation questions, start with the local
`cce-latent-music-terrarium` semantic search and request 3–8 focused hits.
After identifying a symbol, use `codebase-memory` only for structural questions
such as callers, callees, dependencies, impact, or execution paths. Then read
the exact source before editing it.

Prefer direct Grep/Glob/Read for known paths, exact strings, generated timeline
fields, shader bindings, query parameters, and dynamic references. Do not send
repository source to the remote `claude.ai Era Context` server.
