# Agent instructions

Read `CLAUDE.md` and `docs/plan.md` before making changes; their project
orientation and settled decisions apply to Codex as well.

## Indexed retrieval and session memory

Route by question type. Questions about this project's code, docs, plan, or
architecture → `cce-latent-music-terrarium` `context_search` first, 3–8 focused
hits (the index covers `docs/`, so plan questions belong there, not grep).
Questions about project history or past decisions → `session_recall` first.
After identifying a symbol, use `codebase-memory` only for structural questions
such as callers, callees, dependencies, impact, or execution paths. Then read
the exact source before editing it.

Prefer direct Grep/Glob/Read for known paths, exact strings, generated timeline
fields, shader bindings, query parameters, and dynamic references. After
settling a non-obvious decision, record it with `record_decision`; after
tracing a non-obvious flow, `record_code_area`. Do not send repository source
to the remote `claude.ai Era Context` server.
