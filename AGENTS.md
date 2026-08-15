# Agent instructions

Read `CLAUDE.md` and `docs/plan.md` before making changes; their project
orientation and settled decisions apply to Codex as well.

## Indexed retrieval

After identifying a symbol, use `codebase-memory` only for structural questions
such as callers, callees, dependencies, impact, or execution paths. Then read
the exact source before editing it.

Prefer direct Grep/Glob/Read for known paths, exact strings, docs and plan
questions, generated timeline fields, shader bindings, query parameters, and
dynamic references. For project history and past decisions, use git log. Do
not send repository source to the remote `claude.ai Era Context` server.
