# AGENT RULES (trade_lab)

1. Work step-by-step only.
2. Perform ONE action at a time, then run an explicit verification step.
3. Before any code change, create a rollback checkpoint:
   - Run `git status`
   - Run `git diff`
   - Commit: `checkpoint: <reason>`
4. Do not introduce extra changes beyond the request.
5. If you see useful additions, suggest them briefly and wait for approval.
6. Keep explanations minimal.
7. Include only a brief conceptual "what/why" when needed.
8. Never run destructive commands without explicit confirmation:
   - `rm -rf`
   - `git reset --hard`
   - `git checkout -- <path>`
   - force-delete/rewrite equivalents
9. Keep the repo clean. Never commit:
   - `data/*.csv`
   - `__pycache__/`
   - `runs.sqlite3`
   - `.DS_Store`
   - `.venv/`
10. Prefer reversible, minimal diffs.
11. Preserve existing behavior unless the task explicitly asks to change it.
12. If blocked, report exact blocker and next safe step.
13. After each completed task:
   - Show short diff summary
   - Show `git status`
   - Stop for user confirmation before commit/push
