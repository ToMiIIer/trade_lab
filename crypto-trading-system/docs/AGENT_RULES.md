# Agent Rules

## Workflow
- One change-set at a time.
- Create a git checkpoint before substantive edits.
- Run tests after each implemented change-set.
- Prefer additive, reversible changes.

## Safety and Reliability
- Risk logic is hardcoded in Engine; YAML provides numeric params only.
- Any failure path must degrade to `NO_TRADE`.
- Never commit or log secrets.
- Use `.env.example` and environment variables for credentials.
- Ensure runs are idempotent (`run_id`, unique constraints, upserts/guards).

## Boundary Discipline
- Keep Engine vs Artifact separation clean.
- Do not place core control logic into YAML.
- Keep execution paper-only for Phase 1 MVP.
