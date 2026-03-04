# Agent Context

## Engine vs Artifact
- **Engine (Python code):** deterministic execution, risk logic, data validation, persistence, scheduling, and fail-safe behavior.
- **Artifact (YAML config):** numeric parameters, agent metadata/prompts, data source settings, pair/timeframe lists, consensus/risk thresholds, and schedules.
- Boundary rule: Artifact configures behavior; Engine enforces control logic and safety invariants.

## Phase 1 MVP Pipeline (Paper Trading)
1. Load config artifacts.
2. Collect market data (OHLCV + ticker).
3. Compute indicators.
4. Run two agents (technical + sentiment) via mock/default LLM layer.
5. Build hypotheses and weighted consensus.
6. Apply engine risk checks.
7. If approved, create simulated trade only.
8. Persist run artifacts (hypotheses, consensus, trades, snapshots, logs).
9. Send Telegram run summary.

## Phase 1 Scope
- Symbols: BTC/USDC, ETH/USDC, SOL/USDC.
- Timeframes: 1h, 4h, 1d.
- Consensus: weighted voting only.
- Execution: paper trading only, no exchange order placement.

## Invariants
- Any missing data or exception must result in `NO_TRADE`.
- Risk gate is fail-safe and enforced in Engine.
- All runs and decisions are logged and persisted.
- Idempotent runs: avoid duplicate side effects with `run_id` and upsert/guards.
