"""Persistence repository with idempotent upsert guards."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.agents.base_agent import AgentResult
from src.graph.nodes.consensus import ConsensusDecision
from src.storage.models import AgentPerformance, Base, ConsensusLog, Hypothesis, PortfolioSnapshot, Trade


class StorageRepository:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, future=True, pool_pre_ping=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_hypotheses(self, hypotheses: list[AgentResult]) -> None:
        if not hypotheses:
            return

        with self.session() as session:
            for hypothesis in hypotheses:
                existing = session.scalar(
                    select(Hypothesis).where(
                        Hypothesis.run_id == hypothesis.run_id,
                        Hypothesis.agent_id == hypothesis.agent_id,
                    )
                )
                if existing:
                    existing.action = hypothesis.action
                    existing.confidence = hypothesis.confidence
                    existing.reasoning = hypothesis.reasoning
                    existing.risk_notes = hypothesis.risk_notes
                    continue

                session.add(
                    Hypothesis(
                        run_id=hypothesis.run_id,
                        pair=hypothesis.pair,
                        timeframe=hypothesis.timeframe,
                        agent_id=hypothesis.agent_id,
                        action=hypothesis.action,
                        confidence=hypothesis.confidence,
                        reasoning=hypothesis.reasoning,
                        risk_notes=hypothesis.risk_notes,
                    )
                )

    def upsert_consensus(self, decision: ConsensusDecision) -> None:
        with self.session() as session:
            existing = session.scalar(select(ConsensusLog).where(ConsensusLog.run_id == decision.run_id))
            if existing:
                existing.action = decision.action
                existing.weighted_confidence = decision.weighted_confidence
                existing.threshold_passed = decision.threshold_passed
                existing.scores_json = json.dumps(decision.scores, separators=(",", ":"))
                existing.weights_json = json.dumps(decision.weights_used, separators=(",", ":"))
                existing.reasoning = decision.reasoning
                return

            session.add(
                ConsensusLog(
                    run_id=decision.run_id,
                    pair=decision.pair,
                    timeframe=decision.timeframe,
                    action=decision.action,
                    weighted_confidence=decision.weighted_confidence,
                    threshold_passed=decision.threshold_passed,
                    scores_json=json.dumps(decision.scores, separators=(",", ":")),
                    weights_json=json.dumps(decision.weights_used, separators=(",", ":")),
                    reasoning=decision.reasoning,
                )
            )

    def create_simulated_trade(
        self,
        *,
        run_id: str,
        pair: str,
        timeframe: str,
        action: str,
        quantity: float,
        entry_price: float,
        reason: str,
    ) -> Trade:
        with self.session() as session:
            existing = session.scalar(
                select(Trade).where(
                    Trade.run_id == run_id,
                    Trade.pair == pair,
                    Trade.timeframe == timeframe,
                )
            )
            if existing:
                return existing

            trade = Trade(
                run_id=run_id,
                pair=pair,
                timeframe=timeframe,
                action=action,
                quantity=quantity,
                entry_price=entry_price,
                reason=reason,
                status="SIMULATED",
            )
            session.add(trade)
            session.flush()
            return trade

    def upsert_portfolio_snapshot(
        self,
        *,
        run_id: str,
        cash_balance: float,
        equity: float,
        total_exposure: float,
        open_positions: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = json.dumps(metadata or {}, separators=(",", ":"))
        with self.session() as session:
            existing = session.scalar(
                select(PortfolioSnapshot).where(PortfolioSnapshot.run_id == run_id)
            )
            if existing:
                existing.cash_balance = cash_balance
                existing.equity = equity
                existing.total_exposure = total_exposure
                existing.open_positions = open_positions
                existing.metadata_json = payload
                return

            session.add(
                PortfolioSnapshot(
                    run_id=run_id,
                    cash_balance=cash_balance,
                    equity=equity,
                    total_exposure=total_exposure,
                    open_positions=open_positions,
                    metadata_json=payload,
                )
            )

    def upsert_agent_performance(
        self,
        *,
        run_id: str,
        agent_id: str,
        action: str,
        confidence: float,
        pnl: float = 0.0,
        outcome: str = "PENDING",
    ) -> None:
        with self.session() as session:
            existing = session.scalar(
                select(AgentPerformance).where(
                    AgentPerformance.run_id == run_id,
                    AgentPerformance.agent_id == agent_id,
                )
            )
            if existing:
                existing.action = action
                existing.confidence = confidence
                existing.pnl = pnl
                existing.outcome = outcome
                return

            session.add(
                AgentPerformance(
                    run_id=run_id,
                    agent_id=agent_id,
                    action=action,
                    confidence=confidence,
                    pnl=pnl,
                    outcome=outcome,
                )
            )
