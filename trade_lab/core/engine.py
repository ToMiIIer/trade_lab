from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from core.broker_sim import BrokerSim
from core.metrics import add_drawdown, calculate_metrics
from core.risk import RiskManager
from core.types import (
    Bar,
    Fill,
    OrderIntent,
    Position,
    RunConfig,
    RunResult,
    TargetPosition,
)
from strategies import load_strategy
from strategies.base import BaseStrategy


class BacktestEngine:
    def __init__(
        self,
        config: RunConfig,
        broker: BrokerSim | None = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.config = config
        self.execution_mode = getattr(config, "execution_mode", "next_open")
        if self.execution_mode not in {"same_close", "next_open"}:
            raise ValueError(
                f"Invalid execution_mode='{self.execution_mode}'. Expected 'same_close' or 'next_open'."
            )
        self.broker = broker or BrokerSim(fee_rate=config.fee_rate, slippage_bps=config.slippage_bps)
        self.risk_manager = risk_manager or RiskManager(
            max_leverage=config.max_leverage,
            max_notional=config.max_notional,
            long_only=config.long_only,
        )
        self._bars_df: pd.DataFrame | None = None

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy | str,
        strategy_params: dict[str, Any] | None = None,
    ) -> RunResult:
        self._validate_data(data)
        bars_df = data.copy().reset_index(drop=True)
        self._bars_df = bars_df

        if isinstance(strategy, str):
            strategy_obj = load_strategy(strategy, strategy_params or self.config.strategy_params)
        elif isinstance(strategy, BaseStrategy):
            strategy_obj = strategy
        else:
            raise TypeError("strategy must be BaseStrategy instance or strategy module name")

        # initialize may receive full data for precompute; on_bar must rely only on step_state['data'] slice.
        state: dict[str, Any] = {"data": bars_df, "config": self.config}
        strategy_obj.initialize(state)

        cash = float(self.config.initial_cash)
        position = Position()
        open_trade: dict[str, Any] | None = None

        equity_rows: list[dict[str, Any]] = []
        fill_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []

        for i, row in bars_df.iterrows():
            bar = Bar.from_row(row)
            marked_equity = cash + (position.quantity * bar.close)
            # Defensive guard: do not trigger ATR stop on the same bar the position was entered.
            atr_stop_hit = False
            if position.entry_timestamp is None or pd.Timestamp(position.entry_timestamp) != bar.timestamp:
                atr_stop_hit = self.risk_manager.is_stop_hit(bar, position)

            step_state = {
                # Only expose history up to current bar; no future bars in on_bar.
                "data": bars_df.iloc[: i + 1].copy(),
                "position": replace(position),
                "equity": marked_equity,
                "cash": cash,
                "bar_index": i,
                "atr_stop_hit": atr_stop_hit,
            }

            if self.execution_mode == "next_open":
                # For next-open execution, the decision on bar i is applied to bar i+1 open.
                # Equity at bar i close must be recorded before executing the queued decision.
                equity_rows.append(self._equity_row(bar=bar, cash=cash, position=position))

            if self.risk_manager.should_kill(step_state):
                decision: TargetPosition | list[OrderIntent] | None = TargetPosition(
                    target_qty=0.0,
                    reason="kill_switch",
                )
            elif atr_stop_hit and position.quantity > 0:
                decision = TargetPosition(target_qty=0.0, reason="atr_stop_hit")
            else:
                decision = strategy_obj.on_bar(i, bar, step_state)

            if isinstance(decision, TargetPosition):
                new_fills, new_trades, open_trade, cash = self._execute_target(
                    bar=bar,
                    bar_index=i,
                    target=decision,
                    cash=cash,
                    position=position,
                    open_trade=open_trade,
                )
                fill_rows.extend(new_fills)
                trade_rows.extend(new_trades)
            elif isinstance(decision, list):
                new_fills, new_trades, open_trade, cash = self._execute_intents(
                    bar=bar,
                    bar_index=i,
                    intents=decision,
                    cash=cash,
                    position=position,
                    open_trade=open_trade,
                )
                fill_rows.extend(new_fills)
                trade_rows.extend(new_trades)
            elif decision is not None:
                raise TypeError(
                    "Strategy on_bar must return None, TargetPosition, or list[OrderIntent]"
                )

            if self.execution_mode == "same_close":
                equity_rows.append(self._equity_row(bar=bar, cash=cash, position=position))

        equity_curve = add_drawdown(pd.DataFrame(equity_rows))
        fills = pd.DataFrame(fill_rows)
        trades = pd.DataFrame(trade_rows)

        if trades.empty:
            trades = pd.DataFrame(
                columns=[
                    "entry_time",
                    "exit_time",
                    "side",
                    "qty",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "pnl_pct",
                    "bars_held",
                    "reason",
                ]
            )

        metrics = calculate_metrics(equity_curve=equity_curve, trades=trades, timeframe=self.config.timeframe)

        return RunResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            fills=fills,
        )

    def _execute_target(
        self,
        bar: Bar,
        bar_index: int,
        target: TargetPosition,
        cash: float,
        position: Position,
        open_trade: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, float]:
        execution_bar = self._resolve_execution_bar(bar_index)
        if execution_bar is None:
            # In next_open mode there is no executable next bar (last row), so skip order execution.
            return [], [], open_trade, cash
        execution_price = self._resolve_execution_price(execution_bar)
        execution_index = self._resolve_execution_index(bar_index)

        current_equity = cash + (position.quantity * execution_price)
        target_qty = self.risk_manager.cap_target_qty(
            target_qty=target.target_qty,
            price=execution_price,
            equity=current_equity,
        )
        delta = target_qty - position.quantity
        if abs(delta) <= 1e-12:
            return [], [], open_trade, cash

        side = "buy" if delta > 0 else "sell"
        order_qty = abs(delta)
        order_qty = self.risk_manager.cap_order_qty(
            requested_qty=order_qty,
            side=side,
            current_qty=position.quantity,
            price=execution_price,
            equity=current_equity,
        )
        if order_qty <= 1e-12:
            return [], [], open_trade, cash

        fill = self.broker.execute_market_order(
            timestamp=execution_bar.timestamp,
            side=side,
            quantity=order_qty,
            reference_price=execution_price,
            reason=target.reason,
        )

        before_qty = position.quantity
        cash = self._apply_fill(fill, position, cash)
        trade_updates, open_trade = self._track_trade(
            fill=fill,
            before_qty=before_qty,
            after_qty=position.quantity,
            open_trade=open_trade,
            bar_index=execution_index,
        )

        if side == "buy" and target.stop_distance is not None and position.quantity > 0:
            stop_price = self.risk_manager.compute_atr_stop_price(
                entry_price=fill.price,
                stop_distance=target.stop_distance,
                side="long",
            )
            if stop_price is not None:
                if position.stop_price is None:
                    position.stop_price = stop_price
                else:
                    # Keep the tighter stop for long positions.
                    position.stop_price = max(position.stop_price, stop_price)

        fill_row = self._fill_to_row(fill, position_qty=position.quantity)
        return [fill_row], trade_updates, open_trade, cash

    def _execute_intents(
        self,
        bar: Bar,
        bar_index: int,
        intents: list[OrderIntent],
        cash: float,
        position: Position,
        open_trade: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, float]:
        execution_bar = self._resolve_execution_bar(bar_index)
        if execution_bar is None:
            # In next_open mode there is no executable next bar (last row), so skip order execution.
            return [], [], open_trade, cash
        execution_price = self._resolve_execution_price(execution_bar)
        execution_index = self._resolve_execution_index(bar_index)

        fills: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []

        for intent in intents:
            current_equity = cash + (position.quantity * execution_price)
            order_qty = self.risk_manager.cap_order_qty(
                requested_qty=intent.quantity,
                side=intent.side,
                current_qty=position.quantity,
                price=execution_price,
                equity=current_equity,
            )
            if order_qty <= 1e-12:
                continue

            fill = self.broker.execute_market_order(
                timestamp=execution_bar.timestamp,
                side=intent.side,
                quantity=order_qty,
                reference_price=execution_price,
                reason=intent.reason,
            )

            before_qty = position.quantity
            cash = self._apply_fill(fill, position, cash)
            trade_updates, open_trade = self._track_trade(
                fill=fill,
                before_qty=before_qty,
                after_qty=position.quantity,
                open_trade=open_trade,
                bar_index=execution_index,
            )

            fills.append(self._fill_to_row(fill, position_qty=position.quantity))
            trades.extend(trade_updates)

        return fills, trades, open_trade, cash

    def _apply_fill(self, fill: Fill, position: Position, cash: float) -> float:
        if fill.side == "buy":
            cash -= fill.notional + fill.fee
            if position.quantity >= 0:
                new_qty = position.quantity + fill.quantity
                if new_qty > 0:
                    if position.quantity > 0:
                        weighted_cost = (
                            position.avg_entry_price * position.quantity
                            + fill.price * fill.quantity
                        )
                        position.avg_entry_price = weighted_cost / new_qty
                    else:
                        position.avg_entry_price = fill.price
                position.quantity = new_qty
                if position.entry_timestamp is None and position.quantity > 0:
                    position.entry_timestamp = fill.timestamp
            else:
                # Placeholder for short-cover behavior.
                position.quantity += fill.quantity

        else:
            cash += fill.notional - fill.fee
            if position.quantity > 0:
                new_qty = position.quantity - fill.quantity
                if new_qty <= 1e-12:
                    position.quantity = 0.0
                    position.avg_entry_price = 0.0
                    position.entry_timestamp = None
                    position.stop_price = None
                else:
                    position.quantity = new_qty
            else:
                # Placeholder for opening/adding shorts in the future.
                position.quantity -= fill.quantity
                if position.quantity < 0 and position.avg_entry_price == 0:
                    position.avg_entry_price = fill.price

        return cash

    def _track_trade(
        self,
        fill: Fill,
        before_qty: float,
        after_qty: float,
        open_trade: dict[str, Any] | None,
        bar_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        closed: list[dict[str, Any]] = []

        if fill.side == "buy":
            if before_qty <= 1e-12 and after_qty > 1e-12:
                open_trade = {
                    "entry_time": fill.timestamp,
                    "entry_price": fill.price,
                    "entry_fee": fill.fee,
                    "qty": fill.quantity,
                    "entry_index": bar_index,
                }
            elif open_trade is not None and after_qty > before_qty:
                total_qty = open_trade["qty"] + fill.quantity
                open_trade["entry_price"] = (
                    open_trade["entry_price"] * open_trade["qty"] + fill.price * fill.quantity
                ) / total_qty
                open_trade["entry_fee"] += fill.fee
                open_trade["qty"] = total_qty
            return closed, open_trade

        if fill.side == "sell" and open_trade is not None and before_qty > 1e-12:
            closing_qty = min(fill.quantity, open_trade["qty"])
            if closing_qty <= 1e-12:
                return closed, open_trade

            fee_share = open_trade["entry_fee"] * (closing_qty / open_trade["qty"])
            pnl = closing_qty * (fill.price - open_trade["entry_price"]) - fee_share - fill.fee
            base_cost = open_trade["entry_price"] * closing_qty
            pnl_pct = pnl / base_cost if base_cost else 0.0

            closed.append(
                {
                    "entry_time": open_trade["entry_time"],
                    "exit_time": fill.timestamp,
                    "side": "long",
                    "qty": closing_qty,
                    "entry_price": open_trade["entry_price"],
                    "exit_price": fill.price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "bars_held": int(max(0, bar_index - open_trade["entry_index"])),
                    "reason": fill.reason,
                }
            )

            remaining_qty = open_trade["qty"] - closing_qty
            if remaining_qty <= 1e-12 or after_qty <= 1e-12:
                open_trade = None
            else:
                open_trade["qty"] = remaining_qty
                open_trade["entry_fee"] -= fee_share

        return closed, open_trade

    @staticmethod
    def _fill_to_row(fill: Fill, position_qty: float) -> dict[str, Any]:
        return {
            "timestamp": fill.timestamp,
            "side": fill.side,
            "qty": fill.quantity,
            "price": fill.price,
            "notional": fill.notional,
            "fee": fill.fee,
            "slippage": fill.slippage,
            "reason": fill.reason,
            "position_qty_after": position_qty,
        }

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required.difference(set(data.columns))
        if missing:
            raise ValueError(f"Input data missing required columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("Input data is empty")

    @staticmethod
    def _equity_row(bar: Bar, cash: float, position: Position) -> dict[str, Any]:
        market_value = position.quantity * bar.close
        unrealized = position.quantity * (bar.close - position.avg_entry_price)
        equity = cash + market_value
        return {
            "timestamp": bar.timestamp,
            "equity": equity,
            "cash": cash,
            "market_value": market_value,
            "unrealized": unrealized,
            "position_qty": position.quantity,
            "close": bar.close,
        }

    def _resolve_execution_bar(self, bar_index: int) -> Bar | None:
        if self._bars_df is None:
            raise RuntimeError("Backtest bars are not initialized")
        execute_index = self._resolve_execution_index(bar_index)
        if execute_index >= len(self._bars_df):
            return None
        return Bar.from_row(self._bars_df.iloc[execute_index])

    def _resolve_execution_index(self, bar_index: int) -> int:
        if self.execution_mode == "next_open":
            return bar_index + 1
        return bar_index

    def _resolve_execution_price(self, execution_bar: Bar) -> float:
        if self.execution_mode == "next_open":
            return execution_bar.open
        return execution_bar.close
