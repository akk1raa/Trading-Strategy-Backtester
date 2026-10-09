"""Strategy spec: the strict schema the LLM must fill in.

The LLM never writes backtest code. It emits this JSON, we validate it, and
the engine maps it onto tested indicator functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INDICATORS = {"sma", "ema", "rsi", "atr", "bbands", "macd", "donchian", "vwap", "close", "high", "low", "open"}
OPERATORS = {">", "<", ">=", "<=", "crosses_above", "crosses_below"}


class SpecError(ValueError):
    pass


@dataclass
class Operand:
    """Either a number, or an indicator reference like {'ind': 'ema', 'period': 20}."""
    value: float | None = None
    ind: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    field_: str | None = None  # sub-output e.g. bbands 'upper', macd 'signal'

    @staticmethod
    def parse(raw: Any) -> "Operand":
        if isinstance(raw, (int, float)):
            return Operand(value=float(raw))
        if isinstance(raw, dict) and "ind" in raw:
            ind = raw["ind"]
            if ind not in INDICATORS:
                raise SpecError(f"Unknown indicator '{ind}'. Allowed: {sorted(INDICATORS)}")
            params = {k: v for k, v in raw.items() if k not in ("ind", "field")}
            return Operand(ind=ind, params=params, field_=raw.get("field"))
        raise SpecError(f"Bad operand: {raw!r}")


@dataclass
class Condition:
    left: Operand
    op: str
    right: Operand

    @staticmethod
    def parse(raw: dict) -> "Condition":
        op = raw.get("op")
        if op not in OPERATORS:
            raise SpecError(f"Unknown operator '{op}'. Allowed: {sorted(OPERATORS)}")
        return Condition(Operand.parse(raw["left"]), op, Operand.parse(raw["right"]))


@dataclass
class Rules:
    all_of: list[Condition]  # AND
    any_of: list[Condition]  # OR (optional, ANDed with all_of if both present)

    @staticmethod
    def parse(raw: dict | None) -> "Rules":
        raw = raw or {}
        return Rules(
            [Condition.parse(c) for c in raw.get("all", [])],
            [Condition.parse(c) for c in raw.get("any", [])],
        )

    def empty(self) -> bool:
        return not self.all_of and not self.any_of


@dataclass
class Stop:
    kind: str = "none"  # none | fixed_points | atr_multiple | percent
    value: float = 0.0
    atr_period: int = 14


@dataclass
class StrategySpec:
    name: str
    timeframe: str  # e.g. "1h", "1d", "15m"
    direction: str  # long | short | both
    long_entry: Rules
    long_exit: Rules
    short_entry: Rules
    short_exit: Rules
    stop: Stop
    target: Stop
    session_utc: tuple[int, int] | None  # (start_hour, end_hour) trade window, UTC
    contracts: int
    uncodifiable: list[str]  # things the LLM could not turn into rules
    confidence: str  # high | medium | low

    @staticmethod
    def from_dict(d: dict) -> "StrategySpec":
        if d.get("direction") not in ("long", "short", "both"):
            raise SpecError("direction must be long|short|both")

        def stop(raw):
            raw = raw or {}
            kind = raw.get("kind", "none")
            if kind not in ("none", "fixed_points", "atr_multiple", "percent"):
                raise SpecError(f"Bad stop/target kind '{kind}'")
            return Stop(kind, float(raw.get("value", 0)), int(raw.get("atr_period", 14)))

        sess = d.get("session_utc")
        spec = StrategySpec(
            name=d.get("name", "unnamed"),
            timeframe=d.get("timeframe", "1h"),
            direction=d["direction"],
            long_entry=Rules.parse(d.get("long_entry")),
            long_exit=Rules.parse(d.get("long_exit")),
            short_entry=Rules.parse(d.get("short_entry")),
            short_exit=Rules.parse(d.get("short_exit")),
            stop=stop(d.get("stop")),
            target=stop(d.get("target")),
            session_utc=tuple(sess) if sess else None,
            contracts=int(d.get("contracts", 1)),
            uncodifiable=list(d.get("uncodifiable", [])),
            confidence=d.get("confidence", "low"),
        )
        if spec.direction in ("long", "both") and spec.long_entry.empty():
            raise SpecError("Long side enabled but no long_entry rules were extracted.")
        if spec.direction in ("short", "both") and spec.short_entry.empty():
            raise SpecError("Short side enabled but no short_entry rules were extracted.")
        return spec
