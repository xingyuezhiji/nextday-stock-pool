# -*- coding: utf-8 -*-
"""Standalone AlphaGPT formula evaluator used by the Pages generator.

This file intentionally implements only the operators and features required by
the four published AlphaGPT strategies. It avoids importing the TSP backend so
GitHub Actions can refresh static JSON with the lightweight nextday repo.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import warnings

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "n_buy": 30,
    "n_hold": 45,
    "min_listed_days": 60,
    "amount_window": 20,
    "min_avg_amount": 30_000_000.0,
}

TRADING_RULE = {
    "buy": "weekly rotation, buy top n_buy by score at next open",
    "sell": "sell positions outside n_hold buffer at next open",
    "sizing": "equal weight, QMT should apply local cash/position limits",
}


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    label: str
    name: str
    description: str
    formula_tokens: tuple[str, ...]
    params: dict
    backtest: dict


STRATEGIES = [
    StrategySpec(
        strategy_id="ai_alphagpt_round937_c02_amp_gate_jump",
        label="AlphaGPT round937 c02 amp gate jump",
        name="AlphaGPT round937 candidate #2: amplitude gate jump.",
        description="Round937 AlphaGPT formula gating MA60 repair and delayed low-crowding by low-amplitude jump confirmation.",
        formula_tokens=(
            "CV_CORR30_LOW", "DECAY", "MA60_BIAS_LOW", "CA_CORR20_LOW", "DELAY1",
            "ABS", "ADD", "AMP20_LOW", "JUMP", "GATE", "DECAY", "JUMP",
        ),
        params=dict(DEFAULT_PARAMS),
        backtest={
            "strategy_id": "ai_alphagpt_round937_c02_amp_gate_jump",
            "grid": "stop10_trail12_exp50_pos5",
            "total_return": 1.1830,
            "annual_return": 0.4990,
            "max_drawdown": -0.0916,
            "sharpe": 2.77,
            "sortino": 4.25,
            "calmar": 5.45,
            "win_rate": None,
            "profit_factor": None,
            "n_trades": 311,
            "final_equity": 2183010.32,
            "max_exposure": None,
            "formula": None,
            "rank": None,
        },
    ),
    StrategySpec(
        strategy_id="ai_alphagpt_round935_c12_jump_delay",
        label="AlphaGPT round935 c12 jump delay",
        name="AlphaGPT round935 candidate #12: jump with delayed gate.",
        description="Round935 AlphaGPT formula preserving only delayed low-crowding jump confirmations.",
        formula_tokens=(
            "CV_CORR30_LOW", "MAX3", "MA60_BIAS_LOW", "CA_CORR20_LOW",
            "MA60_BIAS_LOW", "JUMP", "NEG", "GATE", "CV_CORR30_LOW",
            "DELAY1", "GATE", "JUMP",
        ),
        params=dict(DEFAULT_PARAMS),
        backtest={
            "strategy_id": "ai_alphagpt_round935_c12_jump_delay",
            "grid": "c12_stop10_trail12_exp50_pos5",
            "total_return": 1.7862,
            "annual_return": 0.6938,
            "max_drawdown": -0.1382,
            "sharpe": 3.18,
            "sortino": 5.51,
            "calmar": 5.02,
            "win_rate": 0.5995,
            "profit_factor": 1.57,
            "n_trades": 397,
            "final_equity": 2786159.58,
            "max_exposure": 0.5562,
            "formula": None,
            "rank": None,
        },
    ),
    StrategySpec(
        strategy_id="ai_alphagpt_round943_c02_delay_bias_ma",
        label="AlphaGPT round943 c02 delay bias ma",
        name="AlphaGPT round943 candidate #2: delayed bias with MA jump gate.",
        description="Round943 c12 mutation combining delayed crowding with MA60 bias jump rejection.",
        formula_tokens=(
            "CV_CORR30_LOW", "DELAY1", "MA60_BIAS_LOW", "CV_CORR30_LOW",
            "MA60_BIAS_LOW", "JUMP", "NEG", "GATE", "CV_CORR30_LOW",
            "DELAY1", "GATE", "JUMP",
        ),
        params={**DEFAULT_PARAMS, "n_buy": 10, "n_hold": 15},
        backtest={
            "strategy_id": "tmp_round943_2y06_c08",
            "grid": None,
            "total_return": 2.4591,
            "annual_return": 0.9107,
            "max_drawdown": -0.1309,
            "sharpe": 3.75,
            "sortino": 6.23,
            "calmar": 6.96,
            "win_rate": 0.6143,
            "profit_factor": 1.76,
            "n_trades": 643,
            "final_equity": 3459051.80,
            "max_exposure": 0.6392,
            "formula": "CV_CORR30_LOW DELAY1 MA60_BIAS_LOW CV_CORR30_LOW MA60_BIAS_LOW JUMP NEG GATE CV_CORR30_LOW DELAY1 GATE JUMP",
            "rank": 8,
        },
    ),
    StrategySpec(
        strategy_id="ai_alphagpt_round944_c02_delay_cv_max3",
        label="AlphaGPT round944 c02 delay cv max3",
        name="AlphaGPT round944 candidate #2: delayed crowding with max3 fallback.",
        description="Round944 two-year mutation using delayed low crowding and max3 crowding fallback for lower drawdown.",
        formula_tokens=(
            "CV_CORR30_LOW", "DELAY1", "MA60_BIAS_LOW", "CV_CORR30_LOW",
            "CV_CORR30_LOW", "JUMP", "NEG", "GATE", "CV_CORR30_LOW",
            "MAX3", "GATE", "JUMP",
        ),
        params={**DEFAULT_PARAMS, "n_buy": 10, "n_hold": 15},
        backtest={
            "strategy_id": "tmp_round944_2y06_c30",
            "grid": None,
            "total_return": 2.2250,
            "annual_return": 0.8421,
            "max_drawdown": -0.0970,
            "sharpe": 3.73,
            "sortino": 6.19,
            "calmar": 8.69,
            "win_rate": 0.6062,
            "profit_factor": 1.83,
            "n_trades": 579,
            "final_equity": 3225014.86,
            "max_exposure": 0.6367,
            "formula": "CV_CORR30_LOW DELAY1 MA60_BIAS_LOW CV_CORR30_LOW CV_CORR30_LOW JUMP NEG GATE CV_CORR30_LOW MAX3 GATE JUMP",
            "rank": 30,
        },
    ),
]


def _delay(values: np.ndarray, bars: int = 1) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if bars < values.shape[0]:
        out[bars:] = values[:-bars]
    return out


def _rolling_mean(values: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    min_periods = window if min_periods is None else min_periods
    df = pd.DataFrame(values)
    return df.rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float32)


def _rolling_std(values: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    min_periods = window if min_periods is None else min_periods
    df = pd.DataFrame(values)
    return df.rolling(window, min_periods=min_periods).std(ddof=0).to_numpy(dtype=np.float32)


def _rolling_corr(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    x = pd.DataFrame(left, dtype="float64")
    y = pd.DataFrame(right, dtype="float64")
    mean_x = x.rolling(window, min_periods=window).mean()
    mean_y = y.rolling(window, min_periods=window).mean()
    mean_xy = (x * y).rolling(window, min_periods=window).mean()
    std_x = x.rolling(window, min_periods=window).std(ddof=0)
    std_y = y.rolling(window, min_periods=window).std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (mean_xy - mean_x * mean_y) / (std_x * std_y + 1e-9)
    return corr.to_numpy(dtype=np.float32)


def _ema(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    alpha = np.float32(2.0 / (window + 1.0))
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for col in range(values.shape[1]):
        prev = np.nan
        for row in range(values.shape[0]):
            if not valid[row, col]:
                continue
            v = np.float32(values[row, col])
            prev = v if not np.isfinite(prev) else alpha * v + (1.0 - alpha) * prev
            out[row, col] = prev
    return out


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / (denominator + 1e-9)


def _cs_robust_norm(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            median = np.nanmedian(values, axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(values - median), axis=1, keepdims=True) + 1e-6
            norm = (values - median) / mad
    return np.nan_to_num(np.clip(norm, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def _ts_jump(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(values, axis=0, keepdims=True)
        std = np.nanstd(values, axis=0, keepdims=True) + 1e-6
        z = (values - mean) / std
    return np.nan_to_num(np.maximum(z - 3.0, 0.0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _max3(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, np.maximum(_delay(values, 1), _delay(values, 2))).astype(np.float32)


def _decay(values: np.ndarray) -> np.ndarray:
    return (values + np.float32(0.8) * _delay(values, 1) + np.float32(0.6) * _delay(values, 2)).astype(np.float32)


def _weekly_signal_mask(labels: list[str]) -> np.ndarray:
    out = np.zeros(len(labels), dtype=bool)
    if not labels:
        return out
    days = [date.fromisoformat(label[:4] + "-" + label[4:6] + "-" + label[6:8]) for label in labels]
    weeks = [day.isocalendar()[:2] for day in days]
    for idx in range(len(labels) - 1):
        if weeks[idx + 1] != weeks[idx]:
            out[idx] = True
    out[-1] = out[-1] or days[-1].weekday() >= 3
    return out


def _score_rank(eligible: np.ndarray, value: np.ndarray) -> np.ndarray:
    score = np.zeros(value.shape, dtype=np.float32)
    for row in range(value.shape[0]):
        asset_ids = np.flatnonzero(eligible[row] & np.isfinite(value[row]))
        if asset_ids.size == 0:
            continue
        order = asset_ids[np.argsort(-value[row, asset_ids], kind="stable")]
        if order.size == 1:
            score[row, order[0]] = np.float32(100.0)
        else:
            ranks = np.arange(order.size, dtype=np.float32)
            score[row, order] = (np.float32(1.0) - ranks / np.float32(order.size - 1)) * np.float32(100.0)
    return score


def _top_mask(eligible: np.ndarray, value: np.ndarray, top_n: int) -> np.ndarray:
    out = np.zeros(eligible.shape, dtype=bool)
    for row in range(value.shape[0]):
        asset_ids = np.flatnonzero(eligible[row] & np.isfinite(value[row]))
        if asset_ids.size:
            selected = asset_ids[np.argsort(-value[row, asset_ids], kind="stable")[:top_n]]
            out[row, selected] = True
    return out


def build_matrix(panel: pd.DataFrame) -> dict:
    dates = sorted(panel["trade_date"].astype(str).unique())
    codes = sorted(panel["ts_code"].astype(str).unique())
    shape = (len(dates), len(codes))
    date_i = {d: i for i, d in enumerate(dates)}
    code_i = {c: i for i, c in enumerate(codes)}
    fields: dict[str, np.ndarray] = {}
    for field in ("open", "high", "low", "close", "vol", "amount", "turnover_rate", "is_paused", "is_st"):
        fields[field] = np.full(shape, np.nan, dtype=np.float32)
    for row in panel.itertuples(index=False):
        r = date_i[str(row.trade_date)]
        c = code_i[str(row.ts_code)]
        fields["open"][r, c] = row.open
        fields["high"][r, c] = row.high
        fields["low"][r, c] = row.low
        fields["close"][r, c] = row.close
        fields["vol"][r, c] = row.vol
        fields["amount"][r, c] = row.amount
        fields["turnover_rate"][r, c] = row.turnover_rate
        fields["is_paused"][r, c] = row.is_paused
        fields["is_st"][r, c] = row.is_st
    return {"dates": dates, "codes": codes, "fields": fields}


def build_features(matrix: dict, params: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    f = matrix["fields"]
    close = f["close"]
    open_ = f["open"]
    high = f["high"]
    low = f["low"]
    volume = f["vol"]
    amount = f["amount"]
    valid = (
        np.isfinite(close)
        & np.isfinite(open_)
        & (close > 0)
        & (open_ > 0)
        & np.isfinite(volume)
        & (volume > 0)
        & (np.nan_to_num(f["is_paused"], nan=1.0) == 0)
        & (np.nan_to_num(f["is_st"], nan=1.0) == 0)
    )
    amount_valid = np.isfinite(amount) & (amount > 0)
    amount_window = int(params.get("amount_window", 20))
    min_listed = int(params.get("min_listed_days", 60))
    min_avg_amount = float(params.get("min_avg_amount", 30_000_000.0))
    avg_amount = _rolling_mean(np.where(amount_valid, amount, np.nan), amount_window, amount_window)
    listed_days = np.cumsum(valid, axis=0)

    corr_ca20 = _rolling_corr(close, amount, 20)
    corr_cv30 = _rolling_corr(close, volume, 30)
    ema60 = _ema(close, valid, 60)
    ma60_bias = _safe_ratio(close, ema60) - 1.0
    amp1 = _safe_ratio(high - low, close)
    amp20 = _rolling_mean(amp1, 20, 20)
    features = {
        "CA_CORR20_LOW": -_cs_robust_norm(corr_ca20),
        "CV_CORR30_LOW": -_cs_robust_norm(corr_cv30),
        "MA60_BIAS_LOW": -_cs_robust_norm(ma60_bias),
        "AMP20_LOW": -_cs_robust_norm(amp20),
    }
    eligible = (
        valid
        & amount_valid
        & np.isfinite(avg_amount)
        & np.isfinite(corr_ca20)
        & np.isfinite(corr_cv30)
        & np.isfinite(ma60_bias)
        & np.isfinite(amp20)
        & (listed_days >= min_listed)
        & (avg_amount >= min_avg_amount)
    )
    return features, eligible


def evaluate_rpn(tokens: tuple[str, ...], features: dict[str, np.ndarray]) -> np.ndarray:
    stack: list[np.ndarray] = []
    for token in tokens:
        if token in features:
            stack.append(features[token])
        elif token == "ADD":
            y, x = stack.pop(), stack.pop()
            stack.append((x + y).astype(np.float32))
        elif token == "NEG":
            stack.append((-stack.pop()).astype(np.float32))
        elif token == "ABS":
            stack.append(np.abs(stack.pop()).astype(np.float32))
        elif token == "GATE":
            y, x, condition = stack.pop(), stack.pop(), stack.pop()
            stack.append(np.where(condition > 0, x, y).astype(np.float32))
        elif token == "JUMP":
            stack.append(_ts_jump(stack.pop()))
        elif token == "DECAY":
            stack.append(_decay(stack.pop()))
        elif token == "DELAY1":
            stack.append(_delay(stack.pop(), 1))
        elif token == "MAX3":
            stack.append(_max3(stack.pop()))
        else:
            raise ValueError(f"unsupported AlphaGPT token: {token}")
    if len(stack) != 1:
        raise ValueError(f"invalid formula stack size: {len(stack)}")
    return np.nan_to_num(stack[0], nan=-1e9, posinf=1e9, neginf=-1e9).astype(np.float32)


def compute_strategy(matrix: dict, spec: StrategySpec) -> dict:
    features, eligible = build_features(matrix, spec.params)
    value = evaluate_rpn(spec.formula_tokens, features)
    score = _score_rank(eligible, value)
    buy_pool = _top_mask(eligible, value, int(spec.params["n_buy"]))
    buffer_pool = _top_mask(eligible, value, int(spec.params["n_hold"]))
    signal_days = _weekly_signal_mask(matrix["dates"])
    row = len(matrix["dates"]) - 1
    return {
        "score": score[row],
        "value": value[row],
        "eligible": eligible[row],
        "buy_mask": buy_pool[row] & signal_days[row],
        "buffer_mask": buffer_pool[row],
        "signal_enabled": bool(signal_days[row]),
    }


def strategy_base(spec: StrategySpec, as_of: str, updated_at: str | None = None) -> dict:
    return {
        "strategy_id": spec.strategy_id,
        "label": spec.label,
        "name": spec.name,
        "description": spec.description,
        "as_of": as_of,
        "updated_at": updated_at,
        "formula_tokens": list(spec.formula_tokens),
        "params": spec.params,
        "trading_rule": TRADING_RULE,
        "backtest": spec.backtest,
    }
