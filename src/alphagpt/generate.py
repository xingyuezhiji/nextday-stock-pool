# -*- coding: utf-8 -*-
"""Generate AlphaGPT static JSON for GitHub Pages and QMT readers."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_config
from ..data_api import ZZClient
from ..backtest import fetch_panel
from .formulas import STRATEGIES, build_matrix, compute_strategy, strategy_base


def _fmt(d: str) -> str:
    d = str(d).replace("-", "")
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _now() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _latest_complete_day(client: ZZClient, cal: list[str]) -> tuple[str, pd.DataFrame]:
    for d in reversed(cal[-8:]):
        df = client.market_day(d)
        if len(df) >= 3000:
            return d, df
    raise RuntimeError("近8个交易日均无完整行情数据, AlphaGPT无法生成")


def _row_payload(code: str, day_df: pd.DataFrame, names: dict[str, str],
                 score: float, rank: int, selected: bool) -> dict:
    row = day_df.loc[code]
    return {
        "rank": rank,
        "symbol": code,
        "code": code,
        "name": names.get(code, code),
        "close": round(float(row["close"]), 2) if pd.notna(row["close"]) else None,
        "change_pct": round(float(row["pct"]), 2) if pd.notna(row["pct"]) else None,
        "amount": round(float(row["amount"]) / 1e8, 2) if pd.notna(row["amount"]) else None,
        "amount_yi": round(float(row["amount"]) / 1e8, 2) if pd.notna(row["amount"]) else None,
        "turnover_rate": round(float(row["turnover_rate"]), 2) if pd.notna(row["turnover_rate"]) else None,
        "score": round(float(score), 4),
        "selected": bool(selected),
    }


def generate(days: int = 100) -> dict:
    cfg = load_config()
    client = ZZClient(cfg)
    cal = client.trade_days(days=days + 10)
    signal_date, day_df = _latest_complete_day(client, cal)
    pos = cal.index(signal_date)
    panel_dates = cal[max(0, pos - days + 1): pos + 1]
    panel = fetch_panel(client, panel_dates, log_every=50, float32=False)
    matrix = build_matrix(panel)

    names_df = client.stock_names()
    names = dict(zip(names_df["ts_code"], names_df["name"]))
    latest_day = panel[panel["trade_date"].astype(str) == signal_date].copy()
    latest_rows = latest_day.set_index("ts_code", drop=False)

    updated_at = _now()
    out_dir = ROOT / "output" / "alphagpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies_payload = []
    qmt_plans = []

    for spec in STRATEGIES:
        result = compute_strategy(matrix, spec)
        score = result["score"]
        eligible_ids = [i for i in range(len(matrix["codes"])) if bool(result["eligible"][i])]
        ranked_ids = sorted(eligible_ids, key=lambda i: (-float(score[i]), matrix["codes"][i]))
        buffer_ids = [i for i in ranked_ids if bool(result["buffer_mask"][i])]
        buy_ids = [i for i in ranked_ids if bool(result["buy_mask"][i])]

        rows = []
        for rank, idx in enumerate(buy_ids, 1):
            code = matrix["codes"][idx]
            if code not in latest_rows.index:
                continue
            rows.append(_row_payload(code, latest_rows, names, score[idx], rank, True))

        buffer_rows = []
        for rank, idx in enumerate(buffer_ids[: int(spec.params["n_hold"])], 1):
            code = matrix["codes"][idx]
            if code not in latest_rows.index:
                continue
            buffer_rows.append(_row_payload(code, latest_rows, names, score[idx], rank, code in {r["code"] for r in rows}))

        payload = strategy_base(spec, _fmt(signal_date), updated_at)
        payload["latest"] = {
            "total": len(rows),
            "rows": rows,
            "buffer_total": len(buffer_rows),
            "buffer_rows": buffer_rows,
            "signal_enabled": bool(result["signal_enabled"]),
            "error": None,
        }
        sdir = out_dir / spec.strategy_id
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        strategies_payload.append({k: v for k, v in payload.items() if k != "latest"})
        qmt_plans.append({
            "strategy_id": spec.strategy_id,
            "label": spec.label,
            "as_of": _fmt(signal_date),
            "updated_at": updated_at,
            "signal_enabled": bool(result["signal_enabled"]),
            "buy_candidates": rows,
            "buffer_candidates": buffer_rows,
            "sell_rule": "sell positions outside n_hold buffer at next open",
            "params": spec.params,
            "latest_url": f"data/alphagpt/{spec.strategy_id}/latest.json",
        })

    overview = {
        "updated_at": updated_at,
        "as_of": _fmt(signal_date),
        "strategies": strategies_payload,
    }
    qmt = {
        "as_of": _fmt(signal_date),
        "updated_at": updated_at,
        "plans": qmt_plans,
    }
    (out_dir / "latest.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "strategies.json").write_text(json.dumps(strategies_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "qmt_order_plans.json").write_text(json.dumps(qmt, ensure_ascii=False, indent=2), encoding="utf-8")
    return qmt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=100)
    args = parser.parse_args()
    result = generate(days=args.days)
    print(f"AlphaGPT generated: as_of={result['as_of']} strategies={len(result['plans'])}")


if __name__ == "__main__":
    main()
