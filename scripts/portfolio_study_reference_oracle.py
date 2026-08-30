"""Independent membership/signal/boundary oracle; no production/numpy/pandas imports.

Uses the separately reviewed exact-rational sign-region ledger oracle. This
prints reviewable software fixtures, never investment performance evidence.
"""

import json
import math
import statistics
from datetime import date
from fractions import Fraction as F

from portfolio_reference_oracle import LEDGER_COLUMNS, solve


def signals(case):
    symbols = sorted(case["prices"])
    rule, manifest = case["config"], case["membership"]
    active = set(manifest["initial_members"])
    events = sorted(manifest["events"], key=lambda event: (event["effective_on"], event["ticker"]))
    masks, scores, targets, baseline, flags = [], [], [], [], []
    event_at, previous_period = 0, None
    for row, session in enumerate(case["dates"]):
        while event_at < len(events) and events[event_at]["effective_on"] <= session:
            event = events[event_at]
            if event["action"] == "add":
                active.add(event["ticker"])
            else:
                active.remove(event["ticker"])
            event_at += 1
        masks.append([symbol in active for symbol in symbols])
        warmed = row >= rule["lookback"]
        values = {
            symbol: F(str(case["prices"][symbol][row - rule["skip_recent"]]))
            / F(str(case["prices"][symbol][row - rule["lookback"]]))
            - 1
            for symbol in symbols
            if warmed and symbol in active
        }
        scores.append([float(values[symbol]) if symbol in values else None for symbol in symbols])
        day = date.fromisoformat(session)
        period = (day.year, day.month) if rule["rebalance"] == "monthly" else day.isocalendar()[:2]
        changed = row > 0 and masks[-1] != masks[-2]
        scheduled = warmed and (
            row == rule["lookback"] or rule["rebalance"] == "daily" or period != previous_period or changed
        )
        flags.append(scheduled)
        previous_period = period
        if not scheduled:
            targets.append(None)
            baseline.append(None)
            continue
        selected = sorted(
            (
                symbol
                for symbol in values
                if rule["absolute_threshold"] is None or values[symbol] > F(str(rule["absolute_threshold"]))
            ),
            key=lambda symbol: (-values[symbol], symbol),
        )[: rule["top_k"]]
        weight = min(F(1, rule["top_k"]), F(str(rule["max_weight"])))
        targets.append([float(weight) if symbol in selected else 0.0 for symbol in symbols])
        weight = min(F(1, len(active)), F(str(rule["max_weight"]))) if active else F(0)
        baseline.append([float(weight) if symbol in active else 0.0 for symbol in symbols])
    return masks, scores, targets, baseline, flags


def boundary(full, split):
    frames = {name: [list(row) for row in values[split - 1 :]] for name, values in full.items()}
    columns = {name: index for index, name in enumerate(LEDGER_COLUMNS)}
    records = frames["ledger"]
    starting = records[0][columns["nav"]]
    for row in records:
        row[columns["equity"]] = row[columns["nav"]] / starting
    for name in (
        "asset_pnl",
        "cash_interest",
        "transaction_cost",
        "traded_notional",
        "turnover",
        "gross_return",
        "return",
        "rebalance_executed",
    ):
        records[0][columns[name]] = 0.0
    records[0][columns["pre_trade_nav"]] = starting
    frames["trades"][0] = [0.0] * len(frames["trades"][0])
    frames["executed_targets"][0] = [None] * len(frames["executed_targets"][0])
    returns = [row[columns["return"]] for row in records[1:]]
    equities = [row[columns["equity"]] for row in records]
    volatility = statistics.stdev(returns) * math.sqrt(252)
    peak, drawdown = 1.0, 0.0
    for equity in equities:
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
    metrics = {
        "starting_nav": starting,
        "return_intervals": len(returns),
        "final_nav": records[-1][columns["nav"]],
        "total_return": equities[-1] - 1,
        "cagr": math.expm1(math.log(equities[-1]) * 252 / len(returns)),
        "sharpe": statistics.mean(returns) * 252 / volatility if volatility > 1e-12 else None,
        "volatility": volatility,
        "max_drawdown": drawdown,
        "average_cash_weight": statistics.mean(row[columns["cash_weight"]] for row in records),
    }
    for key, column in (
        ("transaction_costs", "transaction_cost"),
        ("traded_notional", "traded_notional"),
        ("turnover", "turnover"),
        ("rebalances", "rebalance_executed"),
    ):
        metrics[key] = sum(row[columns[column]] for row in records)
    metrics["rebalances"] = int(metrics["rebalances"])
    return frames, metrics


def evaluate(case):
    masks, scores, targets, baseline, flags = signals(case)
    execution = {name: case["config"][name] for name in ("initial_capital", "cost_bps")}
    full, tests, metrics = {}, {}, {}
    for name, instructions in (("result", targets), ("benchmark", baseline)):
        solved = solve({"dates": case["dates"], "prices": case["prices"], "targets": instructions, "config": execution})
        full[name] = solved["expected"]
        full[name]["executed_targets"] = [[None] * len(case["prices"])] + solved["targets"][:-1]
        tests[name], metrics[name] = boundary(full[name], case["test_split"])
    return {
        **case,
        "eligibility": masks,
        "scores": scores,
        "flags": flags,
        "targets": [[None] * len(case["prices"]) if row is None else row for row in targets],
        "expected": full,
        "test_expected": tests,
        "test_metrics": metrics,
    }


def reference():
    membership = {
        "schema_version": 1,
        "universe_id": "synthetic-software-oracle",
        "source": "Project-generated synthetic events",
        "license": "MIT; software tests only",
        "coverage_start": "2024-01-01",
        "coverage_end": "2024-12-31",
        "initial_known_on": "2023-12-29",
        "initial_members": ["AAA", "BBB"],
        "events": [],
    }
    rule = {
        "lookback": 2,
        "skip_recent": 0,
        "top_k": 1,
        "rebalance": "monthly",
        "absolute_threshold": None,
        "max_weight": 0.75,
        "initial_capital": 1000,
        "cost_bps": 10,
    }
    cases = [
        {
            "id": "entry_exit_and_carried_test",
            "dates": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-15",
            ],
            "prices": {
                "AAA": [100, 110, 120, 115, 117, 125, 118, 130, 128, 135],
                "BBB": [100, 99, 98, 100, 103, 105, 120, 122, 119, 125],
                "CCC": [50, 55, 60, 65, 70, 75, 70, 80, 84, 90],
            },
            "config": rule,
            "test_split": 6,
            "membership": {
                **membership,
                "events": [
                    {"ticker": "CCC", "known_on": "2024-01-03", "effective_on": "2024-01-08", "action": "add"},
                    {"ticker": "AAA", "known_on": "2024-01-08", "effective_on": "2024-01-10", "action": "remove"},
                    {"ticker": "CCC", "known_on": "2024-01-10", "effective_on": "2024-01-12", "action": "remove"},
                ],
            },
        },
        {
            "id": "month_boundary_to_all_cash",
            "dates": ["2024-01-25", "2024-01-26", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"],
            "prices": {"AAA": [100, 110, 120, 130, 125, 115, 105], "BBB": [100, 100, 105, 110, 115, 125, 130]},
            "config": {**rule, "skip_recent": 1, "top_k": 2, "max_weight": 0.4, "cost_bps": 25},
            "test_split": 4,
            "membership": {
                **membership,
                "events": [
                    {"ticker": ticker, "known_on": "2024-01-31", "effective_on": "2024-02-01", "action": "remove"}
                    for ticker in ("AAA", "BBB")
                ],
            },
        },
    ]
    return {
        "schema_version": 1,
        "oracle": "Independent Fraction signal / sign-region accounting / scalar boundary oracle",
        "ledger_columns": LEDGER_COLUMNS,
        "cases": [evaluate(case) for case in cases],
    }


if __name__ == "__main__":
    print(json.dumps(reference(), indent=2, allow_nan=False))
