"""Exact rational turnover ledgers; no project, NumPy or pandas imports.

Uses sign-region algebra for the unrestricted target and a convex combination
of the original and fully rebalanced accounts for constrained fills. Production
uses floating-point bisection and scales trades. Never rebaseline from its output.
"""

import json
from datetime import date
from fractions import Fraction as F

from portfolio_reference_oracle import LEDGER_COLUMNS, post_fee_value

EXECUTION_COLUMNS = [
    "instruction_processed",
    "requested_notional",
    "filled_notional",
    "unfilled_notional",
    "fill_fraction",
    "turnover_limited",
]


def solve(case):
    symbols = sorted(case["prices"])
    rule = case["config"]
    capital = F(str(rule["initial_capital"]))
    rate = (
        F(str(rule["cost_bps"])) + F(str(rule.get("slippage_bps", 0))) + F(str(rule.get("spread_bps", 0))) / 2
    ) / 10000
    annual = F(str(rule.get("cash_rate", 0)))
    cap = None if rule["max_turnover"] is None else F(str(rule["max_turnover"]))
    lag = rule.get("execution_lag", 1)
    cash = prior_nav = capital
    shares = [F(0)] * len(symbols)
    previous = [F(0)] * len(symbols)
    result = {
        name: []
        for name in (
            "ledger",
            "weights",
            "holdings",
            "asset_values",
            "trades",
            "executed_targets",
            "execution",
            "requested_trades",
            "unfilled_trades",
        )
    }
    for i, day in enumerate(case["dates"]):
        quote = [F(str(case["prices"][symbol][i])) for symbol in symbols]
        elapsed = (date.fromisoformat(day) - date.fromisoformat(case["dates"][i - 1])).days if i else 0
        assert not annual or elapsed % 365 == 0
        interest = cash * ((1 + annual) ** (elapsed // 365) - 1) if annual else F(0)
        pnl = sum(n * (p - old) for n, p, old in zip(shares, quote, previous)) if i else F(0)
        cash += interest
        marked = [n * p for n, p in zip(shares, quote)]
        before = cash + sum(marked)
        target = case["targets"][i - lag] if i >= lag else None
        requested = fills = [F(0)] * len(symbols)
        fraction = F(0)
        fees = F(0)
        limited = False
        if target is not None:
            weights = [F(str(w)) for w in target]
            full_nav = post_fee_value(before, marked, weights, rate)
            full_values = [w * full_nav for w in weights]
            requested = [new - old for new, old in zip(full_values, marked)]
            gross = sum(abs(v) for v in requested)
            fraction = min(F(1), cap * before / gross) if cap is not None and gross else F(1)
            limited = fraction < 1
            full_fee = rate * gross
            full_cash = before - full_fee - sum(full_values)
            # Convex combination proves feasibility without prioritizing assets.
            next_values = [(1 - fraction) * old + fraction * new for old, new in zip(marked, full_values)]
            cash = (1 - fraction) * cash + fraction * full_cash
            fills = [new - old for new, old in zip(next_values, marked)]
            fees = fraction * full_fee
            shares = [v / p for v, p in zip(next_values, quote)]
            marked = next_values
        remaining = [want - fill for want, fill in zip(requested, fills)]
        filled = sum(abs(v) for v in fills)
        want = sum(abs(v) for v in requested)
        nav = cash + sum(marked)
        assert cash >= 0 and nav > 0 and all(n >= 0 for n in shares)
        assert cap is None or filled <= cap * before
        assert nav == before - fees
        values = {
            "ledger": [
                nav,
                nav / capital,
                cash,
                cash / nav,
                pnl,
                interest,
                before,
                fees,
                filled,
                filled / before,
                before / prior_nav - 1,
                nav / prior_nav - 1,
                F(target is not None),
            ],
            "weights": [v / nav for v in marked],
            "holdings": shares,
            "asset_values": marked,
            "trades": fills,
            "requested_trades": requested,
            "unfilled_trades": remaining,
            "execution": [F(target is not None), want, filled, want - filled, fraction, F(limited)],
        }
        for name, row in values.items():
            result[name].append([float(v) for v in row])
        result["executed_targets"].append(target if target is not None else [None] * len(symbols))
        prior_nav, previous = nav, quote
    return {
        **case,
        "targets": [row if row is not None else [None] * len(symbols) for row in case["targets"]],
        "expected": result,
    }


def reference():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
    prices = {"AAA": [10, 10, 20, 20, 10, 15], "BBB": [20, 20, 20, 10, 20, 10]}
    cases = [
        ("partial_entry_hold_exit_and_unexecuted_last_signal", [[1, 0], None, [0, 0], None, None, [0, 1]], 0.25, 0, {}),
        ("rotation_with_shared_cash_and_paid_fees", [[1, 0], [0, 1], [0.5, 0.25], [0, 0], [1, 0], None], 0.4, 1000, {}),
        (
            "longer_lag_high_cost_and_cash_slots",
            [[1, 0], [0, 1], [0.25, 0.25], [0, 0], None, None],
            0.2,
            8950,
            {"execution_lag": 2, "slippage_bps": 25, "spread_bps": 50},
        ),
        ("zero_budget_never_buys", [[1, 0]] * 6, 0, 100, {}),
        ("nonbinding_two_sided_limit", [[1, 0], [0, 1], [0.25, 0.25], [0, 0], None, [1, 0]], 2, 100, {}),
        ("explicit_unlimited_matches_old_book", [[1, 0], [0, 1], [0.25, 0.25], [0, 0], None, [1, 0]], None, 100, {}),
        ("satisfied_cash_instructions", [[0, 0]] * 6, 0, 0, {}),
    ]
    resolved = [
        solve(
            {
                "id": name,
                "dates": dates,
                "prices": prices,
                "targets": targets,
                "config": {"initial_capital": 1000, "cost_bps": cost, "max_turnover": cap, **extra},
            }
        )
        for name, targets, cap, cost, extra in cases
    ]
    resolved.append(
        solve(
            {
                "id": "budget_after_cash_interest",
                "dates": ["2020-01-01", "2020-12-31", "2021-12-31"],
                "prices": {"AAA": [10, 10, 20], "BBB": [20, 20, 20]},
                "targets": [[1, 0], [0, 1], None],
                "config": {"initial_capital": 1000, "cost_bps": 100, "max_turnover": 0.2, "cash_rate": 0.1},
            }
        )
    )
    return {
        "schema_version": 1,
        "oracle": "scripts/portfolio_turnover_reference_oracle.py; Fraction sign regions and convex accounts, no production imports",
        "ledger_columns": LEDGER_COLUMNS,
        "execution_columns": EXECUTION_COLUMNS,
        "cases": resolved,
    }


if __name__ == "__main__":
    print(json.dumps(reference(), indent=2, allow_nan=False))
