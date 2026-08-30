"""Independent exact-rational portfolio ledger; stdout is a reviewable fixture.

No project, NumPy or pandas imports. Transaction costs are solved algebraically
by enumerating buy/sell sign regions, not with the production bisection solver.
Cash-interest cases use whole ACT/365 years so Fraction arithmetic stays exact.
Never update the frozen fixture merely to make an engine regression pass.
"""

import json
from datetime import date
from fractions import Fraction as F
from itertools import product

LEDGER_COLUMNS = [
    "nav",
    "equity",
    "cash",
    "cash_weight",
    "asset_pnl",
    "cash_interest",
    "pre_trade_nav",
    "transaction_cost",
    "traded_notional",
    "turnover",
    "gross_return",
    "return",
    "rebalance_executed",
]


def post_fee_value(value, marked, target, rate):
    """Solve each linear region exactly and accept a sign-consistent root."""
    for signs in product((-1, 1), repeat=len(marked)):
        numerator = value + rate * sum(sign * leg for sign, leg in zip(signs, marked))
        denominator = 1 + rate * sum(sign * weight for sign, weight in zip(signs, target))
        candidate = numerator / denominator
        if 0 <= candidate <= value and all(
            sign * (weight * candidate - leg) >= 0 for sign, weight, leg in zip(signs, target, marked)
        ):
            return candidate
    raise AssertionError("No feasible self-financing sign region")


def solve(case):
    symbols = sorted(case["prices"])
    count = len(symbols)
    capital = F(str(case["config"].get("initial_capital", 1000)))
    cost = F(str(case["config"].get("cost_bps", 0)))
    slip = F(str(case["config"].get("slippage_bps", 0)))
    spread = F(str(case["config"].get("spread_bps", 0)))
    rate = (cost + slip + spread / 2) / 10000
    cash_rate = F(str(case["config"].get("cash_rate", 0)))
    lag = case["config"].get("execution_lag", 1)
    cash, previous_nav = capital, capital
    shares = [F(0)] * count
    previous_quote = [F(0)] * count
    expected = {name: [] for name in ("ledger", "weights", "holdings", "asset_values", "trades")}
    for row, session in enumerate(case["dates"]):
        quote = [F(str(case["prices"][symbol][row])) for symbol in symbols]
        days = (date.fromisoformat(session) - date.fromisoformat(case["dates"][row - 1])).days if row else 0
        if cash_rate and days % 365:
            raise AssertionError("Exact cash fixture needs whole ACT/365 years")
        interest = cash * ((1 + cash_rate) ** (days // 365) - 1) if cash_rate else F(0)
        asset_pnl = sum(n * (p - old) for n, p, old in zip(shares, quote, previous_quote)) if row else F(0)
        cash += interest
        marked = [n * p for n, p in zip(shares, quote)]
        before = cash + sum(marked)
        target = case["targets"][row - lag] if row >= lag else None
        traded = [F(0)] * count
        fees = F(0)
        if target is not None:
            weights = [F(str(weight)) for weight in target]
            after = post_fee_value(before, marked, weights, rate)
            desired = [weight * after for weight in weights]
            traded = [new - old for new, old in zip(desired, marked)]
            fees = rate * sum(abs(leg) for leg in traded)
            cash = before - fees - sum(desired)
            assert cash >= 0
            shares = [leg / price for leg, price in zip(desired, quote)]
            marked = desired
        nav = cash + sum(marked)
        assert nav > 0
        notional = sum(abs(leg) for leg in traded)
        record = [
            nav,
            nav / capital,
            cash,
            cash / nav,
            asset_pnl,
            interest,
            before,
            fees,
            notional,
            notional / before,
            before / previous_nav - 1,
            nav / previous_nav - 1,
            F(target is not None),
        ]
        values = {
            "ledger": record,
            "weights": [leg / nav for leg in marked],
            "holdings": shares,
            "asset_values": marked,
            "trades": traded,
        }
        for name, entries in values.items():
            expected[name].append([float(value) for value in entries])
        previous_nav, previous_quote = nav, quote
    return {
        **case,
        "targets": [[None] * count if row is None else row for row in case["targets"]],
        "expected": expected,
    }


def reference():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    cases = [
        {
            "id": "delayed_rotation_and_pending_exit",
            "dates": dates,
            "prices": {"AAA": [100, 110, 120, 90, 100], "BBB": [100, 100, 100, 110, 105]},
            "targets": [[1, 0], None, [0, 1], [0.5, 0.25], [0, 0]],
            "config": {"initial_capital": 1000, "cost_bps": 100},
        },
        {
            "id": "hold_shares_and_weight_drift",
            "dates": dates[:4],
            "prices": {"AAA": [100, 100, 200, 400], "BBB": [100, 100, 100, 100]},
            "targets": [[0.5, 0.25], None, None, None],
            "config": {"initial_capital": 1000, "cost_bps": 0},
        },
        {
            "id": "fully_liquidate_after_fee_paid_entry",
            "dates": dates[:3],
            "prices": {"AAA": [100, 100, 100], "BBB": [100, 100, 100]},
            "targets": [[0.5, 0.5], [0, 0], [1, 0]],
            "config": {"initial_capital": 100, "cost_bps": 200},
        },
        {
            "id": "cash_effective_annual_compounding",
            "dates": ["2020-01-01", "2020-12-31", "2021-12-31"],
            "prices": {"AAA": [100, 150, 200], "BBB": [100, 80, 90]},
            "targets": [None, None, None],
            "config": {"initial_capital": 100, "cost_bps": 0, "cash_rate": 0.1},
        },
        {
            "id": "high_cost_rotation_and_longer_delay",
            "dates": dates,
            "prices": {"AAA": [10, 11, 12, 13, 14], "BBB": [20, 19, 18, 17, 16]},
            "targets": [[1, 0], [0, 1], [0.25, 0.25], [1, 0], None],
            "config": {
                "initial_capital": 1000,
                "cost_bps": 8950,
                "slippage_bps": 25,
                "spread_bps": 50,
                "execution_lag": 2,
            },
        },
        {
            "id": "repeated_flat_targets_do_not_churn",
            "dates": dates[:4],
            "prices": {"AAA": [10, 10, 10, 10], "BBB": [20, 20, 20, 20]},
            "targets": [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4], None],
            "config": {"initial_capital": 1000, "cost_bps": 10},
        },
    ]
    return {
        "schema_version": 1,
        "oracle": "scripts/portfolio_reference_oracle.py; exact Fraction sign-region algebra, no production imports",
        "ledger_columns": LEDGER_COLUMNS,
        "cases": [solve(case) for case in cases],
    }


if __name__ == "__main__":
    print(json.dumps(reference(), indent=2, allow_nan=False))
