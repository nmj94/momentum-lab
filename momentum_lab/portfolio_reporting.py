"""Offline reports for the explicitly full-history portfolio workflow."""

import html

from .reporting import _display, _markdown_cell

LIMITATIONS = (
    "Fixed ex-post universe or user-declared membership, not verified point-in-time constituent or delisting coverage. "
    "Daily closes/fills are assumed synchronized; equal session-date labels do not prove synchronized markets. "
    "No FX, leverage, shorts, borrow, capacity, nonlinear impact, minimum fees, taxes or broker execution. "
    "Adjustment, source and license fields are user declarations, not independently verified. "
    "Whole-history and development metrics include warm-up cash intervals; test reports use the documented prior-close anchor. "
    "The initial structural zero return is excluded. "
    "Target caps apply at rebalances; actual asset weights drift between fills. "
    "The last signal may still be pending: no fill is invented after the final data bar. "
    "Cash uses an effective annual ACT/365 rate; this differs from the single-asset engine's simple ACT/365.25 convention."
)
_METRICS = (
    ("final_nav", "Final NAV", False),
    ("total_return", "Total return", True),
    ("cagr", "CAGR", True),
    ("sharpe", "Sharpe", False),
    ("volatility", "Volatility", True),
    ("max_drawdown", "Max drawdown", True),
    ("transaction_costs", "Paid transaction costs", False),
    ("turnover", "Two-sided turnover / pre-trade NAV", False),
    ("rebalances", "Executed rebalance instructions", False),
    ("average_cash_weight", "Average cash weight", True),
)


def _tables(summary, metadata):
    metrics = [
        (
            label,
            _display(summary["metrics"].get(key), percent=percent),
            _display(summary["benchmark_metrics"].get(key), percent=percent),
        )
        for key, label, percent in _METRICS
    ]
    if "starting_nav" in summary["metrics"]:
        metrics.insert(
            0,
            (
                "Period starting NAV",
                _display(summary["metrics"]["starting_nav"]),
                _display(summary["benchmark_metrics"]["starting_nav"]),
            ),
        )
    allocations = [
        (
            ticker,
            _display(summary["latest_weights"][ticker], percent=True),
            _display(summary["last_signal_targets"][ticker], percent=True),
            _display(summary["last_signal_scores"][ticker], percent=True),
        )
        for ticker in summary["assets"]
    ]
    allocations.append(
        (
            "Uninvested cash",
            _display(summary["latest_cash_weight"], percent=True),
            _display(max(0.0, 1.0 - sum(summary["last_signal_targets"].values())), percent=True),
            "—",
        )
    )
    settings = [(key, _display(value)) for key, value in metadata["recipe"].items()]
    settings += [
        ("execution_model", metadata["execution_model"]),
        ("cash_convention", metadata["cash_convention"]),
        ("annualization", _display(metadata["annualization"])),
        ("currency", metadata["currency"]),
    ]
    sources = [
        (ticker, value["source"], value["license"], value["price_adjustment"], value["contract_sha256"])
        for ticker, value in summary["data_provenance"].items()
    ]
    tables = [
        (
            f"Performance ({summary.get('performance_scope', 'whole history')})",
            ("Metric", "Momentum portfolio", summary.get("benchmark_label", "Equal-weight buy-and-hold")),
            metrics,
        ),
        (
            "Final holdings and last signal",
            ("Asset", "Final realized weight", "Last signal target", "Last signal score"),
            allocations,
        ),
        ("Fixed recipe and assumptions", ("Setting", "Value"), settings),
        (
            "Declared data provenance",
            ("Asset", "Source", "Usage terms", "Adjustment", "Dataset contract SHA-256"),
            sources,
        ),
    ]
    if summary.get("membership"):
        tables.append(("Declared membership history", ("Field", "Value"), list(summary["membership"].items())))
    return tables


def render_portfolio_markdown(summary, metadata):
    lines = [
        "# Momentum Lab portfolio research",
        "",
        summary["history_notice"],
        "",
        f"- Run: {_markdown_cell(summary['run_id'])}",
        f"- Sessions: {summary['data_start']} to {summary['data_end']}; {summary['n_bars']} bars",
        f"- Last signal: {summary['last_signal_date']}; last executed instruction: {summary['last_execution_date']}",
        f"- Contract SHA-256: `{summary['contract_sha256']}`",
        "",
    ]
    for title, columns, rows in _tables(summary, metadata):
        lines.extend([f"## {title}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)])
        lines.extend("| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in rows)
        lines.append("")
    lines.extend(["## Interpretation limits", "", LIMITATIONS, ""])
    return "\n".join(lines)


def render_portfolio_html(summary, metadata):
    def esc(value):
        return html.escape(str(value), quote=True)

    sections = []
    for title, columns, rows in _tables(summary, metadata):
        headings = "".join(f"<th>{esc(value)}</th>" for value in columns)
        body = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows)
        sections.append(
            f"<section><h2>{esc(title)}</h2><div class='scroll'><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum Lab portfolio — {esc(summary["run_id"])}</title><style>
body{{margin:0;background:#f5f7fa;color:#18202b;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:30px auto;padding:0 20px 40px}}
header,section{{background:white;border:1px solid #dbe1e8;border-radius:12px;padding:22px;margin-bottom:18px}}
h1,h2{{margin:0 0 12px}}h1{{font-size:26px}}h2{{font-size:20px}}table{{width:100%;border-collapse:collapse}}.scroll{{overflow-x:auto}}
th,td{{text-align:left;padding:9px;border-bottom:1px solid #dbe1e8;overflow-wrap:anywhere;min-width:90px}}code,p{{overflow-wrap:anywhere}}.notice{{border-left:4px solid #b46d10;padding-left:14px}}
</style></head><body><main><header><h1>Momentum Lab portfolio research</h1>
<p class="notice">{esc(summary["history_notice"])}</p><p>Run: {esc(summary["run_id"])}<br>
Sessions: {esc(summary["data_start"])} to {esc(summary["data_end"])} · {summary["n_bars"]} bars<br>
Last signal: {esc(summary["last_signal_date"])} · Last executed instruction: {esc(summary["last_execution_date"])}</p>
<p>Contract SHA-256: <code>{esc(summary["contract_sha256"])}</code></p></header>
{"".join(sections)}<section><h2>Interpretation limits</h2><p>{esc(LIMITATIONS)}</p></section></main></body></html>"""
