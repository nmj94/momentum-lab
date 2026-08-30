"""Dependency-free research report rendering."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping

_METRICS = (
    ("sharpe", "Sharpe", False),
    ("cagr", "CAGR", True),
    ("total_return", "Total return", True),
    ("max_drawdown", "Max drawdown", True),
    ("volatility", "Volatility", True),
    ("trade_count", "Trades", False),
    ("turnover", "Turnover", False),
    ("requested_turnover", "Requested turnover", False),
    ("fill_ratio", "Fill ratio", True),
    ("transaction_cost_drag", "Transaction cost drag", True),
    ("max_participation", "Maximum participation", True),
    ("capacity_constrained_bars", "Capacity-constrained bars", False),
    ("borrow_blocked_bars", "Borrow-blocked bars", False),
)

_ASSUMPTIONS = (
    ("execution_model", "Execution model"),
    ("cost_bps", "Commission (bps)"),
    ("slippage_bps", "Slippage (bps)"),
    ("spread_bps", "Quoted spread (bps)"),
    ("impact_bps", "Impact at reference participation (bps)"),
    ("impact_exponent", "Impact exponent"),
    ("impact_reference_participation", "Reference participation"),
    ("max_participation", "Maximum participation"),
    ("initial_capital", "Initial capital"),
    ("min_fee", "Minimum fee"),
    ("cash_rate", "Cash rate"),
    ("financing_rate", "Financing base rate"),
    ("financing_spread", "Financing spread"),
    ("borrow_bps", "Borrow fee (bps)"),
    ("short_rebate_rate", "Short rebate rate"),
    ("max_leverage", "Maximum leverage"),
)


def _display(value, *, percent=False):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if percent:
            return f"{value:.2%}"
        return f"{value:,.4f}"
    return str(value)


def _metric_rows(best, benchmark):
    validation = best.get("val_metrics", {}) if best else {}
    test = best.get("test_metrics", {}) if best else {}
    benchmark = benchmark or {}
    return [
        (
            label,
            _display(validation.get(key), percent=percent),
            _display(test.get(key), percent=percent),
            _display(benchmark.get(key), percent=percent),
        )
        for key, label, percent in _METRICS
    ]


def _diagnostic_rows(summary):
    best = summary.get("best") or {}
    selected = best.get("selection_diagnostics") or {}
    overall = summary.get("selection_diagnostics") or {}
    ci = selected.get("validation_sharpe_ci_95") or [None, None]
    pbo = overall.get("pbo") or {}
    walk_forward = overall.get("walk_forward_selection") or {}
    return [
        ("Selection metric", overall.get("selection_metric")),
        ("Trials", overall.get("trials")),
        ("Full-development trials", overall.get("full_development_trials")),
        ("Sharpe-dispersion source", overall.get("sharpe_std_source")),
        ("Eligible candidates", overall.get("eligible_candidates")),
        ("Deflated Sharpe probability", _display(selected.get("deflated_sharpe_probability"), percent=True)),
        ("Validation Sharpe 95% CI (analytic)", f"[{_display(ci[0])}, {_display(ci[1])}]"),
        ("CSCV/PBO", _display(pbo.get("probability"), percent=True)),
        ("Walk-forward mean outer Sharpe", _display(walk_forward.get("mean_outer_sharpe"))),
        ("Walk-forward worst outer Sharpe", _display(walk_forward.get("worst_outer_sharpe"))),
    ]


def _search_rows(summary, run_config):
    search = summary.get("search_diagnostics") or {}
    cache = summary.get("indicator_cache") or {}
    rows = [
        ("Method", search.get("method", run_config.get("search_method", "grid"))),
        ("Final full-development candidates", summary.get("n_results", 0)),
    ]
    if search.get("method") == "successive_halving":
        rows.extend(
            [
                ("Initial candidates", search.get("initial_candidates")),
                ("Validation resources (bars)", " → ".join(map(str, search.get("resource_bars") or []))),
                ("Total stage evaluations", search.get("total_stage_evaluations")),
                ("Eliminated candidates", search.get("eliminated_candidates")),
            ]
        )
    rows.extend(
        [
            ("Indicator-cache entries/process", cache.get("max_entries_per_process")),
            ("Observed cache hit rate", _display(cache.get("hit_rate"), percent=True)),
            ("Observed cache hits", cache.get("hits")),
            ("Observed cache misses", cache.get("misses")),
        ]
    )
    return rows


def _bootstrap_tables(summary):
    diagnostic = summary.get("bootstrap_diagnostics") or {}
    periods = diagnostic.get("periods") or {}
    settings, intervals = [], []
    metrics = (
        ("strategy_annualized_mean", "Strategy annualized mean", True),
        ("strategy_sharpe", "Strategy Sharpe (unrounded)", False),
        ("benchmark_annualized_mean", "Buy & hold annualized mean", True),
        ("benchmark_sharpe", "Buy & hold Sharpe (unrounded)", False),
        ("annualized_mean_excess", "Annualized mean excess vs buy & hold", True),
    )

    def number(value, percent):
        if value is None:
            return "—"
        return f"{value * 100:.6g}%" if percent else f"{value:.6g}"

    for label, key in (("Validation", "validation"), ("Sealed test", "test")):
        period = periods.get(key)
        if not period:
            continue
        settings.append(
            (
                label,
                period.get("n_observations"),
                period.get("required_observations"),
                period.get("block_length"),
                f"{period.get('completed_resamples', 0)}/{period.get('n_resamples', 0)}",
                number(period.get("confidence_level"), True),
                period.get("seed"),
                period.get("status"),
            )
        )
        for name, metric_label, percent in metrics:
            statistic = (period.get("statistics") or {}).get(name) or {}
            bounds = statistic.get("ci") or [None, None]
            interval = (
                f"[{number(bounds[0], percent)}, {number(bounds[1], percent)}]"
                if all(value is not None for value in bounds)
                else "Not estimated"
            )
            status = statistic.get("status", "unavailable")
            if statistic.get("reason"):
                status += f": {statistic['reason']}"
            intervals.append((label, metric_label, number(statistic.get("estimate"), percent), interval, status))
    note = diagnostic.get("warning") or (
        "Post-selection return diagnostics, not a correction for strategy selection or repeated test access. "
        "Annualized means are arithmetic, not CAGR."
    )
    return diagnostic.get("status", "unavailable"), note, settings, intervals


def _markdown_cell(value):
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown_report(summary: Mapping, run_config: Mapping) -> str:
    """Render a portable Markdown report from final, post-selection results."""
    best = summary.get("best") or {}
    strategy = best.get("strategy") or "No eligible selection"
    params = json.dumps(best.get("params") or {}, ensure_ascii=False, sort_keys=True)
    lines = [
        f"# Momentum Lab research report: {run_config.get('ticker', 'unknown')}",
        "",
        "> Research evidence only. The sealed test result is reported once after selection and is not trading advice.",
        "",
        "## Run overview",
        "",
        f"- Run ID: `{summary.get('run_id', 'unknown')}`",
        f"- Package: `{run_config.get('package_version', 'unknown')}`",
        f"- Data: `{run_config.get('data_start', 'unknown')}` to `{run_config.get('data_end', 'unknown')}`",
        f"- Experiments: {summary.get('n_results', 0)} ({summary.get('n_errors', 0)} errors)",
        f"- Selected strategy: `{strategy}`",
        f"- Parameters: `{params}`",
        "",
        "## Performance evidence",
        "",
        "| Metric | Validation | Sealed test | Buy & hold test |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {label} | {validation} | {test} | {benchmark} |"
        for label, validation, test, benchmark in _metric_rows(best, summary.get("benchmark_metrics"))
    )
    lines.extend(["", "## Selection diagnostics", "", "| Diagnostic | Value |", "|---|---:|"])
    lines.extend(f"| {label} | {_display(value)} |" for label, value in _diagnostic_rows(summary))
    status, note, settings, intervals = _bootstrap_tables(summary)
    lines.extend(
        [
            "",
            "## Paired block-bootstrap uncertainty",
            "",
            f"Status: {_markdown_cell(status)}",
            "",
            _markdown_cell(note),
            "",
        ]
    )
    if settings:
        lines.extend(
            [
                "| Window | Observations | Required | Block bars | Completed/requested | Confidence | Seed | Status |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        lines.extend("| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in settings)
        lines.extend(
            ["", "| Window | Statistic | Estimate | Percentile interval | Status / reason |", "|---|---|---:|---|---|"]
        )
        lines.extend("| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in intervals)
    lines.extend(["", "## Search efficiency", "", "| Diagnostic | Value |", "|---|---:|"])
    lines.extend(f"| {label} | {_display(value)} |" for label, value in _search_rows(summary, run_config))
    lines.extend(["", "## Execution and financing assumptions", "", "| Assumption | Value |", "|---|---:|"])
    lines.extend(f"| {label} | {_display(run_config.get(key))} |" for key, label in _ASSUMPTIONS if key in run_config)

    sensitivity = summary.get("parameter_sensitivity") or {}
    if sensitivity:
        lines.extend(
            [
                "",
                "## Local parameter sensitivity",
                "",
                f"- Grade: `{sensitivity.get('grade', 'unavailable')}`",
                f"- Verdict: {sensitivity.get('verdict', sensitivity.get('error', 'unavailable'))}",
                f"- Neighbors evaluated: {sensitivity.get('n_neighbors', 0)}",
                f"- Isolated peak: {_display(sensitivity.get('isolated_peak'))}",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "Selection-bias diagnostics reduce but do not remove overfitting risk. Capacity is estimated from bar "
                "volume, not an order-book replay; taxes, halts, queue position, and live operational risk remain "
                "outside this model."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_html_report(summary: Mapping, run_config: Mapping) -> str:
    """Render a self-contained, offline HTML report."""
    best = summary.get("best") or {}
    strategy = best.get("strategy") or "No eligible selection"
    params = json.dumps(best.get("params") or {}, ensure_ascii=False, sort_keys=True)

    def esc(value):
        return html.escape(str(value), quote=True)

    metric_rows = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(validation)}</td><td>{esc(test)}</td><td>{esc(benchmark)}</td></tr>"
        for label, validation, test, benchmark in _metric_rows(best, summary.get("benchmark_metrics"))
    )
    diagnostics = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(_display(value))}</td></tr>" for label, value in _diagnostic_rows(summary)
    )
    search_efficiency = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(_display(value))}</td></tr>"
        for label, value in _search_rows(summary, run_config)
    )
    bootstrap_status, bootstrap_note, bootstrap_settings, bootstrap_intervals = _bootstrap_tables(summary)
    bootstrap_section = (
        f"<section><h2>Paired block-bootstrap uncertainty</h2><p>Status: {esc(bootstrap_status)}</p>"
        f'<p class="note">{esc(bootstrap_note)}</p>'
    )
    if bootstrap_settings:
        bootstrap_section += (
            "<table><thead><tr><th>Window</th><th>Observations</th><th>Required</th><th>Block bars</th>"
            "<th>Completed/requested</th><th>Confidence</th><th>Seed</th><th>Status</th></tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in bootstrap_settings
            )
            + "</tbody></table><table><thead><tr><th>Window</th><th>Statistic</th><th>Estimate</th>"
            "<th>Percentile interval</th><th>Status / reason</th></tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in bootstrap_intervals
            )
            + "</tbody></table>"
        )
    bootstrap_section += "</section>"
    assumptions = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(_display(run_config.get(key)))}</td></tr>"
        for key, label in _ASSUMPTIONS
        if key in run_config
    )
    sensitivity = summary.get("parameter_sensitivity") or {}
    sensitivity_section = ""
    if sensitivity:
        sensitivity_section = f"""
        <section><h2>Local parameter sensitivity</h2>
        <dl><dt>Grade</dt><dd>{esc(sensitivity.get("grade", "unavailable"))}</dd>
        <dt>Verdict</dt><dd>{esc(sensitivity.get("verdict", sensitivity.get("error", "unavailable")))}</dd>
        <dt>Neighbors</dt><dd>{esc(sensitivity.get("n_neighbors", 0))}</dd>
        <dt>Isolated peak</dt><dd>{esc(_display(sensitivity.get("isolated_peak")))}</dd></dl></section>
        """

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum Lab report — {esc(run_config.get("ticker", "unknown"))}</title>
<style>
:root{{--bg:#f6f8fb;--card:#fff;--ink:#18202b;--muted:#5d6878;--line:#dbe1e8;--accent:#1769aa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1000px;margin:32px auto;padding:0 18px 48px}}section,header{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px;margin:16px 0}}
h1{{margin:0 0 8px;font-size:28px}}h2{{margin:0 0 14px;font-size:20px}}p.note{{color:var(--muted)}}code{{overflow-wrap:anywhere}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}thead th{{color:var(--muted)}}dl{{display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:8px 18px}}dt{{color:var(--muted)}}dd{{margin:0}}
</style></head><body><main>
<header><h1>Momentum Lab research report</h1><p class="note">Research evidence only. The sealed test is reported once after selection and is not trading advice.</p>
<dl><dt>Ticker</dt><dd>{esc(run_config.get("ticker", "unknown"))}</dd><dt>Run ID</dt><dd>{esc(summary.get("run_id", "unknown"))}</dd>
<dt>Package</dt><dd>{esc(run_config.get("package_version", "unknown"))}</dd><dt>Data</dt><dd>{esc(run_config.get("data_start", "unknown"))} to {esc(run_config.get("data_end", "unknown"))}</dd>
<dt>Experiments</dt><dd>{esc(summary.get("n_results", 0))} ({esc(summary.get("n_errors", 0))} errors)</dd><dt>Selected strategy</dt><dd>{esc(strategy)}</dd><dt>Parameters</dt><dd><code>{esc(params)}</code></dd></dl></header>
<section><h2>Performance evidence</h2><table><thead><tr><th>Metric</th><th>Validation</th><th>Sealed test</th><th>Buy &amp; hold test</th></tr></thead><tbody>{metric_rows}</tbody></table></section>
<section><h2>Selection diagnostics</h2><table><tbody>{diagnostics}</tbody></table></section>
{bootstrap_section}
<section><h2>Search efficiency</h2><table><tbody>{search_efficiency}</tbody></table></section>
<section><h2>Execution and financing assumptions</h2><table><tbody>{assumptions}</tbody></table></section>
{sensitivity_section}
<section><h2>Interpretation limits</h2><p>Selection-bias diagnostics reduce but do not remove overfitting risk. Capacity is estimated from bar volume, not an order-book replay; taxes, halts, queue position, and live operational risk remain outside this model.</p></section>
</main></body></html>"""
