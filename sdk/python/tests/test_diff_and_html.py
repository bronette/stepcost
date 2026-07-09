"""Tests for the cost-diff/budget gate and the HTML report (betas)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path


from stepcost import StepCost, Provider, agent_step, llm_call
from stepcost.cli import main as cli_main
from stepcost.diff import Budget, compute_diff, render_markdown


def _make_db(tmp_path: Path, name: str, *, input_tokens: int, feature: str = "chat") -> str:
    db = tmp_path / name
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    with cc.trace(feature_id=feature):
        with agent_step("answer"):
            with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
                call.record_usage(input_tokens=input_tokens, output_tokens=100)
    cc.flush()
    return str(db)


def test_diff_totals_and_direction(tmp_path: Path):
    base = _make_db(tmp_path, "base.db", input_tokens=100_000)   # $0.015 + out
    head = _make_db(tmp_path, "head.db", input_tokens=200_000)   # roughly double
    diff = compute_diff(base, head)
    assert diff.head_total > diff.base_total
    assert diff.delta == diff.head_total - diff.base_total
    assert 90 < diff.pct < 110  # ~2x input cost, same output
    assert diff.passed  # no budget → no violations


def test_budget_violations(tmp_path: Path):
    base = _make_db(tmp_path, "base.db", input_tokens=100_000)
    head = _make_db(tmp_path, "head.db", input_tokens=200_000)
    budget = Budget(
        max_total_usd=Decimal("0.001"),
        max_trace_usd=Decimal("0.001"),
        max_increase_pct=10.0,
    )
    diff = compute_diff(base, head, budget)
    assert not diff.passed
    assert len(diff.violations) == 3  # total, trace, increase all tripped


def test_budget_loads_from_toml(tmp_path: Path):
    toml = tmp_path / ".stepcost.toml"
    toml.write_text('[budget]\nmax_total_usd = 5.00\nmax_increase_pct = 20\n')
    b = Budget.load(toml)
    assert b.max_total_usd == Decimal("5.00")
    assert b.max_increase_pct == 20.0
    assert b.max_trace_usd is None


def test_markdown_output_shape(tmp_path: Path):
    base = _make_db(tmp_path, "base.db", input_tokens=100_000)
    head = _make_db(tmp_path, "head.db", input_tokens=200_000, feature="chat")
    diff = compute_diff(base, head, Budget(max_increase_pct=10.0))
    md = render_markdown(diff)
    assert md.startswith("## 💸 StepCost report")
    assert "budget exceeded" in md
    assert "| chat |" in md
    assert "| answer |" in md


def test_cli_diff_exit_codes(tmp_path: Path, capsys):
    base = _make_db(tmp_path, "base.db", input_tokens=100_000)
    head = _make_db(tmp_path, "head.db", input_tokens=200_000)
    assert cli_main(["diff", base, head]) == 0
    capsys.readouterr()

    toml = tmp_path / ".stepcost.toml"
    toml.write_text("[budget]\nmax_increase_pct = 10\n")
    assert cli_main(["diff", base, head, "--budget", str(toml), "--markdown"]) == 3
    out = capsys.readouterr().out
    assert "budget exceeded" in out

    assert cli_main(["diff", base, head, "--budget", str(tmp_path / "missing.toml")]) == 2


def test_cli_diff_zero_base(tmp_path: Path, capsys):
    """Empty-cost base (e.g. first run) must not divide by zero."""
    base_path = tmp_path / "base.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{base_path}")
    with cc.trace(feature_id="chat"):
        with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
            call.record_usage(input_tokens=0, output_tokens=0)
    cc.flush()
    base = str(base_path)
    head = _make_db(tmp_path, "head.db", input_tokens=100_000)
    assert cli_main(["diff", base, head]) == 0
    assert "n/a" in capsys.readouterr().out


def test_html_report_self_contained(tmp_path: Path, capsys):
    db = _make_db(tmp_path, "h.db", input_tokens=50_000)
    out = tmp_path / "report.html"
    assert cli_main(["report", db, "--html", str(out)]) == 0
    html = out.read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert "StepCost" in html and "By feature" in html and "chat" in html
    assert "http" not in html.split("stepcost.com")[0].split("</style>")[0]  # no external assets in CSS
    assert "agent_step: answer" in html or "answer" in html


def test_html_report_escapes_user_content(tmp_path: Path, capsys):
    db = tmp_path / "x.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    with cc.trace(feature_id="<script>alert(1)</script>"):
        with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
            call.record_usage(input_tokens=1000)
    cc.flush()
    out = tmp_path / "r.html"
    assert cli_main(["report", str(db), "--html", str(out)]) == 0
    html = out.read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_diff_unpriced_spans_fail_the_budget_gate(tmp_path: Path):
    base = _make_db(tmp_path, "base.db", input_tokens=1_000)
    head_path = tmp_path / "head.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{head_path}")
    with cc.trace(feature_id="chat"):
        with llm_call(model="totally-unknown-model", provider=Provider.OTHER) as call:
            call.record_usage(input_tokens=50_000)
    cc.flush()
    diff = compute_diff(base, str(head_path), Budget(max_total_usd=Decimal("100")))
    assert not diff.passed
    assert any("unpriced" in v for v in diff.violations)
    assert "totally-unknown-model" in render_markdown(diff)
    # Without a budget it's surfaced but not a violation
    assert compute_diff(base, str(head_path)).passed


def test_markdown_cells_escape_pipes(tmp_path: Path):
    base_path = tmp_path / "b.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{base_path}")
    with cc.trace(feature_id="x | injected | row"):
        with agent_step("step|pipe"):
            with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
                call.record_usage(input_tokens=1_000)
    cc.flush()
    diff = compute_diff(str(base_path), str(base_path))
    md = render_markdown(diff)
    assert "x \\| injected \\| row" in md
    assert "step\\|pipe" in md
