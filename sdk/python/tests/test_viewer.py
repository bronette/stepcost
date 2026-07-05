"""Tests for the report data layer (report.py) and waste heuristics (waste.py)."""

from __future__ import annotations

from decimal import Decimal

from stepcost import StepCost, agent_step, llm_call
from stepcost import report as R
from stepcost.cli import main
from stepcost.waste import detect_waste


def _make_db(tmp_path) -> tuple[str, str]:
    """Return (sink_url, db_path). Two traces: a looping cache-less one + a growing-context one."""
    db = tmp_path / "t.db"
    url = f"sqlite://{db}"  # abs path → clean 3-slash sqlite URL
    cc = StepCost(project="t", sink=url, default_feature="f")

    with cc.trace(customer_id="acme", feature_id="support"):
        for _ in range(4):
            with agent_step("plan"):
                with llm_call(model="claude-sonnet-4-20250514", provider="anthropic") as c:
                    c.record_usage(input_tokens=6000, output_tokens=400)
            with agent_step("retrieve"):
                with cc.span(kind="retrieval", name="pinecone"):
                    pass

    with cc.trace(customer_id="beta", feature_id="chat"):
        for toks in (2000, 4000, 6000):
            with agent_step("chat"):
                with llm_call(model="gpt-4o-mini", provider="openai") as c:
                    c.record_usage(input_tokens=toks, output_tokens=150)

    cc.flush()
    return url, str(db)


def _trace_for(spans, feature):
    for ss in R.group_by_trace(spans).values():
        if any(s.feature_id == feature for s in ss):
            return ss
    raise AssertionError(f"no trace with feature={feature}")


def test_load_and_summary(tmp_path):
    url, _ = _make_db(tmp_path)
    spans = R.load_spans(url)
    summary = R.build_summary(spans)
    assert summary.n_traces == 2
    assert summary.total_usd > 0
    assert "support" in summary.by_feature
    assert "acme" in summary.by_customer
    assert summary.by_kind["llm_generation"] == summary.total_usd  # only llm carries cost


def test_by_step_attribution(tmp_path):
    url, _ = _make_db(tmp_path)
    support = _trace_for(R.load_spans(url), "support")
    by_step = R.rollup_by_step(support)
    assert "plan" in by_step
    assert by_step["plan"] == Decimal("0.096")  # 4 × (0.018 in + 0.006 out)


def test_tree_subtree_sums(tmp_path):
    url, _ = _make_db(tmp_path)
    support = _trace_for(R.load_spans(url), "support")
    roots = R.build_tree(support)
    assert len(roots) == 1
    root = roots[0]
    assert root.span.kind.value == "trace"
    assert root.subtree_usd == Decimal("0.096")
    assert all(c.span.kind.value == "agent_step" for c in root.children)
    # each llm leaf sits under an agent_step
    plan_nodes = [c for c in root.children if c.span.name == "plan"]
    assert len(plan_nodes) == 4
    assert all(pn.subtree_usd == Decimal("0.024") for pn in plan_nodes)


def test_waste_missing_cache_and_loop(tmp_path):
    url, _ = _make_db(tmp_path)
    support = _trace_for(R.load_spans(url), "support")
    waste = {w.code: w for w in detect_waste(support)}
    assert "missing_cache" in waste
    assert "retry_loop" in waste
    loop = waste["retry_loop"]
    assert "plan" in loop.message  # keyed on the looping step, not model/tool
    assert loop.est_usd and loop.est_usd > 0
    # zero-cost 'retrieve' loop must NOT be flagged as a cost leak
    assert all("retrieve" not in w.message for w in detect_waste(support) if w.code == "retry_loop")


def test_waste_oversized_context_and_no_false_loop(tmp_path):
    url, _ = _make_db(tmp_path)
    chat = _trace_for(R.load_spans(url), "chat")
    codes = {w.code for w in detect_waste(chat)}
    assert "oversized_context" in codes
    assert "retry_loop" not in codes  # 'chat' ran 3× (< 4 threshold), not a loop


def test_model_oversized(tmp_path):
    db = tmp_path / "m.db"
    url = f"sqlite://{db}"
    cc = StepCost(project="m", sink=url)
    with cc.trace(feature_id="classify"):
        with agent_step("classify"):
            with llm_call(model="gpt-4o", provider="openai") as c:
                c.record_usage(input_tokens=800, output_tokens=20)
    cc.flush()
    waste = {w.code for w in detect_waste(R.load_spans(url))}
    assert "model_oversized" in waste


def test_cli_summary_and_trace(tmp_path, capsys):
    url, db_path = _make_db(tmp_path)
    assert main(["report", db_path]) == 0
    out = capsys.readouterr().out
    assert "StepCost report" in out
    assert "Waste signals" in out
    assert "By feature" in out

    tid = R.load_spans(url)[0].trace_id
    assert main(["report", db_path, "--trace", tid]) == 0
    tout = capsys.readouterr().out
    assert "Trace" in tout
    assert "By step" in tout
    assert "tok" in tout  # tree shows per-node token totals, not just $


def test_cli_missing_db_returns_error(tmp_path, capsys):
    assert main(["report", str(tmp_path / "nope.db")]) == 2
