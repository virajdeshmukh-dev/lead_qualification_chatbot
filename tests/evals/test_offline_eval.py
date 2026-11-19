import os
from pathlib import Path

import pytest

from lead_qualification_chatbot.data.repositories import load_data
from lead_qualification_chatbot.evals.fakes import build_offline_bot
from lead_qualification_chatbot.evals.models import Scenario, Turn
from lead_qualification_chatbot.evals.runner import run_scenario
from lead_qualification_chatbot.evals.scenarios import build_scenarios
from lead_qualification_chatbot.leads.sink import MemoryLeadSink

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
DATA = ROOT / "src" / "lead_qualification_chatbot" / "resources"
OFFICE = "020 7946 0100"
pytestmark = pytest.mark.skipif(
    not (DATA / "faq.csv").exists() and not os.environ.get("CI"),
    reason="packaged resource data/ not present",
)


@pytest.fixture(scope="module")
def data():
    return load_data(DATA)


def test_scenarios_cover_every_lane(data):
    lanes = {s.lane for s in build_scenarios(data, OFFICE)}
    assert {"A", "B", "C", "D", "E", "INPUT", "FLOW", "SAFETY"} <= lanes


async def test_all_offline_scenarios_pass_on_real_data(data):
    scenarios = [s for s in build_scenarios(data, OFFICE) if not s.needs_llm]
    assert len(scenarios) >= 20
    bot = build_offline_bot(data, OFFICE)
    results = [await run_scenario(bot, s) for s in scenarios]
    failed = {r.scenario.id: r.failures for r in results if not r.passed}
    assert not failed, failed


async def test_runner_detects_failures(data):
    bot = build_offline_bot(data, OFFICE)
    wrong = Scenario(id="x", lane="A", description="", turns=[Turn(say="buy", step="nope")])
    result = await run_scenario(bot, wrong)
    assert not result.passed and "expected 'nope'" in result.failures[0]


async def test_offline_scenarios_emit_valid_leads(data):
    sink = MemoryLeadSink()
    bot = build_offline_bot(data, OFFICE, lead_sink=sink)
    scenarios = [s for s in build_scenarios(data, OFFICE) if not s.needs_llm]
    full = [s for s in scenarios if s.qualified and s.done]  # complete journeys
    for s in scenarios:
        await run_scenario(bot, s)
    assert len(sink.leads) >= len(full) > 0
    assert all(
        lead.phone.startswith("+44") and "@" in lead.email and lead.name for lead in sink.leads
    )
