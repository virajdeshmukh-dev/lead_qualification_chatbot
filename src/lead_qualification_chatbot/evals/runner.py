import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from lead_qualification_chatbot.evals.models import Scenario, Turn
from lead_qualification_chatbot.graph.orchestrator import ChatReply

_NON_DIGITS = re.compile(r"\D+")


class Chatter(Protocol):
    async def chat(self, session_id: str, text: str) -> ChatReply: ...


@dataclass(slots=True)
class ScenarioResult:
    scenario: Scenario
    failures: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    transcript: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def _digits(text: str) -> str:
    return _NON_DIGITS.sub("", text)


def _check_turn(turn: Turn, reply: ChatReply, index: int) -> list[str]:
    tag = f"turn {index} {turn.say!r}"
    joined = "\n".join(reply.messages)
    low, digits = joined.lower(), _digits(joined)
    failures: list[str] = []
    if reply.degraded:
        failures.append(f"{tag}: degraded reply")
    if turn.step is not None and reply.step != turn.step:
        failures.append(f"{tag}: step {reply.step!r}, expected {turn.step!r}")
    if turn.route is not None and reply.route != turn.route:
        failures.append(f"{tag}: route {reply.route!r}, expected {turn.route!r}")
    failures += [f"{tag}: missing {s!r}" for s in turn.contains if s.lower() not in low]
    failures += [f"{tag}: unexpected {s!r}" for s in turn.not_contains if s.lower() in low]
    failures += [
        f"{tag}: missing digits of {d!r}"
        for d in turn.contains_digits
        if _digits(d) and _digits(d) not in digits
    ]
    if turn.contains_any and not any(
        s.lower() in low or (len(_digits(s)) >= 7 and _digits(s) in digits)
        for s in turn.contains_any
    ):
        failures.append(f"{tag}: none of {turn.contains_any!r} present")
    return failures


async def run_scenario(bot: Chatter, scenario: Scenario) -> ScenarioResult:
    result = ScenarioResult(scenario)
    session_id = f"eval-{uuid4().hex}"
    reply: ChatReply | None = None
    for index, turn in enumerate(scenario.turns, start=1):
        started = time.perf_counter()
        reply = await bot.chat(session_id, turn.say)
        result.latencies_ms.append((time.perf_counter() - started) * 1000)
        result.transcript.append((turn.say, reply.messages))
        result.failures += _check_turn(turn, reply, index)
    if reply is not None:
        if scenario.closing is not None and reply.closing != scenario.closing:
            result.failures.append(
                f"final: closing {reply.closing!r}, \
                                   expected {scenario.closing!r}"
            )
        if scenario.qualified is not None and reply.qualified != scenario.qualified:
            result.failures.append(
                f"final: qualified {reply.qualified}, \
                                   expected {scenario.qualified}"
            )
        if scenario.done is not None and reply.done != scenario.done:
            result.failures.append(f"final: done {reply.done}, expected {scenario.done}")
    return result


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def pass_rate(results: list[ScenarioResult]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 0.0


def format_report(results: list[ScenarioResult]) -> str:
    lines = ["", f"{'lane':8}{'passed':>8}{'total':>8}"]
    for lane in sorted({r.scenario.lane for r in results}):
        group = [r for r in results if r.scenario.lane == lane]
        lines.append(f"{lane:8}{sum(r.passed for r in group):>8}{len(group):>8}")
    latencies = [ms for r in results for ms in r.latencies_ms]
    lines.append(
        f"\nPASS RATE {pass_rate(results):.1%} ({sum(r.passed for r in results)}/{len(results)}) | "
        f"turn latency p50 {_pct(latencies, 0.5):.0f} ms, p95 {_pct(latencies, 0.95):.0f} ms"
    )
    for r in results:
        if not r.passed:
            lines.append(f"\nFAIL {r.scenario.id}: {r.scenario.description}")
            lines += [f"  - {f}" for f in r.failures]
            if r.scenario.needs_llm:
                lines.append(
                    "  (LLM scenario: with llm_retry or interpret_degraded in the "
                    "logs this was a provider rate limit, not a logic failure)"
                )
    return "\n".join(lines)


def write_transcripts(
    results: list[ScenarioResult],
    directory: Path,
    lanes: tuple[str, ...] = ("A", "B", "C", "D", "FAQ"),
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r.scenario.lane not in lanes:
            continue
        blocks = [
            f"# {r.scenario.id}",
            f"{r.scenario.description} ({'PASS' if r.passed else 'FAIL'})",
            "**Bot:** How may I help you?",
        ]
        for visitor, messages in r.transcript:
            blocks.append(f"**Visitor:** {visitor}")
            blocks += [f"**Bot:** {m}" for m in messages]
        (directory / f"{r.scenario.id}.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
