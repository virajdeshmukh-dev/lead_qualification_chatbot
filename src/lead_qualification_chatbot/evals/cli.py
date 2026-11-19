import argparse
import asyncio
import json
from pathlib import Path

from lead_qualification_chatbot.app import build_orchestrator
from lead_qualification_chatbot.config import settings
from lead_qualification_chatbot.data.repositories import load_data
from lead_qualification_chatbot.evals.fakes import build_offline_bot
from lead_qualification_chatbot.evals.runner import (
    Chatter,
    ScenarioResult,
    format_report,
    pass_rate,
    run_scenario,
    write_transcripts,
)
from lead_qualification_chatbot.evals.scenarios import build_scenarios
from lead_qualification_chatbot.leads.sink import MemoryLeadSink
from lead_qualification_chatbot.logging_config import setup_logging
from lead_qualification_chatbot.observability import build_tracer


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scripted conversations and report a pass rate")
    p.add_argument(
        "--mode",
        choices=["fake", "live"],
        default="fake",
        help="fake: no LLM, flow scenarios only. live: real LLM, all scenarios",
    )
    p.add_argument("--only", help="lane (e.g. A) or id substring")
    p.add_argument("--min-pass", type=float, help="exit 1 below this rate (fake 1.0, live 0.85)")
    p.add_argument("--out", type=Path, help="write a JSON report here")
    p.add_argument("--transcripts", type=Path, help="write markdown transcripts here")
    p.add_argument("--langsmith", action="store_true", help="also run a LangSmith experiment")
    p.add_argument(
        "--throttle",
        type=float,
        default=0.0,
        help="seconds to pause after each LLM scenario (free-tier rate limits)",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    setup_logging()
    data = load_data(settings.data_dir)
    scenarios = build_scenarios(data, settings.office_phone)
    sink: MemoryLeadSink | None = None
    if args.only:
        scenarios = [
            s for s in scenarios if args.only.lower() in s.id.lower() or args.only.upper() == s.lane
        ]
    bot: Chatter
    if args.mode == "fake":
        bot = build_offline_bot(data, settings.office_phone)
        scenarios = [s for s in scenarios if not s.needs_llm]
    else:
        sink = MemoryLeadSink()
        bot = await build_orchestrator(settings, lead_sink=sink)

    results: list[ScenarioResult] = []
    for scenario in scenarios:  # sequential: local models handle one request at a time
        results.append(await run_scenario(bot, scenario))
        if scenario.needs_llm and args.throttle:
            await asyncio.sleep(args.throttle)
    print(format_report(results))
    if sink is not None:
        print(f"leads captured: {len(sink.leads)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "model": f"{settings.llm_provider}:{settings.llm_model}",
                    "pass_rate": pass_rate(results),
                    "results": [
                        {
                            "id": r.scenario.id,
                            "lane": r.scenario.lane,
                            "passed": r.passed,
                            "failures": r.failures,
                            "latencies_ms": r.latencies_ms,
                        }
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.transcripts:
        write_transcripts(results, args.transcripts)
    if args.langsmith:
        tracer = build_tracer(settings)
        if tracer is None:
            print("LangSmith is not enabled (see LANGSMITH_* settings); skipping experiment.")
        else:
            from lead_qualification_chatbot.evals.langsmith_sync import run_experiment

            print("LangSmith dataset:", await run_experiment(bot, scenarios, tracer.client))

    threshold = (
        args.min_pass if args.min_pass is not None else (1.0 if args.mode == "fake" else 0.85)
    )  # noqa: E501
    return 0 if pass_rate(results) >= threshold else 1


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(_run(_args(argv))))


if __name__ == "__main__":
    main()
