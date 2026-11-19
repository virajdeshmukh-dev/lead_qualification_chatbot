import hashlib
import json
from typing import Any

from langsmith import Client, aevaluate

from lead_qualification_chatbot.evals.models import Scenario
from lead_qualification_chatbot.evals.runner import Chatter, run_scenario


def dataset_name(scenarios: list[Scenario]) -> str:
    payload = json.dumps([s.model_dump() for s in scenarios], sort_keys=True)
    return f"estate-scenarios-{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def ensure_dataset(client: Client, scenarios: list[Scenario]) -> str:
    """Content-addressed: changed scenarios create a new dataset version."""
    name = dataset_name(scenarios)
    if not client.has_dataset(dataset_name=name):
        dataset = client.create_dataset(name, description="Lead qualification conversation scenarios")
        client.create_examples(
            inputs=[{"scenario": s.model_dump()} for s in scenarios],
            outputs=[{"id": s.id, "lane": s.lane} for s in scenarios],
            dataset_id=dataset.id,
        )
    return name


async def run_experiment(
    bot: Chatter, scenarios: list[Scenario], client: Client, prefix: str = "lead-qualification"
) -> str:
    name = ensure_dataset(client, scenarios)

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = await run_scenario(bot, Scenario.model_validate(inputs["scenario"]))
        return {
            "passed": result.passed,
            "failures": result.failures,
            "max_turn_latency_ms": max(result.latencies_ms, default=0.0),
        }

    def scenario_pass(run: Any, example: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"key": "scenario_pass", "score": 1.0 if (run.outputs or {}).get("passed") else 0.0}

    await aevaluate(
        target,
        data=name,
        evaluators=[scenario_pass],
        experiment_prefix=prefix,
        client=client,
        max_concurrency=1,
    )
    return name
