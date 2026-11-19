import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langsmith import Client, evaluate
from pydantic import BaseModel, Field

from lead_qualification_chatbot.app import build_orchestrator
from lead_qualification_chatbot.config import Settings, settings
from lead_qualification_chatbot.graph.orchestrator import ChatOrchestrator, ChatReply

DEFAULT_TRANSCRIPTS = Path(__file__).resolve().parent / "resources" / "transcripts.jsonl"


class ExpectedOutcome(BaseModel):
    qualified: bool
    done: bool
    closing: str | None = None


class Transcript(BaseModel):
    id: str
    turns: list[str] = Field(min_length=1)
    expected: ExpectedOutcome


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    transcript_id: str
    reply: ChatReply


def load_transcripts(path: Path = DEFAULT_TRANSCRIPTS) -> list[Transcript]:
    return [Transcript.model_validate_json(line) for line in path.read_text().splitlines() if line]


async def run_transcript(bot: ChatOrchestrator, transcript: Transcript) -> TranscriptResult:
    session_id = f"eval-{transcript.id}-{uuid4().hex}"
    reply: ChatReply | None = None
    for turn in transcript.turns:
        reply = await bot.chat(session_id, turn)
    if reply is None:
        raise RuntimeError("Transcript must contain at least one turn")
    return TranscriptResult(transcript.id, reply)


def outcome_score(
    *, outputs: dict[str, object], reference_outputs: dict[str, object]
) -> dict[str, object]:
    keys = ("qualified", "done", "closing")
    matched = all(outputs.get(k) == reference_outputs.get(k) for k in keys)
    return {"key": "outcome_match", "score": int(matched)}


def upload_transcripts(
    client: Client, transcripts: Iterable[Transcript], dataset_name: str
) -> None:
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name,
            description="Regression transcripts for lead qualification journeys.",
        )
    client.create_examples(
        dataset_name=dataset_name,
        examples=[
            {
                "inputs": {"turns": transcript.turns},
                "outputs": transcript.expected.model_dump(),
                "metadata": {"transcript_id": transcript.id},
            }
            for transcript in transcripts
        ],
    )


def run_langsmith_evaluation(
    cfg: Settings = settings, dataset_name: str = "lead-qualification"
) -> None:
    """Upload the local corpus and run deterministic outcome checks in LangSmith."""
    transcripts = load_transcripts()
    client = Client(
        api_url=cfg.langsmith_endpoint,
        api_key=cfg.langsmith_api_key.get_secret_value(),
    )
    upload_transcripts(client, transcripts, dataset_name)

    async def target(inputs: dict[str, object]) -> dict[str, object]:
        bot = await build_orchestrator(cfg)
        turns = inputs.get("turns")
        if not isinstance(turns, list) or not all(isinstance(turn, str) for turn in turns):
            raise ValueError("Dataset input 'turns' must be a list of strings")
        transcript = Transcript(
            id="langsmith",
            turns=turns,
            expected=ExpectedOutcome(qualified=False, done=False),
        )
        result = await run_transcript(bot, transcript)
        return {
            "qualified": result.reply.qualified,
            "done": result.reply.done,
            "closing": result.reply.closing,
        }

    def sync_target(inputs: dict[str, object]) -> dict[str, object]:
        return asyncio.run(target(inputs))

    evaluate(
        sync_target,
        data=dataset_name,
        evaluators=[outcome_score],
        experiment_prefix="lead-qualification-regression",
        client=client,
    )


def transcripts_as_json(path: Path = DEFAULT_TRANSCRIPTS) -> str:
    return json.dumps([t.model_dump() for t in load_transcripts(path)], indent=2)
