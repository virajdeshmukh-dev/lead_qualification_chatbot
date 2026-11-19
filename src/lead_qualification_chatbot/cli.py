import asyncio
from uuid import uuid4

from lead_qualification_chatbot.app import build_orchestrator
from lead_qualification_chatbot.logging_config import setup_logging


async def run() -> None:
    setup_logging()
    bot = await build_orchestrator()
    session_id = uuid4().hex
    print(f"Bot: {bot.greeting()}")
    while True:
        try:
            text = await asyncio.to_thread(input, "You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in {"/quit", "/exit"}:
            break
        reply = await bot.chat(session_id, text)

        for message in reply.messages:
            print(f"Bot: {message}")
        if reply.done:
            break


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
