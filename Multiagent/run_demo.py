from __future__ import annotations

import logging
import sys

from Multiagent.agents import build_app

sys.stdout.reconfigure(encoding="utf-8")

STUDENT_ID = "test_student"


def main() -> None:
    """
    Sends one question to the supervisor app without server or frontend and prints the conversation.

    Usage: ``python -m Multiagent.run_demo ["<question>"]``.
    """
    logging.basicConfig(level=logging.INFO)
    question = sys.argv[1] if len(sys.argv) > 1 else "Erkläre mir Sharding."

    app = build_app()
    result = app.invoke(
        {"messages": [{"role": "user", "content": question}]},
        context={"student_id": STUDENT_ID},
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
