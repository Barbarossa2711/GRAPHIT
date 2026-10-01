from __future__ import annotations

from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from Multiagent.llm_endpoint import build_chat_openai, use_openai_default

from .prompts import RECOMMENDER_PROMPT, SUPERVISOR_PROMPT, TUTOR_PROMPT
from .state import RuntimeContext
from .tools import (
    find_concept,
    get_concept_material,
    next_steps,
    recommend_next,
)


def build_app(model: str | None = None, *, use_openai: bool | None = None):
    """
    Compiles the supervisor app with tutor and recommender agent.

    The endpoint comes from the ``LLM_*`` variables of the .env. ``temperature`` is
    deliberately left at the LangChain default. ``student_id`` is passed per invoke as
    context, e.g. ``app.invoke({"messages": [...]}, context={"student_id": "..."})``.

    :param model: Model name for supervisor and workers; the configured default when omitted.
    :param use_openai: ``True`` routes the chat to api.openai.com for a comparison run;
        defaults to ``SUPERVISOR_USE_OPENAI`` from the .env.
    :return: The compiled supervisor app.
    """
    if use_openai is None:
        use_openai = use_openai_default()
    llm = build_chat_openai(model=model, use_openai=use_openai)

    tutor_agent = create_agent(
        model=llm,
        name="tutor_agent",
        tools=[find_concept, get_concept_material],
        system_prompt=TUTOR_PROMPT,
        context_schema=RuntimeContext,
    )

    recommender_agent = create_agent(
        model=llm,
        name="recommender_agent",
        tools=[next_steps, find_concept, recommend_next],
        system_prompt=RECOMMENDER_PROMPT,
        context_schema=RuntimeContext,
    )

    workflow = create_supervisor(
        [tutor_agent, recommender_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        context_schema=RuntimeContext,
    )
    return workflow.compile()
