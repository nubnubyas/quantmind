"""Interview Subgraph — parse JD, load profile, generate questions, format output."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agents._utils import last_user_text
from src.agents.state import AgentState, SubgraphOutput
from src.config.llm_client import LLMClient
from src.memory.user_memory import Profile, UserMemory
from src.tools.generate_interview_questions import generate_interview_questions

MIN_JD_LENGTH = 30


class JDParseSchema(BaseModel):
    jd_text: str = Field(description="Full job description text")
    company: str | None = Field(default=None, description="Company name if mentioned")
    focus_areas: list[str] | None = Field(default=None, description="Focus areas to emphasize")


def _merge_focus_areas(
    parsed_areas: list[str] | None,
    profile: Profile | None,
) -> list[str] | None:
    areas: list[str] = list(parsed_areas or [])
    if profile:
        areas.extend(profile.research_interests)
        areas.extend(profile.target_roles)
    if not areas:
        return None
    seen: set[str] = set()
    deduped: list[str] = []
    for item in areas:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.strip())
    return deduped or None


def _format_questions(questions: list[dict]) -> str:
    by_category: dict[str, list[dict]] = {}
    for q in questions:
        category = q.get("category", "general")
        by_category.setdefault(category, []).append(q)

    lines = [f"面试题（共 {len(questions)} 题）", ""]
    for category, items in sorted(by_category.items()):
        lines.append(f"## {category}")
        for i, q in enumerate(items, 1):
            difficulty = q.get("difficulty", "medium")
            lines.append(f"{i}. [{difficulty}] {q.get('question', '')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def compile_interview_subgraph(
    llm_client: LLMClient,
    memory: UserMemory | None = None,
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build and compile the Interview subgraph on AgentState."""
    mem = memory or UserMemory()
    _ctx: dict = {}

    def parse_jd(state: AgentState) -> dict:
        _ctx.clear()
        user_text = last_user_text(state)
        try:
            raw = llm_client.chat_structured(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract job description details from the user message. "
                            "Return jd_text, optional company, and optional focus_areas."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                JDParseSchema,
            )
            jd_text = (raw.get("jd_text") or "").strip()
            if len(jd_text) < MIN_JD_LENGTH:
                _ctx["error"] = "Job description is too short to generate interview questions"
                return {}
            _ctx["jd"] = raw
        except Exception as exc:  # noqa: BLE001
            _ctx["error"] = str(exc)
        return {}

    def load_profile(state: AgentState) -> dict:
        if _ctx.get("error"):
            return {}
        _ctx["profile"] = mem.get_profile(state.get("user_id", "eval_user"))
        return {}

    def generate_questions(state: AgentState) -> dict:
        if _ctx.get("error"):
            return {}
        jd = _ctx.get("jd") or {}
        profile = _ctx.get("profile")
        focus_areas = _merge_focus_areas(jd.get("focus_areas"), profile)
        result = generate_interview_questions(
            jd_text=jd.get("jd_text", ""),
            company=jd.get("company"),
            focus_areas=focus_areas,
            llm=llm_client,
        )
        if not result.ok:
            _ctx["error"] = result.error or "Failed to generate interview questions"
            return {}
        _ctx["questions"] = result.data
        return {}

    def format_questions(state: AgentState) -> dict:
        error = _ctx.get("error")
        if error:
            final = f"无法生成面试题：{error}"
            subgraph_output: SubgraphOutput = {
                "mode": "interview",
                "result": final,
                "citations": [],
                "error": error,
            }
            return {
                "final_response": final,
                "subgraph_outputs": {"interview": subgraph_output},
            }

        data = _ctx.get("questions") or {}
        questions = data.get("questions") or []
        final = _format_questions(questions)
        subgraph_output = {
            "mode": "interview",
            "result": final,
            "citations": [],
            "error": None,
        }
        return {
            "final_response": final,
            "subgraph_outputs": {"interview": subgraph_output},
        }

    builder = StateGraph(AgentState)
    builder.add_node("parse_jd", parse_jd)
    builder.add_node("load_profile", load_profile)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("format_questions", format_questions)

    builder.add_edge(START, "parse_jd")
    builder.add_edge("parse_jd", "load_profile")
    builder.add_edge("load_profile", "generate_questions")
    builder.add_edge("generate_questions", "format_questions")
    builder.add_edge("format_questions", END)

    return builder.compile(checkpointer=checkpointer)
