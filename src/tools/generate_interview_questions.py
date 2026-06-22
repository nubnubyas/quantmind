"""T5: Generate interview questions from a job description."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config.llm_client import LLMClient, create_llm_client
from src.tools._helpers import fail, ok
from src.tools.types import ToolResult

TOOL_NAME = "generate_interview_questions"


class InterviewQuestion(BaseModel):
    question: str
    category: str
    difficulty: str


class InterviewQuestionsSchema(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)


def generate_interview_questions(
    jd_text: str,
    company: str | None = None,
    focus_areas: list[str] | None = None,
    num_questions: int = 10,
    *,
    llm: LLMClient | None = None,
) -> ToolResult:
    if focus_areas is not None and len(focus_areas) == 0:
        focus_areas = None

    client = llm or create_llm_client()
    user_parts = [
        f"Generate exactly {num_questions} technical interview questions for this job description:",
        jd_text,
    ]
    if company:
        user_parts.append(f"Company: {company}")
    if focus_areas:
        user_parts.append(f"Focus areas: {', '.join(focus_areas)}")

    try:
        parsed = client.chat_structured(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate technical interview questions for quantitative finance / "
                        "ML engineering roles. Each question needs question, category, "
                        "and difficulty (easy/medium/hard)."
                    ),
                },
                {"role": "user", "content": "\n\n".join(user_parts)},
            ],
            InterviewQuestionsSchema,
        )
    except Exception as exc:  # noqa: BLE001
        return fail(TOOL_NAME, str(exc), "LLM_ERROR", retryable=True)

    questions = parsed.get("questions", [])
    return ok(
        TOOL_NAME,
        "interview_questions",
        questions=questions,
        count=len(questions),
    )
