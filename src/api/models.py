"""FastAPI request/response models (interface contract §C5)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.agents.state import Citation


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class InterruptPayload(BaseModel):
    interrupt_id: str
    type: Literal["confirm_strategy"]
    strategy_spec: dict


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["ok", "interrupt"]
    response: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    interrupt: Optional[InterruptPayload] = None


class ResumeRequest(BaseModel):
    thread_id: str
    interrupt_id: str
    edited_spec: dict
