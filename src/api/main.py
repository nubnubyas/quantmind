"""FastAPI integration layer for the QuantMind LangGraph agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.agents.codegen_agent import compile_codegen_subgraph
from src.agents.interview_agent import compile_interview_subgraph
from src.agents.planning_agent import compile_planning_subgraph
from src.agents.research_agent import compile_research_subgraph
from src.agents.supervisor import compile_parent_graph
from src.api.models import ChatRequest, ChatResponse, InterruptPayload, ResumeRequest
from src.config.llm_client import LLMClient
from src.memory import UserMemory
from src.sandbox import SandboxRunner
from src.vector_store.qdrant_client_wrapper import VectorStore


def _strategy_spec_from_interrupt_value(value: Any) -> dict:
    if isinstance(value, dict) and "strategy_spec" in value:
        return value["strategy_spec"]
    if isinstance(value, dict):
        return value
    return {"value": value}


def _state_to_response(thread_id: str, state: dict) -> ChatResponse:
    interrupts = state.get("__interrupt__", [])
    if interrupts:
        iv = interrupts[0]
        return ChatResponse(
            thread_id=thread_id,
            status="interrupt",
            interrupt=InterruptPayload(
                interrupt_id=iv.id,
                type="confirm_strategy",
                strategy_spec=_strategy_spec_from_interrupt_value(iv.value),
            ),
        )
    return ChatResponse(
        thread_id=thread_id,
        status="ok",
        response=state.get("final_response"),
        citations=state.get("citations") or [],
    )


def create_app(*, graph: CompiledStateGraph | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if graph is not None:
            app.state.graph = graph
            yield
        else:
            with SqliteSaver.from_conn_string("quantmind_dev.sqlite") as checkpointer:
                vector_store = VectorStore()
                llm_client = LLMClient()
                sandbox_runner = SandboxRunner()
                memory = UserMemory()

                research = compile_research_subgraph(
                    vector_store, llm_client, checkpointer=checkpointer
                )
                codegen = compile_codegen_subgraph(
                    llm_client, sandbox_runner, checkpointer=checkpointer
                )
                planning = compile_planning_subgraph(
                    llm_client, memory, checkpointer=checkpointer
                )
                interview = compile_interview_subgraph(
                    llm_client, memory, checkpointer=checkpointer
                )

                app.state.graph = compile_parent_graph(
                    research,
                    codegen,
                    planning,
                    interview,
                    llm_client,
                    checkpointer=checkpointer,
                )
                yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        graph = request.app.state.graph
        config = {"configurable": {"thread_id": req.thread_id}}
        state = graph.invoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "user_id": req.user_id,
            },
            config=config,
        )
        return _state_to_response(req.thread_id, state)

    @app.post("/resume", response_model=ChatResponse)
    async def resume(req: ResumeRequest, request: Request) -> ChatResponse:
        graph = request.app.state.graph
        config = {"configurable": {"thread_id": req.thread_id}}
        state = graph.invoke(
            Command(resume={req.interrupt_id: req.edited_spec}),
            config=config,
        )
        return _state_to_response(req.thread_id, state)

    return app


app = create_app()
