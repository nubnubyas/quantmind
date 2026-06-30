"""§C6 LLM Client abstraction (interface contract v1.0)."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

try:
    from fastembed import TextEmbedding
except ImportError:  # pragma: no cover
    TextEmbedding = None  # type: ignore[misc, assignment]


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict
    latency_ms: int


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)


def _to_lc_messages(messages: list[dict]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _extract_usage(response: AIMessage) -> dict:
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if not usage and hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage = {
            "input_tokens": getattr(um, "input_tokens", 0) or um.get("input_tokens", 0),
            "output_tokens": getattr(um, "output_tokens", 0) or um.get("output_tokens", 0),
            "total_tokens": getattr(um, "total_tokens", 0) or um.get("total_tokens", 0),
        }
    return {
        "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0))),
        "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0))),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


class LLMClient:
    """DeepSeek V3 via OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        verify_fail_once: bool = False,
    ) -> None:
        _load_env()
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._default_model = default_model or os.getenv("LLM_FAST", "deepseek-v4-flash")
        self._strong_model = os.getenv("LLM_STRONG", "deepseek-v4-pro")
        self._verify_fail_once = verify_fail_once
        self._verify_failed = False

    def _chat_model(self, model: str | None = None, **kw: Any) -> ChatOpenAI:
        return ChatOpenAI(
            model=model or self._default_model,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=kw.get("temperature", 0.2),
            **{k: v for k, v in kw.items() if k != "temperature"},
        )

    def chat(self, messages: list[dict], model: str | None = None, **kw: Any) -> LLMResult:
        llm = self._chat_model(model, **kw)
        start = time.perf_counter()
        response = llm.invoke(_to_lc_messages(messages))
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = response.content if isinstance(response.content, str) else str(response.content)
        used_model = model or self._default_model
        return LLMResult(
            text=text,
            model=used_model,
            usage=_extract_usage(response),
            latency_ms=latency_ms,
        )

    def chat_structured(
        self,
        messages: list[dict],
        schema: type[BaseModel] | dict,
        model: str | None = None,
    ) -> dict:
        if self._verify_fail_once and not self._verify_failed:
            if isinstance(schema, type) and schema.__name__ == "VerificationJudgeSchema":
                self._verify_failed = True
                return {
                    "answers_question": False,
                    "claims_grounded": False,
                    "no_hallucination": False,
                    "uncertainty_stated": False,
                    "score": 0.3,
                    "failure_reasons": ["simulated first verification failure for retry demo"],
                }

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        else:
            schema_hint = json.dumps(schema, ensure_ascii=False)

        json_messages = list(messages) + [
            {
                "role": "system",
                "content": (
                    "Respond with a single valid JSON object only, no markdown. "
                    f"Schema: {schema_hint}"
                ),
            }
        ]
        llm_json = self._chat_model(model, model_kwargs={"response_format": {"type": "json_object"}})
        response = llm_json.invoke(_to_lc_messages(json_messages))
        content = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(content)

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                validated = schema.model_validate(data)
                data = validated.model_dump()
            except Exception as e:
                defaults_filled = {}
                for field_name, field_info in schema.model_fields.items():
                    if field_name not in data:
                        if field_info.default is not None and field_info.default is not ...:
                            defaults_filled[field_name] = field_info.default
                        elif field_info.default_factory is not None:
                            defaults_filled[field_name] = field_info.default_factory()
                data.update(defaults_filled)
                if defaults_filled:
                    logging.warning(
                        "chat_structured: filled missing fields %s for %s (LLM omitted them). "
                        "Validation error: %s",
                        list(defaults_filled.keys()),
                        schema.__name__,
                        e,
                    )
        return data

    def chat_stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        llm = self._chat_model(model)
        for chunk in llm.stream(_to_lc_messages(messages)):
            if chunk.content:
                yield chunk.content if isinstance(chunk.content, str) else str(chunk.content)

    @property
    def strong_model(self) -> str:
        return self._strong_model


class EmbeddingClient:
    model: str
    dim: int

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        _load_env()
        self.model = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        if TextEmbedding is None:
            raise ImportError("fastembed is required for EmbeddingClient")
        self._embedder = TextEmbedding(model_name=self.model)
        if dim is not None:
            self.dim = dim
        else:
            test_vec = list(self._embedder.embed(["test"]))[0]
            self.dim = len(test_vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._embedder.embed(texts)]


def create_llm_client(**kwargs: Any) -> LLMClient:
    return LLMClient(**kwargs)


def create_embedding_client(**kwargs: Any) -> EmbeddingClient:
    return EmbeddingClient(**kwargs)
