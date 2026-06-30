"""Streamlit chat UI for QuantMind (interface contract §C5 consumer)."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import requests
import streamlit as st
from requests.exceptions import RequestException

from src.api.models import ChatResponse, InterruptPayload

API_BASE = os.environ.get("QUANTMIND_API_URL", "http://localhost:8000")
DEFAULT_USER_ID = "default-user"


def api_chat(user_id: str, message: str, thread_id: str) -> ChatResponse | None:
    """POST /chat and return parsed response, or None on failure."""
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"user_id": user_id, "thread_id": thread_id, "message": message},
            timeout=120,
        )
        resp.raise_for_status()
        return ChatResponse(**resp.json())
    except RequestException as e:
        st.error(f"API 请求失败: {e}")
        return None


def api_resume(
    thread_id: str,
    interrupt: InterruptPayload,
    edited_spec: dict,
) -> ChatResponse | None:
    """POST /resume after user confirms interrupt payload."""
    try:
        resp = requests.post(
            f"{API_BASE}/resume",
            json={
                "thread_id": thread_id,
                "interrupt_id": interrupt.interrupt_id,
                "edited_spec": edited_spec,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return ChatResponse(**resp.json())
    except RequestException as e:
        st.error(f"Resume 请求失败: {e}")
        return None


def process_chat_response(response: ChatResponse) -> dict[str, Any]:
    """Map ChatResponse to a message dict for session history."""
    if response.status == "interrupt" and response.interrupt is not None:
        return {
            "role": "assistant",
            "content": "请确认以下策略参数后继续生成代码。",
            "citations": None,
            "interrupt": response.interrupt.model_dump(),
        }
    return {
        "role": "assistant",
        "content": response.response or "",
        "citations": [dict(c) for c in response.citations],
        "interrupt": None,
    }


def build_edited_spec(
    spec: dict,
    name: str,
    framework: str,
    parameters_str: str,
    signal_logic: str,
) -> tuple[dict | None, str | None]:
    """Merge form fields into strategy spec; return (spec, error_message)."""
    try:
        parameters = json.loads(parameters_str)
    except json.JSONDecodeError as e:
        return None, f"参数 JSON 格式错误: {e}"
    if not isinstance(parameters, dict):
        return None, "参数必须是 JSON 对象"
    return {
        **spec,
        "name": name,
        "framework": framework,
        "parameters": parameters,
        "signal_logic": signal_logic,
    }, None


def apply_chat_response(
    response: ChatResponse,
    pending_interrupt: InterruptPayload | None,
) -> tuple[list[dict[str, Any]], InterruptPayload | None]:
    """Apply ChatResponse to message list and pending interrupt state."""
    message = process_chat_response(response)
    pending = pending_interrupt
    if response.status == "interrupt" and response.interrupt is not None:
        pending = response.interrupt
    elif response.status == "ok":
        pending = None
    return [message], pending


def render_citations(citations: list[dict] | None) -> None:
    if not citations:
        return
    with st.expander("📚 References", expanded=False):
        for c in citations:
            title = c.get("title", "N/A")
            score = c.get("relevance_score", 0)
            st.markdown(f"- {title} (relevance={score:.2f})")


def _init_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = DEFAULT_USER_ID


def _start_new_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_interrupt = None


def _render_message(msg: dict[str, Any]) -> None:
    role = msg.get("role", "assistant")
    if role == "user":
        st.markdown(f"**👤 用户:** {msg.get('content', '')}")
    else:
        st.markdown("**🤖 QuantMind:**")
        st.markdown(msg.get("content", ""))
        render_citations(msg.get("citations"))


def _render_interrupt_form() -> None:
    interrupt = st.session_state.pending_interrupt
    if interrupt is None:
        return

    spec = interrupt.strategy_spec
    with st.form("strategy_confirmation"):
        st.subheader("📋 确认策略参数")
        name = st.text_input("策略名称", value=spec.get("name", ""))
        framework_options = ["backtrader", "vectorbt"]
        default_framework = spec.get("framework", "backtrader")
        framework_index = (
            framework_options.index(default_framework)
            if default_framework in framework_options
            else 0
        )
        framework = st.selectbox("回测框架", framework_options, index=framework_index)
        parameters_str = st.text_area(
            "参数 (JSON)",
            value=json.dumps(spec.get("parameters", {}), indent=2, ensure_ascii=False),
        )
        signal_logic = st.text_area(
            "信号逻辑",
            value=spec.get("signal_logic", ""),
            height=100,
        )
        submitted = st.form_submit_button("✅ 确认并生成代码")
        if submitted:
            edited_spec, error = build_edited_spec(
                spec, name, framework, parameters_str, signal_logic
            )
            if error:
                st.error(error)
                return
            response = api_resume(
                st.session_state.thread_id,
                interrupt,
                edited_spec,
            )
            if response is None:
                return
            new_messages, pending = apply_chat_response(
                response, st.session_state.pending_interrupt
            )
            st.session_state.messages.extend(new_messages)
            st.session_state.pending_interrupt = pending
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="QuantMind", page_icon="📈", layout="wide")
    _init_session_state()

    st.title("QuantMind — 量化研究 AI Copilot")

    with st.sidebar:
        st.caption(f"会话 ID: `{st.session_state.thread_id[:8]}...`")
        if st.button("🔄 新对话", use_container_width=True):
            _start_new_conversation()
            st.rerun()

    for msg in st.session_state.messages:
        _render_message(msg)

    if st.session_state.pending_interrupt:
        _render_interrupt_form()
        st.info("请先确认策略参数后再发送新消息。")
    else:
        user_input = st.chat_input("输入消息...")
        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input, "citations": None, "interrupt": None}
            )
            response = api_chat(
                st.session_state.user_id,
                user_input,
                st.session_state.thread_id,
            )
            if response is not None:
                new_messages, pending = apply_chat_response(
                    response, st.session_state.pending_interrupt
                )
                st.session_state.messages.extend(new_messages)
                st.session_state.pending_interrupt = pending
                st.rerun()


if __name__ == "__main__":
    main()
