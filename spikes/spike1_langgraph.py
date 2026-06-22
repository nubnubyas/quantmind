"""
Spike 1: LangGraph 四件套验证
=================================
目标：在写核心代码前，用最小 demo 验证 4 个机制：
  Q1. Subgraph 组合 — 父图如何调用子图，State 字段如何映射？
  Q2. interrupt() + Command(resume=) — 暂停/恢复机制？
  Q3. Checkpointer + Store 共存 — 配置方式和各自职责？
  Q4. Supervisor fan-out (Send) — 跨域并行路由写法？

运行：
    cd quant-project
    source .venv/bin/activate
    python spikes/spike1_langgraph.py

结论写在文件末尾，也会打印到 stdout。
"""

from __future__ import annotations

import os
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Send, interrupt

DIVIDER = "\n" + "=" * 60 + "\n"


# ──────────────────────────────────────────────────────────────
# Q1: Subgraph 组合
# ──────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """Research 子图的私有状态（键名与父图可以不同）。"""
    query: str
    answer: str


def search_papers(state: ResearchState) -> dict:
    return {"answer": f"[Hybrid search result for: {state['query']}]"}


def verify_answer(state: ResearchState) -> dict:
    return {"answer": state["answer"] + " ✓verified"}


def _build_research_subgraph() -> StateGraph:
    b = StateGraph(ResearchState)
    b.add_node("search", search_papers)
    b.add_node("verify", verify_answer)
    b.add_edge(START, "search")
    b.add_edge("search", "verify")
    b.add_edge("verify", END)
    return b.compile()


# Parent graph 通过 wrapper node 调用子图（显式字段映射）
class ParentState(TypedDict):
    user_query: str
    research_answer: str   # 接收子图结果
    final_response: str


research_subgraph = _build_research_subgraph()


def call_research(state: ParentState) -> dict:
    """Wrapper node: 显式映射 parent_key <-> subgraph_key。"""
    result = research_subgraph.invoke(
        {"query": state["user_query"], "answer": ""},
    )
    return {"research_answer": result["answer"]}


def format_response(state: ParentState) -> dict:
    return {"final_response": f"QuantMind answer: {state['research_answer']}"}


def test_q1_subgraph():
    print(DIVIDER + "Q1: Subgraph 组合")
    b = StateGraph(ParentState)
    b.add_node("research", call_research)
    b.add_node("format", format_response)
    b.add_edge(START, "research")
    b.add_edge("research", "format")
    b.add_edge("format", END)

    graph = b.compile()
    result = graph.invoke(
        {"user_query": "momentum factor", "research_answer": "", "final_response": ""}
    )
    print("结果:", result["final_response"])
    print("""
✅ 结论 Q1:
  - 键名不同时：用 wrapper node 做显式字段映射（research_subgraph.invoke({"query": ...})）
  - 键名相同时：可直接 builder.add_node("sub", compiled_subgraph) 自动映射
  - 推荐方案：QuantMind 用 wrapper node，字段映射清晰可控
    """)


# ──────────────────────────────────────────────────────────────
# Q2: interrupt() + Command(resume=)
# ──────────────────────────────────────────────────────────────

class CodeGenState(TypedDict):
    strategy: str
    confirmed_spec: Optional[dict]
    generated_code: str


def parse_strategy(state: CodeGenState) -> dict:
    spec = {"signal": "MA crossover", "period": "20/60", "asset": "equity"}
    return {"confirmed_spec": spec}


def confirm_with_user(state: CodeGenState) -> dict:
    """
    INTERRUPT 节点：暂停执行，把 strategy_spec 交给用户确认。
    resume 时 interrupt() 的返回值 = Command(resume=...) 中传入的 value。
    """
    user_response = interrupt({"strategy_spec": state["confirmed_spec"]})
    # user_response 就是用户确认/修改后的 spec
    return {"confirmed_spec": user_response}


def generate_code(state: CodeGenState) -> dict:
    spec = state["confirmed_spec"]
    code = f"# Backtrader strategy\n# Signal: {spec.get('signal')}, Period: {spec.get('period')}\npass"
    return {"generated_code": code}


def test_q2_interrupt():
    print(DIVIDER + "Q2: interrupt() + Command(resume=)")
    b = StateGraph(CodeGenState)
    b.add_node("parse", parse_strategy)
    b.add_node("confirm", confirm_with_user)
    b.add_node("generate", generate_code)
    b.add_edge(START, "parse")
    b.add_edge("parse", "confirm")
    b.add_edge("confirm", "generate")
    b.add_edge("generate", END)

    checkpointer = InMemorySaver()
    graph = b.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "codegen-demo-1"}}

    # 第一次 invoke：在 confirm 节点暂停
    r1 = graph.invoke(
        {"strategy": "20/60 MA", "confirmed_spec": None, "generated_code": ""},
        config=config,
    )
    interrupts = r1.get("__interrupt__", [])
    print("第一次 invoke 后：")
    print("  __interrupt__ value:", interrupts[0].value if interrupts else "(none)")
    print("  next nodes:", graph.get_state(config).next)

    # 用户看到 interrupt value 后，修改 spec 并 resume
    edited_spec = {**interrupts[0].value["strategy_spec"], "approved": True}
    r2 = graph.invoke(Command(resume=edited_spec), config=config)
    print("\nresume 后：")
    print("  confirmed_spec:", r2.get("confirmed_spec"))
    print("  generated_code:", r2.get("generated_code"))
    print("""
✅ 结论 Q2:
  - interrupt() 不抛异常，返回值放在 __interrupt__ 键（list[Interrupt]）
  - graph.get_state(config).next 可看到挂起的节点
  - Command(resume=value) 恢复执行，value 成为 interrupt() 的返回值
  - Streamlit 端：检测到 __interrupt__ -> 渲染确认表单 -> 用户提交后调 invoke(Command(resume=...))
  - thread_id 是持久化的 session 标识，必须用 checkpointer
    """)


# ──────────────────────────────────────────────────────────────
# Q3: Checkpointer + Store 共存
# ──────────────────────────────────────────────────────────────

class MemoryState(TypedDict):
    user_id: str
    query: str
    answer: str


def write_to_store(state: MemoryState) -> dict:
    """写入 Store（跨线程长期记忆）。用 get_store() 获取注入的 store 实例。"""
    store = get_store()
    store.put(
        ("user", state["user_id"], "profile"),
        "research_interests",
        {"interests": ["momentum", "stat_arb"], "updated": "2026-06-20"},
    )
    return {"answer": "stored user profile"}


def read_from_store(state: MemoryState) -> dict:
    """从 Store 读取用户 profile，注入个性化上下文。"""
    store = get_store()
    results = store.search(("user", state["user_id"], "profile"))
    profile = results[0].value if results else {}
    interests = profile.get("interests", [])
    return {"answer": f"Personalized for interests: {interests}"}


def test_q3_checkpointer_store():
    print(DIVIDER + "Q3: Checkpointer + Store 共存")

    # InMemoryStore (开发) / 生产替换为 PostgresStore
    store = InMemoryStore()
    # InMemorySaver (测试) / 生产替换为 SqliteSaver 或 PostgresSaver
    checkpointer = InMemorySaver()

    b = StateGraph(MemoryState)
    b.add_node("write", write_to_store)
    b.add_node("read", read_from_store)
    b.add_edge(START, "write")
    b.add_edge("write", "read")
    b.add_edge("read", END)

    graph = b.compile(checkpointer=checkpointer, store=store)

    # Thread 1: 写入 profile
    r1 = graph.invoke(
        {"user_id": "user_alice", "query": "momentum", "answer": ""},
        config={"configurable": {"thread_id": "session-1"}},
    )
    print("Thread 1 result:", r1["answer"])

    # Thread 2（不同 session，相同 user_id）：读取同一 profile
    r2 = graph.invoke(
        {"user_id": "user_alice", "query": "stat arb", "answer": ""},
        config={"configurable": {"thread_id": "session-2"}},
    )
    print("Thread 2 result:", r2["answer"])
    print("""
✅ 结论 Q3:
  - compile(checkpointer=cp, store=store) 同时接受两个参数，共存无冲突
  - 节点内用 get_store() 获取 store 实例（不是函数参数注入）
  - Checkpointer 管同一 thread_id 的对话历史（短期，线程级）
  - Store 管跨 thread_id 的用户数据（长期，用户级）
  - 生产：SqliteSaver.from_conn_string(path) 必须用 with 上下文管理器
  - Store.put(namespace_tuple, key, value_dict)
  - Store.search(namespace_prefix_tuple) -> list[SearchItem]，item.value 取数据
    """)

    # 演示 SqliteSaver 的正确用法
    print("SqliteSaver 用法演示（生产模式）：")
    with SqliteSaver.from_conn_string(":memory:") as sqlite_cp:
        g2 = StateGraph(MemoryState)
        g2.add_node("w", write_to_store)
        g2.add_edge(START, "w")
        g2.add_edge("w", END)
        graph2 = g2.compile(checkpointer=sqlite_cp, store=InMemoryStore())
        r3 = graph2.invoke(
            {"user_id": "u1", "query": "q", "answer": ""},
            config={"configurable": {"thread_id": "t1"}},
        )
        history = list(graph2.get_state_history({"configurable": {"thread_id": "t1"}}))
        print(f"  ✅ SqliteSaver with context manager: {len(history)} checkpoints persisted")


# ──────────────────────────────────────────────────────────────
# Q4: Supervisor fan-out with Send (multi-mode routing)
# ──────────────────────────────────────────────────────────────

def _merge_outputs(a: dict, b: dict) -> dict:
    """Reducer：合并并发分支写入同一 key 的 dict。"""
    return {**a, **b}


class OrchestratorState(TypedDict):
    query: str
    modes: list[str]
    multi_mode: bool
    subgraph_outputs: Annotated[dict, _merge_outputs]  # 并发写入需要 reducer


def research_handler(state: OrchestratorState) -> dict:
    return {"subgraph_outputs": {"research": f"[Research] {state['query']}"}}


def interview_handler(state: OrchestratorState) -> dict:
    return {"subgraph_outputs": {"interview": f"[Interview] {state['query']}"}}


def merge_node(state: OrchestratorState) -> dict:
    outputs = state["subgraph_outputs"]
    combined = "\n".join(f"  [{k}]: {v}" for k, v in outputs.items())
    return {"query": f"Combined:\n{combined}"}


def supervisor_route(state: OrchestratorState) -> list[Send]:
    """fan-out：根据 modes 并行发送到多个子图。"""
    if state["multi_mode"]:
        return [Send("research", state), Send("interview", state)]
    mode = state["modes"][0] if state["modes"] else "research"
    return [Send(mode, state)]


def test_q4_fanout():
    print(DIVIDER + "Q4: Supervisor fan-out (Send)")
    b = StateGraph(OrchestratorState)
    b.add_node("supervisor", lambda s: s)  # 路由前的 pass-through
    b.add_node("research", research_handler)
    b.add_node("interview", interview_handler)
    b.add_node("merge", merge_node)
    b.add_conditional_edges("supervisor", supervisor_route)
    b.add_edge("research", "merge")
    b.add_edge("interview", "merge")
    b.add_edge(START, "supervisor")
    b.add_edge("merge", END)

    graph = b.compile()

    # 单域
    r1 = graph.invoke({
        "query": "动量因子原理", "modes": ["research"],
        "multi_mode": False, "subgraph_outputs": {},
    })
    print("单域 (research only):", r1["query"])

    # 多域（S9 场景）
    r2 = graph.invoke({
        "query": "Citadel quant role + 我的动量项目",
        "modes": ["research", "interview"],
        "multi_mode": True,
        "subgraph_outputs": {},
    })
    print("多域 (research + interview):", r2["query"])
    print("""
✅ 结论 Q4:
  - Send(node_name, state) 用于 fan-out，从 conditional_edges 返回 list[Send]
  - 并发分支写同一 State key 会报 InvalidUpdateError，需要 Annotated[T, reducer]
  - 推荐 reducer：Annotated[dict, merge_dicts] 合并两个分支的 subgraph_outputs
  - merge 节点接收所有分支合并后的 state
  - 从 langgraph.types import Send（constants 里的已 deprecated）
    """)


# ──────────────────────────────────────────────────────────────
# 汇总结论
# ──────────────────────────────────────────────────────────────

CONCLUSIONS = """
╔══════════════════════════════════════════════════════════════╗
║          Spike 1 结论汇总（LangGraph 1.2.6）                 ║
╚══════════════════════════════════════════════════════════════╝

Q1  子图 State 映射
    → 键名不同：用 wrapper node 显式调用 subgraph.invoke({"key": ...})
    → 键名相同：可直接 add_node("sub", compiled_subgraph) 自动映射
    → QuantMind 采用 wrapper node 方案（清晰可控）

Q2  interrupt() + Command(resume=)
    → interrupt() 不抛异常，结果放在 invoke 返回值的 __interrupt__ 键
    → Command(resume=value) 恢复，value 成为 interrupt() 的返回值
    → 需要 checkpointer 才能跨 invoke 调用 resume
    → Streamlit 端：检测 __interrupt__ -> 渲染表单 -> POST /resume

Q3  Checkpointer + Store 共存
    → compile(checkpointer=cp, store=store) 两个参数同时传，无冲突
    → 节点内用 get_store() 获取 store（不是参数注入）
    → SqliteSaver 必须用 with 上下文管理器
    → Store.put(namespace, key, dict) / Store.search(namespace_prefix)
    → 三层分离：Checkpointer(线程短期) / Store(跨线程长期) / PostgreSQL(业务)

Q4  Supervisor fan-out
    → conditional_edges 返回 list[Send] 实现 fan-out
    → 并发分支写同一 key 必须加 Annotated[T, reducer]
    → from langgraph.types import Send（已从 constants 迁移）

对接口契约的影响：
    - C0 AgentState.subgraph_outputs 需要 Annotated[dict, merge_dicts] reducer
    - C5 FastAPI /chat 返回 __interrupt__ 原样透传，/resume 调 Command(resume=)
    - SqliteSaver 用法（with 上下文）需要在 FastAPI lifespan 中管理
"""


if __name__ == "__main__":
    test_q1_subgraph()
    test_q2_interrupt()
    test_q3_checkpointer_store()
    test_q4_fanout()
    print(CONCLUSIONS)
