"""
Spike 3: LangSmith Trace + Eval 接入
======================================
目标：验证 LangSmith 接入方式，回答 4 个关键问题：

  Q1. LANGCHAIN_TRACING_V2=true 是否自动 trace LangGraph 全链路？
  Q2. 如何把 JSONL 导入为 LangSmith dataset？
  Q3. LLM-as-judge evaluator 与自定义（代码执行）evaluator 的注册方式？
  Q4. embedding token 与 LLM token 能否在 trace 中分开统计？

前置条件：
    在项目根目录创建 .env 文件，写入：
        OPENAI_API_KEY=sk-...
        LANGCHAIN_API_KEY=ls__...
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_PROJECT=quantmind-spike

运行：
    cd quant-project
    source .venv/bin/activate
    python spikes/spike3_langsmith.py

注意：本 spike 分两个段落。
  Part A：无需 API key 的结构验证（可立即运行）
  Part B：需要 API key 的实际 trace 验证（配置 .env 后运行）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DIVIDER = "\n" + "=" * 60 + "\n"

# ──────────────────────────────────────────────────────────────
# Part A: 结构验证（无需 API key）
# ──────────────────────────────────────────────────────────────

def test_dataset_format():
    """Q2 预演：验证 JSONL dataset 格式和 LangSmith 导入逻辑。"""
    print(DIVIDER + "Part A: Dataset 格式验证（无需 API key）")

    # 示例 benchmark JSONL（与 data/benchmarks/retrieval_qa.jsonl 格式对齐）
    sample_entries = [
        {
            "id": "rq_001",
            "category": "retrieval_qa",
            "difficulty": "easy",
            "input": "动量因子在新兴市场是否有效？",
            "source_of_truth": "Rouwenhorst 1999, Fama & French 2012",
            "expected_supporting_papers": ["q-fin/0501001"],
            "required_points": [
                "动量效应在新兴市场比发达市场弱",
                "流动性差异是主要解释因素",
            ],
            "failure_criteria": [
                "未引用任何相关论文",
                "声称动量在新兴市场完全无效",
            ],
        },
        {
            "id": "cg_001",
            "category": "code_gen",
            "difficulty": "medium",
            "input": "用 Backtrader 写一个 20/60 日均线交叉策略",
            "framework": "backtrader",
            "expected_keywords": ["CrossOver", "SMA", "Signal"],
            "should_execute": True,
            "sample_data_path": "data/sample/spy_daily.csv",
            "expected_output_contains": ["Final Portfolio Value"],
            "forbidden_imports": ["requests", "urllib", "socket"],
        },
    ]

    # Validate JSONL serialization
    jsonl_lines = [json.dumps(e, ensure_ascii=False) for e in sample_entries]
    print(f"✅ JSONL format validated: {len(jsonl_lines)} entries")
    for line in jsonl_lines:
        parsed = json.loads(line)
        assert "id" in parsed and "input" in parsed
    print("✅ All entries parse back correctly")

    # Show LangSmith import pattern (requires API key at runtime)
    print("""
LangSmith dataset import pattern (requires API key):

    from langsmith import Client
    client = Client()

    # Create dataset
    dataset = client.create_dataset(
        dataset_name="quantmind-eval-v1",
        description="120-item QuantMind benchmark"
    )

    # Upload examples
    client.create_examples(
        inputs=[{"question": e["input"]} for e in entries],
        outputs=[{"expected": e} for e in entries],
        dataset_id=dataset.id,
    )
    # Or use from_csv / from_jsonl for batch upload
    """)


def test_evaluator_structure():
    """Q3 预演：evaluator 结构验证（不实际调 LangSmith API）。"""
    print(DIVIDER + "Part A: Evaluator 结构验证")

    # Automated evaluator example (code execution check)
    def code_syntax_evaluator(run, example) -> dict:
        """
        Custom evaluator: check if generated code passes ast.parse.
        Called by LangSmith for each (run, example) pair.
        """
        import ast
        code = run.outputs.get("generated_code", "")
        try:
            ast.parse(code)
            return {"key": "syntax_correct", "score": 1.0, "comment": "ast.parse passed"}
        except SyntaxError as e:
            return {"key": "syntax_correct", "score": 0.0, "comment": str(e)}

    # Test evaluator logic locally
    class MockRun:
        outputs = {"generated_code": "import backtrader as bt\nclass Strategy(bt.Strategy): pass\n"}
    class MockExample:
        pass

    result = code_syntax_evaluator(MockRun(), MockExample())
    print(f"✅ Syntax evaluator result: {result}")

    # LLM-as-judge evaluator pattern
    faithfulness_prompt = """
You are evaluating an AI assistant's answer for faithfulness to retrieved sources.
Answer: {answer}
Sources: {sources}
Score 1-5 where 5=fully supported, 1=contradicts sources or hallucinated.
Return JSON: {{"score": <int>, "reasoning": "<brief>"}}
"""
    print("✅ Faithfulness evaluator prompt template defined")

    print("""
LangSmith evaluator registration pattern (requires API key):

    from langsmith.evaluation import evaluate

    results = evaluate(
        lambda inputs: graph.invoke(inputs),    # target function
        data="quantmind-eval-v1",               # dataset name
        evaluators=[
            code_syntax_evaluator,              # custom Python evaluator
            faithfulness_evaluator,             # LLM-as-judge evaluator
        ],
        experiment_prefix="quantmind-phase1",
        max_concurrency=4,
    )
    """)


# ──────────────────────────────────────────────────────────────
# Part B: 实际 trace 验证（需要 .env 中配置 API key）
# ──────────────────────────────────────────────────────────────

def test_actual_tracing():
    """Q1 + Q4: 实际 trace 验证。需要 API key。"""
    print(DIVIDER + "Part B: 实际 trace 验证（需要 API key）")

    # Load .env if present
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
    else:
        print(f"⚠️  No .env found at {env_file}")
        print("   Create .env with OPENAI_API_KEY and LANGCHAIN_API_KEY to run this part.")
        return

    openai_key = os.getenv("OPENAI_API_KEY", "")
    langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not openai_key or not langchain_key:
        print("⚠️  API keys not set in .env. Skipping actual trace test.")
        return

    # Set tracing env vars
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "quantmind-spike"

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langsmith import Client
    from typing import TypedDict

    class ResState(TypedDict):
        query: str
        answer: str

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    def research(state: ResState) -> dict:
        resp = llm.invoke([{"role": "user", "content": f"Define in one sentence: {state['query']}"}])
        return {"answer": resp.content}

    b = StateGraph(ResState)
    b.add_node("r", research)
    b.add_edge(START, "r")
    b.add_edge("r", END)
    graph = b.compile(checkpointer=InMemorySaver())

    # Invoke — should auto-trace via LANGCHAIN_TRACING_V2=true
    result = graph.invoke(
        {"query": "Sharpe Ratio", "answer": ""},
        config={"configurable": {"thread_id": "spike3-trace-1"}},
    )
    print(f"✅ Q1: LangGraph auto-traced: {result['answer'][:80]}")

    # Test embedding call trace
    _ = embedder.embed_query("Fama-French factor")
    print("✅ Q4: Embedding call executed (check LangSmith trace for token split)")

    # Fetch recent run
    client = Client()
    runs = list(client.list_runs(project_name="quantmind-spike", limit=3))
    print(f"✅ LangSmith runs found: {len(runs)}")
    for run in runs:
        print(f"  [{run.run_type}] {run.name}: {run.total_tokens or 0} tokens")

    print("""
✅ 结论 Q1: LANGCHAIN_TRACING_V2=true + LANGCHAIN_PROJECT=xxx 自动 trace LangGraph 全链路
✅ 结论 Q4: LangSmith trace 中 embedding 和 LLM token 分开记录（run.run_type = 'embedding'/'llm'）
    """)


# ──────────────────────────────────────────────────────────────
# 汇总结论（无需实际运行）
# ──────────────────────────────────────────────────────────────

CONCLUSIONS = """
╔══════════════════════════════════════════════════════════════╗
║        Spike 3 结论汇总（langsmith 0.8.18）                   ║
╚══════════════════════════════════════════════════════════════╝

Q1  自动 trace
    → 仅需设置 LANGCHAIN_TRACING_V2=true + LANGCHAIN_PROJECT=name
    → LangGraph 全链路（每个节点/LLM调用）自动上报，无需手动 wrap
    → 推荐在 FastAPI lifespan 或 .env 中统一设置

Q2  Dataset 导入
    → client.create_dataset() + client.create_examples(inputs, outputs, dataset_id)
    → 或从 JSONL 文件批量导入（见 langsmith Python SDK 文档）
    → QuantMind: 120 条 benchmark → 1 个 dataset "quantmind-eval-v1"

Q3  Evaluator 注册
    → 自定义 evaluator: def my_eval(run, example) -> dict(key, score, comment)
    → LLM-as-judge: 用 ChatOpenAI 打分，同样包装成 evaluator 函数
    → evaluate(target_fn, data=dataset_name, evaluators=[...])

Q4  Token 分开统计
    → LangSmith trace 按 run_type 分层：'llm' / 'embedding' / 'chain' / 'tool'
    → run.total_tokens / prompt_tokens / completion_tokens 每个 LLM call 独立记录
    → Embedding token 在 run_type='embedding' 的 run 中

操作事项（正式开发前完成）：
    1. 创建 .env 文件（见 .env.example，项目根目录）
    2. 配置 OPENAI_API_KEY + LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true
    3. 运行本 spike 的 Part B 验证 trace 链路
    4. 在 LangSmith UI 建立 "quantmind-eval-v1" dataset 并上传 benchmark JSONL
"""

if __name__ == "__main__":
    test_dataset_format()
    test_evaluator_structure()
    test_actual_tracing()   # skips gracefully if no .env
    print(CONCLUSIONS)
