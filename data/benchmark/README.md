# QuantMind Benchmark v1

`benchmark_v1.jsonl` 是 QuantMind Agent 的离线评估数据集，共 **120 条**，覆盖 S1–S10 十个用户场景。供 E2 evaluator 与 LangSmith offline eval 使用。

## 文件

| 文件 | 说明 |
|------|------|
| `benchmark_v1.jsonl` | 评估数据，每行一个 JSON 对象 |

## 数据格式

每行一个 JSON 对象，字段如下：

```json
{
  "id": "bench_001",
  "scenario": "S1",
  "query": "用户会真实提问的自然语言问题",
  "expected_behavior": "一段话描述理想回答应包含什么（非标准答案）",
  "difficulty": "easy",
  "eval_criteria": {
    "factual_grounding": true,
    "cites_sources": true,
    "uncertainty_stated": false,
    "requires_code": false
  },
  "tags": ["momentum", "factor_investing"]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识，格式 `bench_NNN`（三位数字，001–120） |
| `scenario` | string | 用户场景，取值 `S1`–`S10` |
| `query` | string | 模拟用户输入；中文为主，S2 代码生成类可用英文 |
| `expected_behavior` | string | 可评估的行为描述：理想回答应覆盖的要点、结构或验收方式 |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `eval_criteria` | object | 四个布尔标记，辅助 evaluator 自动评分 |
| `tags` | string[] | 主题标签（snake_case 英文），便于按主题分析 recall |

### `eval_criteria` 字段含义

| 字段 | 含义 |
|------|------|
| `factual_grounding` | 回答应基于检索结果或知识库事实，不应凭空编造 |
| `cites_sources` | 回答应给出论文、文档或数据来源引用 |
| `uncertainty_stated` | 证据不足或存在争议时，应明确说明局限或不确定性 |
| `requires_code` | 期望输出包含可执行代码（S2 代码生成场景专用） |

### S2 代码生成类 `expected_behavior` 要求

除上述通用字段外，S2 的 `expected_behavior` 应写明：

1. **框架**：`backtrader` 或 `vectorbt`
2. **核心组件**：如 `Strategy` 类、indicator 定义、`next()` 或等价信号逻辑
3. **验收方式**：代码可通过 `ast.parse()` 检查且无语法错误；可引用 `data/sample/spy_daily.csv` 作为 sample 数据

## 场景分布

| 场景 | 数量 | 重点 |
|------|------|------|
| S1 策略探索 | 20 | 动量/均值回归/配对交易/因子组合等 |
| S2 代码生成 | 30 | 回测代码；动量 10 / 均值回归 10 / 其他 10 |
| S3 概念解释 | 20 | Sharpe/Alpha/Beta/VaR/Drawdown 等 |
| S4 论文问答 | 10 | 特定论文的方法或结论 |
| S5 因子研究 | 10 | Fama-French/动量/低波动率等 |
| S6 面试准备 | 15 | Quant 面试高频问题 |
| S7 研究规划 | 5 | 制定学习计划 |
| S8 长期记忆 | 3 | 基于历史偏好的个性化推荐 |
| S9 跨域问答 | 4 | Research + Interview 混合 |
| S10 求职追踪 | 3 | 申请状态记录与下一步提醒 |

## 难度分布

全数据集目标分布：

| 难度 | 比例 | 数量 |
|------|------|------|
| easy | 30% | 36 |
| medium | 50% | 60 |
| hard | 20% | 24 |

## 验收命令

```bash
cd quant-project

# 行数应为 120
wc -l data/benchmark/benchmark_v1.jsonl

# 逐行 JSON 合法性 + 分布统计
python -m json.tool data/benchmark/benchmark_v1.jsonl > /dev/null 2>&1 || \
python -c "
import json, collections
from pathlib import Path
lines = Path('data/benchmark/benchmark_v1.jsonl').read_text().splitlines()
assert len(lines) == 120, f'expected 120 lines, got {len(lines)}'
counts = collections.Counter()
diff = collections.Counter()
for i, line in enumerate(lines, 1):
    o = json.loads(line)
    assert o['id'] == f'bench_{i:03d}', f'id mismatch at line {i}'
    assert o['scenario'] in {f'S{j}' for j in range(1, 11)}
    counts[o['scenario']] += 1
    diff[o['difficulty']] += 1
    for k in ['factual_grounding','cites_sources','uncertainty_stated','requires_code']:
        assert isinstance(o['eval_criteria'][k], bool)
print('scenario:', dict(sorted(counts.items())))
print('difficulty:', dict(sorted(diff.items())))
"
```

分布目标（±2 条容差）：S1=20, S2=30, S3=20, S4=10, S5=10, S6=15, S7=5, S8+S9+S10=10；easy 34–38, medium 58–62, hard 22–26。

## 版本说明

- **v1**：按场景（S1–S10）schema，路径为 `data/benchmark/`（单数）。旧提案中的 `data/benchmarks/` 四分类格式已废弃，E2 接入时以本文件为准。
