# 面试回答指南：LangGraph / LangSmith / RAG 实战经验

> 基于 QuantMind 项目的真实经历，用 STAR 法则（情境→任务→行动→结果）组织。每个回答都来自你实际做过的工程决策，不是背的理论。

---

## 一、核心叙事线（30 秒电梯演讲）

> 我独立做了一个基于 LangGraph 的多子图量化研究 AI Agent。它用 Qdrant 做混合检索 RAG，用 LangGraph 的子图和 interrupt 实现人机协作的代码生成流程，用 LangSmith 做全链路 tracing 来诊断生产问题。整个项目 100 个测试全绿，支持四子图并行 fan-out，Docker 一键部署。

---

## 二、LangGraph 相关

### Q1: "说说你对 LangGraph 的理解，为什么要用它而不是直接调 LLM？"

**回答要点**：从"单次 LLM 调用"到"有状态的 Agent 工作流"的演进。

> 直接调 LLM 的问题是：复杂任务需要多次调用、条件判断、错误重试、状态管理，如果都用 if-else 写，很快变成意大利面代码。
>
> LangGraph 让我把 Agent 的行为建模成一个**有向图**——节点是操作（检索、调用 LLM、执行代码），边是流程（成功就继续、失败就重试）。重点是：
>
> 1. **State 是图的共享内存**。所有节点读写同一个 State，但它不是简单的 dict——带 reducer，处理并发写冲突。比如我的项目里 `retry_counts` 被多个并行分支写，reducer 取 max 而不是后写覆盖前写。
>
> 2. **条件边让 AI 做决策**。我的 Router 节点用 LLM 判断用户意图，返回 `RouteDecision`，然后条件边根据这个决定走哪个子图。不是硬编码的 `if "code" in text`。
>
> 3. **子图嵌套**让每个能力模块独立开发、独立测试。我的 Research、CodeGen、Planning、Interview 四个子图各自编译，最后被父图挂载。加新能力不改已有代码。
>
> 4. **Checkpointer 自动持久化**。每次节点执行后 State 自动存 SQLite，所以对话可以跨请求继续，服务器重启也不丢状态。

### Q2: "你有没有用过 LangGraph 的 interrupt？讲讲具体怎么实现的？"

**这是最能体现你实战深度的回答，可以讲得很细。**

> 用过。我的 CodeGen 子图用 interrupt 实现了一个 Human-in-the-Loop 策略确认流程。
>
> **业务场景**：用户说"帮我写一个 RSI 策略回测代码"，Agent 先用 LLM 解析出策略参数（RSI 周期、买卖阈值、框架选择），然后停下来让用户确认或修改，确认后再生成代码和在沙箱里执行。
>
> **实现细节**：
>
> ```python
> def confirm_with_user(state):
>     confirmed = interrupt(state["strategy_spec"])  # 抛给用户
>     return {"strategy_spec": confirmed}
> ```
>
> **恢复的时候有坑**：必须用 `Command(resume={interrupt_id: value})`，不能传裸 value。因为 LangGraph 支持多 interrupt，不用 id 映射会恢复错。这是看文档和踩坑才发现的。
>
> **还有一个关键约束**：resume 后节点会**从头重新执行**，不是从 interrupt 那行继续。所以 interrupt 之前不能有任何不可逆的副作用——不能写数据库、不能发网络请求。所有副作用必须在 interrupt 拿到返回值之后做。我在契约文档里明确写了这个约束，确保开发 AI 不踩坑。
>
> **API 层的处理**：`/chat` 返回的 `ChatResponse` 有一个 `status` 字段，`"ok"` 和 `"interrupt"` 互斥——`"interrupt"` 时 `response` 是 None，前端靠这个判断要不要显示编辑表单。这个设计比前端靠 `interrupt is None` 推断更可靠。

### Q3: "你怎么用 LangGraph 处理并行任务的？"

> 我的 Supervisor Router 判断一个问题是跨域的（比如"如何在 Citadel 面试中展示动量因子研究"），就通过 Send API 并行 fan-out：
>
> ```python
> def route_to_subgraphs(state):
>     modes = ["research", "interview"]
>     return [Send(mode, {}) for mode in modes]
> ```
>
> 两个子图同时跑，各自往 `subgraph_outputs` 写结果（用 mode 做 key 分桶），最后 merge 节点合并。
>
> **这里踩过一个坑**：并行写同一个 State key 必须加 reducer，否则 `InvalidUpdateError`。我的 `retry_counts` 加了 `_merge_retry_counts` reducer（取 max），`subgraph_outputs` 加了 `_merge_named_outputs` reducer（按 key 合并）。这个在第二轮接口审核时被外部 AI 审核员指出来修复了。

### Q4: "你怎么设计 Agent 的状态管理的？"

> 我用了三层持久化分离，每一层有明确的职责边界：
>
> 1. **Checkpointer（SQLite/Postgres）**：自动管理线程级对话状态和中间结果。开发者不手动操作这层，LangGraph 自动存。
> 2. **Store（LangGraph Memory Store）**：跨线程的长期用户记忆——用户偏好、收藏的论文、研究计划摘要。不是大量数据，是"跨 session 需要记住的小东西"。
> 3. **PostgreSQL（SQLAlchemy ORM）**：大量结构化业务数据——求职记录的状态机、研究笔记全文、ingestion 日志。这层不存临时对话状态。
>
> 这样做的好处：每层独立扩展，不会互相污染。比如换 Checkpointer 从 SQLite 到 Postgres，不影响 Store 和业务数据。

---

## 三、LangSmith 相关

### Q5: "你用 LangSmith 做什么？有什么实际的工程价值？"

**这是面试官最想听的——不是"我有 tracing"，而是"tracing 帮我发现了什么"。**

> 不只是打开 tracing 看看漂亮的可视化。LangSmith 在我的项目里是**诊断工具**——跑 120 条 benchmark 评估时，我通过 trace 发现了 8 个生产问题。举几个例子：
>
> 1. **检索延时 14 秒**：trace 里 hybrid_search 节点的 latency 异常高。排查发现是 Qdrant 只有 5 篇种子上来，检索结果太少导致后续频繁 fallback 到 web search。修复是把论文从 5 篇扩到 25 篇，延时降到 6.7 秒。
>
> 2. **CodeEvaluator 崩溃**：某条 trace 在 execute_sandbox 节点直接抛 TypeError。看 trace 的堆栈才发现是评估器里一个 `del` 语句误删了属性，砂箱代码本身没问题。
>
> 3. **并行写冲突**：Supervisor 级别的 trace 报了 `InvalidUpdateError`。原因是 `final_response` 字段没有 reducer，两个并行子图同时写同一个 key 就炸了。加上 `_merge_final_response` reducer 修复。
>
> 4. **CodeGen 联网误判**：Sandbox 的 AST 检测报了 `FORBIDDEN_IMPORT`，但 trace 显示代码并没有 import 网络库。排查发现是 `parse_strategy` 误把用户描述的"buy and hold"策略解析成了另一套策略参数，生成的代码引了不存在的东西。
>
> **关键心得**：LangSmith 本身不解决任何问题，它让你**看到问题在哪**。没有 tracing，你只能看到"评估指标不对"，但不知道为什么。有 tracing，你能精确到"这个节点的这个字段在这个输入下出错了"。

### Q6: "LangSmith 集成有什么坑吗？"

> 最大的坑是**区域问题**。我的账号在 APAC 区，endpoint 必须是 `https://apac.api.smith.langchain.com`。用了默认的美国 endpoint 会静默失败——API 返回 403 但错误信息里没有任何地区提示。而且 project 必须在 Dashboard 先手动建再用 API 调用，否则也 403。
>
> 还有个注意点：LangSmith 在本地开发时很有用，但生产环境要注意成本——每个 LLM 调用都被 trace 的 overhead 大约是 5-10%。

---

## 四、RAG 相关

### Q7: "讲讲你的 RAG 系统的设计？为什么选 Qdrant + Hybrid Search？"

> 我的 RAG 做的是量化金融论文的检索增强生成。技术选型是 Qdrant + Hybrid RRF（Dense + Sparse）。
>
> **为什么 Hybrid 而不纯 Dense？** 纯语义向量找"意思相近"的，但金融领域术语精确性很重要——"momentum factor"和"cross-sectional momentum"在向量空间里很近，但对用户来说可能不是同一个东西。Sparse BM25 做的是关键词精确匹配，补上了 dense 的短板。
>
> **为什么 Qdrant？** 原生支持 hybrid fusion（RFF 内置），不需要自己实现 rank 合并逻辑。1.18+ 版本统一用 `query_points()` 入口，不再有多个 search API。
>
> **Embedding 为什么用本地 fastembed 而不是 API？**
> - BGE-small-en-v1.5 在英文论文检索上够用，384-dim
> - 免费，不消耗 API token。2,202 个 chunks 的 embedding 本地跑几分钟
> - 缺点是对中文支持弱（BM25 不支持中文分词），这个我记录为已知局限，升级方案是加中文 tokenizer 或 query 翻译
>
> **Pipeline 设计**：
> - 检索：query → parse intent → hybrid search → check confidence
> - confidence < 0.5 → fallback 到 web search（DuckDuckGo）
> - confidence 够 → synthesize answer → verify（LLM-as-judge）
> - verify 不通过 → retry 一次 → 仍不通过 → format_doubtful（带警告头输出）
>
> 这个"检索→验证→重试→优雅降级"的闭环，确保用户要么拿到有来源的高质量答案，要么被告知"信息不足，这里有几个你可以探索的方向"。

### Q8: "你的 RAG 系统怎么评估的？怎么知道检索质量够不够？"

> 我建了一个 120 条数据的 benchmark，覆盖 10 个用户场景。评估用三维度：
>
> 1. **LLM-as-judge**：LLM 对 output 评分（0-1），评估回答是否准确、是否引用来源
> 2. **代码验证**：CodeGen 的输出必须在真实沙箱中执行成功
> 3. **引用检查**：输出中的 citation 是否指向实际存在的论文
>
> 评估是离线跑的，不是单元测试那种 pass/fail。评估系统命名为 E2 任务包。
>
> **评估暴露的问题**：初始的 5 篇论文数据导致 recall 很低，很多检索返回空或低分。数据扩到 25 篇后检索覆盖面明显改善。Judge 的评分阈值也有偏高的问题（给的打分都很低），这个需要后续 prompt 校准，我记录为已知局限。

### Q9: "你的 RAG 遇到过幻觉问题吗？怎么解决的？"

> 有。我的解决方案是**四层防御**，而不是单靠 prompt：
>
> 1. **检索验证**：`check_confidence` 节点看检索结果的 fusion score，低于 0.5 就不强行生成，走 web search fallback 或 graceful decline
> 2. **结构化验证**：`verify_answer` 节点用 LLM 对生成答案做四维度评判——是否回答了问题、每个声明是否有来源支撑、是否引入了无来源的新声明、证据不足时是否说明了不确定
> 3. **重试机制**：验证不通过回到检索重试一次（不是无限，耗 token）。重试后仍不通过走 format_doubtful，给答案加上"⚠️ 以下内容未经充分验证，请谨慎参考"的警告头
> 4. **citation 强制**：system prompt 要求引用使用 `[1]` `[2]` 格式，输出时自动附加参考文献列表
>
> 核心思路是：**不让 Agent 在证据不足时硬编**。宁可告诉用户"我不知道"，也不给一个写得漂亮但是假的东西。

---

## 五、通用工程问题

### Q10: "这个项目你怎么保证代码质量的？"

> 三个层面：
>
> 1. **接口契约先行**：在写代码之前，我和三轮外部 AI 审核把 7 个模块的接口全部定义好并锁定。所有签名、类型、错误约定在 `模块接口契约.md` 里白纸黑字写清楚。这轮审核发现了 14 个阻塞问题——比如并行写冲突的 reducer、中断恢复的 interrupt_id 映射，都是审核发现的。
>
> 2. **测试驱动**：最终的测试覆盖是 100 tests collected、0 crash。每个模块有自己的单元测试，用 Mock 隔离外部依赖。
>
> 3. **评估驱动诊断**：120 条 benchmark 不是跑完就完了——我对比了评估指标的前后变化（修数据饥饿前 code execution 0.00→修后 1.00），通过 LangSmith trace 定位瓶颈。这种"离线评估→trace 诊断→定点修复→指标验证"的循环，是测试做不到的。

### Q11: "这个项目你学到了什么？有什么后悔的？"

> 最大的收获：
> 1. **LangGraph 的图思维**：把 Agent 拆成节点和边之后，复杂逻辑变得可测试、可观测。以前用 prompt engineering 调半天，现在直接看 trace 找哪个节点卡了。
> 2. **契约驱动的并行开发**：5 个工作流同时推进的前提是接口先锁。没有契约，并行是幻觉。
> 3. **Tracing 不只是"有监控"**：LangSmith 的价值在诊断，不是展示。监控不会告诉你为什么检索慢了 10 秒，trace 会。
>
> 如果可以重来：
> 1. 上来就应该用至少 20 篇种子论文做数据——5 篇种子上来导致评估指标失真，修数据花了一轮维护。
> 2. 评估系统应该和功能开发同步做，而不是最后加。做晚了才发现 CodeGen 有 parse_strategy 误判的问题，修晚了。

---

## 六、回答速查表

| 如果面试官问 | 用哪个回答 | 关键词 |
|-------------|-----------|--------|
| "讲讲 LangGraph" | Q1 + Q2 | 有向图、State reducer、条件边、interrupt、Checkpointer |
| "你怎么做状态管理" | Q4 | 三层分离、Checkpointer vs Store vs DB |
| "interrupt 怎么用" | Q2 | interrupt()、Command(resume=)、节点重入约束 |
| "并行怎么处理" | Q3 | Send API、reducer、InvalidUpdateError |
| "LangSmith 有什么用" | Q5 + Q6 | trace 诊断、APAC 端点、实际发现的问题 |
| "讲讲你的 RAG" | Q7 | Qdrant Hybrid RRF、Dense+Sparse、本地 embedding |
| "怎么处理幻觉" | Q9 | 检索验证→结构化验证→重试→降级 |
| "怎么评估质量" | Q8 | 三维 benchmark、LLM-as-judge |
| "项目学到了什么" | Q11 | 图思维、契约驱动、tracing 诊断 |

---

*编写日期：2026-06-26*
