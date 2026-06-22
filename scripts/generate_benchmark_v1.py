#!/usr/bin/env python3
"""One-off generator for data/benchmark/benchmark_v1.jsonl — run from quant-project root."""

from __future__ import annotations

import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

OUT = Path("data/benchmark/benchmark_v1.jsonl")

# ── S1: 策略探索 (20) ──────────────────────────────────────────────
S1 = [
    {
        "scenario": "S1",
        "query": "解释动量因子的构建方法及其在 A 股市场的有效性",
        "expected_behavior": "应返回动量因子的定义、计算公式（如过去 12-1 个月收益率）、截面排序方法，并基于检索到的论文给出 A 股市场实证结论或局限性说明",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["momentum", "factor_investing", "China_market"],
    },
    {
        "scenario": "S1",
        "query": "配对交易在美股和港股上的适用条件有什么不同？",
        "expected_behavior": "应说明配对交易的基本逻辑（协整、spread 均值回归）、选股配对方法，并对比不同市场的流动性、做空机制与交易成本对策略的影响",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["pairs_trading", "stat_arb", "market_structure"],
    },
    {
        "scenario": "S1",
        "query": "均值回归策略在高波动市场会不会失效？有哪些文献讨论过？",
        "expected_behavior": "应解释均值回归的经济直觉与常见信号（z-score、布林带），引用相关研究说明波动率 regime 切换对策略的影响，并给出失效场景或风险",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["mean_reversion", "volatility", "regime_switch"],
    },
    {
        "scenario": "S1",
        "query": "多因子组合时应该怎么处理因子之间的相关性？",
        "expected_behavior": "应讨论因子相关性带来的冗余暴露、正交化或风险平价加权思路，并提及实践中常见的组合优化方法（如最大化 IR 或约束跟踪误差）",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["factor_combination", "portfolio_construction", "correlation"],
    },
    {
        "scenario": "S1",
        "query": "市场中性策略的对冲比例应该怎么设定？",
        "expected_behavior": "应说明 beta 对冲、 dollar-neutral 与风险中性等概念，讨论对冲比例与残差风险、交易成本的权衡，并给出实务中的常见做法",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False},
        "tags": ["market_neutral", "hedging", "beta"],
    },
    {
        "scenario": "S1",
        "query": "时间序列动量和截面动量在新兴市场表现差异大吗？",
        "expected_behavior": "应区分 TS momentum 与 CS momentum 的定义与构建方式，结合文献讨论新兴市场流动性、散户占比对两类动量效应的影响",
        "difficulty": "hard",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["momentum", "emerging_markets", "time_series"],
    },
    {
        "scenario": "S1",
        "query": "统计套利和配对交易是一回事吗？有什么区别？",
        "expected_behavior": "应厘清两者关系：配对交易是 stat arb 的子集，说明协整检验、spread 建模与 basket trading 的差异，并举例常见应用场景",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False},
        "tags": ["stat_arb", "pairs_trading", "strategies"],
    },
    {
        "scenario": "S1",
        "query": "低波动率异象在中国股市是否成立？和美股有何不同？",
        "expected_behavior": "应定义低波动率因子（如最低波动组合 vs 市场）、引用实证研究对比中美市场，并讨论 A 股涨跌停、散户行为等本地因素",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["low_volatility", "anomaly", "China_market"],
    },
    {
        "scenario": "S1",
        "query": "因子择时（factor timing）值得做吗？学术上怎么看的？",
        "expected_behavior": "应介绍 factor timing 的基本思路（宏观指标、估值价差驱动），引用文献说明其可行性与过拟合风险，给出审慎结论",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["factor_timing", "macro", "factor_investing"],
    },
    {
        "scenario": "S1",
        "query": "日内动量策略在期货上可行吗？主要挑战是什么？",
        "expected_behavior": "应说明日内动量信号构建（开盘效应、隔夜收益延续等），讨论期货杠杆、滑点与保证金约束，并指出数据频率与执行难度",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["intraday", "momentum", "futures"],
    },
    {
        "scenario": "S1",
        "query": "价值因子和动量因子同时配置时会不会互相抵消？",
        "expected_behavior": "应解释 value 与 momentum 的历史负相关关系、组合分散化收益，并讨论多因子框架下如何同时持有两类因子暴露",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["value", "momentum", "factor_combination"],
    },
    {
        "scenario": "S1",
        "query": "行业中性化对因子策略回测结果影响大吗？",
        "expected_behavior": "应说明行业中性化的目的（剥离行业 beta）、常见实现方式（行业内排序或回归残差），并讨论对 Sharpe 与换手的影响",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False},
        "tags": ["industry_neutral", "factor_investing", "backtesting"],
    },
    {
        "scenario": "S1",
        "query": "加密货币市场的动量效应和股票市场一样吗？",
        "expected_behavior": "应对比 crypto 与 equity 动量文献结论，讨论 24/7 交易、高波动与流动性碎片化带来的差异，并说明证据局限性",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["crypto", "momentum", "cross_asset"],
    },
    {
        "scenario": "S1",
        "query": "用期权数据能不能增强股票多空策略？有哪些思路？",
        "expected_behavior": "应列举期权隐含波动率偏斜、put-call ratio、gamma exposure 等信号如何用于股票选股或对冲，并引用相关研究",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["options", "equity_long_short", "alternative_data"],
    },
    {
        "scenario": "S1",
        "query": "小盘股溢价策略现在还有效吗？",
        "expected_behavior": "应回顾 SMB 因子起源、近年衰减讨论与可能的解释（流动性溢价、机构持仓变化），并给出有条件结论",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["size_premium", "SMB", "factor_decay"],
    },
    {
        "scenario": "S1",
        "query": "趋势跟踪 CTA 策略在股债双杀年份表现如何？",
        "expected_behavior": "应说明趋势跟踪的时间序列动量逻辑、与股票 beta 的低相关性，引用 crisis period 表现研究并讨论尾部风险",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["CTA", "trend_following", "crisis_performance"],
    },
    {
        "scenario": "S1",
        "query": "基本面量化（Quantamental）和传统因子投资有什么本质区别？",
        "expected_behavior": "应对比 fundamental data 驱动的信号（盈利质量、分析师修正）与 price-based factor 的差异，讨论数据频率与 alpha 衰减",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False},
        "tags": ["quantamental", "fundamental_data", "factor_investing"],
    },
    {
        "scenario": "S1",
        "query": "东南亚小盘股的动量效应值得研究吗？数据上有什么障碍？",
        "expected_behavior": "应讨论新兴市场小盘动量文献、流动性与数据可得性限制，并建议可行的研究切入点（如新加坡/泰国市场对比）",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["momentum", "Southeast_Asia", "small_cap"],
    },
    {
        "scenario": "S1",
        "query": "反转策略（短期 reversal）和动量策略怎么区分？能同时用吗？",
        "expected_behavior": "应区分短期 reversal（1周-1月）与中期 momentum（3-12月）的时间尺度，说明两者在不同 horizon 共存的原因及组合注意事项",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False},
        "tags": ["reversal", "momentum", "horizon"],
    },
    {
        "scenario": "S1",
        "query": "机器学习选股和传统线性因子模型相比优劣势在哪？",
        "expected_behavior": "应比较 ML 的非线性拟合能力与过拟合风险，讨论可解释性、数据需求与样本外表现，引用近期 quant ML 文献",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False},
        "tags": ["machine_learning", "factor_models", "overfitting"],
    },
]

# ── S2: 代码生成 (30) — 动量10 / 均值回归10 / 其他10 ─────────────
S2_MOMENTUM = [
    {
        "scenario": "S2",
        "query": "用 backtrader 写一个双均线策略：20 日均线上穿 60 日均线买入，下穿卖出",
        "expected_behavior": "返回可运行的 backtrader 代码，包含 Strategy 类、SMA(20)/SMA(60) indicator、CrossOver 或等价买卖信号逻辑、cerebro 初始化与 data/sample/spy_daily.csv 加载；代码可通过 ast.parse() 且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["moving_average", "backtrader", "momentum"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader strategy using 12-month ROC: buy top ROC, sell when ROC turns negative",
        "expected_behavior": "Return runnable backtrader code with Strategy class, ROC indicator (period=252), next() signal logic for momentum entry/exit, cerebro setup loading data/sample/spy_daily.csv; passes ast.parse() with no syntax errors",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["ROC", "backtrader", "momentum"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 实现唐奇安通道突破策略：突破 20 日最高价买入，跌破 10 日最低价卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、Highest/Lowest 或 DonchianChannel indicator、next() 突破信号、cerebro 与 CSV 数据加载；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["donchian", "breakout", "backtrader"],
    },
    {
        "scenario": "S2",
        "query": "用 vectorbt 写一个简单的时间序列动量策略：过去 6 个月收益为正则做多",
        "expected_behavior": "返回可运行的 vectorbt 代码，含收益率计算、6 个月滚动窗口信号生成、Portfolio.from_signals 回测逻辑；使用 data/sample/spy_daily.csv；可通过 ast.parse() 且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["time_series_momentum", "vectorbt", "momentum"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写 EMA 交叉策略：EMA(12) 上穿 EMA(26) 买入，下穿卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、ExponentialMovingAverage indicator、CrossOver 信号、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["EMA", "backtrader", "momentum"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader dual momentum strategy: hold SPY if 12-month return > 0, else hold cash",
        "expected_behavior": "Return backtrader Strategy with 12-month return calculation, conditional position logic in next(), cerebro setup with data/sample/spy_daily.csv; passes ast.parse() with no syntax errors",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["dual_momentum", "backtrader", "absolute_momentum"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 实现 ATR 通道突破：收盘价突破前一日高点+1.5*ATR 买入",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、ATR indicator、突破信号逻辑、cerebro 与数据加载；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["ATR", "breakout", "backtrader"],
    },
    {
        "scenario": "S2",
        "query": "用 vectorbt 实现截面动量轮动：每月选过去 3 个月收益最高的标的等权持有",
        "expected_behavior": "返回 vectorbt 代码，含多标的收益率矩阵、3 个月滚动排名、月度调仓信号与 Portfolio 回测；可通过 ast.parse() 且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["cross_sectional", "vectorbt", "rotation"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写价格创新高策略：收盘价创 50 日新高买入，跌破 20 日均线卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、Highest(50) 与 SMA(20) indicator、next() 买卖逻辑、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["new_high", "backtrader", "momentum"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader strategy: go long when Close > SMA(200), flat otherwise (trend filter)",
        "expected_behavior": "Return backtrader Strategy with SMA(200) trend filter, position logic in next(), cerebro with data/sample/spy_daily.csv; passes ast.parse() with no syntax errors",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["trend_filter", "SMA200", "backtrader"],
    },
]

S2_MEAN_REVERSION = [
    {
        "scenario": "S2",
        "query": "用 backtrader 写一个 RSI 策略：RSI < 30 买入，RSI > 70 卖出",
        "expected_behavior": "返回可运行的 backtrader 代码，包含 Strategy 类、RSI indicator 定义、next() 方法中的买卖信号逻辑、cerebro 初始化代码；可通过 ast.parse() 且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["RSI", "backtrader", "mean_reversion"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 实现布林带均值回归：价格触及下轨买入，触及上轨卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、BollingerBands indicator、next() 触及上下轨信号、cerebro 与 data/sample/spy_daily.csv 加载；可通过 AST 检查且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["bollinger", "backtrader", "mean_reversion"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader pairs trading strategy: trade spread z-score, enter at |z|>2, exit at |z|<0.5",
        "expected_behavior": "Return backtrader Strategy with spread calculation, z-score logic, entry/exit thresholds in next(), cerebro setup; passes ast.parse() with no syntax errors",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["pairs_trading", "z_score", "backtrader"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写 Stochastic 策略：%K < 20 买入，%K > 80 卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、Stochastic indicator、next() 超买超卖信号、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["stochastic", "backtrader", "mean_reversion"],
    },
    {
        "scenario": "S2",
        "query": "用 vectorbt 实现 z-score 均值回归：价格偏离 20 日均线 2 个标准差时反向开仓",
        "expected_behavior": "返回 vectorbt 代码，含 rolling mean/std 计算、z-score 信号、Portfolio.from_signals 回测；使用 data/sample/spy_daily.csv；可通过 ast.parse() 且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["z_score", "vectorbt", "mean_reversion"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写 CCI 策略：CCI < -100 买入，CCI > 100 卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、CCI indicator、next() 信号逻辑、cerebro 与 CSV 加载；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["CCI", "backtrader", "mean_reversion"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader mean reversion strategy: buy when Close < SMA(20) - 2*StdDev(20)",
        "expected_behavior": "Return backtrader Strategy with SMA and StdDev indicators, entry when price below lower band, exit at mean, cerebro setup; passes ast.parse() with no syntax errors",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["bollinger", "stddev", "backtrader"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 实现 Williams %R 策略：%R < -80 买入，%R > -20 卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、WilliamsR indicator、next() 买卖逻辑、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["williams_r", "backtrader", "mean_reversion"],
    },
    {
        "scenario": "S2",
        "query": "用 vectorbt 写网格交易策略：价格每下跌 2% 加仓，每上涨 2% 减仓",
        "expected_behavior": "返回 vectorbt 代码，含价格变化检测、网格加仓减仓信号、Portfolio 回测逻辑；可通过 ast.parse() 且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["grid_trading", "vectorbt", "mean_reversion"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写短期反转策略：昨日跌幅超过 2% 则今日开盘买入，持有 5 日后卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、日收益率计算、持有期计数与卖出逻辑、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["short_term_reversal", "backtrader", "mean_reversion"],
    },
]

S2_OTHER = [
    {
        "scenario": "S2",
        "query": "用 backtrader 写 MACD 策略：MACD 线上穿信号线买入，下穿卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、MACD indicator、CrossOver 信号逻辑、cerebro 与 data/sample/spy_daily.csv 加载；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["MACD", "backtrader", "signal_crossover"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader strategy with ATR-based stop loss: exit if price drops 2*ATR from entry",
        "expected_behavior": "Return backtrader Strategy with ATR indicator, entry signal, trailing stop logic in next(), cerebro setup; passes ast.parse() with no syntax errors",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["ATR", "stop_loss", "backtrader"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 实现波动率目标策略：根据 20 日历史波动率动态调整仓位，目标年化波动 15%",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、年化波动率计算、仓位缩放逻辑、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["volatility_targeting", "backtrader", "risk_management"],
    },
    {
        "scenario": "S2",
        "query": "用 vectorbt 写一个简单的买入持有基准策略，并输出总收益和 Sharpe Ratio",
        "expected_behavior": "返回 vectorbt 代码，含 Portfolio.from_holding、收益与 Sharpe 计算输出；使用 data/sample/spy_daily.csv；可通过 ast.parse() 且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["buy_and_hold", "vectorbt", "benchmark"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写 OBV 策略：OBV 突破 20 日 OBV 均线买入，跌破卖出",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、OnBalanceVolume indicator、SMA on OBV、next() 信号、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["OBV", "backtrader", "volume"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader walk-forward backtest skeleton: train on 252 days, test on 63 days, roll forward",
        "expected_behavior": "Return backtrader code with walk-forward loop structure, train/test window split, cerebro re-instantiation per window; passes ast.parse() with no syntax errors",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["walk_forward", "backtrader", "backtesting"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写 ADX 过滤策略：仅当 ADX > 25 时执行均线交叉信号",
        "expected_behavior": "返回 backtrader 代码，含 Strategy 类、ADX 与 SMA CrossOver indicator、ADX 过滤条件、cerebro 初始化；可通过 AST 检查且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["ADX", "backtrader", "filter"],
    },
    {
        "scenario": "S2",
        "query": "用 vectorbt 实现月度再平衡的等权双资产组合回测框架",
        "expected_behavior": "返回 vectorbt 代码，含多资产收盘价、月度调仓日期检测、等权权重分配与 Portfolio 回测；可通过 ast.parse() 且无语法错误",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["rebalancing", "vectorbt", "portfolio"],
    },
    {
        "scenario": "S2",
        "query": "用 backtrader 写带手续费和滑点的回测模板：commission=0.1%, slippage=0.05%",
        "expected_behavior": "返回 backtrader 代码，含 cerebro.broker.setcommission 与 slippage 设置、简单 Strategy 骨架、data 加载；可通过 AST 检查且无语法错误",
        "difficulty": "easy",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["commission", "slippage", "backtrader"],
    },
    {
        "scenario": "S2",
        "query": "Write a backtrader strategy combining RSI and MACD: buy when RSI<30 AND MACD>signal, sell when RSI>70",
        "expected_behavior": "Return backtrader Strategy with RSI and MACD indicators, combined entry/exit conditions in next(), cerebro with data/sample/spy_daily.csv; passes ast.parse() with no syntax errors",
        "difficulty": "medium",
        "eval_criteria": {"factual_grounding": False, "cites_sources": False, "uncertainty_stated": False, "requires_code": True},
        "tags": ["RSI", "MACD", "backtrader"],
    },
]

S2 = S2_MOMENTUM + S2_MEAN_REVERSION + S2_OTHER

# ── S3: 概念解释 (20) ──────────────────────────────────────────────
S3 = [
    {"scenario": "S3", "query": "什么是 Sharpe Ratio？怎么计算，多少算好？", "expected_behavior": "应给出 Sharpe 的直觉解释、公式 (R_p - R_f) / σ_p、年化处理说明，并说明不同策略类型的合理 Sharpe 范围", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["Sharpe_ratio", "risk_metrics", "performance"]},
    {"scenario": "S3", "query": "Alpha 和 Beta 在量化投资里分别是什么意思？", "expected_behavior": "应解释 Beta 为系统性风险暴露、Alpha 为超额收益，给出 CAPM 框架下的关系，并举例实际应用", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["alpha", "beta", "CAPM"]},
    {"scenario": "S3", "query": "解释 Fama-French 三因子模型的三个因子分别代表什么", "expected_behavior": "应说明 MKT-RF、SMB、HML 的定义与构建方法，解释各自捕捉的风险溢价，并提及模型局限", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["Fama_French", "factor_models", "SMB"]},
    {"scenario": "S3", "query": "VaR（Value at Risk）和 CVaR 有什么区别？", "expected_behavior": "应定义 VaR 为给定置信水平下的最大损失、CVaR 为尾部条件期望损失，比较两者对极端风险的刻画能力", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["VaR", "CVaR", "risk_metrics"]},
    {"scenario": "S3", "query": "Maximum Drawdown 怎么算？为什么比波动率更能反映投资者体验？", "expected_behavior": "应给出最大回撤计算公式（peak-to-trough）、举例说明，并解释路径依赖与投资者心理", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["drawdown", "risk_metrics", "performance"]},
    {"scenario": "S3", "query": "Sortino Ratio 和 Sharpe Ratio 的核心区别是什么？", "expected_behavior": "应说明 Sortino 仅惩罚下行波动（使用 downside deviation），解释其相对 Sharpe 的优势与适用场景", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["Sortino_ratio", "Sharpe_ratio", "downside_risk"]},
    {"scenario": "S3", "query": "什么是 Information Ratio？和 Sharpe 有什么关系？", "expected_behavior": "应定义 IR = 主动收益 / 跟踪误差，说明其衡量主动管理技能，并与 Sharpe 对比（基准选择差异）", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["information_ratio", "active_management", "risk_metrics"]},
    {"scenario": "S3", "query": "Kelly Criterion 在仓位管理里怎么用？有什么风险？", "expected_behavior": "应给出 Kelly 公式 f* = (bp - q) / b 的直觉与推导思路，说明 full Kelly 波动过大，实践中常用 fractional Kelly", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": True, "requires_code": False}, "tags": ["Kelly_criterion", "position_sizing", "portfolio_theory"]},
    {"scenario": "S3", "query": "协整（Cointegration）和 correlation 有什么区别？", "expected_behavior": "应说明 correlation 衡量短期线性关系、cointegration 衡量长期均衡关系，解释为何配对交易需要协整而非仅高相关", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["cointegration", "correlation", "econometrics"]},
    {"scenario": "S3", "query": "GARCH 模型是用来做什么的？", "expected_behavior": "应解释 GARCH 建模条件异方差（波动率聚集），给出基本方程形式，并说明在 VaR 与期权定价中的应用", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["GARCH", "volatility", "econometrics"]},
    {"scenario": "S3", "query": "什么是 Overfitting？回测里怎么检测？", "expected_behavior": "应定义过拟合为样本内过度拟合噪声，列举检测方法（样本外测试、walk-forward、deflated Sharpe、参数敏感性分析）", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["overfitting", "backtesting", "validation"]},
    {"scenario": "S3", "query": "Walk-Forward Analysis 是什么？为什么比单次回测更可靠？", "expected_behavior": "应说明滚动训练/测试窗口方法、避免前视偏差与过拟合的原理，并描述基本实施步骤", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["walk_forward", "backtesting", "validation"]},
    {"scenario": "S3", "query": "Smart Beta 和 traditional factor investing 是一回事吗？", "expected_behavior": "应说明 Smart Beta 为规则化因子暴露产品，对比纯学术因子研究与 ETF 化实现，讨论透明性与费用", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["smart_beta", "factor_investing", "ETF"]},
    {"scenario": "S3", "query": "Risk Parity 策略的核心思想是什么？", "expected_behavior": "应解释等风险预算分配（非等资金权重）、与均值方差优化的区别，并提及桥水 All Weather 等实例", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["risk_parity", "portfolio_theory", "allocation"]},
    {"scenario": "S3", "query": "滑点（Slippage）在回测里应该怎么建模？", "expected_behavior": "应定义滑点为预期与成交价差，讨论固定比例、成交量占比（market impact）等建模方式及对回测结果的影响", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["slippage", "execution", "backtesting"]},
    {"scenario": "S3", "query": "UMD 动量因子在 Carhart 四因子模型里怎么定义？", "expected_behavior": "应说明 Up Minus Down 组合构建（赢家减输家）、典型 12 个月形成期，并解释其作为第四因子的作用", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["UMD", "Carhart", "momentum"]},
    {"scenario": "S3", "query": "Kalman Filter 在量化金融里有哪些典型应用？", "expected_behavior": "应列举动态 beta 估计、配对交易信号提取、状态空间模型等应用，并简要说明递归更新机制", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["Kalman_filter", "state_space", "econometrics"]},
    {"scenario": "S3", "query": "Order Book Imbalance 是什么？为什么能预测短期价格？", "expected_behavior": "应定义订单簿买卖量不平衡指标、微观结构理论（价格压力），并说明高频场景下的预测局限", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": True, "requires_code": False}, "tags": ["order_book", "microstructure", "HFT"]},
    {"scenario": "S3", "query": "Volatility Targeting 策略的原理是什么？", "expected_behavior": "应说明根据已实现或预测波动率缩放仓位以维持目标风险水平，解释杠杆随波动率反向调整的逻辑", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["volatility_targeting", "risk_management", "position_sizing"]},
    {"scenario": "S3", "query": "Factor Investing 和传统主动选股有什么本质区别？", "expected_behavior": "应对比规则化因子暴露 vs 基本面选股、系统性风险溢价 harvesting vs alpha 挖掘，并讨论成本与容量", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["factor_investing", "active_management", "systematic"]},
]

# ── S4: 论文问答 (10) ──────────────────────────────────────────────
S4 = [
    {"scenario": "S4", "query": "arXiv:1208.2775 这篇关于 price momentum 的论文核心结论是什么？", "expected_behavior": "应基于论文内容说明动量策略的风险调整收益、波动率管理方法（如 scaling by realized vol），并引用论文具体发现", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["1208.2775", "momentum", "paper_qa"]},
    {"scenario": "S4", "query": "Fama-French 三因子回归模型那篇 arXiv:2006.02467 用了什么方法论？", "expected_behavior": "应说明论文中三因子回归框架、时间序列与横截面测试方法，并总结主要实证结果", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["2006.02467", "Fama_French", "paper_qa"]},
    {"scenario": "S4", "query": "arXiv:2002.01800 关于 Sharpe Ratio 高维分析的主要贡献是什么？", "expected_behavior": "应说明论文对高维环境下 Sharpe 比率估计偏差与推断问题的分析，引用其核心定理或结论", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["2002.01800", "Sharpe_ratio", "paper_qa"]},
    {"scenario": "S4", "query": "arXiv:0801.4047 讨论的无套利条件对交易策略设计有什么启示？", "expected_behavior": "应基于论文解释无套利条件的数学表述、与交易策略可行性的关系，并给出实务启示", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["0801.4047", "no_arbitrage", "paper_qa"]},
    {"scenario": "S4", "query": "arXiv:1511.07101 这篇风险收益实证研究的关键发现有哪些？", "expected_behavior": "应总结论文的统计方法、样本特征与主要风险收益关系发现，并正确引用论文来源", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["1511.07101", "risk_return", "paper_qa"]},
    {"scenario": "S4", "query": "Jegadeesh and Titman (1993) 动量论文的核心实证结果是什么？", "expected_behavior": "应说明 3-12 个月形成期买入赢家卖出输家的策略收益、持有期效应，并注明该论文可能未完全入库应基于已知学术共识回答", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["Jegadeesh_Titman", "momentum", "classic_paper"]},
    {"scenario": "S4", "query": "Daniel and Moskowitz 的 Momentum Crashes 论文主要解释了什么现象？", "expected_behavior": "应解释动量策略在市场反转时期的剧烈回撤机制（panic state、volatility regime），并说明证据局限性", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["momentum_crash", "Daniel_Moskowitz", "classic_paper"]},
    {"scenario": "S4", "query": "Fama and French (1992) 交叉截面股票收益论文发现了什么？", "expected_behavior": "应说明市值与账面市值比效应的发现、方法论（FMB 回归），并注明可能需依赖检索或学术共识", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["Fama_French_1992", "cross_section", "classic_paper"]},
    {"scenario": "S4", "query": "Barroso and Santa-Clara 2015 关于动量波动率管理的论文提出了什么方法？", "expected_behavior": "应说明通过 realized volatility scaling 管理动量组合风险的方法与绩效改善，关联 1208.2775 若可检索", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["vol_scaling", "momentum", "Barroso"]},
    {"scenario": "S4", "query": "Lo and MacKinlay (1990) 对 contrarian 策略的解释是什么？", "expected_behavior": "应说明短期反转与长期动量的关系、lead-lag 效应及过度反应假说，并注明文献检索局限", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["contrarian", "reversal", "classic_paper"]},
]

# ── S5: 因子研究 (10) ──────────────────────────────────────────────
S5 = [
    {"scenario": "S5", "query": "Fama-French 五因子模型比三因子多了哪两个因子？实证表现如何？", "expected_behavior": "应说明 RMW（盈利能力）与 CMA（投资风格）因子的定义，引用文献对比三因子与五因子解释力", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["Fama_French", "five_factor", "factor_research"]},
    {"scenario": "S5", "query": "质量因子（Quality）一般怎么构建？有哪些学术定义？", "expected_behavior": "应列举 ROE、盈利稳定性、低杠杆等质量指标，引用 Asness 等质量因子研究及组合构建方法", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["quality", "factor_construction", "factor_research"]},
    {"scenario": "S5", "query": "低波动率因子异象的主要解释有哪些？", "expected_behavior": "应讨论杠杆约束、彩票偏好、行业集中等解释假说，引用 Baker、Ang 等相关研究", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["low_volatility", "anomaly", "factor_research"]},
    {"scenario": "S5", "query": "动量因子在不同形成期（3/6/12 个月）下表现差异大吗？", "expected_behavior": "应比较不同 formation period 的动量收益与衰减特征，引用 Jegadeesh-Titman 及后续研究结论", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["momentum", "formation_period", "factor_research"]},
    {"scenario": "S5", "query": "有哪些基于情绪或文本数据的量化因子研究？", "expected_behavior": "应列举新闻情绪、社交媒体、财报语调等另类数据因子，引用相关论文并说明数据处理方法", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["sentiment", "alternative_data", "factor_research"]},
    {"scenario": "S5", "query": "价值因子（HML）近年来衰减了吗？原因是什么？", "expected_behavior": "应讨论 value premium 衰减证据、可能的 crowding 与结构性变化解释，引用近年 factor decay 研究", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["value", "factor_decay", "HML"]},
    {"scenario": "S5", "query": "规模因子 SMB 在 A 股市场的表现和美股一致吗？", "expected_behavior": "应对比中美 SMB 效应差异，讨论 A 股壳价值、散户结构等本地因素，引用新兴市场研究", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": True, "requires_code": False}, "tags": ["SMB", "China_market", "size_premium"]},
    {"scenario": "S5", "query": "盈利因子（RMW）和价值投资因子有什么区别？", "expected_behavior": "应区分 profitability（高 ROE/毛利率）与 value（高 B/M）的经济含义与相关性，引用五因子框架", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["RMW", "value", "profitability"]},
    {"scenario": "S5", "query": "因子拥挤（factor crowding）怎么度量？对策略有什么影响？", "expected_behavior": "应说明 crowding 度量指标（配对相关性、做空成本、因子估值价差），讨论收益衰减与 tail risk", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["factor_crowding", "capacity", "factor_research"]},
    {"scenario": "S5", "query": "投资因子（CMA，保守减激进）捕捉的是什么风险溢价？", "expected_behavior": "应解释 CMA 基于资产增长率的组合构建、投资效应的理论解释（过度投资假说），引用 Fama-French 五因子", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["CMA", "investment_factor", "five_factor"]},
]

# ── S6: 面试准备 (15) ──────────────────────────────────────────────
S6 = [
    {"scenario": "S6", "query": "量化面试常考的 Python 数据结构题：如何实现一个滚动窗口最大值？", "expected_behavior": "应给出 deque 或单调队列解法、时间复杂度分析，并说明在量化场景（滚动极值计算）中的应用", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["python", "coding_interview", "data_structures"]},
    {"scenario": "S6", "query": "面试题：抛硬币直到连续两次正面，期望抛几次？", "expected_behavior": "应给出状态机或条件期望解法、推导过程与答案（6次），并展示清晰的概率推理思路", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["probability", "brain_teaser", "interview"]},
    {"scenario": "S6", "query": "Citadel 量化研究岗面试一般会考哪些类型的题目？", "expected_behavior": "应涵盖概率统计、线性代数、金融市场直觉、编程题、因子研究案例题，并给出备考建议", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["Citadel", "interview_prep", "quant_research"]},
    {"scenario": "S6", "query": "如何向面试官解释我的动量因子回测项目？给一个结构化的介绍框架", "expected_behavior": "应给出 STAR 或问题-方法-结果-反思框架，涵盖因子定义、数据、回测设计、关键指标与局限性", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["project_presentation", "momentum", "interview"]},
    {"scenario": "S6", "query": "面试题：什么是 p-value？在因子检验里怎么用，有什么误区？", "expected_behavior": "应定义 p-value、假设检验流程，说明多重检验问题（Bonferroni/FDR），并警告 p-hacking 风险", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["statistics", "hypothesis_testing", "interview"]},
    {"scenario": "S6", "query": "Two Sigma 面试里的机器学习题一般考什么难度？", "expected_behavior": "应说明 ML 基础（bias-variance、正则化、交叉验证）、特征工程、过拟合防范，并给出典型题目类型", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": True, "requires_code": False}, "tags": ["Two_Sigma", "machine_learning", "interview"]},
    {"scenario": "S6", "query": "量化面试：如何检测时间序列是否平稳？有哪些检验方法？", "expected_behavior": "应介绍 ADF 检验、KPSS 检验、可视化方法，说明原假设与实务中的预处理步骤", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["stationarity", "time_series", "interview"]},
    {"scenario": "S6", "query": "Jane Street 面试的概率题：100 个囚犯和灯泡问题怎么分析？", "expected_behavior": "应给出问题澄清、概率建模思路（或已知结论），展示结构化推理过程而非仅给答案", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["Jane_Street", "probability", "brain_teaser"]},
    {"scenario": "S6", "query": "面试题：回测中 look-ahead bias 是什么？举三个具体例子", "expected_behavior": "应定义前视偏差、举例（ survivorship bias、使用未来财报数据、全样本标准化），并说明防范方法", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["look_ahead_bias", "backtesting", "interview"]},
    {"scenario": "S6", "query": "DE Shaw 量化开发岗面试的编程题有什么特点？", "expected_behavior": "应说明偏重算法效率、数据结构、Python/C++ 能力，可能含金融场景题，并给出练习建议", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": True, "requires_code": False}, "tags": ["DE_Shaw", "programming", "interview"]},
    {"scenario": "S6", "query": "面试题：解释中心极限定理，它和量化金融有什么关系？", "expected_behavior": "应陈述 CLT 内容、收敛条件，说明在 VaR 估计、期权定价、蒙特卡洛模拟中的应用", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["CLT", "statistics", "interview"]},
    {"scenario": "S6", "query": "量化研究面试：如何设计一个 A/B 测试来验证新因子是否有增量 alpha？", "expected_behavior": "应说明假设设定、样本分割、out-of-sample 验证、多重检验校正与实务中的数据 snooping 防范", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["AB_testing", "factor_research", "interview"]},
    {"scenario": "S6", "query": "面试题：Black-Scholes 模型的五个假设是什么？哪个最不现实？", "expected_behavior": "应列出常数波动率、无摩擦、连续交易、无套利、对数正态分布等假设，并批判性讨论常数波动率假设", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["Black_Scholes", "options", "interview"]},
    {"scenario": "S6", "query": "AQR 面试会问因子投资方面的什么问题？", "expected_behavior": "应涵盖因子定义、组合构建、因子衰减、交易成本、学术与实务差异，体现 AQR 学术导向风格", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["AQR", "factor_investing", "interview"]},
    {"scenario": "S6", "query": "面试题：你有 10 个球，1 个较重，用天平最少称几次能找到？", "expected_behavior": "应给出三分法解法（3 次）、推理过程，并推广到 n 个球的一般思路", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["brain_teaser", "logic", "interview"]},
]

# ── S7: 研究规划 (5) ──────────────────────────────────────────────
S7 = [
    {"scenario": "S7", "query": "我想从零开始学习统计套利，帮我规划一个 8 周的学习路径", "expected_behavior": "应产出分阶段计划（协整基础→配对交易→回测→风险管理），每步含推荐阅读论文、代码练习与里程碑", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["stat_arb", "learning_plan", "research_planning"]},
    {"scenario": "S7", "query": "帮我制定一个因子投资方向的 3 个月研究计划，目标是能写一份完整的 factor report", "expected_behavior": "应分解为文献综述、因子构建、回测验证、稳健性检验、报告撰写等阶段，每步有具体 deliverable", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["factor_investing", "learning_plan", "research_planning"]},
    {"scenario": "S7", "query": "我想入门机器学习量化，应该先学什么再学什么？", "expected_behavior": "应给出有序路径（Python 基础→统计学习→特征工程→模型选择→回测验证），推荐资源与练习项目", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["machine_learning", "learning_plan", "research_planning"]},
    {"scenario": "S7", "query": "帮我规划市场微观结构方向的学习路线，我对 order book 数据很感兴趣", "expected_behavior": "应规划从微观结构理论、LOB 数据格式、特征工程到预测模型的渐进路径，含论文与数据实践建议", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["market_microstructure", "order_book", "research_planning"]},
    {"scenario": "S7", "query": "我每周只有 10 小时，想系统学 Python 量化回测，怎么安排？", "expected_behavior": "应给出按周划分的 realistic 计划（pandas→回测框架→策略实现→评估），考虑时间约束与优先级", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["python", "backtesting", "research_planning"]},
]

# ── S8/S9/S10 (10) ──────────────────────────────────────────────
S8 = [
    {"scenario": "S8", "query": "根据我之前研究过的动量和均值回归策略，你觉得我接下来适合看什么方向？", "expected_behavior": "应引用用户历史研究偏好（Store 中的 profile/bookmarks），推荐相关延伸方向（如因子组合、regime switching），并说明推荐理由", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["long_term_memory", "personalization", "strategy_recommendation"]},
    {"scenario": "S8", "query": "我收藏过的那些动量论文里，哪几篇最适合深入做 follow-up 研究？", "expected_behavior": "应读取用户 bookmarks 列表，对每篇论文给出一句话摘要与 follow-up 建议（如换市场、加 vol scaling）", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["bookmarks", "long_term_memory", "momentum"]},
    {"scenario": "S8", "query": "结合我的研究兴趣偏好，推荐 3 个适合我的 arXiv 论文关键词搜索", "expected_behavior": "应基于 Store 中用户 profile（如偏好 factor investing、emerging markets），给出个性化搜索关键词与预期收获", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["user_profile", "long_term_memory", "paper_search"]},
]

S9 = [
    {"scenario": "S9", "query": "Citadel 的 quant researcher 面试会考什么？我的动量因子回测项目怎么介绍？", "expected_behavior": "应同时覆盖面试题型（概率、因子、编程）与动量项目介绍框架，合并 Research 知识与 Interview 技巧", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": True, "requires_code": False}, "tags": ["cross_domain", "Citadel", "momentum"]},
    {"scenario": "S9", "query": "Two Sigma 面试准备：我做过配对交易回测，应该怎么讲才能体现 research depth？", "expected_behavior": "应结合配对交易的技术要点（协整、spread 建模）与面试表达技巧，给出结构化自我介绍与追问应对", "difficulty": "hard", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["cross_domain", "pairs_trading", "interview"]},
    {"scenario": "S9", "query": "申请 Jane Street 需要准备什么？我目前的因子研究经历够不够？", "expected_behavior": "应评估用户研究经历与 JS 要求（数学、编程、交易直觉）的匹配度，给出差距分析与补强建议", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": True, "requires_code": False}, "tags": ["cross_domain", "Jane_Street", "career"]},
    {"scenario": "S9", "query": "面试问到 Fama-French 模型时，怎么结合我读过的论文和我的回测经验回答？", "expected_behavior": "应融合 FF 理论解释、论文引用与用户个人回测经验，展示跨 Research 与 Interview 的综合表达能力", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": True, "uncertainty_stated": False, "requires_code": False}, "tags": ["cross_domain", "Fama_French", "interview"]},
]

S10 = [
    {"scenario": "S10", "query": "帮我记录一下我申请了 Two Sigma 的 Quant Researcher 职位，状态是 Phone Screen", "expected_behavior": "应调用求职追踪工具更新 applications 记录（公司、职位、状态 Phone Screen），并确认写入成功", "difficulty": "easy", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["application_tracking", "Two_Sigma", "job_search"]},
    {"scenario": "S10", "query": "我目前所有求职申请的状态汇总一下，哪些需要跟进？", "expected_behavior": "应查询 applications 表，按状态分类汇总，标注需要跟进的申请（如超过一周无回复的）", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["application_tracking", "status_summary", "job_search"]},
    {"scenario": "S10", "query": "Citadel 的申请状态更新为 Final Round，接下来我应该准备什么？", "expected_behavior": "应更新申请状态为 Final Round，关联 Citadel 面试题库，并给出 final round 准备建议（深度技术、案例研究）", "difficulty": "medium", "eval_criteria": {"factual_grounding": True, "cites_sources": False, "uncertainty_stated": False, "requires_code": False}, "tags": ["application_tracking", "Citadel", "interview_prep"]},
]

S8_S9_S10 = S8 + S9 + S10


def interleave_round_robin(groups: list[list[dict]]) -> list[dict]:
    """Round-robin interleave scenario groups."""
    result: list[dict] = []
    for batch in zip_longest(*groups):
        for item in batch:
            if item is not None:
                result.append(item)
    return result


def fix_scenario_clustering(entries: list[dict], max_passes: int = 500) -> list[dict]:
    """Break up tail clusters when round-robin leaves extra items from large groups."""
    for _ in range(max_passes):
        swapped = False
        for i in range(len(entries) - 1):
            if entries[i]["scenario"] == entries[i + 1]["scenario"]:
                for j in range(len(entries)):
                    if j in (i, i + 1):
                        continue
                    if entries[j]["scenario"] != entries[i]["scenario"]:
                        if j > 0 and entries[j - 1]["scenario"] == entries[i]["scenario"]:
                            continue
                        if j + 1 < len(entries) and entries[j + 1]["scenario"] == entries[i]["scenario"]:
                            continue
                        entries[i + 1], entries[j] = entries[j], entries[i + 1]
                        swapped = True
                        break
            if swapped:
                break
        if not swapped:
            break
    return entries


def fix_adjacent_tags(entries: list[dict], max_passes: int = 500) -> list[dict]:
    """Swap entries to avoid consecutive rows sharing any tag."""
    for _ in range(max_passes):
        swapped = False
        for i in range(len(entries) - 1):
            a = set(entries[i]["tags"])
            b = set(entries[i + 1]["tags"])
            if a & b:
                for j in range(i + 2, len(entries)):
                    c = set(entries[j]["tags"])
                    if not (a & c) and (j + 1 >= len(entries) or not (c & set(entries[j + 1]["tags"]))):
                        entries[i + 1], entries[j] = entries[j], entries[i + 1]
                        swapped = True
                        break
        if not swapped:
            break
    return entries


def validate(entries: list[dict]) -> None:
    counts = Counter(e["scenario"] for e in entries)
    diff = Counter(e["difficulty"] for e in entries)
    assert len(entries) == 120
    assert counts["S1"] == 20
    assert counts["S2"] == 30
    assert counts["S3"] == 20
    assert counts["S4"] == 10
    assert counts["S5"] == 10
    assert counts["S6"] == 15
    assert counts["S7"] == 5
    assert counts["S8"] == 3
    assert counts["S9"] == 4
    assert counts["S10"] == 3
    assert 34 <= diff["easy"] <= 38
    assert 58 <= diff["medium"] <= 62
    assert 22 <= diff["hard"] <= 26
    for e in entries:
        if e["scenario"] == "S2":
            assert e["eval_criteria"]["requires_code"] is True


def main() -> None:
    groups = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10]
    for g in groups:
        assert len(g) == len(set(x["query"] for x in g)), "duplicate queries in group"

    interleaved = interleave_round_robin(groups)
    interleaved = fix_scenario_clustering(interleaved)
    interleaved = fix_adjacent_tags(interleaved)

    for i, entry in enumerate(interleaved, 1):
        entry["id"] = f"bench_{i:03d}"

    validate(interleaved)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for entry in interleaved:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    counts = Counter(e["scenario"] for e in interleaved)
    diff = Counter(e["difficulty"] for e in interleaved)
    print(f"Wrote {len(interleaved)} entries to {OUT}")
    print("scenario:", dict(sorted(counts.items())))
    print("difficulty:", dict(sorted(diff.items())))


if __name__ == "__main__":
    main()
