import re
import textwrap

import pytest

from src.sandbox import SandboxResult, SandboxRunner

BACKTRADER_SAMPLE_CODE = textwrap.dedent("""\
import backtrader as bt
import datetime

class SmaCross(bt.Strategy):
    params = dict(pfast=20, pslow=60)

    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.params.pfast)
        sma_slow = bt.ind.SMA(period=self.params.pslow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.sell()

import pandas as pd
import numpy as np

np.random.seed(42)
n = 300
dates = pd.date_range('2020-01-01', periods=n, freq='B')
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
data_df = pd.DataFrame({
    'open': close * 0.999,
    'high': close * 1.005,
    'low': close * 0.995,
    'close': close,
    'volume': np.random.randint(1_000_000, 5_000_000, n),
    'openinterest': 0,
}, index=dates)

data = bt.feeds.PandasData(dataname=data_df)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(SmaCross)
cerebro.broker.setcash(100_000)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0, _name='sharpe')

results = cerebro.run()
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
""")


@pytest.fixture
def runner():
    return SandboxRunner()


def test_forbidden_import(runner):
    result = runner.run("import requests", sample_data_path="/tmp/data.csv")

    assert isinstance(result, SandboxResult)
    assert result.success is False
    assert result.phase == "syntax"
    assert result.error_code == "FORBIDDEN_IMPORT"


def test_backtrader_execution(runner):
    result = runner.run(BACKTRADER_SAMPLE_CODE, sample_data_path="/tmp/data.csv", timeout_s=30)

    assert result.success is True
    assert result.phase == "output"
    assert re.search(r"\d", result.stdout)


def test_syntax_error(runner):
    result = runner.run("if True\n    print('broken')", sample_data_path="/tmp/data.csv")

    assert result.success is False
    assert result.phase == "syntax"
    assert result.error_code == "SYNTAX_ERROR"


def test_empty_output(runner):
    result = runner.run("x = 1", sample_data_path="/tmp/data.csv")

    assert result.success is False
    assert result.phase == "output"
    assert result.error_code == "OUTPUT_VALIDATION_FAILED"


def test_timeout(runner):
    result = runner.run("while True: pass", sample_data_path="/tmp/data.csv", timeout_s=2)

    assert result.success is False
    assert result.phase == "execute"
    assert result.timed_out is True
    assert result.error_code == "TIMEOUT"
