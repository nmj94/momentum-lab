# Momentum Lab

**可复现的动量策略研究与参数敏感性评估工具。**

[English](README.md) | [中文](README_CN.md)

Momentum Lab 用于在明确的数据、成交、成本和风险假设下，对比经典动量规则、实验性机器学习模型和市场状态策略。

本项目不再宣称能够找到普遍适用的“最优策略”。当大量参数被反复测试时，漂亮的历史结果可能来自选择偏差或过拟合。输出是研究证据，不是投资建议。

## 安全的快速开始

项目目前应从源码安装。PyPI 上的 `momentum-lab` 名称属于一个无关项目，请勿运行 `pip install momentum-lab`。

```bash
git clone https://github.com/nmj94/momentum-lab.git
cd momentum-lab
pip install -e .

# 默认：非 ML 策略各取 5 个确定性的拉丁超立方样本，并将结果流式写入磁盘
momentum-lab GLD

# 指定策略
momentum-lab SPY --strategies tsmom,ma_cross,rsi,regime_aware

# 非 ML 完整网格：159,668 次实验，必须显式启用
momentum-lab GLD --exhaustive --workers 4
```

机器学习策略目前属于实验功能，不再默认安装或运行：

```bash
# scikit-learn 策略
pip install -e ".[ml]"
momentum-lab SPY --strategies ml_logreg,ml_rf

# XGBoost 策略
pip install -e ".[xgb]"
momentum-lab SPY --strategies ml_xgb

# 全部 26 个策略；完整 ML 网格可能运行数日
momentum-lab SPY --all-strategies
momentum-lab SPY --all-strategies --exhaustive --workers 8
```

未来发布使用的 distribution name 为 `momentum-research-lab`；Python import 和命令行名称仍为 `momentum_lab` 与 `momentum-lab`。

## v0.6 主要修复

- 搜索进程只获得训练和验证边界；全部候选与 Top 文件不再包含测试指标，完成选型后才对复制出的最终候选评估一次测试期。
- 选型加入最小样本、交易次数和敞口约束，并使用 Deflated Sharpe 概率；同时输出时间折、95% Sharpe 区间、扩展式 walk-forward 选型回放及 CSCV/PBO 估计。
- 账户破产后净值固定为零，不可能因后续杠杆收益“复活”。
- 默认采用次日收盘成交，并显式支持同收盘、次日开盘和延迟收盘模型。
- 融资与借券费用按真实日历天数计提；目标权重漂移产生再平衡换手；做空现金、抵押品返息和借券费分别核算。
- SQLite 成为事务性续跑日志；CSV/JSON 原子导出；运行清单记录 lockfile 及完整运行环境。
- ML 可以预测未来标签未知的尾部数据；已确认晚上市的资产可以正常复用缓存。
- 快速模式改用固定种子的拉丁超立方抽样；不同平滑参数可复用昂贵的基础信号。

## 研究流程

```text
股票代码 + 版本化运行配置
        -> 复权日频 OHLCV 数据
        -> 40% 训练 / 40% 时间折验证 / 20% 封存测试
        -> 候选评估阶段无法获得测试边界
        -> 约束后的 Deflated Sharpe 选型及 walk-forward/PBO 诊断
        -> 仅对复制出的最终候选报告一次测试期表现
        -> 局部参数敏感性分析
        -> 断点、摘要、基准和运行清单
```

这些措施只能降低、不能消除数据挖掘风险。独立数据验证和前向模拟仍是 1.0 的发布门槛。

## 策略范围

### 经典动量策略（15）

| 策略 | 说明 |
|---|---|
| TSMOM | 时间序列动量 |
| MA Cross | SMA/EMA/WMA/DEMA 均线交叉 |
| MACD | 交叉、柱体变化和零轴过滤 |
| RSI | 动量与反转模式 |
| ROC | 变化率动量 |
| Bollinger | 突破与均值回归 |
| Donchian | 带退出通道的持续突破持仓 |
| Dual Momentum | 单资产绝对动量 |
| Triple MA | 三均线排列 |
| Vol Scale | 动量加波动率缩放 |
| Acceleration | 长短周期收益加速度 |
| Z-Score | 有状态的动量/回归规则 |
| Heikin Ashi | 平滑 K 线方向 |
| Supertrend | 基于 ATR 的趋势规则 |
| Multi Breakout | 多周期突破投票 |

### 实验性 ML 策略（8）

逻辑回归、随机森林、XGBoost、KNN、SVM、Gaussian Naive Bayes、AdaBoost 和 Extra Trees。Walk-forward 训练会 purge 与预测边界重叠的标签；网格初始训练窗口不低于 252 个样本；KNN 中样本不足的组合会在运行前剔除。

### 组合与市场状态策略（3）

Ensemble、Stacked 和 Regime Aware。

v0.6 完整空间为 291,188 个组合，其中非 ML 159,668 个、ML 131,520 个。组合数量不是研究质量指标。

## Python API

```python
from momentum_lab import SearchConfig, run_search

config = SearchConfig(
    ticker="GLD",
    strategies=["tsmom", "ma_cross", "regime_aware"],
    quick=True,
    risk_free_rate=0.0,
    cash_rate=0.0,
    financing_rate=0.05,
    max_leverage=1.5,
    execution_model="next_close",
    validation_folds=4,
    min_validation_trades=2,
    result_dir="experiments",
    run_id="gld-research-v1",
    keep_all_results=False,
)

result = run_search(config=config)
print(result["best"])
print(result["benchmark_metrics"])
print(result["parameter_sensitivity"])

# 只有源码树、运行环境、数据、策略空间与会计模型完全一致才允许续跑；Git SHA 仅作记录
resumed = run_search(config=config, resume=True)
print(resumed["n_skipped"], resumed["n_errors"])
```

单策略接口仍然可用：

```python
from momentum_lab import backtest, evaluate, get_strategy, prepare_data

data, frame = prepare_data("BTC-USD")  # 自动使用 365 年化
positions = get_strategy("tsmom").run(data, lookback=63, threshold=0.01, long_short=False)
simulation = backtest(
    positions,
    frame["close"],
    cost_bps=2,
    slippage_bps=3,
    cash_rate=0.03,
    financing_rate=0.06,
    short_rebate_rate=0.01,
    max_leverage=1.0,
    execution_lag=1,
    annualization=365,
)
print(evaluate(simulation["returns"], risk_free_rate=0.03, annualization=365))
```

## 命令行参数

```text
momentum-lab TICKER [选项]

  --quick                 每个策略取 5 个代表性组合（默认）
  --exhaustive            完整参数网格，必须显式启用
  --strategies NAMES      逗号分隔策略名（默认仅非 ML）
  --all-strategies        加入实验性 ML 策略
  --config PATH           加载完整 SearchConfig JSON
  --resume                对完全一致的 run-id 续跑
  --workers N             并行进程数（默认 1）
  --cost BPS              线性交易成本（默认 1）
  --slippage BPS          额外线性滑点（默认 0）
  --cash-rate RATE        年化现金收益（默认 0）
  --financing-rate RATE   超过 1 倍敞口的融资利率（默认 0）
  --borrow-bps BPS        年化做空借券费（默认 0）
  --short-rebate-rate R   年化做空抵押品返息（默认 0）
  --max-leverage X        最终绝对敞口上限（默认 2）
  --execution-model M     same_close、next_close、next_open、delayed_close
  --execution-lag N       仅供 delayed_close 使用的延迟 bar 数
  --annualization N       覆盖自动推导的 252/365
  --risk-free-rate RATE   评价指标门槛利率（默认 0）
  --validation-folds N    偶数个时间验证折（默认 4）
  --min-val-bars N        最少验证样本（默认 60）
  --min-val-trades N      最少验证交易（默认 1）
  --min-val-exposure X    最低平均绝对敞口（默认 0.01）
  --start DATE            开始日期（默认 2004-01-01）
  --end DATE              包含在结果中的结束日期
  --refresh               忽略市场数据缓存
  --top N                 Top 结果数量（默认 50）
  --result-dir DIR        运行产物父目录
  --run-id ID             稳定运行目录名
  --keep-all              在内存保留全部结果（默认流式写盘）
  --robust / --no-robust  开关局部参数敏感性分析
  --robust-frac F         局部扰动比例（默认 0.2）
  --list                  列出策略和完整网格数量
  --version               显示版本
```

## 输出文件

每次运行保存在 `experiments/<run_id>/`：

- `run_config.json`：包/schema/Git 信息、源码/数据/lockfile/环境哈希、区间、策略、成本和风险假设。
- `results.sqlite3`：事务性的正式断点与续跑日志。
- `all_results.csv`：原子导出的可读结果，仅含训练/验证指标。
- `top_results.csv`：按验证证据排序的候选，不含测试指标。
- `robustness.csv`：启用时输出局部参数敏感性摘要。
- `summary.json`：选定结果、买入持有基准、敏感性结果、实验数、错误数和续跑统计。

## 当前方法局限

- Deflated Sharpe 使用偏保守的独立试验近似；基于时间折的 CSCV/PBO 只能作为诊断，不能证明没有过拟合。
- 如果不断新建 run-id 并反复观察最终测试报告，人为层面仍会形成测试集泄漏；代码不能替代研究预注册制度。
- 局部参数敏感性仍是描述性指标，本身不能校正多重检验。
- 线性成本模型尚未覆盖买卖价差、市场冲击、容量、停牌、借券可得性和税费。
- yfinance 适合个人研究，不能直接作为商业数据再分发层。

在独立验证、现实成交假设和 forward/paper testing 完成前，不应将结果直接用于真实资金配置。

## 开发与测试

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest -m "not network" -q
```

当前 CI 覆盖 Python 3.10—3.13，并在全新环境构建、安装 wheel，执行覆盖率门槛及每周数据源契约测试。自动发布仍在路线图中。

## 许可证

MIT
