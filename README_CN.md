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

# 默认：18 个非 ML 策略各取 5 个代表性参数，并将结果流式写入磁盘
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

## v0.5 主要修复

- 默认改为快速、非 ML、流式结果，避免一次命令触发数日计算和 1GB 以上内存占用。
- 风险仓位不再作为 alpha 参数参与搜索，避免 Sharpe 机械偏好更大杠杆。
- 所有策略在回测出口统一执行最终杠杆上限，内部缩放不能再把 2 倍上限放大为 4 倍。
- 闲置现金可以获得显式现金收益；融资成本只对超过 1 倍的敞口收取。
- 加密资产默认按 365 年化，其他日频资产默认按 252 年化，也可手动覆盖。
- 对用户承诺的结束日期真正包含当天；内部自动处理 yfinance 的排他性 `end`。
- Yahoo 下载增加有限次数的指数退避重试。
- 数据缓存迁移到操作系统用户缓存目录，并采用原子写入。
- 断点文件使用固定字段；数据、策略、搜索模式、包版本、schema、Git SHA、成本或风险设置不同均拒绝混合续跑。
- scikit-learn 和 XGBoost 改为可选依赖。
- 原“稳健性等级”明确改称局部参数敏感性；它不能替代多重检验校正。

## 研究流程

```text
股票代码 + 版本化运行配置
        -> 复权日频 OHLCV 数据
        -> 互不重叠的训练 / 验证 / 测试区间
        -> 策略与参数评估
        -> 按验证集 Sharpe 排名
        -> 对选定结果报告一次测试集表现
        -> 局部参数敏感性分析
        -> 断点、摘要、基准和运行清单
```

当前 train/validation/test 流程只是研究基线，并非最终方法。正式 1.0 版本之前，路线图将加入嵌套 walk-forward、Deflated Sharpe Ratio 和 Probability of Backtest Overfitting。

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

v0.5 完整空间为 291,188 个组合，其中非 ML 159,668 个、ML 131,520 个。组合数量不是研究质量指标。

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
    result_dir="experiments",
    run_id="gld-research-v1",
    keep_all_results=False,
)

result = run_search(config=config)
print(result["best"])
print(result["benchmark_metrics"])
print(result["parameter_sensitivity"])

# 只有源码、数据、策略集合、搜索模式与成本模型完全一致才允许续跑
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
    max_leverage=1.0,
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
  --max-leverage X        最终绝对敞口上限（默认 2）
  --annualization N       覆盖自动推导的 252/365
  --risk-free-rate RATE   评价指标门槛利率（默认 0）
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

- `run_config.json`：包/schema/Git 版本、源码与数据哈希、区间、策略、成本和风险假设。
- `all_results.csv`：固定字段、增量写入的断点文件。
- `top_results.csv`：按验证集 Sharpe 排名的 Top 结果。
- `robustness.csv`：启用时输出局部参数敏感性摘要。
- `summary.json`：选定结果、买入持有基准、敏感性结果、实验数、错误数和续跑统计。

## 当前方法局限

- 当前仍然在单一验证窗口上从较大策略族中选择结果。
- 局部参数敏感性不能校正多重检验和选择偏差。
- 收盘价信号对应下一段 close-to-close 收益，尚未提供精确的 MOC/次日开盘成交模型。
- 线性成本模型尚未覆盖买卖价差、市场冲击、容量、停牌、借券可得性和税费。
- yfinance 适合个人研究，不能直接作为商业数据再分发层。

在独立验证、现实成交假设和 forward/paper testing 完成前，不应将结果直接用于真实资金配置。

## 开发与测试

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest -m "not network" -q
```

当前 CI 覆盖 Python 3.10—3.13，并在全新环境构建、安装 wheel，同时执行覆盖率门槛。定时联网测试和自动发布仍在路线图中。

## 许可证

MIT
