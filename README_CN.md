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

# 预算搜索：每策略先取 256 个候选，再按 3 倍比例确定性淘汰
momentum-lab GLD --successive-halving --workers 4
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

## v0.14.1 完整性修复

修复特殊资产代码的缓存串用、CSV 浮点精度漂移、恒定收益与破产统计错误、下一开盘成交的敏感性分析口径不一致、缺失来源配置仍可续跑、重跑备份不完整，以及单资产揭示结果缓存顺序错误。补充严格参数／配置校验和独立临时文件的原子写入，CI 强制使用已提交的依赖锁；原有 24 组冻结核算基准保持不变。

复现证据、测试、兼容性和限制见[审计记录](AUDIT_2026-09-03.md)。升级后应新建实验／研究协议；旧实验保留原来的源码、环境和访问历史，不绕过版本及来源校验。

## v0.14 功能升级

- 新增**组合级研究封存**：先固定规则并评估开发期，再通过单独命令显式揭示测试期。计算测试结果前，以一个事务记录所有候选资产的访问，兼容已有单资产及全历史组合研究记录。
- 支持带公告日、生效日的历史成分名单。动量策略和比较基准都遵守当时的成员资格；名单变化可触发额外调仓，但仍延迟到下一交易日收盘成交。
- 测试期承接已有持仓、现金及待执行指令，计入第一天收益，分别披露策略和基准的期初净值。同一研究再次查看复用已固定的结果，并记录重放，不重新回测。
- 新增 2 组独立核算的名单／测试边界基准、完整流程与并发审计测试，以及独立安装包验收；原有 16 + 6 组软件基准不变。

```bash
# 示例中的 AAA/BBB 和名单事件为合成示例；请替换为合法取得、严格对齐的数据。
momentum-lab portfolio study --config examples/portfolio_study_config.json --run-id development
momentum-lab portfolio study status relative-momentum-study-v1
momentum-lab portfolio study --config examples/portfolio_study_config.json --run-id test-reveal --reveal-test
momentum-lab portfolio study benchmark
```

详细用法见 [PORTFOLIO_STUDIES.md](PORTFOLIO_STUDIES.md) 和 [MEMBERSHIP.md](MEMBERSHIP.md)。封存是本地流程与审计控制，不证明数据从未被观察过；历史名单是用户声明，并非独立验证的历史数据。当前仍要求所有候选资产在完整区间内都有正价格，包括非成员期间；**IPO 前缺价、退市缺价及清算分配尚不支持**，不会凭空补价或假设零回收。尚无组合参数搜索或实盘交易。

## v0.13 功能升级

- 新增多资产横截面动量：比较资产间的历史涨幅，按排名选择前 K 个持仓，可配置正动量过滤、目标权重上限及日/周/月频调仓；不够入选的名额保留为现金。
- 新增共享现金的只做多组合账户：信号延迟至下一根日线收盘成交，计入买卖两侧费用和现金利息；非调仓日保持持仓数量，权重随行情漂移。
- 输出组合资金账、持仓、交易、目标与实际权重，并生成离线 HTML/Markdown 报告；基准为相同预热期和成本口径下的等权买入持有组合。
- 首版明确限定为**全历史探索研究**，不是组合样本外检验或实盘。每次调用必须显式确认；计算前将每个资产的完整使用区间登记为开发期访问，避免以后把这些历史重新称为未观察过的测试数据。
- 新增 6 组独立精确核算、SHA-256 锁定的组合基准；原有 16 组单资产回归基准保持不变。

```bash
# 先导入有权使用、日期严格对齐且同币种的离线数据集。
# 按自己的文件修改示例中的资产、路径、日期及成本假设。
momentum-lab portfolio --config examples/portfolio_config.json --acknowledge-history
momentum-lab portfolio benchmark
```

完整公式、Python API、数据契约和输出说明见 [PORTFOLIOS.md](PORTFOLIOS.md)。权重上限只约束调仓目标，实际权重可能漂移超限。组合现金采用 ACT/365 年有效利率复利，与原单资产引擎的 ACT/365.25 单利不同。v0.14 增加了声明式历史名单和固定规则组合封存；已验证的历史数据、完整退市处理、组合参数选择、外汇换算和实盘仍属后续工作。合成基准用于检验软件，不是收益承诺。

## v0.12 功能升级

- 新增离线 CSV 数据集：明确记录来源、使用条款、计价货币、日频日历和复权口径，不再只依赖 Yahoo 在线下载；文件有问题时不会自动替换为线上数据。
- 校验 OHLCV、日期、重复行和 SHA-256；保留原始 CSV 字节，导入时新建目录，不覆盖已有数据。
- 数据来源信息写入 JSON、Markdown 和 HTML 报告，并锁定到研究协议与断点恢复；移动未变动的数据集不影响恢复，修改数据或声明则需新建实验。
- 离线数据沿用测试封存、显式揭示和重复访问审计。来源、使用条款和复权标签是用户声明，不代表已验证授权、质量或真实历史收益。

```bash
momentum-lab data import my-prices.csv --output datasets/spy-v1 --ticker SPY \
  --source "自行合法取得的数据导出" --license "仅限内部研究" \
  --currency USD --calendar exchange --price-adjustment split_and_dividend_adjusted
momentum-lab data inspect datasets/spy-v1/manifest.json
momentum-lab SPY --dataset datasets/spy-v1/manifest.json --start 2020-01-01 \
  --study-id spy-local-v1 --run-id spy-local-v1
```

CSV 格式为 `date,open,high,low,close[,volume]`，日期使用 `YYYY-MM-DD`。本地模式未指定结束日期时使用文件的最后一天，不是今天。请根据自己的文件选择代码和日期，确保有权使用；项目未捆绑商业历史行情。完整契约、复权/成交量限制、揭示和 Python API 见 [DATASETS.md](DATASETS.md)。

## v0.11 功能升级

- 新增研究登记模式：使用 `--study-id` 固定数据、策略范围与评估规则，默认不计算或展示测试成绩。
- 共享 SQLite 登记库按股票/资产代码和日期重叠识别历史访问；换实验编号、结果目录或数据版本不会清除同一登记库中的记录。以前用作训练/验证的日期也会被识别。
- 揭示前先提交访问记录；中断仍算可能已访问。重复计算需要明确确认及理由，同一固定研究再次查看则复用已有结果并记录访问，不标成新的首次测试。
- 报告区分封存、首次记录、已经揭示、重复使用及历史未知。旧命令仍可运行并自动测试，但现在会记录访问、明确提示历史限制。

```bash
# 第一步：登记、选型，只查看开发期证据
momentum-lab GLD --study-id gld-2026q3 --run-id gld-dev --end 2026-08-28

# 查看登记状态，不显示缓存的测试成绩
momentum-lab study status gld-2026q3

# 第二步：保持配置和数据不变，明确揭示已冻结策略
momentum-lab GLD --study-id gld-2026q3 --run-id gld-dev --end 2026-08-28 \
  --resume --reveal-test

# 查看历史、导入已知旧实验（不改写旧文件）
momentum-lab study history --ticker GLD
momentum-lab study import-legacy experiments/old-run
```

已知重叠会阻止新的登记模式测试；确需历史复核时，加 `--allow-test-reuse --test-reuse-reason "历史复核，非新增样本外证据"`。这些确认参数不能写进 JSON 配置自动启用。不要换登记库规避记录，注意备份；旧版本实验不会被自动扫描。

“首次记录”不等于“从未查看过”：登记库之外的历史始终未知，本地文件也不是防篡改或加密托管系统。登记库路径、兼容迁移、崩溃恢复和完整方法见 [GOVERNANCE.md](GOVERNANCE.md)。

## v0.10 功能升级

- 对最终选定策略新增配对区块 Bootstrap 置信区间，覆盖算术年化平均收益、夏普比率，以及相对买入持有的年化平均超额收益；验证期与测试期分别计算。
- 新结果只作选型后的诊断，不改变候选排名、现有解析夏普区间、多重检验门槛或回测账本。
- 报告记录随机种子、区块长度、样本量和无法估计的原因；不悄悄删除缺失观测或未定义的重采样结果，续跑时锁定相关参数。

```bash
# 默认：2000 次重采样、10 根 K 线的循环区块、95% 区间、种子 42
momentum-lab GLD

# 请在查看结果之前确定参数，不要根据测试期表现调整
momentum-lab GLD --bootstrap-resamples 2000 --bootstrap-block-length 20 \
  --bootstrap-confidence 0.95 --bootstrap-seed 42

# 关闭区间估计，候选排名保持不变
momentum-lab GLD --no-bootstrap
```

结果位于最终 `summary.json` 和 Python 返回值的 `bootstrap_diagnostics` 中，同时展示在 Markdown/HTML 报告里。默认至少需要 60 个观测及 5 个名义区块；样本不足、零波动、退化重采样或超出计算预算时会说明原因，不编造精确区间。

方法假设收益近似平稳，**不能修正策略筛选偏差或反复查看测试集造成的泄漏**。算术年化平均收益不是 CAGR。完整口径及 Python API 见 [UNCERTAINTY.md](UNCERTAINTY.md)。

## v0.9 功能升级

- 新增 `momentum-lab benchmark`：4 类离线合成行情 × 4 个固定策略，共 16 个案例，覆盖交易日/自然日、趋势反转、跳空、借券受限及流动性压力。
- 用 SHA-256 锁定行情与参数假设；比较每根 K 线的完整账本和指标，不只比较期末收益或四舍五入后的夏普比率。
- 自动生成版本对比 JSON 与 Markdown 报告，记录耗时和内存分配峰值。数值变好或变坏都需要审查，性能阈值可选，参考结果不会自动覆盖。
- CI 在 Python 3.10—3.13 执行冻结回归并上传报告，还会验证干净 wheel 在源码目录之外能读取内置基准。

### 运行冻结回归（无需联网）

```bash
# 与随包提供的已复核软件回归参考结果比较
momentum-lab benchmark --output experiments/benchmarks/check-090

# 在自己的机器上比较两个版本，保留旧版本快照
momentum-lab benchmark --repeat 3 --output experiments/benchmarks/before
# 安装待比较的新版本后，使用新的输出目录
momentum-lab benchmark --repeat 3 \
  --compare experiments/benchmarks/before/snapshot.json \
  --output experiments/benchmarks/after
```

每次输出 `snapshot.json`、`comparison.json` 和 `report.md`。退出码：`0` 表示结果兼容，`1` 表示数值或启用的性能指标发生超限变化，`2` 表示输入无效或不可比较。已有输出目录会拒绝覆盖。

可追加 `--max-slowdown 1.5 --max-memory-growth 1.5` 设置单案例的资源比值上限；两侧必须都有测量结果，随包参考结果不含机器相关的性能数字。内存是 `tracemalloc` 跟踪的分配峰值，不是进程总 RSS。

这些是**合成软件测试数据，不是真实历史收益或样本外有效性的证据**，不会参与策略搜索与选型。详细口径、Python API 和参考结果更新规则见 [BENCHMARKS.md](BENCHMARKS.md)。

## v0.8 功能升级

- 新增可配置候选预算、淘汰比例和验证资源阶段的确定性 Successive Halving；只有完成全部开发期评估的幸存者才会进入正式排名，但所有阶段评估仍会计入多重检验门槛。
- 阶段结果以事务方式写入 SQLite，并导出 `search_stages.csv`；中断后可从未完成阶段继续，无需重复计算。
- 候选进程现在只能获得开发期观测，而不只是看不到测试边界；封存测试数据只在最终选型后释放一次。
- 新增跨策略共享、按进程有界的指标 DAG 缓存，可复用收益率、均线、波动率、RSI、通道、ATR、ADX 及其依赖节点；报告会展示阶段搜索和缓存效率。

## v0.7 功能升级

- 回测可限制单根 K 线成交量参与率；容量不足时按 bar 逐步部分成交，不再默认流动性无限。
- 成交模型新增完整买卖价差、基于参与率的非线性市场冲击、初始资金规模和单次最低手续费。
- 现金、融资、融资利差、借券费、抵押品返息及借券可用性可使用带日期的 pandas Series；稀疏数据只使用已知历史值前向填充，不会回填未来信息。
- 回测结果新增目标换手、实际换手、实际成交持仓、交易成本、参与率、容量受限和借券受限序列；搜索指标以实际成交路径为准。
- 每次完成的搜索除机器可读文件外，还会自动生成便携的 `report.md` 和自包含 `report.html` 研究报告。

## v0.6 主要修复

- 搜索进程只获得开发期观测及训练/验证边界；全部候选与 Top 文件不再包含测试指标，完成选型后才对复制出的最终候选评估一次测试期。
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
        -> 候选评估阶段无法获得测试期观测
        -> 可选：只用验证期递增资源进行分阶段淘汰
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

v0.7 完整空间为 291,188 个组合，其中非 ML 159,668 个、ML 131,520 个。组合数量不是研究质量指标。

## Python API

```python
from momentum_lab import SearchConfig, run_search

config = SearchConfig(
    ticker="GLD",
    strategies=["tsmom", "ma_cross", "regime_aware"],
    search_method="successive_halving",
    candidate_budget=256,
    halving_factor=3,
    halving_stages=3,
    indicator_cache_size=256,
    risk_free_rate=0.0,
    cash_rate=0.0,
    financing_rate=0.05,
    financing_spread=0.01,
    spread_bps=4.0,
    impact_bps=2.0,
    max_participation=0.05,
    initial_capital=1_000_000,
    min_fee=0.50,
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
import pandas as pd

from momentum_lab import backtest, evaluate, get_strategy, prepare_data

data, frame = prepare_data("BTC-USD")  # 自动使用 365 年化
positions = get_strategy("tsmom").run(data, lookback=63, threshold=0.01, long_short=False)
simulation = backtest(
    positions,
    frame["close"],
    volume=frame["volume"],
    cost_bps=2,
    slippage_bps=3,
    spread_bps=4,
    impact_bps=2,
    impact_reference_participation=0.01,
    max_participation=0.05,
    initial_capital=1_000_000,
    min_fee=0.50,
    cash_rate=0.03,
    financing_rate=0.06,
    financing_spread=0.01,
    short_rebate_rate=0.01,
    max_leverage=1.0,
    execution_lag=1,
    annualization=365,
)
print(evaluate(simulation["returns"], risk_free_rate=0.03, annualization=365))
print(simulation["capacity_constrained"].sum(), "个容量受限 bar")
```

如果需要使用历史资金或借券条件，可以传入有序 pandas Series。首个观测必须覆盖首根价格 bar，后续稀疏数据只会前向填充：

```python
cash_curve = pd.Series([0.01, 0.05], index=pd.to_datetime(["2020-01-01", "2022-03-17"]))
borrow_available = pd.Series([True, False], index=pd.to_datetime(["2020-01-01", "2021-01-28"]))
simulation = backtest(positions, frame["close"], cash_rate=cash_curve, borrow_available=borrow_available)
```

## 命令行参数

```text
momentum-lab TICKER [选项]

  --quick                 每个策略取 5 个代表性组合（默认）
  --exhaustive            完整参数网格，必须显式启用
  --successive-halving    确定性的分阶段预算搜索
  --candidate-budget N    每策略初始候选数（默认 256）
  --halving-factor N      每阶段候选缩减比例（默认 3）
  --halving-stages N      最大验证资源阶段数（默认 3）
  --indicator-cache-size N
                          每进程可复用指标节点数；0 表示关闭
  --strategies NAMES      逗号分隔策略名（默认仅非 ML）
  --all-strategies        加入实验性 ML 策略
  --config PATH           加载完整 SearchConfig JSON
  --resume                对完全一致的 run-id 续跑
  --workers N             并行进程数（默认 1）
  --cost BPS              线性交易成本（默认 1）
  --slippage BPS          额外线性滑点（默认 0）
  --spread-bps BPS        完整买卖价差（默认 0）
  --impact-bps BPS        参考参与率下的市场冲击
  --impact-exponent X     非线性冲击指数（默认 0.5）
  --impact-reference-participation X
                          冲击报价对应的参与率（默认 0.01）
  --max-participation X   单 bar 最大成交量参与率
  --initial-capital N     用于容量和最低费的初始净值（默认 1,000,000）
  --min-fee N             每次再平衡的最低货币手续费（默认 0）
  --cash-rate RATE        年化现金收益（默认 0）
  --financing-rate RATE   超过 1 倍敞口的融资利率（默认 0）
  --financing-spread R    融资基准利率之上的年化利差（默认 0）
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
  --no-report             不生成 Markdown/HTML 报告
  --robust / --no-robust  开关局部参数敏感性分析
  --robust-frac F         局部扰动比例（默认 0.2）
  --list                  列出策略和完整网格数量
  --version               显示版本
```

## 输出文件

每次运行保存在 `experiments/<run_id>/`：

- `run_config.json`：包/schema/Git 信息、源码/数据/lockfile/环境哈希、区间、策略、成本和风险假设。
- `results.sqlite3`：事务性的正式断点与续跑日志。
- `search_stages.csv`：Successive Halving 的验证期阶段证据与晋级决定。
- `all_results.csv`：原子导出的可读结果，仅含训练/验证指标。
- `top_results.csv`：按验证证据排序的候选，不含测试指标。
- `robustness.csv`：启用时输出局部参数敏感性摘要。
- `summary.json`：选定结果、买入持有基准、敏感性结果、实验数、错误数和续跑统计。
- `report.md` 与 `report.html`：包含假设、诊断、验证期、封存测试期及基准结果的便携研究报告。

## 当前方法局限

- Deflated Sharpe 使用偏保守的独立试验近似；基于时间折的 CSCV/PBO 只能作为诊断，不能证明没有过拟合。
- 如果不断新建 run-id 并反复观察最终测试报告，人为层面仍会形成测试集泄漏；代码不能替代研究预注册制度。
- 局部参数敏感性仍是描述性指标，本身不能校正多重检验。
- 流动性模型使用 bar 级汇总成交量和参数化冲击曲线，不是订单簿回放；排队位置、bar 内路径、场所分散、停牌和税费仍未覆盖。
- 搜索配置目前使用标量市场假设；带日期的利率及借券可用性序列通过直接 Python 回测 API 提供。
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
