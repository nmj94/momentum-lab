# Momentum Lab

**只需提供一个股票代码，自动找到最优动量交易策略。**

[English](README.md) | [中文](README_CN.md)

Momentum Lab 自动测试 26 种策略、超过 135 万组参数组合，为你的标的找到表现最优的动量策略。包含经典动量指标、机器学习模型、以及根据市场状态自动切换子策略的自适应 Regime 策略。

## 快速开始

```bash
# 安装
pip install -e .

# 搜索黄金 ETF 的最优策略
momentum-lab GLD

# 快速搜索标普 500
momentum-lab SPY --quick

# 4 线程并行搜索比特币
momentum-lab BTC-USD --workers 4

# 只搜索指定策略
momentum-lab AAPL --strategies tsmom,ma_cross,rsi,regime_aware
```

### 配置驱动与断点续跑（P1）

把一次研究运行写成 JSON，可以在本地、CI 或后续任务中复现同一组参数：

```bash
momentum-lab --config examples/search_config.json
momentum-lab --config examples/search_config.json --resume
```

`--resume` 会读取同一 `result_dir/run_id/all_results.csv` 中已经完成的组合，只运行缺失组合；因此需要在配置中固定 `run_id`。配置文件中的字段优先于命令行默认值，支持的字段与 `SearchConfig` 一致。也可以直接在 Python API 中传入 `SearchConfig`、字典或 JSON 路径：

```python
from momentum_lab import SearchConfig, run_search

config = SearchConfig(ticker="GLD", strategies=["tsmom"], quick=True,
                      robust=False, result_dir="experiments", run_id="gld-p1")
result = run_search(config=config)
resumed = run_search(config=config, resume=True)
print(resumed["n_skipped"], "experiments loaded from checkpoint")
```

## 工作原理

```
输入一个代码（如 GLD）
        |
        v
  从 Yahoo Finance 下载数据
        |
        v
  划分 训练集 / 验证集 / 测试集
        |
        v
  测试 26 种策略 × 数千组参数
 （经典动量、机器学习、自适应 Regime）
        |
        v
  按验证集 Sharpe 排名
        |
        v
  在测试集上评估最优策略
        |
        v
  稳健性检验：扰动最优参数
  （检测过拟合/孤立峰值）
        |
        v
  输出：最优策略 + 参数 + 指标
```

## 包含的策略

### 经典动量策略（15 种）
| 策略 | 说明 | 关键参数 |
|------|------|---------|
| TSMOM | 时间序列动量 | lookback, threshold, skip_recent |
| MA Cross | 均线交叉 | fast, slow, ma_type (SMA/EMA/WMA/DEMA) |
| MACD | MACD 信号交叉 | fast, slow, signal, mode |
| RSI | RSI 动量/反转 | period, 买卖阈值, 平滑 |
| ROC | 变化率动量 | period, threshold, smoothing |
| Bollinger | 布林带突破 | period, num_std, band_width_filter |
| Donchian | 唐奇安通道突破 | period, exit_period, confirmation |
| Dual Momentum | 绝对动量 | lookback, threshold, smoothing |
| Triple MA | 三均线系统 | fast, medium, slow, ma_type |
| Vol Scale | 动量 + 波动率缩放 | lookback, vol_target, vol_lookback |
| Acceleration | 价格加速动量 | short_lb, long_lb, threshold |
| Z-Score | Z 分数动量/回归 | lookback, entry_z, exit_z |
| Heikin Ashi | HA K 线动量 | smooth, confirmation |
| Supertrend | 超级趋势 | atr_period, multiplier |
| Multi Breakout | 多周期突破投票 | periods, vote_threshold |

### 机器学习策略（8 种）
| 策略 | 说明 |
|------|------|
| ML LogReg | 逻辑回归（Walk-forward 训练） |
| ML RF | 随机森林 |
| ML XGB | XGBoost 梯度提升 |
| ML KNN | K 近邻 |
| ML SVM | 支持向量机 |
| ML NB | 朴素贝叶斯 |
| ML AdaBoost | AdaBoost 决策树桩 |
| ML Extra Trees | 极端随机树 |

### 自适应策略（3 种）
| 策略 | 说明 |
|------|------|
| Ensemble | 多策略投票 |
| Stacked | 策略 + 动量过滤叠加 |
| **Regime Aware** | 自动检测市场状态（趋势/震荡/危机），动态切换子策略 |

## Regime 自适应策略

旗舰策略使用 4 个指标（ADX 趋势强度、波动率比率、均线排列、动量信号）检测市场状态，并自动切换：

| 市场状态 | 策略行为 |
|---------|---------|
| 趋势 + 多头 | 全仓，波动率缩放仓位 |
| 震荡 + 多头 | 全仓（捕捉非趋势上涨） |
| 趋势 + 空头 | 空仓或做空（可选） |
| 危机 | 降仓 + 极低波动率目标 |
| 中性震荡 | 空仓 |

另附快速止损机制：当 N 日收益跌破阈值时立即削减仓位。

## Python API

```python
from momentum_lab import download_data, backtest, evaluate, get_strategy, run_search

# 运行搜索
results = run_search("GLD", quick=True)
print(f"最优策略: {results['best']['strategy']}")
print(f"最优参数: {results['best']['params']}")

# 稳健性检验结果（过拟合检测）
rob = results.get("robustness") or {}
if rob.get("grade"):
    print(f"等级: {rob['grade']} ({rob.get('verdict', 'n/a')})")
    print(f"基线验证 Sharpe: {rob['baseline']:.4f}")
    stats = rob.get("stats") or {}
    if stats:
        print(f"邻域中位数: {stats['median']:.4f}")

# 单独测试某个策略
from momentum_lab.data import prepare_data
data, df = prepare_data("SPY")
strategy = get_strategy("regime_aware")
positions = strategy.run(data, adx_trend_threshold=15, mom_lookback=63,
                          vol_target_normal=0.12, position_size=2.0)
result = backtest(positions, df["close"])
metrics = evaluate(result["returns"])
print(f"Sharpe: {metrics['sharpe']}")
```

## 命令行参数

```
momentum-lab TICKER [选项]

参数:
  TICKER              Yahoo Finance 代码（GLD, SPY, BTC-USD, AAPL, ...）

选项:
  --quick             快速模式：每种策略只测 5 组参数
  --config PATH       从 JSON 加载完整搜索配置（配置值优先）
  --resume            从指定 run-id 的 all_results.csv 断点续跑
  --strategies STR    指定策略名称（逗号分隔）
  --workers N         并行进程数（默认 1）
  --cost BPS          交易成本，基点（默认 1.0）
  --slippage BPS      额外滑点，基点（默认 0）
  --financing-rate R  年化融资利率（小数，默认 0）
  --borrow-bps BPS    年化做空借券费，基点（默认 0）
  --annualization N   年化周期数（股票 252，加密货币 365）
  --risk-free-rate R  年化无风险利率，小数形式（默认 0.04）
  --start DATE        数据开始日期（默认 2004-01-01）
  --end DATE          数据结束日期（默认今天）
  --refresh           忽略缓存，强制从 Yahoo 重新下载
  --top N             保留前 N 个结果（默认 50）
  --result-dir DIR    结果产物父目录（默认 ./experiments）
  --run-id ID         自定义运行目录名（默认自动生成）
  --no-keep-all       结果只流式写入 CSV，不在内存中保留全量
                      （全量网格建议开启：否则内存占用 1 GB+）
  --robust            对最优参数做稳健性检验（默认开启）
  --no-robust         跳过稳健性检验
  --robust-frac F     参数扰动比例（默认 0.2）
  --list              列出所有策略并退出
  --version           显示版本号
```

## 稳健性检验

搜索可能会落在某个碰巧表现好的参数尖峰上，无法泛化。找到最优策略后，momentum-lab
会把每个数值参数扰动 ±20%（整数 ±1）并重新评估每个邻域的验证集 Sharpe：

- **等级 A（稳健）**：邻域结果接近最优 —— 参数高原宽阔
- **等级 B**：相对稳定
- **等级 C**：脆弱 —— 结果高度依赖精确参数
- **等级 D / 孤立峰值**：最优点是尖峰，结果属于过拟合

如果看到 *ISOLATED PEAK - likely overfit*（疑似孤立峰值/过拟合），请降低对实盘的预期，
并考虑收窄参数搜索空间。

## 安装

### 从源码安装
```bash
git clone https://github.com/nmj94/momentum-lab.git
cd momentum-lab
pip install -e .
```

> **macOS 提示：** 可选的 XGBoost 策略需要 `libomp`（`brew install libomp`）。
> 缺少它时 `ml_xgb` 实验会优雅失败，其余策略不受影响。

### 环境要求
- Python 3.10+
- 可访问 Yahoo Finance 的网络（支持股票、ETF、加密货币、指数）

## 输出结果

运行后，结果保存在 `experiments/<run_id>/` 目录，每次运行独立保存：
- `run_config.json` - 数据区间、成本模型、参数、切分配置、Git SHA 和数据快照哈希
- `all_results.csv` - 所有实验结果（含训练/验证/测试指标），也是断点续跑检查点
- `top_results.csv` - 按验证集 Sharpe 排序的前 N 个策略
- `robustness.csv` - 最优策略的稳健性检验汇总
- 控制台输出最优策略的参数和测试集表现

## 示例结果（黄金 GLD）

由全部 18 个非 ML 策略的穷举搜索产生（952,824 个实验，数据区间
2004-11-18 ~ 2026-08-14）。买入持有基准与策略收取相同的一次性建仓成本：

| 策略 | 验证集 Sharpe | 测试集 Sharpe | 测试集年化 | 测试集最大回撤 |
|------|-------------|-------------|-----------|-------------|
| TSMOM（验证集最优） | 1.29 | 0.27 | 7.9% | -38.3% |
| Acceleration | 1.28 | -0.07 | 0.8% | -32.1% |
| RSI | 1.23 | 0.48 | 13.8% | -35.8% |
| 买入持有 | - | 0.86 | 20.4% | -26.4% |

验证集最优（TSMOM）的稳健性等级为 B（邻域稳定、非孤峰），但在测试窗口
跑输买入持有——该窗口金价单边大牛市，只做多过滤策略反而落后。这种量级的
验证->测试衰减属正常现象，也正是本工具单独报告未触碰测试集的原因。

## 免责声明

本工具仅供研究使用，不构成投资建议。历史回测不代表未来收益。交易前请自行研究并考虑交易成本、滑点和税费。

## 许可证

MIT
