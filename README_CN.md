# Momentum Lab

**只需提供一个股票代码，自动找到最优动量交易策略。**

[English](README.md) | [中文](README_CN.md)

Momentum Lab 自动测试 26 种策略、超过 160 万组参数组合，为你的标的找到表现最优的动量策略。包含经典动量指标、机器学习模型、以及根据市场状态自动切换子策略的自适应 Regime 策略。

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
rob = results.get("robustness")
print(f"等级: {rob['grade']} ({rob['verdict']})")
print(f"基线验证 Sharpe: {rob['baseline']:.4f}")
print(f"邻域中位数: {rob['stats']['median']:.4f}")

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
  --strategies STR    指定策略名称（逗号分隔）
  --workers N         并行进程数（默认 1）
  --cost BPS          交易成本，基点（默认 1.0）
  --start DATE        数据开始日期（默认 2004-01-01）
  --end DATE          数据结束日期（默认今天）
  --refresh           忽略缓存，强制从 Yahoo 重新下载
  --top N             保留前 N 个结果（默认 50）
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

### 环境要求
- Python 3.10+
- 可访问 Yahoo Finance 的网络（支持股票、ETF、加密货币、指数）

## 输出结果

运行后，结果保存在 `experiments/` 目录：
- `all_results.csv` - 所有实验结果（含训练/验证/测试指标）
- `top_results.csv` - 按验证集 Sharpe 排序的前 N 个策略
- `robustness.csv` - 最优策略的稳健性检验汇总
- 控制台输出最优策略的参数和测试集表现

## 示例结果（黄金 GLD）

测试了 97 万+ 组参数组合，覆盖 23 种策略：

| 策略 | 验证集 Sharpe | 测试集 Sharpe | 测试集年化 | 测试集最大回撤 |
|------|-------------|-------------|-----------|-------------|
| Regime Aware | 0.90 | 1.31 | 36.3% | -19.0% |
| Vol Scale Mom | 0.89 | 1.49 | 54.1% | -20.3% |
| TSMOM | 0.06 | 1.55 | 48.8% | -23.9% |
| 买入持有 | 0.56 | 1.18 | 24.2% | -21.0% |

## 免责声明

本工具仅供研究使用，不构成投资建议。历史回测不代表未来收益。交易前请自行研究并考虑交易成本、滑点和税费。

## 许可证

MIT
