# FlowForge

**世界级归一化流 / 可逆生成建模系统** · 纯 NumPy · CPU · 离线可跑 · 逐位可复现

> **作者：晨星**（CJX0712） · MIT License

[![CI](https://github.com/CJX0712/flowforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/flowforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/flowforge?label=Release)](https://github.com/CJX0712/flowforge/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-A%20(production)-brightgreen.svg)](docs/model_card.md)

---

## 一句话

在**完全已知真值密度**的二维分布上，用纯 NumPy 手写的仿射耦合 / 掩码自回归 /
分段线性样条三类双射器，加一个**自适应混合归一化流**（FlowFuse），
把 `KL(p_true ‖ p_model)` 相对最强单流基线压低 **52.20%**，同时在 **5/5** 个数据集上非劣。

## 为什么这个系统不一样

归一化流最容易被"看起来收敛了"骗过去。本系统给每个目标分布配**可解析的 `log p_true(x)`**，
于是主指标有**绝对下界 0**：

$$\mathrm{KL}(p_{\text{true}}\|p_{\text{model}}) = \mathbb{E}\big[\log p_{\text{true}} - \log p_{\text{model}}\big] \ge 0$$

这不是"比最强基线低 X%"这种需要外部参照的口径，而是有理论下界的绝对标尺。
**开发过程中它真的抓到了 bug**：`banana` 分布采样用方差 1.6、`log_prob` 却按单位方差计算，
导致 KL 估计系统性偏 −0.065 nats —— 如果没有这条不变量，这个错误会一路沉默到最终结论。

配套五类硬不变量，全部进 CI 门禁：

| 不变量 | 判据 | 实测 |
|---|---|---|
| I1 往返可逆性 | `‖f⁻¹(f(x)) − x‖∞` | ≤ 2.9e-15 |
| I2 logdet 正确性 | 解析 vs 数值雅可比 | ≤ 8.4e-10 |
| I3 梯度正确性 | 解析 vs 中心差分 | ≤ 6.4e-10 |
| I4 密度归一化 | `\|∫p dx − 1\|` | ≤ 9.5e-4 |
| I5 KL 非负 | Gibbs 不等式 | 全部为正 |

## 基准结果

5 个可解析密度 × 3 个种子（7 / 11 / 23），主指标 `KL`（nats，**越低越好**，下界 0）。
所有方法**等参数预算**（8 blocks × hidden 32 ≈ 1040 参数）。

### 逐数据集 KL（3 seeds 均值）

| 数据集 | 形态 | gaussian | affine (RealNVP) | maf | **fusion (FlowFuse)** |
|---|---|---|---|---|---|
| `gmm5` | 5 模态 | 0.68767 | 0.32726 | 0.37742 | **0.13765** |
| `ring8` | 8 模态环 | 1.04434 | 0.54346 | 0.56333 | **0.29220** |
| `banana` | 弯曲单峰 | 0.22293 | 0.05241 | **0.00745** | 0.00832 |
| `funnel` | 变尺度重尾 | 0.59541 | 0.00457 | 0.00454 | **0.00402** |
| `swirl` | 旋转不变 | **0.00112** | 0.00344 | 0.00427 | 0.00293 |

### 聚合

| 方法 | 参数量 | KL (mean ± std) | W2 ↓ | MMD ↓ |
|---|---|---|---|---|
| `gaussian` | 6 | 0.51029 ± 0.37771 | 0.5739 | 0.005807 |
| `affine` | 1040 | 0.18623 ± 0.22588 | 0.4404 | 0.002849 |
| `maf` | 1056 | 0.19140 ± 0.24774 | 0.4463 | 0.003569 |
| **`fusion`** | **1044** | **0.08903 ± 0.11809** | **0.4402** | **0.000824** |

- **G1a** 聚合 KL 相对最强单流基线（`affine`）降低 **52.20%**（门槛 10%）✅
- **G1b** 逐数据集配对显著性：`gmm5` / `ring8` **2/5 显著胜出**（门槛 ≥2）✅
- **G2** 逐数据集非劣（绝对容差 0.005 nats）：**5/5** ✅
- **G5** 消融归因：`K=4` 相对 `K=1` 降低 **51.26%** ✅
- **G3** 确定性：同 seed 两次运行 60 行核心指标**逐位一致**
- **性能预算** demo 端到端 **52.4s**（预算 60s），训练总耗时 29.8s

> **指标诚实说明**：`W2` 在 2D、200 样本下已饱和（fusion 0.4402 vs affine 0.4404，
> 差异淹没在 Sinkhorn 正则偏差里）；**MMD 给出清晰信号（−71%）**。
> 主判据始终是 KL——只有它有精确解析真值作金标准。

## 旗舰：FlowFuse（自适应混合归一化流）

$$\log p(x) = \log\sum_{k=1}^{K}\exp\big(\log\pi_k + \log p_k(x)\big)$$

- **等参数预算**：把 8 blocks 的总预算平均分给 `K` 个分量，因此 `K=1` 时与同块类型基线**结构等价**。
  增益因此**只能归因于"混合"**，归因干净、无参数作弊。
- **验证集自适应**：块类型 ∈ {affine, maf} 与分量数 `K` ∈ {1,2,4} 在**独立验证集**
  （seed 9042，与 train/test 种子不相交）上按验证 NLL 选取。
- **死组件复活**：训练中若某分量责任度长期低于阈值则重置，避免分量塌缩。
- **收益从何而来（消融）**：`K=1 → K=4` 使 KL 降低 **51.26%** —— 增益来自**混合本身**。
  自适应选择相对强制 `K=4` 还有 **−3.21%** 的边际收益（负结果保留，见下）。

## 失败案例（全部从实测结果派生，无预设结论）

| 数据集 | 旗舰 | 最强基线 | 差值 | 归因 |
|---|---|---|---|---|
| `banana` | 0.00832 | `maf` 0.00745 | +0.00087 | 混合流把容量摊薄到 4 个分量，单峰目标上每个分量过浅 |
| `swirl` | 0.00293 | `gaussian` 0.00112 | +0.00181 | **`swirl` 目标本身就是标准高斯**（绕原点旋转不改变高斯），解析高斯是它的精确最优，流只能逼近 |
| `funnel` | 0.00402 | `maf` 0.00454 | −0.00052 | 混合在变尺度重尾上反而略优 |

## 一键复现

```bash
git clone https://github.com/CJX0712/flowforge.git
cd flowforge

python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                   # 或 requirements.lock.txt（钉版本）

# ① 数学不变量自检（I1~I5，秒级）
python -m flowforge.cli --check

# ② 单元测试（58 项）
pytest -q

# ③ 端到端演示：跑基准 + 门禁 + 消融，落盘 benchmark.json（约 52s）
python examples/run_demo.py
```

Docker：

```bash
docker build -t flowforge .
docker run --rm flowforge
```

## 项目结构

```
flowforge/
├── core/         错误码 E1xx~E5xx · dataclass · 配置(ENV 覆盖+校验) · Protocol · seed
├── data/         5 个可解析密度（金标准来源）+ 归一化数值积分自检
├── flows/        双射器(仿射耦合/掩码自回归/线性样条) · 组合流 · 混合流 · 高斯基线
├── training/     Adam + lr 衰减 + 梯度裁剪 · 模型构建 · 死组件复活
├── eval/         KL · 去偏 Sinkhorn 散度 W2 · MMD · 五类硬不变量
├── pipeline/     验证集选参(防泄漏) · 基准编排
├── cli.py        argparse 入口
├── examples/run_demo.py    端到端演示 + 门禁
├── tests/        58 项单测
└── docs/         architecture.md · model_card.md · known_limitations.md
```

## CLI

```bash
python -m flowforge.cli --quick                     # 冒烟
python -m flowforge.cli --check                     # 只跑不变量
python -m flowforge.cli --datasets gmm5 ring8 \
       --methods affine maf fusion --seeds 7 11 23 --json out.json
```

所有配置项支持 `FLOWFORGE_*` 环境变量覆盖，例如 `FLOWFORGE_ITERS=500 python examples/run_demo.py`。

## 复用的世界级方法与开源

本系统**不复用预训练权重、不下载任何模型**（保证离线可复现），而是**忠实实现并对照**
公开发表的世界级方法：

| 方法 | 出处 | 在本系统中的角色 |
|---|---|---|
| RealNVP 仿射耦合 | Dinh et al., *ICML 2017* | 基线 `affine`，同时是 FlowFuse 的可选块 |
| MAF 掩码自回归 | Papamakarios et al., *NeurIPS 2017* | 基线 `maf`，同时是 FlowFuse 的可选块 |
| Neural Spline Flow（线性段） | Durkan et al., *NeurIPS 2019* | 已实现并通过全部不变量检验，**未纳入主基准**（见 known_limitations） |
| 变量变换公式 / Gibbs 不等式 | — | 金标准与 KL 非负不变量 |
| 去偏 Sinkhorn 散度 | Feydy et al., *2019* | W2 指标（裸 Sinkhorn 有熵偏差，不能直接当 W2） |
| NumPy / SciPy / scikit-learn | — | 全部为跨平台 CPU 轮子，零编译 |

## 质量分级

**A（生产级）** —— 详见 [`docs/model_card.md`](docs/model_card.md)。

四项 DoD：性能（多 seed 胜强基线，幅度 52.20% 可量化可复现）· 复现（干净环境一键跑通、
逐位一致）· 工程化（58 单测 + lint 硬门禁 + CI 矩阵 + 依赖锁定 + 离线可跑）·
可信交付（文档齐全、门禁数字全部来自真实运行输出）。

未达 S 级的诚实原因：`spline` 双射器虽已实现且通过全部不变量检验，但在共享训练预算下
优化不稳定，**未纳入主基准**；基准维度限于 2D（5 个分布），未覆盖高维真实数据。

## 许可与合规

MIT。所有依赖（NumPy / SciPy / scikit-learn）均为 BSD 或兼容许可。
仓库不含任何密钥、不含任何预训练权重、运行期零网络请求。

---

**作者：晨星** · [GitHub CJX0712](https://github.com/CJX0712)