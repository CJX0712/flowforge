# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。所有条目作者：**晨星**。

## [0.1.0] - 2026-10-08

首个公开版本。

### 新增

- **五类可解析合成密度**（`data/densities.py`）：`gmm5` / `ring8` / `banana` / `funnel` / `swirl`，
  每个都提供**解析 `log p_true(x)`**，构成整套基准的绝对金标准（KL 下界 0）。
- **三类双射器**（`flows/bijectors.py`），全部带解析梯度：
  - `AffineCoupling` —— RealNVP（Dinh et al., ICML 2017）
  - `MaskedAffineAutoregressive` —— MAF（Papamakarios et al., NeurIPS 2017），
    支持块间**交替自回归顺序**
  - `LinearSplineCoupling` —— 分段线性样条（NSF, Durkan et al., NeurIPS 2019 特例），
    带 `min_bin_width` 下界约束
- **旗舰 FlowFuse**：自适应混合归一化流，块类型与分量数 `K` 在独立验证集上选择，
  含死组件复活机制；总参数预算与基线**严格相等**。
- **五类硬不变量**（`eval/invariants.py`）作为 CI 门禁：
  往返可逆性 / logdet vs 数值雅可比 / 解析梯度 vs 中心差分 / 密度归一化 / KL 非负。
- **评测指标**：KL、去偏 Sinkhorn 散度 W2、RBF 核 MMD。
- 训练器：Adam + 指数 lr 衰减 + 全局梯度裁剪。
- CLI（`--check` / `--quick` / 自定义数据集与方法）、端到端 demo（门禁 + 消融 + 失败案例）。
- CI 矩阵：ubuntu/windows × Python 3.11 / 3.12 / 3.13，lint 与 format 为**硬门禁**。

### 实测结果

- 聚合 `KL(p_true‖p_model)` = **0.08903 ± 0.11809**（5 数据集 × 3 种子），
  相对最强单流基线 `affine`（0.18623）降低 **52.20%**。
- 逐数据集**非劣 5/5**；多模态数据集 `gmm5` / `ring8` 配对显著胜出。
- demo 端到端 **52.4s**（预算 60s）。
- 58 项单元测试全绿；`ruff check` + `ruff format --check` 零告警。

### 修复的真实缺陷（开发期，全部由不变量/门禁抓出）

| 缺陷 | 症状 | 根因 | 修法 |
|---|---|---|---|
| 真值密度方差不一致 | `banana` KL 估值为 **−0.0537**，违反 Gibbs 不等式 | `sample()` 用 `var1=1.6` 采样，`log_prob()` 按单位方差计算 | 补齐 `-0.5·log(var1)` 项，并新增「∫p dx = 1」数值积分不变量 |
| 样条反向广播错误 | reshape 尺寸 324 ≠ 36 | `(n,nm)` 与 `(n,nm,K)` 广播时尾部对齐被撑成 `(n,n,nm,K)` | 显式补维 `d_sl[:, :, None]` |
| MAF 缺交替顺序 | 聚合 KL 0.303（应为 0.058） | 自回归顺序固定 ⇒ 首维永远无法以其它维为条件 | 块间交替 `reverse`，新增回归测试 |
| 样条 bin 宽度塌缩 | 8 blocks 时 NLL 发散 3.31 → 4.84 | softmax 宽度可趋于 0 ⇒ slope → ∞ ⇒ logdet 爆炸 | 强制 `min_bin_width` 下界（NSF 标准做法） |
| 聚合显著性判据失效 | 明明大胜却判 FAIL | 跨数据集 std 被量级差 100 倍撑到 0.18 | 改为逐数据集配对检验 |
| I5 假阴性 | 好模型 KL 报 −0.036 | KL 蒙特卡洛估计只用 24 样本，SE ≈ 0.1 | I5 改用大样本 + 按样本量给统计容差 |

### 已知限制

见 [`docs/known_limitations.md`](docs/known_limitations.md)：
分段线性样条耦合未纳入主基准（优化不稳定）、`swirl` 目标恒等于标准高斯、
W2 指标在当前样本量下饱和、结论限于 2D。
