# FlowForge 架构说明

> 作者：晨星 · MIT · 纯 NumPy / CPU / 离线可跑

## 1. 设计原则

1. **金标准优先**：归一化流最容易被"看起来收敛了"骗过去。本系统给每个目标分布配
   **可解析的 `log p_true(x)`**，于是主指标 `KL(p_true‖p_model)` 有**绝对下界 0**
   （Gibbs 不等式），不需要"更强的基线"作参照物。负的 KL 估计 = 密度算错，直接报错。
2. **可验证优先于可训练**：五类双射器全部带**解析梯度**，并由中心差分梯度检验、
   数值雅可比 logdet 检验、往返可逆性检验三重交叉验证。任何一处推导错误都会
   在 1e-5 量级暴露，而不是静默地让结论失真。
3. **公平对照**：所有方法**等参数预算**（8 blocks × hidden 32 ≈ 1040 参数）。
   旗舰的搜索轴只有（块类型，分量数 K），且 **K=1 时与同块类型基线结构等价**，
   因此相对基线的增益**只能归因于"混合"这一件事**。

## 2. 分层与调用方向

```
cli / examples.run_demo
        ↓
pipeline (benchmark · methods)
        ↓
data · flows · training · eval
        ↓
core (errors · types · config · interfaces · seed)
```

**调用单向无环**：`cli → pipeline → {data, flows, training, eval} → core`。
下层绝不反向 import 上层，保证每个模块可独立测试。

| 目录 | 职责 |
|---|---|
| `core/` | 错误码（E1xx~E5xx）、dataclass、配置（`FLOWFORGE_*` 环境变量覆盖 + schema 校验）、Protocol 契约、全局 seed |
| `data/` | 5 个可解析求密度的合成分布（金标准来源）+ 归一化数值积分自检 |
| `flows/` | 双射器（仿射耦合 / 掩码自回归 / 线性样条）、组合流、混合流、高斯基线 |
| `training/` | Adam + 指数 lr 衰减 + 全局梯度裁剪；模型构建；死组件复活 |
| `eval/` | KL / 去偏 Sinkhorn 散度 W2 / MMD；五类硬不变量 |
| `pipeline/` | 验证集选参（防泄漏）、基准编排 |

## 3. 数学内核

### 3.1 变量变换公式

对可逆映射 $f:\mathbb{R}^d\to\mathbb{R}^d$ 与隐空间基准 $p_z$：

$$\log p_x(x) = \log p_z(f(x)) + \log\left|\det \frac{\partial f}{\partial x}\right|$$

### 3.2 双射器契约

| 双射器 | 参考 | 正向（数据 → 隐空间） | logdet |
|---|---|---|---|
| `AffineCoupling` | RealNVP, Dinh et al. 2017 | $z_m=(x_m-t(x_c))\odot e^{-s(x_c)},\ z_c=x_c$ | $-\sum s$ |
| `MaskedAffineAutoregressive` | MAF, Papamakarios et al. 2017 | $z_i=(x_i-t_i(x_{<i}))e^{-s_i(x_{<i})}$ | $-\sum_i s_i$ |
| `LinearSplineCoupling` | NSF, Durkan et al. 2019（线性段特例） | 分段线性样条，区间外恒等 | $\sum_k \log(\text{slope}_k)$ |

三者的 `backward(gZ, w)` 接口统一：`w` 为每样本权重（单流取 1，混合流取责任度 $r_k$）。

### 3.3 混合流（旗舰 FlowFuse）

$$\log p(x) = \log\sum_{k=1}^{K}\exp\big(\log\pi_k + \log p_k(x)\big)$$

责任度 $r_k(x)=\partial\,\log p(x)/\partial\,\log p_k(x)$，于是分量 $k$ 的梯度权重
恰为 $r_k$；共享主干（若有）的 logdet 权重为 $\sum_k r_k = 1$。

## 4. 五类硬不变量（CI 门禁）

| 编号 | 不变量 | 判据 | 实测 |
|---|---|---|---|
| I1 | 往返可逆性 | $\lVert f^{-1}(f(x))-x\rVert_\infty$ | ≤ 1.8e-15 |
| I2 | logdet 正确性 | 解析 vs 数值雅可比 | ≤ 3.7e-10 |
| I3 | 梯度正确性 | 解析 vs 中心差分 | ≤ 5.7e-10 |
| I4 | 密度归一化 | $\lvert\int p\,dx-1\rvert$ | ≤ 1.0e-3 |
| I5 | KL 非负 | $\mathrm{KL}(p_\text{true}\|p_\text{model})\ge -10^{-6}$ | 全部为正 |

I1~I3 对混合流**逐分量**检验（混合流本身不是单一微分同胚）。

## 5. 防泄漏的数据切分

| 切分 | 采样种子 | 用途 |
|---|---|---|
| train | `seed`（随实验种子变化） | 拟合 |
| val | `hpo_seed = 9042` | **仅**用于选超参，与 train/test 种子不相交 |
| test | `test_seed = 10007`（固定） | 评测；所有方法/种子共用同一 holdout，形成配对比较 |

输入标准化（`pre_mean` / `pre_std`）**只在 train 折统计**，其
$\log|\det| = -\sum\log\sigma$ 计入 `log_prob`。

## 6. 统计口径

- 主指标 `KL`：越低越好，下界 0。
- 跨数据集**聚合**仅用于概览；显著性一律用**逐数据集配对检验**
  （同一 train 种子下 fusion 与基线的 KL 差），判据 $|{\bar\Delta}| > \tfrac12(\sigma_1+\sigma_2)$。
  原因：各数据集 KL 量级相差约 100 倍（gmm5 ≈ 0.29 vs swirl ≈ 0.003），
  聚合 std 会被撑到 0.18，使跨数据集显著性判据失去意义。
- 消融可用较少种子（2），性能门禁保持 3 种子。

## 7. 扩展点

- **换目标分布**：实现 `data.densities._BaseDensity` 的 `sample` / `log_prob` 即可。
  只要能给出解析 `log p_true`，整套门禁与金标准自动适用。
- **换双射器**：实现 `core.interfaces.Bijector` 五个方法，接入 `flows/bijectors.py`。
- **换训练器**：`training/fit.fit()` 只依赖 `params()` / `grads(X)` 两个方法，
  可替换为 scipy `L-BFGS-B` 等任意优化器。