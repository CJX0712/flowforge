"""双射器（bijector）实现：MLP / 仿射耦合 / 掩码自回归 / 线性样条耦合。

契约（core.interfaces.Bijector，硬）：
  forward(X)  -> (Z, logdet)     Z 为隐空间坐标，logdet = log|det dZ/dX|
  inverse(Z)  -> X               必须满足 inverse(forward(X)) == X（机器精度）
  backward(gZ, w) -> (gX, grads) 解析梯度；grads 为「参数名 -> 梯度」，按 batch 求和（未除 n）

每个双射器的解析梯度都由 tests/test_invariants.py 的梯度检验（vs 中心差分）锁死。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import FlowError


# --------------------------------------------------------------------------- MLP
class MLP:
    """单隐层 tanh 网络。参数名固定为 W1/b1/W2/b2。"""

    def __init__(
        self, din: int, hidden: int, dout: int, rng: np.random.Generator, scale: float = 1.0
    ) -> None:
        if din <= 0 or hidden <= 0 or dout <= 0:
            raise FlowError("MLP 维度必须为正", din=din, hidden=hidden, dout=dout)
        s1 = scale * np.sqrt(2.0 / (din + hidden))
        s2 = scale / np.sqrt(hidden)
        self.W1 = rng.normal(0.0, s1, (din, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, s2, (hidden, dout))
        self.b2 = np.zeros(dout)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def P(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def nparams(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._cache["X"] = X
        H = np.tanh(X @ self.W1 + self.b1)
        self._cache["H"] = H
        return H @ self.W2 + self.b2

    def backward(self, dY: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        H = self._cache["H"]
        X = self._cache["X"]
        dW2 = H.T @ dY
        db2 = dY.sum(axis=0)
        dH = dY @ self.W2.T
        dA = dH * (1.0 - H**2)
        dW1 = X.T @ dA
        db1 = dA.sum(axis=0)
        dX = dA @ self.W1.T
        return dX, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ----------------------------------------------------------------- 仿射耦合
class AffineCoupling:
    """RealNVP 仿射耦合 (Dinh et al. 2017)。

    z_m = (x_m - t(x_c)) * exp(-s(x_c));  z_c = x_c;  logdet = -sum(s)
    """

    family = "affine"

    def __init__(self, mask: np.ndarray, hidden: int, rng: np.random.Generator) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.dim = int(self.mask.size)
        self.nm = int(self.mask.sum())
        self.nc = self.dim - self.nm
        if self.nm == 0 or self.nc == 0:
            raise FlowError("耦合 mask 必须非空且非全集", nm=self.nm, nc=self.nc)
        self.net = MLP(self.nc, hidden, 2 * self.nm, rng)

    @property
    def P(self) -> dict[str, np.ndarray]:
        return {f"{k}": v for k, v in self.net.P.items()}

    def nparams(self) -> int:
        return self.net.nparams()

    def params(self) -> list[np.ndarray]:
        return self.net.params()

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, c = self.mask, ~self.mask
        Xc, Xm = X[:, c], X[:, m]
        st = self.net.forward(Xc)
        s = st[:, : self.nm]
        t = st[:, self.nm :]
        es = np.exp(-s)
        Zm = (Xm - t) * es
        Z = X.copy()
        Z[:, m] = Zm
        self._c = {"Xc": Xc, "Xm": Xm, "s": s, "t": t, "es": es, "Zm": Zm}
        return Z, -s.sum(axis=1)

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        m, c = self.mask, ~self.mask
        Zc, Zm = Z[:, c], Z[:, m]
        st = self.net.forward(Zc)
        s = st[:, : self.nm]
        t = st[:, self.nm :]
        X = Z.copy()
        X[:, m] = Zm * np.exp(s) + t
        return X

    def backward(self, gZ: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """w: (n,) 每样本权重。单流 w=1；混合流 w=责任度 r_k。

        f 中 -logdet = +sum(s) 对 s 的直接导数 = +w。
        """
        c = self._c
        m = self.mask
        cc = ~m
        a = gZ[:, m]  # (n, nm)  d f/d Z_m
        ds = -a * c["Zm"] + w[:, None]  # 经由 Zm + logdet 直接项
        dt = -a * c["es"]
        dXm = a * c["es"]
        dST = np.concatenate([ds, dt], axis=1)
        dXc_net, g = self.net.backward(dST)
        dX = np.zeros_like(gZ)
        dX[:, m] = dXm
        dX[:, cc] = gZ[:, cc] + dXc_net
        return dX, g


# ------------------------------------------------------------ 掩码自回归 (MAF)
class MaskedAffineAutoregressive:
    """MAF (Papamakarios et al. 2017)：逐维自回归仿射变换。

    z_i = (x_i - t_i(x_{<i})) * exp(-s_i(x_{<i})),  logdet = -sum_i s_i
    第 0 维为无条件可学习参数；第 i>0 维由 MLP(x_{<i}) 给出 (s_i, t_i)。

    **关键**：自回归顺序必须在块间交替（reverse=True 时反序）。
    否则第 0 维永远只能被常数仿射变换、无法以其它维为条件 ——
    实测该缺陷使 MAF 聚合 KL 从 0.096 恶化到 0.303（结构性表达力缺失）。
    """

    family = "maf"

    def __init__(
        self, dim: int, hidden: int, rng: np.random.Generator, reverse: bool = False
    ) -> None:
        if dim < 2:
            raise FlowError("MAF 需要 dim >= 2", dim=dim)
        self.dim = dim
        self.reverse = bool(reverse)
        self.order = np.arange(dim)[::-1].copy() if reverse else np.arange(dim)
        self.s0 = np.zeros(1)
        self.t0 = np.zeros(1)
        self.nets: list[MLP] = []
        for i in range(1, dim):
            self.nets.append(MLP(i, hidden, 2, rng))

    # ---- 顺序空间 <-> 原始空间 ----
    def _ord(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.order]

    def _unord(self, Xp: np.ndarray) -> np.ndarray:
        out = np.empty_like(Xp)
        out[:, self.order] = Xp
        return out

    @property
    def P(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {"s0": self.s0, "t0": self.t0}
        for i, net in enumerate(self.nets):
            for k, v in net.P.items():
                d[f"net{i}.{k}"] = v
        return d

    def nparams(self) -> int:
        return 2 + sum(n.nparams() for n in self.nets)

    def params(self) -> list[np.ndarray]:
        ps = [self.s0, self.t0]
        for n in self.nets:
            ps += n.params()
        return ps

    def _st(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (s, t)，各 (n, dim)。"""
        n = X.shape[0]
        s = np.empty((n, self.dim))
        t = np.empty((n, self.dim))
        s[:, 0] = self.s0[0]
        t[:, 0] = self.t0[0]
        for i in range(1, self.dim):
            st = self.nets[i - 1].forward(X[:, :i])
            s[:, i] = st[:, 0]
            t[:, i] = st[:, 1]
        return s, t

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xp = self._ord(X)
        s, t = self._st(Xp)
        es = np.exp(-s)
        Zp = (Xp - t) * es
        Z = self._unord(Zp)
        self._c = {"s": s, "t": t, "es": es, "Z": Zp, "X": Xp}
        return Z, -s.sum(axis=1)

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        Zp = self._ord(Z)
        n = Z.shape[0]
        Xp = np.empty((n, self.dim))
        Xp[:, 0] = Zp[:, 0] * np.exp(self.s0[0]) + self.t0[0]
        for i in range(1, self.dim):
            st = self.nets[i - 1].forward(Xp[:, :i])
            Xp[:, i] = Zp[:, i] * np.exp(st[:, 0]) + st[:, 1]
        return self._unord(Xp)

    def backward(self, gZ: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        c = self._c
        es, Zp = c["es"], c["Z"]
        gZp = gZ[:, self.order]  # 转到有序空间
        # f = -log p  =>  df/dz = z；logdet = -sum(s)  =>  df/ds 直接项 = +w
        ds = -gZp * Zp + w[:, None]  # (n, dim)
        dt = -gZp * es  # (n, dim)
        grads: dict[str, np.ndarray] = {}
        # 每个 x_i 的直接路径：dz_i/dx_i = exp(-s_i)
        dXp = gZp * es
        # 第 0 维：无条件参数
        grads["s0"] = np.array([ds[:, 0].sum()])
        grads["t0"] = np.array([dt[:, 0].sum()])
        # 第 i>0 维：(s_i, t_i) = net_{i-1}(x_{<i})，反传会把梯度加到 dXp[:, :i]
        for i in range(1, self.dim):
            dST = np.stack([ds[:, i], dt[:, i]], axis=1)
            dXnet, g = self.nets[i - 1].backward(dST)
            for k, v in g.items():
                grads[f"net{i - 1}.{k}"] = v
            dXp[:, :i] += dXnet
        return self._unord(dXp), grads


# --------------------------------------------------------- 线性样条耦合 (NSF 家族)
class LinearSplineCoupling:
    """线性样条耦合 (Durkan et al. 2019 的 piecewise-linear 特例)。

    在 [-B, B] 内做 K 段线性样条，区间外恒等；每段的宽度/高度由 softmax 归一化。
    logdet = sum(log(slope_k))，slope_k = height_k / width_k。
    """

    family = "spline"

    def __init__(
        self,
        mask: np.ndarray,
        hidden: int,
        n_bins: int,
        rng: np.random.Generator,
        bound: float = 4.0,
        min_bin_width: float = 0.05,
    ) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.dim = int(self.mask.size)
        self.nm = int(self.mask.sum())
        self.nc = self.dim - self.nm
        self.K = int(n_bins)
        self.B = float(bound)
        # 最小 bin 宽度：没有它，某个 bin 宽度可趋于 0 => slope = h/w -> inf
        # => logdet 爆炸、小批量 Adam 下训练发散（实测 8 blocks 时 NLL 3.31 -> 4.84）。
        # Durkan et al. (2019) 的 NSF 实现同样强制该下界。
        self.min_w = float(min_bin_width)
        span = 2.0 * self.B
        if self.min_w * self.K >= span:
            raise FlowError(
                "min_bin_width * n_bins 必须小于 2*bound", min_w=self.min_w, K=self.K, bound=self.B
            )
        if self.K < 2:
            raise FlowError("n_bins 必须 >= 2", n_bins=self.K)
        if self.nm == 0 or self.nc == 0:
            raise FlowError("样条耦合 mask 必须非空且非全集", nm=self.nm, nc=self.nc)
        self.net = MLP(self.nc, hidden, 2 * self.nm * self.K, rng, scale=0.5)

    @property
    def P(self) -> dict[str, np.ndarray]:
        return dict(self.net.P.items())

    def nparams(self) -> int:
        return self.net.nparams()

    def params(self) -> list[np.ndarray]:
        return self.net.params()

    def _softmax(self, raw: np.ndarray) -> np.ndarray:
        m = np.max(raw, axis=-1, keepdims=True)
        e = np.exp(raw - m)
        return e / np.sum(e, axis=-1, keepdims=True)

    def _pieces(self, Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (widths, heights)，形状 (n, nm, K)，均归一到 2B。"""
        n = Xc.shape[0]
        raw = self.net.forward(Xc)
        rw = raw[:, : self.nm * self.K].reshape(n, self.nm, self.K)
        rh = raw[:, self.nm * self.K :].reshape(n, self.nm, self.K)
        span = 2.0 * self.B
        # softmax -> [min_w, ...]，保证每段宽度 >= min_w 且总和仍为 span
        free = span - self.min_w * self.K
        return (self._softmax(rw) * free + self.min_w, self._softmax(rh) * free + self.min_w)

    def _bin_index(self, cum: np.ndarray, u: np.ndarray) -> np.ndarray:
        """cum: (n, nm, K+1)；u: (n, nm)。返回每段所属 bin 下标 (n, nm)。"""
        inner = cum[:, :, 1:-1]  # (n, nm, K-1) 内部边界
        return np.sum(u[:, :, None] > inner, axis=2)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, cc = self.mask, ~self.mask
        Xc, Xm = X[:, cc], X[:, m]
        widths, heights = self._pieces(Xc)  # (n, nm, K)
        slopes = heights / widths  # (n, nm, K)
        cumW = (
            np.concatenate([np.zeros((X.shape[0], self.nm, 1)), np.cumsum(widths, axis=2)], axis=2)
            - self.B
        )
        cumH = (
            np.concatenate([np.zeros((X.shape[0], self.nm, 1)), np.cumsum(heights, axis=2)], axis=2)
            - self.B
        )
        k = self._bin_index(cumW, Xm)  # (n, nm)
        inside = (Xm > -self.B) & (Xm < self.B)
        idx = np.clip(k, 0, self.K - 1)
        ar = np.arange(X.shape[0])[:, None]
        dm = np.arange(self.nm)[None, :]
        w_lo = cumW[ar, dm, idx]
        h_lo = cumH[ar, dm, idx]
        sl = slopes[ar, dm, idx]
        Ym = h_lo + sl * (Xm - w_lo)
        Zm = np.where(inside, Ym, Xm)
        Z = X.copy()
        Z[:, m] = Zm
        logdet = np.where(inside, np.log(sl), 0.0).sum(axis=1)
        self._c = dict(
            Xc=Xc,
            Xm=Xm,
            widths=widths,
            heights=heights,
            slopes=slopes,
            cumW=cumW,
            cumH=cumH,
            k=idx,
            inside=inside,
            Zm=Zm,
        )
        return Z, logdet

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        m, cc = self.mask, ~self.mask
        Zc, Zm = Z[:, cc], Z[:, m]
        widths, heights = self._pieces(Zc)
        slopes = heights / widths
        n = Z.shape[0]
        cumW = (
            np.concatenate([np.zeros((n, self.nm, 1)), np.cumsum(widths, axis=2)], axis=2) - self.B
        )
        cumH = (
            np.concatenate([np.zeros((n, self.nm, 1)), np.cumsum(heights, axis=2)], axis=2) - self.B
        )
        k = self._bin_index(cumH, Zm)
        inside = (Zm > -self.B) & (Zm < self.B)
        idx = np.clip(k, 0, self.K - 1)
        ar = np.arange(n)[:, None]
        dm = np.arange(self.nm)[None, :]
        w_lo = cumW[ar, dm, idx]
        h_lo = cumH[ar, dm, idx]
        sl = slopes[ar, dm, idx]
        Xm = w_lo + (Zm - h_lo) / sl
        X = Z.copy()
        X[:, m] = np.where(inside, Xm, Zm)
        return X

    def backward(self, gZ: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        c = self._c
        m, cc = self.mask, ~self.mask
        n, nm, K = gZ.shape[0], self.nm, self.K
        widths, heights, slopes = c["widths"], c["heights"], c["slopes"]
        cumW, idx = c["cumW"], c["k"]
        Xm, inside = c["Xm"], c["inside"]
        ar = np.arange(n)[:, None]
        dm = np.arange(nm)[None, :]

        gZm = gZ[:, m]  # (n, nm)
        sl = slopes[ar, dm, idx]
        w_lo = cumW[ar, dm, idx]
        # 注意：h_lo 只在前向里用到；反向经 cumH 的前缀和分摊到各段 heights

        # 区间内：Zm = h_lo + sl*(Xm - w_lo)；区间外恒等（梯度直接透传）
        gY = gZm
        d_sl = np.where(inside, gY * (Xm - w_lo), 0.0)
        d_wlo = np.where(inside, -gY * sl, 0.0)
        d_hlo = np.where(inside, gY, 0.0)
        dXm = np.where(inside, gY * sl, gZm)

        # logdet = sum(log slope)（仅区间内）。f = -log p 含 -logdet
        #   => df/d(log slope) = -w  =>  df/d slope = -w / slope
        d_sl = d_sl - np.where(inside, w[:, None] / (sl + 1e-12), 0.0)

        # 展开到逐段 (K,) 的梯度
        d_slopes = np.zeros_like(slopes)
        d_widths = np.zeros_like(widths)
        d_heights = np.zeros_like(heights)
        np.put_along_axis(d_slopes, idx[:, :, None], d_sl[:, :, None], axis=2)
        # cumW_k = -B + sum_{j<k} widths_j  =>  d width_j += d_wlo  (j < k)
        lt = np.arange(K)[None, None, :] < idx[:, :, None]  # (n,nm,K)
        d_widths += np.where(lt, d_wlo[:, :, None], 0.0)
        d_heights += np.where(lt, d_hlo[:, :, None], 0.0)
        # slope_k = height_k / width_k（注意 d_sl 需显式补维，否则 (n,nm) 与
        # (n,nm,K) 广播会因尾部对齐被撑成 (n,n,nm,K)——静默形状 bug）
        eq = np.arange(K)[None, None, :] == idx[:, :, None]
        d_heights = d_heights + np.where(eq, d_sl[:, :, None] / (widths + 1e-12), 0.0)
        d_widths = d_widths + np.where(eq, -d_sl[:, :, None] * heights / (widths**2 + 1e-12), 0.0)

        # softmax 反传：raw -> p = softmax(raw) * span
        def _softmax_bwd(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
            # p = softmax(raw) * free + min_w  =>  d/d raw = softmax'(...) * free
            free = 2.0 * self.B - self.min_w * self.K
            q = dp * free
            s = (p - self.min_w) / free  # = softmax(raw)
            dot = np.sum(q * s, axis=2, keepdims=True)
            return s * (q - dot)

        d_rw = _softmax_bwd(widths, d_widths)
        d_rh = _softmax_bwd(heights, d_heights)
        d_raw = np.concatenate([d_rw.reshape(n, nm * K), d_rh.reshape(n, nm * K)], axis=1)
        dXc_net, g = self.net.backward(d_raw)
        dX = np.zeros_like(gZ)
        dX[:, m] = dXm
        dX[:, cc] = gZ[:, cc] + dXc_net
        return dX, g


__all__ = ["MLP", "AffineCoupling", "LinearSplineCoupling", "MaskedAffineAutoregressive"]
