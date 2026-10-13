# -*- coding: utf-8 -*-
"""
VaR 回测方案的改进与实证 —— 完整复现代码
==========================================
《VaR回测改进方案》完整复现：基于多模型、统计检验与压力测试的系统性风险度量。

数据层  ：baostock，2020-01-01 ~ 2025-12-31 日频收盘价
          标的：上证指数 sh.000001 / 沪深300 sh.000300 / 中证500 sh.000905
模型层  ：7 个 VaR 模型 + 对应 ES
          ① normal 正态（均值方差法）  ② tdist t分布   ③ hist 历史模拟
          ④ hist_w 加权历史模拟(λ=0.97) ⑤ garch GARCH(1,1)
          ⑥ evt 极值理论 EVT-POT       ⑦ fhs 滤波历史模拟
检验层  ：Kupiec 无条件覆盖检验 / Christoffersen 独立性检验 / 联合检验 LRcc
扩展层  ：市场状态分段统计、敏感性分析（窗口/置信水平）、压力测试、全套可视化

运行方式：python var_backtest_v2.py
依赖    ：pandas numpy scipy matplotlib baostock
输出    ：outputs/（CSV结果） figures/（图表）
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")          # 无界面后端，命令行运行不弹窗
import matplotlib.pyplot as plt

# ============================================================
# 全局参数
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

P = 1_000_000                    # 本金 100 万元
WINDOW = 750                     # 滚动估计窗口（约 3 年）
START, END = "2020-01-01", "2025-12-31"
CODES = {"上证指数": "sh.000001", "沪深300": "sh.000300", "中证500": "sh.000905"}
CONFS = [0.95, 0.99]
MODELS = ["normal", "tdist", "hist", "hist_w", "garch", "evt", "fhs"]
MODEL_CN = {"normal": "正态", "tdist": "t分布", "hist": "历史模拟",
            "hist_w": "加权历史", "garch": "GARCH", "evt": "极值理论", "fhs": "FHS"}
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 模块1：数据获取与缓存
# ============================================================
def fetch_close(code, name):
    """从 baostock 拉取日线收盘价，缓存到 outputs/{name}_close.csv"""
    cache = os.path.join(DATA_DIR, f"{name}_close.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["trade_date"])
        print(f"[{name}] 读取缓存 {len(df)} 条")
        return df
    import baostock as bs
    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        code, "date,close", start_date=START, end_date=END,
        frequency="d", adjustflag="3")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    df["close"] = df["close"].astype(float)
    df = df.rename(columns={"date": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    bs.logout()
    print(f"[{name}] 拉取完成 {len(df)} 条，已缓存")
    return df


def build_returns(df):
    """计算对数收益率，返回 ret_all DataFrame"""
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df[["trade_date", "log_return"]].dropna().reset_index(drop=True)


# ============================================================
# 模块2：描述性统计
# ============================================================
def describe_returns(ret):
    """均值/标准差/偏度/超额峰度 + Jarque-Bera 正态性检验"""
    r = ret["log_return"].values
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    skew = stats.skew(r, bias=False)
    ex_kurt = stats.kurtosis(r, fisher=True, bias=False)   # 超额峰度（正态=0）
    jb = n / 6.0 * (skew**2 + ex_kurt**2 / 4.0)
    jb_p = 1 - stats.chi2.cdf(jb, 2)
    return {"样本数": n, "均值": mu, "标准差": sd, "偏度": skew,
            "超额峰度": ex_kurt, "JB_p值": jb_p}


# ============================================================
# 模块3：7 个模型估计器
# 输入：训练窗口收益率 r（np.ndarray），尾部概率 p = 1 - 置信水平
# 输出：(var_r, es_r) 收益率阈值表示（负值），损失金额 = -阈值 × 本金
#
# 【课堂对照】7 个模型的教材术语对照：
#   est_normal —— 均值-方差法 / 参数法 / 方差-协方差法 / 正态法（教材·方案统称"正态"）
#   est_tdist  —— t 分布法（参数法厚尾改进，学生 t 分布）
#   est_hist   —— 历史模拟法（非参数法）
#   est_hist_w —— 加权历史模拟法（指数衰减权重）
#   est_garch  —— GARCH(1,1) 动态波动率法（参数法）
#   est_evt    —— 极值理论 EVT-POT（广义帕累托分布 GPD）
#   est_fhs    —— 滤波历史模拟 FHS（GARCH 滤波 + 历史模拟残差）
# ============================================================
def est_normal(r, p):
    """均值方差法（方案中称"正态"）：VaR = -(μ + z_α·σ)，ES 为正态左尾条件期望
    —— 教材经典参数法：只用均值 μ 与标准差 σ 两个统计量，
       95% 置信取 z=-1.645，99% 取 z=-2.326；表2 中对应 normal 行"""
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(p)                       # 负值，如 p=0.05 → -1.645
    var_r = mu + z * sigma
    es_r = mu - sigma * stats.norm.pdf(z) / p   # 正态条件期望 E[r|r≤VaR]
    return var_r, es_r


def est_tdist(r, p):
    """t 分布：对标准化残差 (r-μ)/σ 做 t 分布 MLE，再反变换回收益分位（厚尾）"""
    mu, sd = r.mean(), r.std(ddof=1)
    if sd <= 0:
        return mu, mu
    df, loc0, scale0 = stats.t.fit((r - mu) / sd)
    loc, scale = mu + loc0 * sd, scale0 * sd
    t_alpha = stats.t.ppf(p, df)
    var_r = loc + t_alpha * scale
    # t 分布条件期望：E[X|X<=t_a] = loc - scale * g(t_a)/p * (df+t_a^2)/(df-1)
    es_r = loc - scale * stats.t.pdf(t_alpha, df) / p * (df + t_alpha**2) / (df - 1)
    es_r = min(es_r, var_r)          # 收益率 ES 比 VaR 更负
    return var_r, es_r


def est_hist(r, p):
    """历史模拟：经验分位数 + 尾部样本均值"""
    var_r = np.quantile(r, p)
    tail = r[r <= var_r]
    es_r = tail.mean() if len(tail) > 0 else var_r
    return var_r, es_r


def est_hist_w(r, p, lam=0.97):
    """加权历史模拟：指数衰减权重 λ=0.97，越近权重越高（RiskMetrics 思路）"""
    n = len(r)
    w = lam ** np.arange(n - 1, -1, -1)         # 最新数据权重最高
    w = w / w.sum()
    order = np.argsort(r)
    rs, ws = r[order], w[order]
    cw = np.cumsum(ws)
    idx = int(np.searchsorted(cw, p))
    idx = min(idx, n - 1)
    var_r = rs[idx]
    tail_w = ws[:idx + 1]
    es_r = (rs[:idx + 1] * tail_w).sum() / tail_w.sum()
    return var_r, es_r


def _garch_fit(e, maxiter=300):
    """GARCH(1,1) 高斯 MLE：σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    参数变换保证 ω>0、α>0、β>0、α+β<1，L-BFGS-B 无约束优化，稳健收敛。
    返回 (omega, alpha, beta, sig2)"""
    n = len(e)
    var0 = e.var(ddof=1)
    if var0 <= 0:
        var0 = 1e-8

    def unpack(x):
        omega = np.exp(x[0])
        ea, eb = np.exp(x[1]), np.exp(x[2])
        denom = 1.0 + ea + eb
        return omega, ea / denom, eb / denom

    def neg_loglik(x):
        omega, alpha, beta = unpack(x)
        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):
            sig2[t] = omega + alpha * e[t - 1] ** 2 + beta * sig2[t - 1]
        if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
            return 1e12
        return 0.5 * np.sum(np.log(sig2) + e ** 2 / sig2)

    # 初始值：ω≈5%无条件方差、α=0.1、β=0.85（α+β=0.95<1）
    omega0, alpha0, beta0 = max(var0 * 0.05, 1e-8), 0.10, 0.85
    c0 = np.log(omega0)
    a0 = np.log(alpha0 / (1 - alpha0 - beta0))
    b0 = np.log(beta0 / (1 - alpha0 - beta0))
    res = minimize(neg_loglik, np.array([c0, a0, b0]), method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-10})
    omega, alpha, beta = unpack(res.x)
    sig2 = np.empty(n)
    sig2[0] = var0
    for t in range(1, n):
        sig2[t] = omega + alpha * e[t - 1] ** 2 + beta * sig2[t - 1]
    return omega, alpha, beta, sig2


def est_garch(r, p):
    """GARCH(1,1) VaR：一步预测条件波动率，捕捉波动率聚集"""
    mu = r.mean()
    e = r - mu
    omega, alpha, beta, sig2 = _garch_fit(e)
    sigma_next = np.sqrt(omega + alpha * e[-1] ** 2 + beta * sig2[-1])
    z = stats.norm.ppf(p)
    var_r = mu + z * sigma_next
    es_r = mu - sigma_next * stats.norm.pdf(z) / p
    return var_r, es_r


def est_evt(r, p, tail_q=0.10):
    """EVT-POT：对左尾超阈值拟合广义帕累托分布 GPD(ξ,σ)
    VaR = u + (σ/ξ)·[((N/N_u)·p)^(-ξ) - 1]，ES 由 GPD 尾部积分"""
    u_r = np.quantile(r, tail_q)                 # 收益率 10% 分位（阈值）
    excess = u_r - r                             # 超阈值部分（>0）
    x = excess[excess > 0]
    n, nu = len(r), len(x)
    if nu < 20:
        var_r = np.quantile(r, p)
        tail = r[r <= var_r]
        return var_r, (tail.mean() if len(tail) else var_r)
    try:
        xi, _, sigma = stats.genpareto.fit(x, loc=0)   # c=ξ, scale=σ
        if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError
    except Exception:
        xi, sigma = 1e-6, x.mean()
    # 约束形状参数到合理厚尾区间，避免外推爆炸
    xi = np.clip(xi, -0.3, 0.5)
    u_loss = -u_r                                # 损失阈值（正值）
    ratio = (n / nu) * p
    if abs(xi) > 1e-8:
        var_loss = u_loss + (sigma / xi) * (ratio ** (-xi) - 1.0)
    else:
        var_loss = u_loss - sigma * np.log(ratio)
    es_loss = (var_loss + sigma - xi * u_loss) / (1.0 - xi)
    es_loss = max(es_loss, var_loss)             # ES 必不小于 VaR（一致性）
    return -var_loss, -es_loss


def est_fhs(r, p):
    """滤波历史模拟：GARCH 滤波标准化残差的经验分位 × 一步预测波动率"""
    mu = r.mean()
    e = r - mu
    omega, alpha, beta, sig2 = _garch_fit(e)
    sigma_next = np.sqrt(omega + alpha * e[-1] ** 2 + beta * sig2[-1])
    z = e / np.sqrt(sig2)
    z_alpha = np.quantile(z, p)
    var_r = mu + z_alpha * sigma_next
    tail_z = z[z <= z_alpha]
    es_r = mu + sigma_next * (tail_z.mean() if len(tail_z) else z_alpha)
    return var_r, es_r


ESTIMATORS = {
    "normal": est_normal, "tdist": est_tdist, "hist": est_hist,
    "hist_w": est_hist_w, "garch": est_garch, "evt": est_evt, "fhs": est_fhs,
}


# ============================================================
# 模块4：滚动样本外回测
# ============================================================
def run_backtest(ret_all, models, confs, window=WINDOW):
    """逐日前滚：仅用当期之前 window 日估计，预测次日 VaR/ES，
    以次日真实收益率判断突破（真实收益 < VaR 阈值即突破）。
    返回长表 DataFrame：date, model, conf, var, es, breach, real_return"""
    recs = []
    r_all = ret_all["log_return"].values
    dates = ret_all["trade_date"].values
    for i in range(window, len(r_all)):
        train = r_all[i - window:i]
        real_r = r_all[i]
        for m in models:
            est = ESTIMATORS[m]
            for conf in confs:
                p = 1.0 - conf
                var_r, es_r = est(train, p)
                # 稳健性保护：单日波动超 ±30% 或非有限值时，回退历史模拟（防个别窗口估计失败）
                if (not np.isfinite(var_r)) or (not np.isfinite(es_r)) or abs(var_r) > 0.30:
                    hist_var = np.quantile(train, p)
                    tail = train[train <= hist_var]
                    var_r, es_r = hist_var, (tail.mean() if len(tail) else hist_var)
                breach = 1 if real_r < var_r else 0
                recs.append({"trade_date": dates[i], "model": m, "conf": conf,
                             "var": var_r, "es": es_r,
                             "real_return": real_r, "breach": breach})
    return pd.DataFrame(recs)


# ============================================================
# 模块5：统计检验
# ============================================================
def kupiec_lr(breaks, p):
    """Kupiec 无条件覆盖检验：LR_uc ~ χ²(1)"""
    T = len(breaks)
    N = int(breaks.sum())
    if N == 0 or N == T:
        return 0.0, 1.0
    lr = -2.0 * np.log((1 - p) ** (T - N) * p ** N) + \
         2.0 * np.log((1 - N / T) ** (T - N) * (N / T) ** N)
    return lr, 1.0 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(breaks):
    """Christoffersen 独立性检验：突破是否聚集，LR_ind ~ χ²(1)"""
    b = breaks.astype(int)
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(b)):
        if b[t - 1] == 0 and b[t] == 0:
            n00 += 1
        elif b[t - 1] == 0 and b[t] == 1:
            n01 += 1
        elif b[t - 1] == 1 and b[t] == 0:
            n10 += 1
        else:
            n11 += 1
    s = n00 + n01 + n10 + n11
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / s if s > 0 else 0.0

    def ll(a, b, prob):
        v = 0.0
        if a > 0:
            v += a * np.log(1 - prob) if (1 - prob) > 0 else 0.0
        if b > 0:
            v += b * np.log(prob) if prob > 0 else 0.0
        return v

    lr = -2.0 * (ll(n00, n01, pi) + ll(n10, n11, pi)) + \
         2.0 * (ll(n00, n01, pi0) + ll(n10, n11, pi1))
    return lr, 1.0 - stats.chi2.cdf(lr, 1)


def lrcc(breaks, p):
    """Christoffersen 联合检验：LR_cc = LR_uc + LR_ind ~ χ²(2)"""
    lr_uc, _ = kupiec_lr(breaks, p)
    lr_ind, _ = christoffersen_ind(breaks)
    return lr_uc + lr_ind, 1.0 - stats.chi2.cdf(lr_uc + lr_ind, 2)


# ============================================================
# 模块6：市场状态分段
# ============================================================
def market_state(ret_all, i, horizon=252):
    """按回测日前 252 日累计对数收益划分市场状态（年化收益代理）"""
    past = ret_all["log_return"].values[i - horizon:i]
    ann = past.sum()
    if ann > 0.05:
        return "牛市"
    if ann < -0.05:
        return "熊市"
    return "震荡市"


# ============================================================
# 模块7：压力测试
# ============================================================
def stress_test(ret_all, results):
    """历史情景：全样本最大单日跌幅 Top5；假设情景：正态概率"""
    r = ret_all["log_return"].values
    idx = np.argsort(r)[:5]
    hist_rows = []
    for k in idx:
        hist_rows.append({
            "日期": str(ret_all["trade_date"].values[k])[:10],
            "单日收益率": r[k], "损失(万元)": -r[k] * P / 10000})
    # 假设情景：以最近 750 日窗口波动率衡量 -3%/-5%/-10% 冲击
    sigma = r[-WINDOW:].std(ddof=1)
    mu = r[-WINDOW:].mean()
    scen = []
    for shock in (-0.03, -0.05, -0.10):
        z = (shock - mu) / sigma
        prob = stats.norm.cdf(z)
        scen.append({"冲击": f"{shock:.0%}", "z值": z,
                     "正态概率": prob, "波动率σ": sigma})
    return pd.DataFrame(hist_rows), pd.DataFrame(scen)


# ============================================================
# 模块8：可视化
# ============================================================
def fig1_dist(ret_all, name):
    """收益率分布直方图 + QQ 图"""
    r = ret_all["log_return"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(r, bins=60, density=True, alpha=0.6, color="steelblue", label="经验分布")
    x = np.linspace(r.min(), r.max(), 300)
    axes[0].plot(x, stats.norm.pdf(x, r.mean(), r.std(ddof=1)), "r-", lw=1.5, label="正态拟合")
    axes[0].set_title(f"{name} 收益率分布 vs 正态", fontsize=13)
    axes[0].legend()
    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title("QQ 图（经验 vs 正态）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig1_dist_{name}.png"), dpi=150)
    plt.close(fig)


def fig2_vol(ret_all, name):
    """滚动年化波动率（窗口 750 日）"""
    r = ret_all["log_return"].values
    vol = pd.Series(r).rolling(WINDOW).std(ddof=1).values * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ret_all["trade_date"].values, vol, color="darkorange", lw=1.0)
    ax.set_title(f"{name} 滚动年化波动率（窗口{WINDOW}日）——波动率聚集与状态切换", fontsize=13)
    ax.set_ylabel("年化波动率")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig2_vol_{name}.png"), dpi=150)
    plt.close(fig)


def fig3_4_var_timeseries(results, name, model, conf):
    """VaR 回测时序：真实收益 + VaR 阈值 + 突破点"""
    sub = results[(results["model"] == model) & (results["conf"] == conf)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(sub["trade_date"].values, sub["real_return"].values, color="gray", lw=0.6, label="每日真实收益率")
    ax.axhline(0, color="black", lw=0.5)
    ax.plot(sub["trade_date"].values, sub["var"].values, color="red", lw=1.0, ls="--",
            label=f"{MODEL_CN[model]} VaR {int(conf*100)}%")
    brk = sub[sub["breach"] == 1]
    ax.scatter(brk["trade_date"].values, brk["real_return"].values, color="red", s=22, zorder=5,
               label=f"突破（{len(brk)}次）")
    ax.set_title(f"{name} {MODEL_CN[model]} VaR 回测时序（{int(conf*100)}% 置信）", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig_{model}_{int(conf*100)}_{name}.png"), dpi=150)
    plt.close(fig)


def fig5_breach_rates(summary):
    """各模型实际突破率 vs 理论值对比（95% 与 99%）"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, conf in zip(axes, CONFS):
        col = f"rate{int(conf * 100)}"
        rates = [summary.loc[summary["model"] == m, col].iloc[0] for m in MODELS]
        bars = ax.bar([MODEL_CN[m] for m in MODELS], rates, color="steelblue", alpha=0.8)
        ax.axhline(1 - conf, color="red", ls="--", lw=1.5, label=f"理论 {1-conf:.0%}")
        ax.set_title(f"{int(conf*100)}% 置信：实际突破率 vs 理论值", fontsize=13)
        ax.set_ylim(0, max(max(rates), 0.07) * 1.2)
        ax.legend()
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0008, f"{v:.2%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_breach_rates.png"), dpi=150)
    plt.close(fig)


def fig6_sensitivity(sens_w, sens_c):
    """敏感性分析：窗口长度（左）与置信水平（右）"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in ["normal", "hist", "hist_w", "tdist"]:
        sub = sens_w[sens_w["model"] == m]
        axes[0].plot(sub["window"], sub["rate"], marker="o", label=MODEL_CN[m])
    axes[0].axhline(0.05, color="red", ls="--", lw=1.2)
    axes[0].set_title("窗口长度敏感性（95% 置信）", fontsize=13)
    axes[0].set_xlabel("估计窗口（交易日）")
    axes[0].set_ylabel("实际突破率")
    axes[0].legend()
    for m in ["normal", "hist", "hist_w", "tdist"]:
        sub = sens_c[sens_c["model"] == m]
        axes[1].plot(1 - sub["conf"], sub["rate"], marker="o", label=MODEL_CN[m])
    axes[1].plot([0.005, 0.1], [0.005, 0.1], "r--", lw=1.2, label="理论值")
    axes[1].set_title("置信水平敏感性（窗口750）", fontsize=13)
    axes[1].set_xlabel("尾部概率 1-置信水平")
    axes[1].set_ylabel("实际突破率")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_sensitivity.png"), dpi=150)
    plt.close(fig)


def fig7_var_es(summary):
    """各模型 VaR 与 ES 对比（95% 置信，100 万本金）"""
    x = np.arange(len(MODELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, [summary.loc[summary["model"] == m, "var95"].iloc[0] for m in MODELS],
           w, label="VaR(万元)", color="steelblue")
    ax.bar(x + w / 2, [summary.loc[summary["model"] == m, "es95"].iloc[0] for m in MODELS],
           w, label="ES(万元)", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_CN[m] for m in MODELS])
    ax.set_ylabel("损失金额（万元，本金100万）")
    ax.set_title("95% 置信：各模型 VaR 与 ES 对比（尾部风险溢价）", fontsize=13)
    for xi, m in zip(x, MODELS):
        v = summary.loc[summary["model"] == m, "var95"].iloc[0]
        e = summary.loc[summary["model"] == m, "es95"].iloc[0]
        ax.text(xi - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_var_es.png"), dpi=150)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("VaR 回测改进方案复现：多模型 × 统计检验 × 压力测试")
    print("=" * 70)

    # ---------- 数据与描述统计 ----------
    rets = {}
    descs = {}
    for name, code in CODES.items():
        df = fetch_close(code, name)
        rets[name] = build_returns(df)
        descs[name] = describe_returns(rets[name])
    desc_df = pd.DataFrame(descs).T.round(6)
    desc_df.to_csv(os.path.join(DATA_DIR, "descriptive_stats.csv"), encoding="utf-8-sig")
    print("\n--- 描述性统计 ---")
    print(desc_df.to_string())

    fig1_dist(rets["上证指数"], "上证指数")
    fig2_vol(rets["上证指数"], "上证指数")

    # ---------- 上证指数全模型回测 ----------
    print("\n" + "=" * 70)
    print("上证指数全模型回测（窗口750，704个样本）")
    print("=" * 70)
    res = run_backtest(rets["上证指数"], MODELS, CONFS)
    res.to_csv(os.path.join(DATA_DIR, "backtest_sh_full.csv"), index=False, encoding="utf-8-sig")

    # 汇总表2
    rows = []
    for m in MODELS:
        row = {"model": m}
        for conf in CONFS:
            sub = res[(res["model"] == m) & (res["conf"] == conf)]
            n_brk = int(sub["breach"].sum())
            row[f"n{int(conf*100)}"] = n_brk
            row[f"rate{int(conf*100)}"] = n_brk / len(sub)
            row[f"var{int(conf*100)}"] = -sub["var"].mean() * P / 10000
            row[f"es{int(conf*100)}"] = -sub["es"].mean() * P / 10000
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(DATA_DIR, "summary_sh.csv"), index=False, encoding="utf-8-sig")
    print("\n表2 上证指数样本外回测汇总（704个样本，本金100万元）")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------- 统计检验（表3） ----------
    test_rows = []
    for m in MODELS:
        for conf in CONFS:
            sub = res[(res["model"] == m) & (res["conf"] == conf)]
            brk = sub["breach"].values.astype(int)
            p = 1 - conf
            lr_uc, p_uc = kupiec_lr(brk, p)
            _, p_ind = christoffersen_ind(brk)
            _, p_cc = lrcc(brk, p)
            test_rows.append({"model": m, "conf": conf, "kupiec_p": p_uc,
                              "ind_p": p_ind, "cc_p": p_cc})
    tests = pd.DataFrame(test_rows)
    tests.to_csv(os.path.join(DATA_DIR, "stat_tests_sh.csv"), index=False, encoding="utf-8-sig")
    print("\n表3 统计检验结果（p值，<0.05 不通过）")
    for conf in CONFS:
        sub = tests[tests["conf"] == conf]
        print(f"--- {int(conf*100)}% 置信 ---")
        for _, r_ in sub.iterrows():
            mark = "通过" if r_["kupiec_p"] >= 0.05 and r_["cc_p"] >= 0.05 else "不通过"
            print(f"  {MODEL_CN[r_['model']]:<6} Kupiec={r_['kupiec_p']:.3f}  "
                  f"LRind={r_['ind_p']:.3f}  LRcc={r_['cc_p']:.3f}  {mark}")

    fig3_4_var_timeseries(res, "上证指数", "normal", 0.95)
    fig3_4_var_timeseries(res, "上证指数", "garch", 0.95)
    fig5_breach_rates(summary)
    fig7_var_es(summary)

    # ---------- 市场状态分段（表4） ----------
    state_rows = []
    for i in range(WINDOW, len(rets["上证指数"])):
        state_rows.append({"i": i, "state": market_state(rets["上证指数"], i)})
    state_df = pd.DataFrame(state_rows)
    print("\n表4 市场状态分布：", state_df["state"].value_counts().to_dict())

    # ---------- 敏感性分析（表5/表6） ----------
    print("\n敏感性分析进行中…")
    sens_w_rows, sens_c_rows = [], []
    for window in [250, 500, 750, 1000, 1250]:
        rr = run_backtest(rets["上证指数"], ["normal", "hist", "hist_w", "tdist"], [0.95], window)
        for m in ["normal", "hist", "hist_w", "tdist"]:
            sub = rr[rr["model"] == m]
            n_brk = int(sub["breach"].sum())
            sens_w_rows.append({"window": window, "model": m, "rate": n_brk / len(sub)})
    for conf in [0.90, 0.95, 0.975, 0.99, 0.995]:
        rr = run_backtest(rets["上证指数"], ["normal", "hist", "hist_w", "tdist"], [conf], WINDOW)
        for m in ["normal", "hist", "hist_w", "tdist"]:
            sub = rr[rr["model"] == m]
            n_brk = int(sub["breach"].sum())
            sens_c_rows.append({"conf": conf, "model": m, "rate": n_brk / len(sub)})
    sens_w = pd.DataFrame(sens_w_rows)
    sens_c = pd.DataFrame(sens_c_rows)
    sens_w.to_csv(os.path.join(DATA_DIR, "sensitivity_window.csv"), index=False, encoding="utf-8-sig")
    sens_c.to_csv(os.path.join(DATA_DIR, "sensitivity_conf.csv"), index=False, encoding="utf-8-sig")
    print("表5 窗口敏感性（95%）：")
    print(sens_w.pivot(index="window", columns="model", values="rate").round(4).to_string())
    print("表6 置信敏感性（窗口750）：")
    print(sens_c.pivot(index="conf", columns="model", values="rate").round(4).to_string())
    fig6_sensitivity(sens_w, sens_c)

    # ---------- 跨资产验证（表7） ----------
    print("\n跨资产验证进行中…")
    cross_rows = []
    for name in ["沪深300", "中证500"]:
        rr = run_backtest(rets[name], ["normal", "tdist", "hist", "hist_w", "garch"], CONFS)
        rr.to_csv(os.path.join(DATA_DIR, f"backtest_{name}.csv"), index=False, encoding="utf-8-sig")
        for m in ["normal", "tdist", "hist", "hist_w", "garch"]:
            row = {"标的": name, "model": m}
            for conf in CONFS:
                sub = rr[(rr["model"] == m) & (rr["conf"] == conf)]
                row[f"rate{int(conf*100)}"] = int(sub["breach"].sum()) / len(sub)
            cross_rows.append(row)
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(os.path.join(DATA_DIR, "cross_asset.csv"), index=False, encoding="utf-8-sig")
    print("表7 跨资产回测突破率：")
    print(cross.pivot(index="标的", columns="model",
                      values=["rate95", "rate99"]).round(4).to_string())

    # ---------- 压力测试（表8） ----------
    hist_scen, hyp_scen = stress_test(rets["上证指数"], res)
    hist_scen.to_csv(os.path.join(DATA_DIR, "stress_hist.csv"), index=False, encoding="utf-8-sig")
    hyp_scen.to_csv(os.path.join(DATA_DIR, "stress_hypo.csv"), index=False, encoding="utf-8-sig")
    print("\n表8 压力测试-历史情景（样本内最大单日跌幅 Top5）")
    print(hist_scen.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("假设情景（最近窗口σ衡量正态概率）")
    print(hyp_scen.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 70)
    print("全部完成。CSV → outputs/，图表 → figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
