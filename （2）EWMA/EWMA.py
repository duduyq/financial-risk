# -*- coding: utf-8 -*-
"""
================================================================================
 题目2：基于 EWMA 模型的上证综合指数在险价值（VaR）测度与回测检验
 课程：《金融风险管理》 2026 级金融专硕 · 小组汇报

 模型逻辑链：
     日收盘价 → 对数收益率 → EWMA 条件方差递推 → 条件标准差
            → VaR / ES  →  违例序列  →  Kupiec / Christoffersen / TUFF 回测
            → λ 敏感性 + 分布假设 + 基准模型对比 + 子样本  稳健性讨论

 本文件为 Stata 版 do 文件（Topic2_EWMA_VaR.do）的 Python 改写版，
 章节编号（§0–§7）与 do 文件一一对应，统计口径尽量保持一致：
   · 偏度/峰度用 Stata 口径（未修正，峰度以正态=3 为基准）
   · 百分位数用线性插值（np.percentile 默认，等价于 Stata/R type-7）
   · 对数收益率 r = 100·ln(P_t/P_{t-1})，损失 loss = -r（正号表示亏损）
   · 标准化 t 分布自由度用网格搜索 ML（df ∈ [2.5, 30]，步长 0.1）
 数据路径与 do 文件保持一致：D:\\EWMA_VaR（data/ 放原始数据，output/ 放结果）

 依赖安装：
     pip install numpy pandas scipy statsmodels arch matplotlib openpyxl
================================================================================
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")                       # 无显示环境也可运行，图形存盘
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

warnings.filterwarnings("ignore")

# =============================================================================
# §0 全局参数设置
# =============================================================================

ROOT = r"D:\EWMA_VaR"                 # 项目根目录（与 do 文件一致，勿改）
RAW  = os.path.join(ROOT, "data")     # 放原始数据
OUT  = os.path.join(ROOT, "output")   # 结果输出（表格 / 图形 / 日志）
for _d in (ROOT, RAW, OUT):
    os.makedirs(_d, exist_ok=True)

# 开启结果日志（答辩时可把 .log 附在材料里）
LOG_FILE = os.path.join(OUT, "Topic2_EWMA_VaR.log")
logger = logging.getLogger("Topic2")
logger.setLevel(logging.INFO)
for _h in list(logger.handlers):
    logger.removeHandler(_h)
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_sh = logging.StreamHandler()
_fmt = logging.Formatter("%(message)s")
_fh.setFormatter(_fmt)
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)
info = logger.info

# --- 数据文件与列名 ---
DATAFILE = "sse_close.csv"            # 数据文件名（CSV，首行为列名）
COL_DATE = "date"                     # 日期列名（不确定时程序会自动识别）
COL_CLOSE = "close"                   # 收盘价列名

# --- 样本区间 ---
START_DATE = "2010-01-04"             # 起始日（本次实证的实际起点）
END_DATE = "2025-12-31"               # 结束日

# --- 模型参数 ---
LAMBDA = 0.94                         # EWMA 衰减因子（RiskMetrics 日频标准值）
NBURN = 50                            # 方差递推的初始化窗口（交易日个数）
CONF = 95                             # 基准置信水平（%）
POSITION = 1                          # 组合市值（元），=1 时 VaR 为"比例"
HOLDDAYS = 1                          # 持有期（天）；多期 VaR 用平方根法则外推


# =============================================================================
# §1 数据导入与清洗
#   数据来源（PPT 与代码注释中同时标明，题目明确要求）：
#     Wind / CSMAR / CEIC：上证综合指数（000001.SH）日行情 → 收盘价
#     导出为 CSV：第 1 列为交易日期，第 2 列为收盘价，首行为英文列名。
# =============================================================================

# ---------- 1.1 读入原始文件（编码兜底：UTF-8 → GB18030 → 系统默认）----------
def read_csv_tolerant(path):
    for enc in ("utf-8", "gb18030", None):
        try:
            return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception:
            continue
    raise IOError(f"无法读取数据文件：{path}")

df = read_csv_tolerant(os.path.join(RAW, DATAFILE))
info(f"读入 {DATAFILE} ｜ 列名：{list(df.columns)} ｜ 原始行数：{len(df)}")

# ---------- 1.2 自动识别日期列 / 收盘价列 ----------
# 主流数据库表头对照：日期 date/tradedate/Trddt/交易日期/日期；收盘 close/Clsindex/收盘价…
# 按优先级顺序逐个关键词扫描列名，命中最靠前的那个（CSV 无变量标签，只查列名）
_PRIO_DATE = ["idxtrd01", "trddt", "trddate", "tradedate", "date", "time",
              "交易日期", "交易日", "日期"]
_PRIO_CLOSE = ["idxtrd05", "clsindex", "clsidx", "clsprc", "close", "cls",
               "收盘指数", "收盘价", "收市价", "收盘"]


def auto_find_col(cols, prio, manual):
    """manual 列存在则用之；否则按 prio 优先级扫描列名（忽略大小写）。"""
    if manual in cols:
        return manual
    lower = {c: str(c).lower() for c in cols}
    for key in prio:
        kl = key.lower()
        for c in cols:
            if kl in lower[c]:
                return c
    raise ValueError(f"未找到目标列，请手工设置参数后重试（候选关键词：{prio}）")


COL_DATE = auto_find_col(df.columns, _PRIO_DATE, COL_DATE)
COL_CLOSE = auto_find_col(df.columns, _PRIO_CLOSE, COL_CLOSE)
info(f"日期列 = {COL_DATE} ｜ 收盘价列 = {COL_CLOSE}")

# ---------- 1.2b 若导出了多只指数，只保留上证综合指数（000001）----------
# CSMAR「指数交易数据」表常同时含沪深多个指数，必须筛选，
# 否则多只指数的收盘价会按日期交错排列，收益率全是假的。
_IDX_CANDS = ["Indexcd", "indexcd", "IndexCD", "IndexCode", "index_code", "指数代码"]
idx_col = next((c for c in _IDX_CANDS if c in df.columns), None)
if idx_col is not None:
    idx_vals = df[idx_col].astype(str).str.strip()
    keep = idx_vals.isin(["000001", "1"]) | idx_vals.str.contains("000001", na=False)
    ni, nt = int(keep.sum()), len(df)
    if 0 < ni < nt:
        info(f"检测到指数代码列 {idx_col}：从 {nt} 条记录中筛出上证综指 {ni} 条")
        df = df[keep].reset_index(drop=True)
    elif ni == 0:
        info("（提示）指数代码列中没有匹配到 000001，请确认代码口径；本次不作筛选")
    else:
        info(f"数据仅含 1 只指数（{ni} 条），无需筛选")

# ---------- 1.3 统一变量名并解析日期 ----------
d_raw = df[COL_DATE].copy()
px_raw = df[COL_CLOSE].copy()

if pd.api.types.is_object_dtype(d_raw) or pd.api.types.is_string_dtype(d_raw):
    # 字符串日期：2020-01-02 / 2020/1/2 / 20200102（YMD 优先）
    d = pd.to_datetime(d_raw, errors="coerce")
    miss = d.isna()
    if miss.any():                               # 兼容 02/01/2020 这类 DMY
        d[miss] = pd.to_datetime(d_raw[miss], format="%d/%m/%Y", errors="coerce")
else:
    # 数值日期：Excel 1900 序列值（等价于 Stata 的 d_raw - 21916，1960 起点）
    d = pd.to_datetime(d_raw, unit="D", origin="1899-12-30")

# 清洗 px：先清除所有非数字、非小数点的异常字符（国泰安 NA/符号/空格专用），再转数值
if pd.api.types.is_object_dtype(px_raw) or pd.api.types.is_string_dtype(px_raw):
    px_clean = px_raw.astype(str).str.replace(r"[^0-9.]", "", regex=True)
else:
    px_clean = px_raw
df["date"] = d
df["px"] = pd.to_numeric(px_clean, errors="coerce")

n_miss_date = int(df["date"].isna().sum())
if n_miss_date > 0:
    raise ValueError(f"有 {n_miss_date} 条日期缺失，请检查 §1.3 日期解析格式")

# ---------- 1.4 清洗：缺失、重复、非法价格、样本区间 ----------
df = df.dropna(subset=["date", "px"])           # 缺失日期/价格
df = df[df["px"] > 0]                            # 删除收盘价 <= 0 的异常观测
dup_n = int(df["date"].duplicated().sum())
info(f"重复交易日数量：{dup_n}（本次数据无重复则无需删除）")
df = df.sort_values("date").reset_index(drop=True)
df = df[(df["date"] >= pd.Timestamp(START_DATE)) &
        (df["date"] <= pd.Timestamp(END_DATE))].reset_index(drop=True)

# ---------- 1.5 构造交易日时间索引 ----------
# 日频金融数据存在周末/节假日缺口，用连续序号 t 作为时间轴是标准做法。
df["t"] = np.arange(1, len(df) + 1)
df = df[["t", "date", "px"]]
info("\n===== 清洗后样本 =====")
info(f"日期范围：{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
info(f"交易日数 N = {len(df)}")
assert len(df) > 200, "样本过短会导致后续检验失效"


# =============================================================================
# §2 变量定义与描述性统计
# =============================================================================

# ---------- 2.1 收益率定义 ----------
# 对数收益率（%）：r_t = 100 * ln(P_t / P_{t-1})；loss = -r（正号=亏损）
df["r"] = 100.0 * (np.log(df["px"]) - np.log(df["px"].shift(1)))
df["loss"] = -df["r"]
df["r2"] = df["r"] ** 2

r_full = df["r"].values                    # 含首行 NaN（Stata 中同样缺失）
r = r_full[~np.isnan(r_full)]              # 有效收益率（统计/GARCH 用）


def stata_desc(x):
    """Stata summarize,detail 口径的描述统计。
    skew/kurt 用未修正矩（Pearson 峰度，正态=3，对应 Stata summarize）；百分位用线性插值。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return dict(
        N=len(x), mean=x.mean(), sd=x.std(ddof=1), min=x.min(), max=x.max(),
        skewness=stats.skew(x, bias=True),
        kurtosis=stats.kurtosis(x, bias=True, fisher=False),
        p1=np.percentile(x, 1), p5=np.percentile(x, 5), p50=np.percentile(x, 50),
        p95=np.percentile(x, 95), p99=np.percentile(x, 99),
    )


# ---------- 2.2 描述性统计 ----------
info("\n===== 表1：收益率描述性统计 =====")
desc = stata_desc(r)
desc_df = pd.DataFrame([desc])
print(desc_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
for k, v in desc.items():
    info(f"{k:9s} = {v:.4f}")

# 偏度/峰度正式检验（D'Agostino 等，原假设：正态）
# scipy 的 skewtest / kurtosistest 与 Stata sktest 同源（D'Agostino 变换）
sk_z, sk_p = stats.skewtest(r)
ku_z, ku_p = stats.kurtosistest(r)
# Jarque-Bera 联合检验：JB = n/6·[S² + (K−3)²/4] ~ χ²(2)
jb_stat = len(r) / 6 * (desc["skewness"] ** 2 + (desc["kurtosis"] - 3) ** 2 / 4)
jb_p = stats.chi2.sf(jb_stat, 2)
info(f"\n偏度检验 p = {sk_p:.6g} ｜ 峰度检验 p = {ku_p:.6g} ｜ 联合检验 p = {jb_p:.6g}")
info(f"Jarque-Bera = {jb_stat:.4f}（χ²(2) 的 p = {jb_p:.6g}）")
info(f"超额峰度 = {desc['kurtosis'] - 3:.4f}（>0 即尖峰厚尾）")

# ---------- 2.3 年化波动率（汇报时的直观口径） ----------
obs_sd = desc["sd"]
info(f"年化波动率 ≈ {obs_sd * np.sqrt(252):.4f} %")

# ---------- 2.4 保存描述性统计（供 §7 导出） ----------
M_desc = pd.DataFrame(
    [[desc["N"], desc["mean"], desc["sd"], desc["min"], desc["max"],
      desc["skewness"], desc["kurtosis"], desc["p1"], desc["p5"],
      desc["p95"], desc["p99"]]],
    index=["日收益率r"],
    columns=["N", "均值", "标准差", "最小", "最大", "偏度", "峰度",
             "p1", "p5", "p95", "p99"],
)


# =============================================================================
# §3 收益率诊断检验（为"为什么用 EWMA"提供实证依据）
#   (a) 收益率平稳       → ADF 单位根检验
#   (b) 收益率自身弱相关 → Ljung-Box（水平）
#   (c) 波动率聚集       → Ljung-Box（平方）+ ARCH-LM
# =============================================================================

# ---------- 3.1 平稳性：ADF 单位根检验 ----------
info("\n===== 表2（上）：ADF 单位根检验 =====")
adf0 = adfuller(r, maxlag=0, autolag=None)
adf5 = adfuller(r, maxlag=5, autolag=None)
info(f"ADF(0) 统计量 = {adf0[0]:.4f}  p 值 = {adf0[1]:.4f}")
info(f"ADF(5) 统计量 = {adf5[0]:.4f}  p 值 = {adf5[1]:.4f}"
     "（p<0.05 拒绝单位根 ⇒ 收益率平稳）")

# ---------- 3.2 自相关：Ljung-Box 白噪声检验 ----------
info("\n===== 表2（中）：Ljung-Box 检验 =====")
lb_r = acorr_ljungbox(r, lags=[10], return_df=True)
lb_r2 = acorr_ljungbox(r ** 2, lags=[10], return_df=True)
info(f"Ljung-Box(r, 10)   Q = {lb_r['lb_stat'].iloc[0]:.4f}  p = {lb_r['lb_pvalue'].iloc[0]:.4f}")
info(f"Ljung-Box(r², 10)  Q = {lb_r2['lb_stat'].iloc[0]:.4f}  p = {lb_r2['lb_pvalue'].iloc[0]:.4f}"
     "（r² 拒绝 ⇒ 波动聚集/ARCH）")

# ---------- 3.3 ARCH 效应：Engle 的 LM 检验 ----------
# 做法：把 r² 对其自身滞后 1–5 期回归，LM = T·R² ~ chi2(5)
r2c = r ** 2
n2 = len(r2c)
X_arch = np.column_stack([np.ones(n2 - 5),
                          r2c[4:-1], r2c[3:-2], r2c[2:-3], r2c[1:-4], r2c[0:-5]])
y_arch = r2c[5:]
beta_arch, *_ = np.linalg.lstsq(X_arch, y_arch, rcond=None)
resid_arch = y_arch - X_arch @ beta_arch
r2_arch = 1 - (resid_arch @ resid_arch) / ((y_arch - y_arch.mean()) ** 2).sum()
LM_arch = (n2 - 5) * r2_arch
LM_p = stats.chi2.sf(LM_arch, 5)
info(f"\n===== 表2（下）：ARCH-LM 检验 =====")
info(f"ARCH-LM(5) = {LM_arch:.4f}    p 值 = {LM_p:.4f}")
info("（p<0.05 ⇒ 存在显著 ARCH 效应，支持使用 EWMA/GARCH 类条件方差模型）")

# ---------- 3.4 GARCH(1,1) 预拟合：读取波动持久性 α+β ----------
# EWMA 等价于 ω=0、α+β=1 的 IGARCH(1,1)；若 α+β 接近 1，支持 EWMA 的持久性设定。
res_g_pre = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
alpha_g = res_g_pre.params["alpha[1]"]
beta_g = res_g_pre.params["beta[1]"]
persist = alpha_g + beta_g
info(f"\nGARCH(1,1) 波动持久性 α+β = {persist:.4f}（接近 1 ⇒ 支持 EWMA 设定）")
info("含义：α+β→1 意味着冲击对波动率的影响几乎永久，正好对应 EWMA 的 λ 高持久性设定")


# =============================================================================
# §4 EWMA 模型：条件方差递推与 VaR / ES 计算
#   模型（RiskMetrics, 1996）：σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
#   VaR_t(c) = σ_t · Φ⁻¹(c) · V（正态假设，均值取 0）
# =============================================================================

lam = LAMBDA
nb = NBURN
c = CONF / 100.0
alpha = 1 - c                                    # 左尾概率（95% → 0.05）
z_c = stats.norm.ppf(c)

n = len(df)

# ---------- 4.1 EWMA 方差递推 ----------
# h_t：用 t-1 及之前信息预测的 t 期方差；用前 nb 期无条件方差初始化（避免预热偏差）
def nanvar(x, ddof=1):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return x.var(ddof=ddof)


h = np.full(n, np.nan)
h[nb] = nanvar(r_full[:nb], ddof=1)              # Stata: sum r in 1/nb → r(Var)
for i in range(nb + 1, n):
    h[i] = lam * h[i - 1] + (1 - lam) * r_full[i - 1] ** 2
df["h"] = h

# 递推正确性自检：与"几何加权和"定义直接对账（任选一期核对）
# h[chk] ≈ (1-λ)·Σ_{i=chk-199}^{chk-1} λ^{chk-i-1}·r²_i
chk = nb + 200                                    # 1-indexed 期号（Stata 同款）
sel = (np.arange(1, n + 1) >= chk - 199) & (np.arange(1, n + 1) <= chk - 1)
wgt = (1 - lam) * lam ** (chk - np.arange(1, n + 1) - 1)
wsum = np.nansum(wgt[sel] * r_full[sel] ** 2)     # 几何加权和用的是 r²（Stata: prod = wgt * r^2）
h_chk = df["h"].iloc[chk - 1]
info(f"\n递推自检：h[{chk}] = {h_chk:.6f} ｜ 几何加权和 = {wsum:.6f}（两者应高度接近）")

# ---------- 4.2 条件标准差与年化波动率 ----------
df["sigma"] = np.sqrt(df["h"])
df["annvol"] = df["sigma"] * np.sqrt(252)        # 年化波动率(%)

# ---------- 4.3 VaR 与 ES（三种分布假设） ----------
df["VaR_n"] = df["sigma"] * z_c                  # ① 正态假设（RiskMetrics 标准做法）

# ② 标准化 Student-t：先对标准化残差 z=r/σ 网格搜索 ML 自由度 df，再取分位数
z = df["r"] / df["sigma"]
z = z / np.sqrt(nanvar(z.values))                # 再标准化，确保方差为 1
df["z"] = z
z_valid = z.dropna().values

best_ll, best_df = -1e300, 5.0
for d_int in range(25, 301):                     # df ∈ [2.5, 30]，步长 0.1
    df_cand = d_int / 10.0
    sc = np.sqrt(df_cand / (df_cand - 2))        # 标准化 t 的缩放因子
    ll = stats.t.logpdf(z_valid * sc, df_cand) + np.log(sc)
    s = ll.sum()
    if s > best_ll:
        best_ll, best_df = s, df_cand
df_t = best_df
info(f"\n标准化 t 分布 ML 估计自由度 df = {df_t:.1f}（df 越小 ⇒ 尾部越厚）")
info(f"对数似然 = {best_ll:.4f}")

# 单位方差标准化 t 的右尾 c 分位数（正数）：
# Stata: q_raw = invttail(df,c)（负）→ q = q_raw·√((df-2)/df) → VaR_t = -σ·q
q_t = stats.t.ppf(c, df_t) * np.sqrt((df_t - 2) / df_t)
df["VaR_t"] = df["sigma"] * q_t

# ③ 滤波历史模拟 FHS：标准化残差的经验分位数 × 当期 σ（不假设参数分布）
q_emp = np.percentile(z_valid, alpha * 100)      # 第 5 百分位（负值）
df["VaR_f"] = -df["sigma"] * q_emp

# ④ 预期损失 ES（正态闭式解）：ES = σ·φ(Φ⁻¹(α))/α
df["ES_n"] = df["sigma"] * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

info(f"\n分位数对照：正态 z_c = {z_c:.4f}"
     f" ｜ 标准化t q_c(df={df_t:.1f}) = {q_t:.4f}"
     f" ｜ 经验 z_c = {-q_emp:.4f}")

# ---------- 4.4 多期 VaR（平方根法则） ----------
df["VaR_h"] = df["VaR_n"] * np.sqrt(HOLDDAYS)

# ---------- 4.5 金额口径 VaR ----------
df["VaR_amt"] = df["VaR_n"] / 100 * POSITION

# ---------- 4.6 描述 VaR 序列 ----------
info("\n===== 表3：VaR / ES 序列特征 =====")
t3 = df[["VaR_n", "VaR_t", "VaR_f", "ES_n"]].describe().loc[
    ["count", "mean", "std", "min", "max"]]
t3.index = ["N", "均值", "标准差", "最小", "最大"]
print(t3.to_string(float_format=lambda v: f"{v:.4f}"))


# =============================================================================
# §5 回测检验：Kupiec / Christoffersen / TUFF
#   若模型正确，违例（loss > VaR）应是"概率为 α 的独立伯努利试验"。
#   · Kupiec LRuc         —— 违例【频率】是否为 α（无条件覆盖）
#   · Christoffersen      —— 违例是否【独立】（有无扎堆）
#   · LRcc = LRuc+LRind   —— 联合检验，~ chi2(2)
#   · TUFF                —— 首次违例出现得是否过早
# =============================================================================

def varback(loss, varr, p, mask=None, label=""):
    """对应 Stata 的 rclass 程序 varback。
    loss 与 VaR 必须以"正号表示亏损、正号表示 VaR"的口径录入。
    返回 dict：T, x, rate, LRuc, p_uc, LRind, p_ind, LRcc, p_cc, v, LRtuff, p_tuff, tick
    """
    loss = np.asarray(loss, dtype=float)
    varr = np.asarray(varr, dtype=float)
    valid = ~(np.isnan(loss) | np.isnan(varr))
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    loss, varr = loss[valid], varr[valid]
    T = len(loss)
    if T == 0:
        raise ValueError("有效样本为 0，请检查 mask 条件")
    hit = (loss > varr).astype(int)
    x = int(hit.sum())
    rate = x / T

    # ---- 5.1 Kupiec 非条件覆盖检验 ----
    lnLr = (T - x) * np.log(1 - p) + x * np.log(p)
    if x == 0 or x == T:                          # 边界情形：非受限似然 = 1
        lnLu = 0.0
    else:
        lnLu = (T - x) * np.log(1 - x / T) + x * np.log(x / T)
    LRuc = max(-2 * (lnLr - lnLu), 0.0)
    p_uc = stats.chi2.sf(LRuc, 1)

    # ---- 5.2 Christoffersen 独立性检验（一阶马尔可夫转移计数） ----
    h0 = hit[:-1]
    h1 = hit[1:]
    n00 = int(((h0 == 0) & (h1 == 0)).sum())
    n01 = int(((h0 == 0) & (h1 == 1)).sum())
    n10 = int(((h0 == 1) & (h1 == 0)).sum())
    n11 = int(((h0 == 1) & (h1 == 1)).sum())
    s0, s1 = n00 + n01, n10 + n11
    nn = s0 + s1
    pi = (n01 + n11) / nn                          # 全样本违例率
    pi0 = n01 / s0 if s0 > 0 else 0.0              # π01
    pi1 = n11 / s1 if s1 > 0 else 0.0              # π11
    if nn == 0 or (n01 + n11) == 0 or (n01 + n11) == nn:
        LRind, p_ind = 0.0, np.nan                 # 退化情形：独立性无法识别
    else:
        lnLr2 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        lnLu2 = (n00 * np.log(1 - pi0) if n00 > 0 else 0.0) \
              + (n01 * np.log(pi0) if n01 > 0 else 0.0) \
              + (n10 * np.log(1 - pi1) if n10 > 0 else 0.0) \
              + (n11 * np.log(pi1) if n11 > 0 else 0.0)
        LRind = max(-2 * (lnLr2 - lnLu2), 0.0)
        p_ind = stats.chi2.sf(LRind, 1)
    LRcc = LRuc + LRind                             # 联合（条件覆盖）检验 ~ chi2(2)
    p_cc = stats.chi2.sf(LRcc, 2)

    # ---- 5.3 TUFF：首次违例时间检验 ----
    if x > 0:
        v = int(np.where(hit == 1)[0][0]) + 1      # 窗口内第 v 天（1-indexed）
        if v > 1:
            LRtuff = max(-2 * (np.log(p) + (v - 1) * np.log(1 - p))
                         + 2 * (np.log(1 / v) + (v - 1) * np.log(1 - 1 / v)), 0.0)
        else:
            LRtuff = max(-2 * np.log(p), 0.0)
        p_tuff = stats.chi2.sf(LRtuff, 1)
    else:
        v, LRtuff, p_tuff = np.nan, np.nan, np.nan

    # ---- 5.4 分位数损失（Tick Loss）：mean((α - I_t)·(VaR_t - loss_t))，越小越好 ----
    tick = float(np.mean((p - hit) * (varr - loss)))

    # ---- 5.5 输出 ----
    info(f"\n----- 回测结果：{label} （α = {p:.4f}） -----")
    info(f"样本期数 T      = {T}")
    info(f"违例次数 x      = {x}   （理论期望 {p * T:.1f} 次）")
    info(f"违例率           = {rate:.4f} （目标 {p:.4f}）")
    info(f"Kupiec  LRuc    = {LRuc:.4f}  p = {p_uc:.4f}")
    info(f"Christ. LRind   = {LRind:.4f}  p = {p_ind:.4f}")
    info(f"联合    LRcc    = {LRcc:.4f}  p = {p_cc:.4f}")
    info(f"TUFF    首次违例v= {v:.0f}     p = {p_tuff:.4f}")
    info(f"平均 Tick Loss   = {tick:.4f}")

    return dict(T=T, x=x, rate=rate, LRuc=LRuc, p_uc=p_uc, LRind=LRind,
                p_ind=p_ind, LRcc=LRcc, p_cc=p_cc, v=v, LRtuff=LRtuff,
                p_tuff=p_tuff, tick=tick)


def bt_row(res):
    return [res["T"], res["x"], res["rate"], res["LRuc"], res["p_uc"],
            res["LRind"], res["p_ind"], res["LRcc"], res["p_cc"], res["tick"]]


_BT_COLS = ["T", "x", "rate", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"]

# ---------- 5.6 对基准模型执行回测（只在 EWMA 有效区间内） ----------
ewin = df["VaR_n"].notna().values
info("\n===== 表4：基准模型 EWMA(λ=%s) 回测 =====" % LAMBDA)
res_n = varback(df["loss"], df["VaR_n"], alpha, mask=ewin, label="EWMA-正态 95%")
res_t = varback(df["loss"], df["VaR_t"], alpha, mask=ewin, label="EWMA-t 95%")
res_f = varback(df["loss"], df["VaR_f"], alpha, mask=ewin, label="EWMA-FHS 95%")
M_bt = pd.DataFrame([bt_row(res_n), bt_row(res_t), bt_row(res_f)],
                    index=["EWMA-正态", "EWMA-t", "EWMA-FHS"], columns=_BT_COLS)


# =============================================================================
# §6 稳健性讨论
# =============================================================================

# ---- 6.1 λ 敏感性分析：λ ∈ {0.90, 0.94, 0.97, 0.99} ----
# λ 越大 → 方差越平滑 → VaR 越"钝" → 波动骤起时更易连续违例；
# λ 越小 → 反应快，但 VaR 频繁跳动，资本占用更高。
info("\n===== 表5：λ 敏感性分析（c = %d%%）=====" % CONF)
lam_list = [0.90, 0.94, 0.97, 0.99]
M_lam_rows = []
for L in lam_list:
    hh = np.full(n, np.nan)
    hh[nb] = nanvar(r_full[:nb], ddof=1)
    for i in range(nb + 1, n):
        hh[i] = L * hh[i - 1] + (1 - L) * r_full[i - 1] ** 2
    vv = np.sqrt(hh) * z_c
    res = varback(df["loss"], vv, alpha, mask=ewin, label=f"EWMA lambda={L}")
    M_lam_rows.append(bt_row(res))
M_lam = pd.DataFrame(M_lam_rows,
                     index=[f"lam{str(L).replace('.', '')}" for L in lam_list],
                     columns=_BT_COLS)
print(M_lam.to_string(float_format=lambda v: f"{v:.4f}"))
for L, row in zip(lam_list, M_lam_rows):
    info(f"λ={L}: 违例率 {row[2]:.4f} ｜ LRuc p={row[4]:.4f} ｜ TickLoss={row[9]:.4f}")

# 按 Tick Loss 准则（越小越好）选出最优 λ
best_tick = M_lam["TickLoss"].min()
best_L = lam_list[int(np.argmin(M_lam["TickLoss"].values))]
info(f"\n按 Tick Loss 准则，最优 λ = {best_L}（TickLoss = {best_tick:.4f}）")
info("说明：Tick Loss 是分位数损失函数，越小代表 VaR 序列整体更贴合真实分位数。")

# ---- 6.2 置信水平敏感性：90% / 95% / 99% ----
info("\n===== 表6：不同置信水平下的 EWMA 回测（λ=%s）=====" % LAMBDA)
conf_list = [90, 95, 99]
M_c_rows = []
for cc in conf_list:
    a2 = 1 - cc / 100.0
    vv = df["sigma"].values * stats.norm.ppf(cc / 100.0)
    res = varback(df["loss"], vv, a2, mask=ewin, label=f"conf={cc}")
    M_c_rows.append(bt_row(res))
M_c = pd.DataFrame(M_c_rows, index=[f"c{cc}" for cc in conf_list], columns=_BT_COLS)
print(M_c.to_string(float_format=lambda v: f"{v:.4f}"))

# ---- 6.3 基准模型横向对比：历史模拟法(HS) 与 GARCH(1,1)-t ----
info("\n===== 表7：与历史模拟法、GARCH(1,1) 的对比 =====")

# (a) 历史模拟法：滚动 250 个交易日取经验分位数（Stata 通用循环路径的等价实现）
W = 250
VaR_hs_raw = np.full(n, np.nan)
for i in range(W, n):
    win = r_full[i - W:i]
    win = win[~np.isnan(win)]                     # Stata summarize 自动忽略缺失
    VaR_hs_raw[i] = np.percentile(win, alpha * 100)
df["VaR_hs"] = -VaR_hs_raw                        # 取负号 → 正的损失口径

# (b) GARCH(1,1)-t：作为 EWMA 的上位模型（自由度退回 §4.3 的 ML 估计兜底）
res_gt = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="t").fit(disp="off")
sigma_g = res_gt.conditional_volatility           # 条件标准差 σ_t（与 r 等长）
df_g = float(res_gt.params["nu"])
if not (df_g > 2) or np.isnan(df_g):
    info(f"（提示）GARCH-t 估计自由度不可用，改用 §4.3 网格搜索 df = {df_t:.1f}")
    df_g = df_t
df["VaR_g"] = np.nan
df.loc[df["r"].notna(), "VaR_g"] = sigma_g * stats.t.ppf(c, df_g) * np.sqrt((df_g - 2) / df_g)
info(f"GARCH(1,1)-t 估计自由度 = {df_g:.2f}")

# (c) 三方对比（口径统一：违例率 + Kupiec p + 平均 Tick Loss）
M_cmp_rows = []
for m in ["n", "hs", "g"]:
    res = varback(df["loss"], df[f"VaR_{m}"], alpha, mask=ewin,
                  label=f"model VaR_{m}")
    M_cmp_rows.append(bt_row(res))
M_cmp = pd.DataFrame(M_cmp_rows, index=["EWMA-N", "HS250", "GARCH-t"],
                     columns=_BT_COLS)
print(M_cmp.to_string(float_format=lambda v: f"{v:.4f}"))

# ---- 6.4 子样本稳健性：极端事件期是否失效？（VaR 模型真正的考验） ----
info("\n===== 表8：子样本（压力期）稳健性 =====")
subs = [
    ("2015股灾", "2015-06-01", "2016-01-31"),
    ("2018熊市", "2018-01-01", "2018-12-31"),
    ("2020疫情", "2020-01-01", "2020-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
]
M_sub_rows = []
for name, s, e in subs:
    m = (df["date"] >= pd.Timestamp(s)) & (df["date"] <= pd.Timestamp(e))
    res = varback(df["loss"], df["VaR_n"], alpha, mask=ewin & m.values, label=name)
    M_sub_rows.append(bt_row(res))
res_all = varback(df["loss"], df["VaR_n"], alpha, mask=ewin, label="全样本")
M_sub_rows.append(bt_row(res_all))
M_sub = pd.DataFrame(M_sub_rows,
                     index=["2015股灾", "2018熊市", "2020疫情", "2024-2025", "全样本"],
                     columns=_BT_COLS)
print(M_sub.to_string(float_format=lambda v: f"{v:.4f}"))

# ---- 6.5 多期 VaR 的合理性检查（平方根法则 vs 直接递推） ----
# √h 法则成立的前提是收益率 i.i.d.；波动聚集会使其低估多期风险。
info("\n===== 表9：多期 VaR（√h 外推，基于最后一期 σ）=====")
sig_last = df["sigma"].iloc[-1]
for hd in [1, 5, 10, 20]:
    info(f"h = {hd:3d} 天  VaR(95%) = {sig_last * z_c * np.sqrt(hd):8.4f} %")


# =============================================================================
# §7 图形绘制与结果导出
# =============================================================================

# 中文字体设置（按系统可用字体自动选取；Windows 一般命中 SimHei/微软雅黑）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei",
                                   "WenQuanYi Zen Hei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 7.1 图1：收益率时序图（展示波动聚集） ----------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["r"], lw=0.6, color="navy", alpha=0.6)
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("日收益率(%)")
ax.set_xlabel("")
ax.set_title("图1  上证指数日收益率：波动聚集特征")
ax.text(0.005, 0.98, f"数据来源：Wind/CSMAR；样本：{START_DATE} 至 {END_DATE}",
        transform=ax.transAxes, va="top", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_returns.png"), dpi=200)
plt.close(fig)

# ---------- 7.2 图2：EWMA 年化波动率 ----------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["annvol"], lw=1.0, color="crimson")
ax.set_ylabel("年化波动率(%)")
ax.set_xlabel("")
ax.set_title(f"图2  EWMA 条件波动率（λ={LAMBDA}）")
ax.text(0.005, 0.98, "年化口径 = 日标准差 × √252", transform=ax.transAxes,
        va="top", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_volatility.png"), dpi=200)
plt.close(fig)

# ---------- 7.3 图3：VaR 与实际损失 + 违例点（回测的核心可视化） ----------
hit_n = (df["loss"] > df["VaR_n"]) & ewin
fig, ax = plt.subplots(figsize=(12, 5))
x = df["date"].values
ax.fill_between(x, df["VaR_n"], df["loss"], where=ewin, alpha=0.15, color="navy")
ax.plot(x, df["VaR_n"], lw=1.2, color="navy", label="VaR(95%, EWMA-正态)")
ax.plot(x, df["loss"], lw=0.6, color="gray", label="实际损失")
ax.scatter(df["date"][hit_n], df["loss"][hit_n], s=14, facecolors="none",
           edgecolors="crimson", label="违例")
ax.set_ylabel("损失 / VaR (%)")
ax.set_xlabel("")
ax.set_title("图3  VaR 回测：损失是否突破 VaR 边界")
ax.legend(loc="lower center", ncol=3, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_var_backtest.png"), dpi=200)
plt.close(fig)

# ---------- 7.4 图4：收益率分布 vs 正态（尖峰厚尾证据） ----------
fig, ax = plt.subplots(figsize=(7, 7))
stats.probplot(r, dist="norm", plot=ax)
ax.set_title("图4  收益率 Q-Q 图：尾部偏离正态")
ax.set_ylabel("样本分位数")
ax.set_xlabel("正态理论分位数")
ax.text(0.05, 0.92, "两端明显偏离 45° 线 ⇒ 尖峰厚尾，正态 VaR 会低估极端风险",
        transform=ax.transAxes, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_qq.png"), dpi=200)
plt.close(fig)

# ---------- 7.5 图5：λ 违例率敏感性（把表5画出来更直观） ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lam_list, M_lam["rate"].values, marker="D", color="navy", label="实际违例率")
ax.axhline(alpha, ls="--", color="crimson", label=f"理论水平 α={alpha}")
ax.set_ylabel("实际违例率")
ax.set_xlabel("衰减因子 λ")
ax.set_title("图5  λ 敏感性：不同衰减因子下的违例率")
ax.legend(loc="center right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_lambda.png"), dpi=200)
plt.close(fig)

# ---------- 7.6 保存清洗后的分析数据集 ----------
df.to_csv(os.path.join(OUT, "analysis_data.csv"), index=False,
          encoding="utf-8-sig")                   # utf-8-sig 便于 Excel 直接打开

# ---------- 7.7 导出结果表（一）：CSV ----------
# 表头/行名用 ASCII，避免 Excel 直接打开 UTF-8 CSV 时中文乱码
def csvmat(mat, rownames, colnames, filename):
    path = os.path.join(OUT, filename)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("rowname," + ",".join(colnames) + "\n")
        for rname, row in zip(rownames, mat.values):
            f.write(rname + "," + ",".join(f"{v:.6f}" for v in row) + "\n")
    info(f"已导出 {filename}")


csvmat(M_desc, ["r"],
       ["N", "mean", "sd", "min", "max", "skewness", "kurtosis", "p1", "p5", "p95", "p99"],
       "表1_描述性统计.csv")
csvmat(M_bt, ["EWMA-normal", "EWMA-t", "EWMA-FHS"],
       ["T", "x", "rate", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
       "表4_基准回测.csv")
csvmat(M_lam, ["lam090", "lam094", "lam097", "lam099"],
       ["T", "x", "rate", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
       "表5_lambda敏感性.csv")
csvmat(M_c, ["c90", "c95", "c99"],
       ["T", "x", "rate", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
       "表6_置信水平.csv")
csvmat(M_cmp, ["EWMA-N", "HS250", "GARCH-t"],
       ["T", "x", "rate", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
       "表7_模型对比.csv")
csvmat(M_sub, ["S2015", "S2018", "S2020", "S2024-2025", "FullSample"],
       ["T", "x", "rate", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
       "表8_子样本.csv")

# ---------- 7.8 导出结果表（二）：Excel 汇总工作簿 ----------
# 布局与 Stata putexcel 一致：A 列行名，B 列起数据，第 1 行列名，数据从第 2 行开始
from openpyxl import Workbook

wb = Workbook()
ws0 = wb.active
ws0.title = "说明"
ws0["A1"] = "题目2：EWMA-VaR 上证指数——结果汇总"
ws0["A2"] = f"样本区间：{START_DATE} ~ {END_DATE} ｜ λ={LAMBDA} ｜ 初始化窗口={NBURN}"


def add_sheet(wb, name, colnames, rownames, mat, col_label="模型"):
    ws = wb.create_sheet(name)
    ws["A1"] = col_label
    for j, cn in enumerate(colnames, start=2):
        ws.cell(row=1, column=j, value=cn)
    for i, rn in enumerate(rownames, start=2):
        ws.cell(row=i, column=1, value=rn)
        for j, v in enumerate(mat.values[i - 2], start=2):
            ws.cell(row=i, column=j, value=float(v))


add_sheet(wb, "描述性统计",
          ["N", "均值", "标准差", "最小", "最大", "偏度", "峰度", "p1", "p5", "p95", "p99"],
          ["日收益率r"], M_desc, col_label="统计量")
add_sheet(wb, "基准回测",
          ["T", "违例数x", "违例率", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
          ["EWMA-正态", "EWMA-t", "EWMA-FHS"], M_bt)
add_sheet(wb, "λ敏感性",
          ["T", "违例数x", "违例率", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
          ["0.90", "0.94", "0.97", "0.99"], M_lam, col_label="λ")
add_sheet(wb, "置信水平",
          ["T", "违例数x", "违例率", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
          ["90%", "95%", "99%"], M_c, col_label="置信水平 c")
add_sheet(wb, "模型对比",
          ["T", "违例数x", "违例率", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
          ["EWMA-N", "HS250", "GARCH-t"], M_cmp)
add_sheet(wb, "子样本",
          ["T", "违例数x", "违例率", "LRuc", "p_uc", "LRind", "p_ind", "LRcc", "p_cc", "TickLoss"],
          ["2015股灾", "2018熊市", "2020疫情", "2024-2025", "全样本"], M_sub)
wb.save(os.path.join(OUT, "结果汇总.xlsx"))
info("已导出 结果汇总.xlsx（6 个工作表）")

# ---------- 7.9 ARCH-LM 辅助回归系数（对应 Stata 的可选 esttab 导出） ----------
info("\n===== ARCH-LM 辅助回归系数（r² 对 L1–L5 滞后项）=====")
coef_names = ["常数项", "L1.r2", "L2.r2", "L3.r2", "L4.r2", "L5.r2"]
for nm, b in zip(coef_names, beta_arch):
    info(f"{nm:8s} = {b:.4f}")

info("\n" + "=" * 66)
info("  全部执行完毕。输出目录：" + OUT)
info("  表格：结果汇总.xlsx ＋ 表1/表4-表8 的 CSV ｜ 图形：fig1-fig5.png")
info(f"  日志：Topic2_EWMA_VaR.log ｜ 数据集：analysis_data.csv")
info("=" * 66)
logging.shutdown()