# -*- coding: utf-8 -*-
"""
================================================================================
 第一步：BTC 5分钟高频数据的清洗与预处理
 ---------------------------------------------------------------------------
 输入：btc_5min_6months.csv   （原始抓取数据，含 datetime/open/high/low/close/
                                volume/quote_volume/trades/log_return）
 输出：btc_5min_clean.csv     （清洗后的建模样本）
       控制台打印一份「数据体检报告」

 为什么不能直接把原始数据丢进 GARCH？
 ---------------------------------------------------------------------------
 GARCH 模型的三条隐含假设，高频数据几乎条条都踩雷：
   1. 样本必须是**等间隔**的时间序列 → 需要检查时间戳是否有断点/重复；
   2. 收益率必须是**平稳**且**无确定性周期**的 → 5分钟数据存在极强的
      「日内季节性」（一天中不同时段波动水平系统性不同），不剔除会让
      GARCH 把周期性误判成波动聚集，参数严重有偏；
   3. 数值尺度要合适 → 收益率原始量级在 1e-3，直接估计方差会让优化器
      收敛困难，行业惯例是放大 100 倍（即换算成「百分点」）后再建模。
================================================================================
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================ 0. 全局参数 ====================================
# ★ 路径不再写死相对路径 —— 以前用相对路径 "btc_5min_6months.csv"，
#   只有当「当前工作目录」恰好等于数据文件所在目录时才找得到；
#   在 VS Code / Jupyter / 终端里从不同目录启动就会 FileNotFoundError。
#   现在改成：以本脚本所在目录为基准，再按候选清单自动搜索。
SCRIPT_DIR = Path(__file__).resolve().parent      # 本脚本所在目录（小组展示）
RAW_NAME = "btc_5min_6months.csv"                 # 原始数据文件名
OUT_FILE = str(SCRIPT_DIR / "btc_5min_clean.csv")  # 清洗后输出路径（写成绝对路径）
FREQ = "5min"                          # 采样频率（一根K线 = 5分钟）
BARS_PER_DAY = 288                     # 一天 24h × 12 根/小时 = 288 根
WINSOR_SIGMA = None                    # 异常值缩尾阈值，None=不缩尾（推荐）
#   说明：高频收益率的「极端值」绝大多数是真实行情，缩尾会人为压低尾部风险，
#         在风险管理场景下宁可保留。若做纯学术研究可设为 10（10倍标准差）。


def find_raw_file() -> Path:
    """按优先级搜索原始数据文件，找不到就给出可操作的报错。"""
    # 1) 命令行参数 / 环境变量（最高优先级，方便临时指定）
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]).expanduser())
    if os.environ.get("BTC_RAW_FILE"):
        candidates.append(Path(os.environ["BTC_RAW_FILE"]).expanduser())

    # 2) 本脚本所在目录（正常情况下 fetch 脚本就写在这里）
    candidates.append(SCRIPT_DIR / RAW_NAME)

    # 3) 当前工作目录
    candidates.append(Path.cwd() / RAW_NAME)

    # 4) 上级目录里的常见兄弟文件夹（历史版本曾把数据写到「小组汇报」）
    for folder in ("小组汇报", "小组展示", "data", "数据"):
        candidates.append(SCRIPT_DIR.parent / folder / RAW_NAME)

    # 5) 兜底：在脚本目录及其上级目录里模糊搜一遍
    for base in (SCRIPT_DIR, SCRIPT_DIR.parent):
        try:
            candidates.extend(sorted(base.glob("*btc*5min*.csv")))
            candidates.extend(sorted(base.glob("*btc*.csv")))
        except Exception:
            pass

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            continue

    print("\n[错误] 找不到原始数据文件 btc_5min_6months.csv")
    print("已搜索过的位置：")
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            print("  ·", s)
    print("\n解决办法（任选其一）：")
    print("  1) 先运行抓取脚本：python fetch_btc_5min.py")
    print("     它现在会把数据直接写到这里：", SCRIPT_DIR / RAW_NAME)
    print("  2) 或直接指定路径：python 01_preprocess_btc_5min.py /你的路径/btc_5min_6months.csv")
    raise SystemExit(1)


RAW_FILE = str(find_raw_file())


def line(title: str = "") -> None:
    """打印分隔线，让控制台报告更好读（纯展示用，不影响建模）"""
    print("\n" + "=" * 78)
    if title:
        print(f"  {title}")
        print("=" * 78)


# ============================ 1. 读取原始数据 ================================
# encoding="utf-8-sig" 是关键：原始 CSV 带有 BOM 头（Excel/Windows 抓取常见），
# 如果不用 utf-8-sig，第一列的列名会变成 "\ufeffdatetime"，后面取列时会报 KeyError。
df = pd.read_csv(RAW_FILE, encoding="utf-8-sig")

# 必备列检查：缺任何一列后面都会崩，这里提前给出明确提示
need_cols = ["datetime", "open", "high", "low", "close", "volume"]
missing_cols = [c for c in need_cols if c not in df.columns]
if missing_cols:
    print(f"\n[错误] 原始数据缺少必备列：{missing_cols}")
    print(f"当前文件的列为：{list(df.columns)}")
    print("请确认该文件是由 fetch_btc_5min.py 生成的 BTC 5 分钟 K 线数据。")
    raise SystemExit(1)

line("1. 原始数据概览")
print(f"数据文件：{RAW_FILE}")
print(f"样本行数：{len(df):,}    列数：{df.shape[1]}")
print(f"列名：{list(df.columns)}")
print(df.head(3).to_string())

# ============================ 2. 时间戳标准化 ================================
# 把字符串解析成 pandas 的 datetime64 类型，后续才能做重采样/时间索引运算。
df["datetime"] = pd.to_datetime(df["datetime"])

# 按时间升序排列：GARCH 是时序模型，样本顺序绝对不能乱。
df = df.sort_values("datetime").reset_index(drop=True)

# 去重：同一个时间戳出现两次（接口重试、网络抖动）会造成虚假的 0 收益。
n_dup = df.duplicated(subset=["datetime"]).sum()
if n_dup > 0:
    df = df.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
print(f"\n重复时间戳：{n_dup} 条（已保留最后一条）")

# ============================ 3. 时间网格完整性检查 ==========================
# 生成一段「理论完整」的 5 分钟网格（从首根到末根，每 5 分钟一个点），
# 再和实际样本做对比，缺失的位置会暴露出来。
df = df.set_index("datetime")
full_grid = pd.date_range(df.index[0], df.index[-1], freq=FREQ)
missing = full_grid.difference(df.index)

line("2. 时间网格完整性")
print(f"时间跨度：{df.index[0]}  →  {df.index[-1]}")
print(f"理论应有一 {len(full_grid):,} 根，实际 {len(df):,} 根，缺失 {len(missing)} 根")
if len(missing) > 0:
    print("缺失时点示例：", list(missing[:5]))
    # 缺失处理策略：
    #   -price 用前值填充（ffill）→ 该期收益率自动为 0，符合"这段时间没成交/没变动"；
    #   -volume/trades 填 0。
    # 注意：本数据集是 7×24 连续交易的加密货币，理论上不应有缺口；
    #       若真有缺口，通常意味着交易所宕机或接口丢包，需要人工复核。
    df = df.reindex(full_grid)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    df[["volume", "quote_volume", "trades"]] = df[["volume", "quote_volume", "trades"]].fillna(0)
    df.index.name = "datetime"

# ============================ 4. 收益率重算与校验 ============================
# 原始文件自带 log_return 列，但「别人算的」不一定可信，这里自己重算一遍做交叉验证。
# 对数收益率定义： r_t = ln(P_t / P_{t-1}) ≈ 百分比收益
# 之所以用对数收益而非简单收益：
#   (a) 多期收益可直接相加，便于把 5 分钟聚合到日/周；
#   (b) 更接近正态分布假设，且天然对称（-50% 与 +100% 互为镜像）。
df["ret"] = np.log(df["close"]).diff()

line("3. 收益率校验")
# 从第 2 行开始比对（第 1 行没有前值，原始文件用了更早一根 K 线，无法比对）
if "log_return" in df.columns:
    diff = (df["ret"] - df["log_return"]).abs().iloc[1:].max()
    print(f"自算收益率与原始 log_return 列的最大绝对偏差：{diff:.3e}")
    print("→ 偏差在 1e-9 以内说明原始列计算正确，可放心使用。")
else:
    print("原始文件没有 log_return 列（不影响建模，本脚本已自行重算 ret）")
# 第一根 K 线的 ret 为 NaN（无前值），直接丢弃，避免后面模型报错。
df = df.iloc[1:].copy()

# ============================ 5. 数据质量体检 ================================
line("4. 数据质量体检")
n = len(df)
zero_vol = (df["volume"] == 0).sum()                    # 零成交量（停牌/无成交）
zero_ret = (df["ret"] == 0).sum()                       # 零收益（价格僵持）
print(f"零成交量 K 线：{zero_vol} 根  （{zero_vol / n:.2%}）")
print(f"零收益率 K 线：{zero_ret} 根  （{zero_ret / n:.2%}）")
print("→ 少量零收益是高频数据的常态（买卖盘僵持、价格未跳动），")
print("  占比 <1% 时属于正常，不必删除；删除反而会破坏时间序列的连续性。")

r = df["ret"]
print(f"\n收益率描述统计（原始单位）：")
print(f"  均值     = {r.mean():.3e}      标准差   = {r.std():.6f}")
print(f"  最小值   = {r.min():.6f}      最大值   = {r.max():.6f}")
print(f"  偏度     = {r.skew():.3f}      峰度     = {r.kurt():.3f}（超额峰度，正态=0）")
print("→ 峰度远高于 0（本例约 20+），说明存在极端尖峰厚尾，")
print("  GARCH 的残差分布应当选 Student-t 或偏态 t，而不是正态。")

# ============================ 6. 异常值处理（可选缩尾） ======================
if WINSOR_SIGMA is not None:
    lo, hi = -WINSOR_SIGMA * r.std(), WINSOR_SIGMA * r.std()
    n_out = ((r < lo) | (r > hi)).sum()
    df["ret_raw"] = df["ret"]                       # 保留未缩尾版本备查
    df["ret"] = df["ret"].clip(lo, hi)
    print(f"\n[缩尾] 将 {n_out} 个超过 ±{WINSOR_SIGMA}σ 的收益率压缩到边界")
else:
    print("\n[缩尾] 未启用（默认保留真实尾部，符合风险管理保守原则）")

# ============================ 7. 日内季节性剔除 ==============================
# ★ 这是高频 GARCH 最关键、也最容易被忽略的一步（Andersen & Bollerslev, 1997）
#
# 现象：5 分钟收益率的波动水平在一天之内呈系统性起伏（UTC 时间下通常表现为
#       欧美交易时段活跃、亚洲凌晨清淡）。这种起伏是「日历效应」，
#       属于可预测的确定性成分，不是 GARCH 要刻画的随机波动聚集。
# 后果：若不剔除，收益率平方的自相关会被季节性主导，GARCH 会估计出
#       虚假的极强持续性（α+β 逼近 1），预测精度大幅下降。
#
# 做法（简洁稳健版）：
#   把一天切成 288 个时段（0~287），对第 i 个时段用 MAD（中位数绝对偏差）的
#   稳健估计算出该时段的典型波动 s_i，再做归一化使 s_i 的均值为 1，
#   最后 r_adj = r / s_i。
line("5. 日内季节性（Intraday Seasonality）剔除")

# 7.1 给每根 K 线打上「当日第几个 5 分钟」的标签 0~287
slot = df.index.hour * 12 + df.index.minute // 5
df["slot"] = slot

# 7.2 计算每个时段的稳健波动尺度
#     MAD = median(|x - median(x)|)，乘以 1.4826 后在正态假设下与标准差可比。
#     用中位数而非均值，是为了不被单个极端行情带偏。
def robust_scale(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

seasonal = df.groupby("slot")["ret"].apply(lambda x: robust_scale(x.values))
# 归一化：让季节因子的平均值为 1，这样只改变"相对形状"，不改变整体波动水平
seasonal = seasonal / seasonal.mean()
# 兜底：万一某个时段样本极少导致 MAD=0，用 1 代替（否则除零）
seasonal = seasonal.replace(0, 1.0)

df["seasonal_factor"] = df["slot"].map(seasonal)
df["ret_deseason"] = df["ret"] / df["seasonal_factor"]      # 剔除季节性后的收益率

# 7.3 效果对比：季节性剔除前后，收益率方差的日内分布是否"变平"
before = df.groupby("slot")["ret"].std()
after = df.groupby("slot")["ret_deseason"].std()
print(f"季节因子范围：{seasonal.min():.3f} ~ {seasonal.max():.3f}"
      f"（>1 表示该时段波动偏大）")
print(f"各时段波动的标准差 / 均值（离散度）：")
print(f"  剔除前 = {before.std() / before.mean():.4f}")
print(f"  剔除后 = {after.std() / after.mean():.4f}")
print("→ 离散度明显下降，说明日内周期性已被有效剥离。")
print("\n波动最活跃的 5 个时段（UTC）：")
top = seasonal.sort_values(ascending=False).head(5)
for s, v in top.items():
    print(f"  {s // 12:02d}:{s % 12 * 5:02d}  季节因子 = {v:.3f}")

# ============================ 8. 数值尺度转换 ================================
# 把收益率 ×100 换算成「百分点」。
# 作用：原始收益率 σ≈0.0012，方差量级 1e-6，优化器在这种尺度下容易
#       数值下溢/收敛缓慢；×100 后方差量级约 0.014，数值稳定性大幅提升。
# 注意：这只是单位换算，所有统计结论不变；最终汇报波动率时记得 ÷100 还原。
df["ret_pct"] = df["ret"] * 100.0                       # 原始收益率（百分点）
df["ret_pct_ds"] = df["ret_deseason"] * 100.0           # 去季节收益率（百分点）

# ============================ 9. 导出清洗结果 ================================
out = df[["open", "high", "low", "close", "volume", "quote_volume",
          "trades", "ret", "ret_deseason", "ret_pct", "ret_pct_ds",
          "slot", "seasonal_factor"]].copy()
out.to_csv(OUT_FILE, encoding="utf-8-sig")

line("6. 预处理完成")
print(f"输出文件：{OUT_FILE}")
print(f"最终样本：{len(out):,} 根 5 分钟 K 线"
      f"（约 {len(out) / BARS_PER_DAY:.1f} 天）")
print(f"建模用列：ret_pct（原始收益率×100）、ret_pct_ds（去季节×100）")
print("\n后续建模建议：")
print("  · 主模型用 ret_pct_ds（已剔除日内季节性，GARCH 结构更干净）")
print("  · 用 ret_pct 做稳健性对照，检验结论是否依赖季节性处理")
print("  · 汇报实际波动率数值时，记得把结果 ÷100 还原成收益率单位")
