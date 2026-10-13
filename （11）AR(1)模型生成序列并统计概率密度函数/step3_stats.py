# -*- coding: utf-8 -*-
"""
第 11 题 · 步骤 3：统计偏度与峰度
运行方式：python step3_stats.py

这一步完成题目要求的第二项：统计两组序列的偏度和峰度。
同时做一个自检：手算结果与 scipy 库函数结果对比。
"""

import os
import numpy as np
from scipy import stats
from q11_core import hand_moments

OUTDIR = "results"


def describe(x, name):
    """输出一组序列的描述性统计"""
    ex_kurt = stats.kurtosis(x)          # 超额峰度（正态 = 0）
    raw_kurt = ex_kurt + 3               # 原始峰度（正态 = 3）
    hs, hk = hand_moments(x)             # 手算结果
    jb, jb_p = stats.jarque_bera(x)
    return {
        "序列": name,
        "偏度": stats.skew(x),
        "峰度": raw_kurt,
        "手算偏度": hs,
        "手算峰度": hk,
        "JB统计量": jb,
        "JB的p值": jb_p,
    }


def main():
    print("=" * 62)
    print("步骤 3 / 6    统计偏度与峰度")
    print("=" * 62)
    print()

    r1 = np.load(os.path.join(OUTDIR, "r1.npy"))
    r2 = np.load(os.path.join(OUTDIR, "r2.npy"))

    rows = [describe(r1, "情况1 恒定方差"), describe(r2, "情况2 随机波动率")]

    print("  （正态分布：偏度 = 0，原始峰度 = 3）")
    print()
    print("  " + "-" * 58)
    print("  {:<18}{:>12}{:>12}{:>16}".format("序列", "偏度", "峰度", "JB的p值"))
    print("  " + "-" * 58)
    for r in rows:
        print("  {:<18}{:>12.6f}{:>12.6f}{:>16.3e}".format(
            r["序列"], r["偏度"], r["峰度"], r["JB的p值"]))
    print("  " + "-" * 58)
    print()

    print("  【结果解读】")
    print("    偏度：两组都接近 0，说明分布左右对称。")
    print("    峰度：情况1 接近 3（正态水平）；情况2 明显大于 3，说明尾部更厚。")
    print("    JB检验：p 值大于 0.05 表示不能拒绝正态假设。")
    print()

    gap_skew = max(abs(r["偏度"] - r["手算偏度"]) for r in rows)
    gap_kurt = max(abs(r["峰度"] - r["手算峰度"]) for r in rows)
    gap = max(gap_skew, gap_kurt)
    print("  【自检：手算 vs scipy 库函数】")
    print(f"    最大差异 = {gap:.2e}")
    if gap < 1e-10:
        print("    >>> 差异接近 0，说明公式实现无误 <<<")
    else:
        print("    !!! 差异过大，请检查实现 !!!")
    print()


if __name__ == "__main__":
    main()
