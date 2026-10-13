# -*- coding: utf-8 -*-
"""
第 11 题 · 步骤 2：生成两组模拟收益率序列
运行方式：python step2_generate.py

这一步完成题目要求的第一项：用 AR(1) 生成两组收益率序列。
生成的数据存到 results 文件夹，供后面几步使用。
"""

import os
import numpy as np
from q11_core import (C, PHI, SIGMA, T, BURN, SEED,
                      ALPHA, BETA, OMEGA,
                      simulate_case1, simulate_case2)

OUTDIR = "results"


def main():
    print("=" * 62)
    print("步骤 2 / 6    生成两组模拟收益率序列")
    print("=" * 62)
    print()

    os.makedirs(OUTDIR, exist_ok=True)

    print("  正在生成情况1（恒定方差）...")
    r1 = simulate_case1(C, PHI, SIGMA, T, BURN, SEED)
    print("  正在生成情况2（GARCH 时变方差）...")
    r2, sigma2_path = simulate_case2(C, PHI, OMEGA, ALPHA, BETA, T, BURN, SEED + 1)
    print()

    print("  【序列长度】")
    print(f"    情况1 = {r1.size}    情况2 = {r2.size}")
    print()

    print("  【情况1 的前 5 个观测】")
    print("   ", np.round(r1[:5], 5))
    print()

    print("  【自检：两组波动水平是否一致】")
    sd1 = r1.std(ddof=1)
    sd2 = r2.std(ddof=1)
    diff = abs(sd1 - sd2) / sd1
    print(f"    情况1 样本标准差 = {sd1:.5f}")
    print(f"    情况2 样本标准差 = {sd2:.5f}")
    print(f"    相对差异         = {diff:.2%}")
    print()
    print("    说明：理论上两组无条件方差相等，样本差异属于抽样波动。")
    print()

    np.save(os.path.join(OUTDIR, "r1.npy"), r1)
    np.save(os.path.join(OUTDIR, "r2.npy"), r2)
    np.save(os.path.join(OUTDIR, "sigma2_path.npy"), sigma2_path)
    print(f"  数据已保存到 {OUTDIR} 文件夹（r1.npy, r2.npy, sigma2_path.npy）")
    print()


if __name__ == "__main__":
    main()
