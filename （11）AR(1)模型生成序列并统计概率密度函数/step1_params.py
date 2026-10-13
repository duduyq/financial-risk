# -*- coding: utf-8 -*-
"""
第 11 题 · 步骤 1：参数设置
运行方式：python step1_params.py

这一步只做一件事：设定参数，并验证"两组长期平均方差相等"这个关键设计。
"""

from q11_core import (C, PHI, SIGMA, T, BURN, SEED,
                      ALPHA, BETA, OMEGA)


def main():
    print("=" * 62)
    print("步骤 1 / 6    参数设置")
    print("=" * 62)
    print()
    print("  【均值方程参数（两组共用）】")
    print(f"    c     = {C}        常数项")
    print(f"    phi   = {PHI}           自回归系数，必须小于 1")
    print(f"    T     = {T}         样本量")
    print(f"    burn  = {BURN}        预烧期")
    print(f"    seed  = {SEED}    随机种子")
    print()
    print("  【情况 1：方差恒定】")
    print(f"    sigma = {SIGMA}         波动率，方差 = {SIGMA**2:.3e}")
    print()
    print("  【情况 2：GARCH(1,1) 时变方差】")
    print(f"    alpha = {ALPHA}          ARCH 项系数")
    print(f"    beta  = {BETA}          GARCH 项系数")
    print(f"    alpha + beta = {ALPHA + BETA}  < 1，满足平稳条件")
    print()
    print("  【关键设计】")
    print(f"    omega = sigma^2 * (1 - alpha - beta)")
    print(f"          = {SIGMA**2:.3e} * {1 - ALPHA - BETA:.2f}")
    print(f"          = {OMEGA:.3e}")
    print()
    print("=" * 62)
    print("  验证：两组的长期平均方差是否相等")
    print("=" * 62)
    var1 = SIGMA ** 2
    var2 = OMEGA / (1 - ALPHA - BETA)
    print(f"    情况1 的方差          = {var1:.3e}")
    print(f"    情况2 的无条件方差    = omega/(1-alpha-beta) = {var2:.3e}")
    print()
    if abs(var1 - var2) < 1e-12:
        print("    >>> 两者完全相等 <<<")
        print()
        print("    因此两组唯一的区别只剩：方差是否随时间变化。")
    else:
        print("    !!! 两者不相等，请检查 omega 的取法 !!!")
    print()


if __name__ == "__main__":
    main()
