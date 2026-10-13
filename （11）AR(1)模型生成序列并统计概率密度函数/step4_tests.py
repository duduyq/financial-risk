# -*- coding: utf-8 -*-
"""
第 11 题 · 步骤 4：AR(1) 参数估计与残差检验
运行方式：python step4_tests.py

这一步把生成的模拟数据当成真实数据，做三项验证：
  1) 用回归把 phi 估回来，确认数据确实是 AR(1) 生成的
  2) 检验残差的自相关，确认均值方程设定正确
  3) 检验"残差平方"的自相关，找波动聚集的统计证据
"""

import os
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

OUTDIR = "results"
PHI_TRUE = 0.6


def ar1_check(x, name):
    y = x[1:]
    X = sm.add_constant(x[:-1])
    res = sm.OLS(y, X).fit()

    lb_resid = acorr_ljungbox(res.resid, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(res.resid ** 2, lags=[10], return_df=True)

    return {
        "序列": name,
        "phi": res.params[1],
        "标准误": res.bse[1],
        "t值": res.tvalues[1],
        "残差LB_p": float(lb_resid["lb_pvalue"].iloc[0]),
        "残差平方LB_p": float(lb_sq["lb_pvalue"].iloc[0]),
    }


def main():
    print("=" * 62)
    print("步骤 4 / 6    AR(1) 参数估计与残差检验")
    print("=" * 62)
    print()

    r1 = np.load(os.path.join(OUTDIR, "r1.npy"))
    r2 = np.load(os.path.join(OUTDIR, "r2.npy"))

    rows = [ar1_check(r1, "情况1"), ar1_check(r2, "情况2")]

    print("  " + "-" * 70)
    print("  {:<8}{:>10}{:>10}{:>10}{:>14}{:>16}".format(
        "序列", "phi", "标准误", "t值", "残差LB_p", "残差平方LB_p"))
    print("  " + "-" * 70)
    for r in rows:
        print("  {:<8}{:>10.6f}{:>10.6f}{:>10.2f}{:>14.4f}{:>16.3e}".format(
            r["序列"], r["phi"], r["标准误"], r["t值"],
            r["残差LB_p"], r["残差平方LB_p"]))
    print("  " + "-" * 70)
    print()
    print(f"  phi 的设定值 = {PHI_TRUE}")
    print()

    print("  【验证 1】phi 估计值是否接近设定值")
    for r in rows:
        ok = abs(r["phi"] - PHI_TRUE) < 0.05
        print(f"    {r['序列']}：估出 {r['phi']:.4f}，"
              f"{'接近' if ok else '偏离'}设定值  ->  "
              f"{'数据确实是 AR(1)' if ok else '需检查'}")
    print()

    print("  【验证 2】残差还有没有自相关（应无）")
    for r in rows:
        ok = r["残差LB_p"] > 0.05
        print(f"    {r['序列']}：p = {r['残差LB_p']:.4f}  ->  "
              f"{'无自相关，均值方程设定正确' if ok else '仍有自相关'}")
    print()

    print("  【验证 3】残差平方有没有自相关（这是关键证据）")
    for r in rows:
        has = r["残差平方LB_p"] < 0.05
        print(f"    {r['序列']}：p = {r['残差平方LB_p']:.3e}  ->  "
              f"{'有波动聚集' if has else '无波动聚集'}")
    print()
    print("  >>> 结论：只有情况2 存在显著的波动聚集 <<<")
    print()


if __name__ == "__main__":
    main()
