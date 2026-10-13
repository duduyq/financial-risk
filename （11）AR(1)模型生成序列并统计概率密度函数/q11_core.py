# -*- coding: utf-8 -*-
"""
第 11 题 · 核心函数模块
本文件只放"模拟数据"和"手算统计量"两个功能，其他步骤脚本会调用它。
"""

import numpy as np

# ============================================================
# 全局参数（唯一的一处定义，其他脚本从这里取）
# ============================================================
C = 0.0005          # AR(1) 常数项
PHI = 0.6           # 自回归系数，必须小于 1
SIGMA = 0.02        # 情况1 的波动率（方差 = 0.0004）
T = 5000            # 样本量
BURN = 1000         # 预烧期
SEED = 20260913     # 随机种子，保证结果可复现

ALPHA = 0.08        # GARCH 的 ARCH 项系数
BETA = 0.90         # GARCH 的 GARCH 项系数
OMEGA = SIGMA ** 2 * (1 - ALPHA - BETA)   # 使两组长期平均方差相等


def simulate_case1(c, phi, sigma, T, burn, seed):
    """情况1：误差项方差恒定。
       r[t] = c + phi*r[t-1] + sigma*z[t]
    """
    rng = np.random.default_rng(seed)
    n = T + burn
    z = rng.standard_normal(n)
    r = np.zeros(n)
    r[0] = c / (1 - phi)
    for t in range(1, n):
        r[t] = c + phi * r[t - 1] + sigma * z[t]
    return r[burn:]


def simulate_case2(c, phi, omega, alpha, beta, T, burn, seed):
    """情况2：误差项方差随时间变化（GARCH(1,1)）。
       sigma2[t] = omega + alpha*u[t-1]^2 + beta*sigma2[t-1]
       u[t]      = sqrt(sigma2[t]) * z[t]
       r[t]      = c + phi*r[t-1] + u[t]
    """
    rng = np.random.default_rng(seed)
    n = T + burn
    z = rng.standard_normal(n)
    r = np.zeros(n)
    u = np.zeros(n)
    sigma2 = np.zeros(n)
    r[0] = c / (1 - phi)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * u[t - 1] ** 2 + beta * sigma2[t - 1]
        u[t] = np.sqrt(sigma2[t]) * z[t]
        r[t] = c + phi * r[t - 1] + u[t]
    return r[burn:], sigma2[burn:]


def hand_moments(x):
    """不调用库函数，按定义手算偏度和峰度（用来和 scipy 对拍）"""
    m = x.mean()
    m2 = np.mean((x - m) ** 2)
    m3 = np.mean((x - m) ** 3)
    m4 = np.mean((x - m) ** 4)
    sd = np.sqrt(m2)
    return m3 / sd ** 3, m4 / sd ** 4
