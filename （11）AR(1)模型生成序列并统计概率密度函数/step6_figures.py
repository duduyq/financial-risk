# -*- coding: utf-8 -*-
"""
第 11 题 · 步骤 6：生成五张结果图
运行方式：python step6_figures.py

这一步完成题目要求的第三项（概率密度函数），并额外生成四张辅助图。
图片保存在 results 文件夹，运行结束后会自动打开该文件夹。
"""

import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")            # 不弹窗，直接存文件
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.stattools import acf

OUTDIR = "results"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def set_ascii_ticks(ax):
    """坐标轴负号统一成 ASCII，避免字体缺字形警告"""
    fmt = mticker.FuncFormatter(lambda v, pos: ("%g" % v).replace("\u2212", "-"))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def fig_series(r1, r2):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(r1, lw=0.6, color="#2c6fd6")
    axes[0].set_title("情况1 恒定方差（波动幅度均匀）", fontsize=12)
    axes[1].plot(r2, lw=0.6, color="#d63c32")
    axes[1].set_title("情况2 随机波动率（大波动成串）", fontsize=12)
    axes[1].set_xlabel("期数 t")
    for ax in axes:
        ax.set_ylabel("收益率")
        ax.grid(alpha=0.3)
        set_ascii_ticks(ax)
    fig.tight_layout()
    return fig, list(axes)


def fig_density(r1, r2):
    s1 = r1.std(ddof=1)
    grid = np.linspace(-0.15, 0.15, 600)
    kde1 = stats.gaussian_kde(r1)
    kde2 = stats.gaussian_kde(r2)
    norm_pdf = stats.norm.pdf(grid, 0, s1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(grid, kde1(grid), lw=2, color="#2c6fd6",
             label="情况1（峰度=%.2f）" % (stats.kurtosis(r1) + 3))
    ax1.plot(grid, kde2(grid), lw=2, color="#d63c32",
             label="情况2（峰度=%.2f）" % (stats.kurtosis(r2) + 3))
    ax1.plot(grid, norm_pdf, lw=1.5, ls="--", color="gray", label="正态分布基准")
    ax1.set_title("概率密度函数对比", fontsize=12)
    ax1.set_xlabel("收益率")
    ax1.set_ylabel("密度")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    set_ascii_ticks(ax1)

    ax2.semilogy(grid, kde1(grid), lw=2, color="#2c6fd6", label="情况1")
    ax2.semilogy(grid, kde2(grid), lw=2, color="#d63c32", label="情况2")
    ax2.semilogy(grid, norm_pdf, lw=1.5, ls="--", color="gray", label="正态分布")
    ax2.set_title("尾部放大（纵轴取对数）", fontsize=12)
    ax2.set_xlabel("收益率")
    ax2.set_ylabel("密度（对数刻度）")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")
    set_ascii_ticks(ax2)

    fig.tight_layout()
    return fig, [ax1, ax2]


def fig_qq(r1, r2):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, x, name, col in zip(axes, [r1, r2],
                                ["情况1 恒定方差", "情况2 随机波动率"],
                                ["#2c6fd6", "#d63c32"]):
        stats.probplot(x, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(col)
        ax.get_lines()[0].set_markeredgecolor(col)
        ax.get_lines()[0].set_markersize(2.5)
        ax.get_lines()[1].set_color("gray")
        ax.set_title(name, fontsize=11)
        # probplot 默认给的是英文轴标签，这里换成中文，和其他图保持一致
        ax.set_xlabel("理论分位数（正态）")
        ax.set_ylabel("样本分位数")
        ax.grid(alpha=0.3)
        set_ascii_ticks(ax)
    fig.tight_layout()
    return fig, list(axes)


def fig_acf(r1, r2):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    lags = 30
    for j, (x, name) in enumerate([(r1, "情况1"), (r2, "情况2")]):
        axes[0, j].stem(range(lags + 1), acf(x, nlags=lags, fft=True), basefmt=" ")
        axes[0, j].set_title(name + "：收益率 ACF", fontsize=11)
        axes[1, j].stem(range(lags + 1), acf(x ** 2, nlags=lags, fft=True), basefmt=" ")
        axes[1, j].set_title(name + "：收益率平方 ACF（波动聚集）", fontsize=11)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_ylim(-0.2, 1.0)
            ax.set_xlabel("滞后阶数")
            ax.grid(alpha=0.3)
            set_ascii_ticks(ax)
    fig.tight_layout()
    return fig, list(axes.ravel())


def fig_kurt_bar(r1, r2):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    labels = ["情况1", "情况2", "正态"]
    values = [stats.kurtosis(r1) + 3, stats.kurtosis(r2) + 3, 3.0]
    bars = ax.bar(labels, values, color=["#2c6fd6", "#d63c32", "gray"], width=0.5)
    ax.axhline(3, color="black", ls="--", lw=1)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, "%.2f" % v,
                ha="center", fontsize=11)
    ax.set_ylabel("原始峰度（正态=3）")
    ax.set_title("峰度对比", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    # 这里不能调用 set_ascii_ticks(ax)：它会给 x 轴装上数值格式化器，
    # 把「情况1/情况2/正态」这些文字标签覆盖成 0/1/2。只处理 y 轴即可。
    fmt = mticker.FuncFormatter(lambda v, pos: ("%g" % v).replace("\u2212", "-"))
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    fig.tight_layout()
    return fig, [ax]


def main():
    print("=" * 62)
    print("步骤 6 / 6    生成五张结果图")
    print("=" * 62)
    print()

    r1 = np.load(os.path.join(OUTDIR, "r1.npy"))
    r2 = np.load(os.path.join(OUTDIR, "r2.npy"))

    jobs = [
        (fig_series, (r1, r2), "fig1_时间序列.png"),
        (fig_density, (r1, r2), "fig2_概率密度.png"),
        (fig_qq, (r1, r2), "fig3_QQ图.png"),
        (fig_acf, (r1, r2), "fig4_自相关.png"),
        (fig_kurt_bar, (r1, r2), "fig5_峰度对比.png"),
    ]

    for func, args, fname in jobs:
        print(f"  正在生成 {fname} ...")
        fig, axes = func(*args)
        fig.savefig(os.path.join(OUTDIR, fname))
        plt.close(fig)

    print()
    print("  五张图已保存到 results 文件夹：")
    for _, _, fname in jobs:
        print("    " + fname)
    print()

    # 统计极端观测个数（比看对数坐标图更直接）
    print("  【补充数据：极端观测个数】")
    for thr in (0.08, 0.10):
        c1 = int((np.abs(r1) > thr).sum())
        c2 = int((np.abs(r2) > thr).sum())
        print(f"    涨跌幅超过 ±{thr*100:.0f}%：情况1 = {c1} 个，情况2 = {c2} 个")
    print()

    # 自动打开文件夹
    try:
        os.startfile(os.path.abspath(OUTDIR))
        print("  已自动打开 results 文件夹。")
    except Exception:
        print(f"  请手动打开 {os.path.abspath(OUTDIR)} 文件夹查看图片。")
    print()


if __name__ == "__main__":
    main()
