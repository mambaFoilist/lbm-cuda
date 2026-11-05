"""make_figures.py — regenerate every figure and table from the CSVs in runs/.

    python scripts/make_figures.py          (from the repo root; needs numpy + matplotlib)

Inputs (each figure is skipped with a note if its CSV is missing):
    runs/poiseuille_tau.csv, poiseuille_profiles.csv, poiseuille_convergence.csv  <- sweep_poiseuille_tau.exe
    runs/strouhal.csv, strouhal_Re*_<precision>.csv                              <- run_strouhal.exe
    runs/schafer_turek.csv                                                        <- run_schafer_turek.exe
    runs/performance.csv                                                          <- check_gpu.exe (copied by hand)
Outputs: figures/*.png (slide-sized fonts) and figures/tables.md (numbers to paste into slides).
"""

import csv
import glob
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS, FIGS = "runs", "figures"
CD_REF, CL_REF = 5.57953523384, 0.010618948146
MAGIC_TAU = 0.5 + math.sqrt(3 / 16)

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16, "legend.fontsize": 13,
    "lines.linewidth": 2, "lines.markersize": 8, "axes.grid": True, "grid.alpha": 0.3,
    "figure.figsize": (7.5, 5.2), "savefig.dpi": 200, "savefig.bbox": "tight",
})
C1, C2, C3, C4 = "#1f77b4", "#d62728", "#2ca02c", "#9467bd"
tables = []


# ---------------------------------------------------------------- helpers
def read_csv(name):
    """rows of runs/<name> as dicts of floats (strings where not numeric); '#' lines are metadata."""
    path = os.path.join(RUNS, name)
    if not os.path.exists(path):
        print(f"  (skipped: {path} not found)")
        return None, []
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    meta = [l[1:].strip() for l in lines if l.startswith("#")]
    body = [l for l in lines if l.strip() and not l.startswith("#")]
    header, rows = None, []
    for r in csv.reader(body):
        if header is None or r == header:  # appended files repeat nothing, but be safe
            header = r
            continue
        if len(r) != len(header):  # half-written last line of a run still in progress
            continue
        d = {}
        for k, v in zip(header, r):
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        rows.append(d)
    return meta, rows


def col(rows, k):
    return np.array([r[k] for r in rows], dtype=float)


def last_per_key(rows, keys):
    """appended CSVs may hold reruns: keep the last row for each key"""
    out = {}
    for r in rows:
        out[tuple(r[k] for k in keys)] = r
    return list(out.values())


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print(f"  -> {FIGS}/{name}")


def guide(ax, x0, y0, xs, slope, label, color="gray"):
    xs = np.array(xs, dtype=float)
    ax.plot(xs, y0 * (xs / x0) ** (-slope), "--", color=color, lw=1.5, label=label)


# ---------------------------------------------------------------- Poiseuille
def poiseuille():
    print("Poiseuille")
    _, prof = read_csv("poiseuille_profiles.csv")
    if prof:
        fig, ax = plt.subplots()
        H = int(prof[0]["H"])
        yy = np.linspace(0.5, H + 0.5, 200)
        # exact profile (nominal walls) is the same shape for every tau once scaled by its max
        for tau, c, mk in [(MAGIC_TAU, C1, "o"), (2.0, C2, "s"), (0.55, C3, "^")]:
            rs = [r for r in prof if abs(r["tau"] - tau) < 1e-3 and 1 <= r["y"] <= H]
            if not rs:
                continue
            y, us, ue = col(rs, "y"), col(rs, "u_sim"), col(rs, "u_exact")
            umax = ue.max()
            ax.plot(us / umax, y, mk, color=c, mfc="none", label=f"LBM, τ = {tau:.3g}")
        ue_line = (yy - 0.5) * (H + 0.5 - yy)
        ax.plot(ue_line / ue_line.max(), yy, "k-", lw=1.5, label="analytic (walls at y = ½, H + ½)")
        ax.axhline(0.5, color="k", lw=0.8); ax.axhline(H + 0.5, color="k", lw=0.8)
        ax.set_xlabel("u / u_max(analytic)"); ax.set_ylabel("y (lattice nodes)")
        ax.set_title(f"Poiseuille channel, H = {H}")
        ax.legend(loc="center left", fontsize=11)
        save(fig, "fig1_poiseuille_profile.png")

    _, rows = read_csv("poiseuille_tau.csv")
    if rows:
        # viscosity
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
        tt = np.linspace(0.52, 2.05, 100)
        a1.plot(tt, (tt - 0.5) / 3, "k-", lw=1.5, label="ν = (τ − ½)/3")
        for H, c, mk in [(16, C1, "o"), (32, C2, "s")]:
            rs = [r for r in rows if r["H"] == H]
            a1.plot(col(rs, "tau"), col(rs, "nu_meas"), mk, color=c, mfc="none", label=f"measured, H = {H}")
            a2.semilogy(col(rs, "tau"), np.abs(col(rs, "nu_rel_err")) + 1e-17, mk + "-", color=c, mfc="none",
                        label=f"H = {H}")
        a1.set_xlabel("τ"); a1.set_ylabel("ν (lattice units)"); a1.legend(); a1.set_title("Viscosity from the fitted curvature")
        a2.set_xlabel("τ"); a2.set_ylabel("|ν_meas / ν_theory − 1|"); a2.set_title("Relative error (round-off)")
        a2.set_ylim(1e-16, 1e-9); a2.legend()
        fig.tight_layout()
        save(fig, "fig2_viscosity_vs_tau.png")

        # wall location: H * shift collapses both H onto (16 Λ - 3)/12
        fig, ax = plt.subplots()
        lam = (tt - 0.5) ** 2
        ax.plot(tt, (16 * lam - 3) / 12, "k-", lw=1.5, label="theory: H·δ ≈ (16Λ − 3)/12")
        for H, c, mk in [(16, C1, "o"), (32, C2, "s")]:
            rs = [r for r in rows if r["H"] == H]
            ax.plot(col(rs, "tau"), H * col(rs, "wall_shift_meas"), mk, color=c, mfc="none", label=f"measured, H = {H}")
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(MAGIC_TAU, color="gray", ls=":", lw=1.5)
        ax.annotate("τ = ½ + √(3/16)\nwall exactly halfway", (MAGIC_TAU, 0), xytext=(1.1, -0.2), fontsize=12,
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_xlabel("τ"); ax.set_ylabel("H · δ   (δ = wall shift per wall, cells)")
        ax.set_title("Effective wall location depends on τ (Λ = (τ − ½)²)")
        ax.legend(loc="upper left")
        save(fig, "fig3_wall_location_vs_tau.png")

        t = ["## Poiseuille τ sweep (H = 32)", "",
             "| τ | ν_meas/ν_th − 1 | H_eff (meas) | H_eff (theory) | wall shift δ (cells) | slip u_wall/u_c |",
             "|---|---|---|---|---|---|"]
        for r in rows:
            if r["H"] == 32:
                t.append(f"| {r['tau']:.3f} | {r['nu_rel_err']:+.1e} | {r['H_eff_meas']:.5f} | {r['H_eff_theory']:.5f} "
                         f"| {r['wall_shift_meas']:+.5f} | {r['slip_meas']:+.2e} |")
        tables.append("\n".join(t))

    _, rows = read_csv("poiseuille_convergence.csv")
    if rows:
        fig, ax = plt.subplots()
        for tau, c, mk in [(0.8, C1, "o"), (1.0, C2, "s")]:
            rs = [r for r in rows if abs(r["tau"] - tau) < 1e-6]
            H, e = col(rs, "H"), col(rs, "l2_error")
            p = np.polyfit(np.log(H), np.log(e), 1)[0]
            ax.loglog(H, e, mk + "-", color=c, mfc="none", label=f"τ = {tau}  (slope {p:.2f})")
        guide(ax, 4, 6e-2, [4, 64], 2, "slope −2")
        ax.set_xlabel("H (nodes across channel)"); ax.set_ylabel("relative L2 error")
        ax.set_title("Poiseuille: second-order convergence")
        ax.legend()
        save(fig, "fig4_poiseuille_convergence.png")


# ---------------------------------------------------------------- Strouhal
def williamson(Re):
    return -3.3265 / Re + 0.1816 + 1.600e-4 * Re


def roshko(Re):
    return 0.212 * (1 - 21.2 / Re)


def st_fft(step, cl, D, U):
    """St from the second half of C_L(t): Hann window, zero-padded FFT, parabolic peak (on log |X|)."""
    n = len(step)
    s, x = step[n // 2:], cl[n // 2:]
    x = x - x.mean()
    dt = s[1] - s[0]
    m = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(m), n=8 * m))
    k = int(np.argmax(X[1:])) + 1
    a, b, c = np.log(X[k - 1]), np.log(X[k]), np.log(X[k + 1])
    kp = k + 0.5 * (a - c) / (a - 2 * b + c)
    f = kp / (8 * m * dt)  # cycles per time step
    return f * D / U


def strouhal():
    print("Strouhal")
    files = sorted(glob.glob(os.path.join(RUNS, "strouhal_Re*_*.csv")))
    if not files:
        print("  (skipped: no runs/strouhal_Re*.csv)")
        return
    _, summ = read_csv("strouhal.csv")
    summ = {(r["Re"], r["precision"]): r for r in (summ or [])}
    pts = {}
    series = {}
    for fn in files:
        meta, rows = read_csv(os.path.basename(fn))
        kv = dict(re.findall(r"(\w+)=([^\s]+)", " ".join(meta)))
        Re, D, U, prec = float(kv["Re"]), float(kv["D"]), float(kv["U"]), kv["precision"]
        if (Re, prec) not in summ:  # no summary row yet: that run is still going
            print(f"  ({os.path.basename(fn)}: run not finished, skipped)")
            continue
        step, cl, cd = col(rows, "step"), col(rows, "C_L"), col(rows, "C_D")
        if len(step) < 100:
            continue
        stf = st_fft(step, cl, D, U)
        sz = summ.get((Re, prec), {}).get("St_zc", float("nan"))
        pts.setdefault(prec, []).append((Re, stf, sz, cd[len(cd) // 2:].mean(), summ.get((Re, prec), {}).get("saturated", 1)))
        series[(Re, prec)] = (step, cl, cd, D, U)

    fig, ax = plt.subplots()
    rr = np.linspace(48, 155, 200)
    ax.plot(rr, williamson(rr), "k-", lw=1.5, label="Williamson (1988)")
    ax.plot(rr, roshko(rr), "k--", lw=1.5, label="Roshko (1954)")
    t = ["## Strouhal number, D = 40, blockage 5%", "",
         "| Re | precision | St (FFT) | St (zero crossings) | Williamson | error | mean C_D |", "|---|---|---|---|---|---|---|"]
    for prec, c, mk in [("double", C2, "o"), ("float", C1, "s")]:
        if prec not in pts:
            continue
        p = sorted(pts[prec])
        Re = np.array([q[0] for q in p])
        ax.plot(Re, [q[1] for q in p], mk, color=c, mfc="none" if prec == "float" else c,
                label=f"LBM D2Q9, {prec}")
        for q in p:
            flag = "" if q[4] else " (not saturated)"
            t.append(f"| {q[0]:.0f} | {prec} | {q[1]:.4f} | {q[2]:.4f} | {williamson(q[0]):.4f} | "
                     f"{100 * (q[1] / williamson(q[0]) - 1):+.1f}%{flag} | {q[3]:.3f} |")
    ax.set_xlabel("Re = U D / ν"); ax.set_ylabel("St = f D / U")
    ax.set_title("Vortex shedding frequency (D = 40, 5% blockage)")
    ax.legend(loc="lower right")
    save(fig, "fig5_strouhal_vs_re.png")
    tables.append("\n".join(t))

    key = (100.0, "double") if (100.0, "double") in series else next(iter(series))
    step, cl, cd, D, U = series[key]
    fig, ax = plt.subplots(figsize=(10, 4))
    tc = step * U / D  # convective time units
    ax.plot(tc, cl, color=C1, lw=1, label="C_L")
    ax.plot(tc, cd, color=C2, lw=1, label="C_D")
    ax.axvline(tc[len(tc) // 2], color="gray", ls=":", label="analysis window starts")
    ax.set_xlabel("t U / D"); ax.set_ylabel("coefficient")
    ax.set_title(f"Force history, Re = {key[0]:.0f} ({key[1]})")
    ax.legend(loc="lower right", ncol=3)
    save(fig, "fig5b_force_history.png")


# ---------------------------------------------------------------- Schafer-Turek
def schafer_turek():
    print("Schafer-Turek convergence")
    _, rows = read_csv("schafer_turek.csv")
    if not rows:
        return
    rows = [r for r in last_per_key(rows, ["N", "ox", "oy"]) if r["converged"] == 1]
    Ns = sorted({int(r["N"]) for r in rows})
    stats = []
    for N in Ns:
        rs = [r for r in rows if r["N"] == N]
        cd = col(rs, "C_D")
        e = np.abs(cd - CD_REF) / CD_REF
        # C_L only from oy = 0: there the centre sits exactly between two node rows, so the staircase
        # is mirror-symmetric top/bottom. Any oy != 0 makes the staircase itself lopsided, and that
        # asymmetry swamps the tiny physical lift (see the per-offset table below).
        sym = [r for r in rs if r["oy"] == 0]
        cl = col(sym, "C_L").mean() if sym else float("nan")
        stats.append(dict(N=N, n=len(rs), cd=cd.mean(), cd_sd=cd.std(ddof=1) if len(rs) > 1 else 0,
                          e=e.mean(), emin=e.min(), emax=e.max(), cl=cl, all_e=e,
                          per_offset=sorted((r["ox"], r["oy"], r["C_D"], r["C_L"]) for r in rs)))

    fig, ax = plt.subplots()
    N = np.array([s["N"] for s in stats], float)
    e = np.array([s["e"] for s in stats])
    for s in stats:
        ax.plot([s["N"]] * len(s["all_e"]), 100 * s["all_e"], "o", color=C1, alpha=0.35, ms=6)
    ax.errorbar(N, 100 * e, yerr=[100 * (e - [s["emin"] for s in stats]), 100 * (np.array([s["emax"] for s in stats]) - e)],
                fmt="o-", color=C1, capsize=5, ms=9, label="LBM, mean over centre offsets (bars: min–max)")
    if len(N) >= 2:
        p = np.polyfit(np.log(N), np.log(e), 1)[0]
        ax.plot([], [], " ", label=f"observed order (fit): {-p:.2f}")
    guide(ax, N[0], 100 * e[0], [N[0], N[-1]], 1, "slope −1", C2)
    guide(ax, N[0], 100 * e[0], [N[0], N[-1]], 2, "slope −2", C3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.minorticks_off()
    ax.set_xlabel("N = D / Δx (cells per diameter)"); ax.set_ylabel("|C_D error|  (%)")
    ax.set_title("Schäfer–Turek 2D-1, τ = 0.65 fixed: staircase cost")
    ax.legend(loc="lower left", fontsize=11)
    save(fig, "fig6_schafer_turek_convergence.png")

    t = ["## Schäfer–Turek 2D-1 (Re = 20), τ = 0.65", "",
         f"Reference: C_D = {CD_REF:.5f}, C_L = {CL_REF:.5f}", "",
         "| N | offsets | C_D (mean ± sd) | C_D error | C_L (symmetric placement) | C_L error | order (vs previous N) |",
         "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(stats):
        order = ""
        if i > 0:
            order = f"{math.log(stats[i - 1]['e'] / s['e']) / math.log(s['N'] / stats[i - 1]['N']):.2f}"
        t.append(f"| {s['N']} | {s['n']} | {s['cd']:.4f} ± {s['cd_sd']:.4f} | {100 * (s['cd'] / CD_REF - 1):+.2f}% | "
                 f"{s['cl']:.5f} | {100 * (s['cl'] / CL_REF - 1):+.1f}% | {order} |")
    t += ["", "Per offset (ox, oy in cells): C_L reacts to the staircase's top/bottom asymmetry far more than C_D", "",
          "| N | (ox, oy) | C_D | C_D error | C_L | C_L / C_L,ref |", "|---|---|---|---|---|---|"]
    for s in stats:
        for ox, oy, cd_, cl_ in s["per_offset"]:
            t.append(f"| {s['N']} | ({ox:.2f}, {oy:+.2f}) | {cd_:.4f} | {100 * (cd_ / CD_REF - 1):+.2f}% | "
                     f"{cl_:+.5f} | {cl_ / CL_REF:+.1f} |")
    if len(stats) >= 3:
        c1, c2, c3 = (s["cd"] for s in stats[-3:])
        r = stats[-1]["N"] / stats[-2]["N"]
        if (c1 - c2) * (c2 - c3) > 0:
            p = math.log((c1 - c2) / (c2 - c3)) / math.log(r)
            cinf = c3 + (c3 - c2) / (r ** p - 1)
            t += ["", f"Richardson (last three N): observed order p = {p:.2f}, extrapolated C_D = {cinf:.4f} "
                      f"({100 * (cinf / CD_REF - 1):+.2f}% vs reference)"]
    tables.append("\n".join(t))


# ---------------------------------------------------------------- performance
def performance():
    print("Performance")
    _, rows = read_csv("performance.csv")
    if not rows:
        return
    peak = 192.0  # GB/s, RTX 4050 Laptop (96-bit GDDR6, 16 Gbps)
    fig, ax = plt.subplots()
    n = col(rows, "size")
    ax.loglog(n ** 2, col(rows, "gpu_float"), "o-", color=C1, label="GPU float")
    ax.loglog(n ** 2, col(rows, "gpu_double"), "s-", color=C2, label="GPU double")
    cpu = [(r["size"], r["cpu_double"]) for r in rows if r["cpu_double"] != ""]
    ax.loglog([c[0] ** 2 for c in cpu], [c[1] for c in cpu], "^-", color=C3, label="CPU double (OpenMP)")
    for T, b, c in [("float", 4, C1), ("double", 8, C2)]:
        ax.axhline(peak * 1e9 / (18 * b) / 1e6, color=c, ls=":", lw=1.5)
    ax.text(7e4, peak * 1e9 / 72 / 1e6 * 1.08, "DRAM peak, float", color=C1, fontsize=11)
    ax.text(7e4, peak * 1e9 / 144 / 1e6 * 1.08, "DRAM peak, double", color=C2, fontsize=11)
    ax.set_xlabel("lattice nodes"); ax.set_ylabel("MLUPS")
    ax.set_title("Throughput (RTX 4050 Laptop vs CPU)")
    ax.legend(loc="lower right")
    save(fig, "fig7_performance.png")

    t = ["## Performance (empty periodic box)", "", f"Peak DRAM bandwidth {peak:.0f} GB/s", "",
         "| size | GPU float MLUPS (GB/s, % peak) | GPU double MLUPS (GB/s, % peak) | CPU double MLUPS | GPU double / CPU |",
         "|---|---|---|---|---|"]
    for r in rows:
        gf, gd = r["gpu_float"] * 72e-3, r["gpu_double"] * 144e-3
        cpu = r["cpu_double"]
        ratio = "" if cpu == "" else f"{r['gpu_double'] / cpu:.0f}×"
        t.append(f"| {int(r['size'])}² | {r['gpu_float']:.0f} ({gf:.0f}, {100 * gf / peak:.0f}%) | "
                 f"{r['gpu_double']:.0f} ({gd:.0f}, {100 * gd / peak:.0f}%) | {cpu if cpu != '' else '–'} | "
                 f"{ratio} |")
    tables.append("\n".join(t))


if __name__ == "__main__":
    poiseuille()
    strouhal()
    schafer_turek()
    performance()
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, "tables.md"), "w", encoding="utf-8") as f:
        f.write("# Tables (generated by scripts/make_figures.py)\n\n" + "\n\n".join(tables) + "\n")
    print(f"  -> {FIGS}/tables.md")
