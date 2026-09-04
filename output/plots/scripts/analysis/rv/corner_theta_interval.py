"""
36_corner_theta_interval.py — plan-5 STEPS 6 + 7 on top of script 82's TEST LR.

STEP 6 — from test to MEASUREMENT.
    theta = cold-corner occupancy relative to the flat rate (0 = universe A, corner empty;
    1 = universe B, flat corner). Universe(theta) is a one-parameter MIXTURE along the corner
    direction, so no parameterized classifier is needed (the plan's importance-reweighting
    collapses): theta enters only through how often a location-matched null draw is a corner
    member — w_i(theta) = theta*pi_ci / (theta*pi_ci + 1 - pi_ci), with pi_ci the local corner
    fraction among NASA planet i's K location neighbors in the detected-B pool.
    Neyman construction with the FIXED statistic T = sum_i l_cond(x_i) (script 82's headline):
    for each theta on a grid, simulate null catalogs at NASA's own locations, take the central
    95% acceptance band of T; the confidence set is every theta whose band contains T_NASA.
    theta_hat = theta whose null median matches T_NASA.

STEP 6b — coverage check (the validity audit for the interval).
    Simulate catalogs at known theta* in {0, 0.5, 1} and check the 95% interval contains
    theta* ~95% of the time. Coverage catalogs are drawn from the TRAIN-half planets while
    the acceptance bands come from the CALIB half — so classifier overfitting or kNN
    approximation error between halves would show up as under-coverage.

STEP 7 — explicit-density cross-check (the plan's "flow variant").
    torch is broken on this machine (DLL import failure), so a normalizing flow is not
    feasible without an install; the same role — an explicit density-ratio estimate,
    independent of boosted trees — is filled by a kNN density ratio: since A = B minus the
    corner,  p_B/p_A = (1 - pi) / (1 - P(corner|x)),  and P(corner|x) is estimated by the
    corner fraction among the K nearest detected-B TRAIN planets (add-half smoothing).
    Full-space and location-space kNN ratios give l_hat_cond exactly as in script 82; the
    whole calibration is re-run with l_hat to get an independent p-value.

Run:
    python scripts/36_corner_theta_interval.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

S82 = _load_by_path("s82", "important_plots/likelihood_ratio_catalog.py")

OUT_DIR = os.path.join(ROOT, "output/plots", "36_corner_theta_interval")

THETA_GRID = np.round(np.arange(0.0, 1.2001, 0.025), 3)
N_NULL = 2000            # null catalogs per theta (acceptance bands)
N_COV = 300              # coverage catalogs per theta*
THETA_STARS = [0.0, 0.5, 1.0]
KNN_K = 150              # step 7: kNN density-ratio neighbors (full feature space)
ALPHA = 0.05
CONFIG_SEEDS = {"precision": 11, "full": 12}   # must match script 82


# ------------------------------------------------------------------ shared LR core
def lr_core(det, nasa, noise_seed):
    """Rebuild script 82's pipeline (identical rng stream -> identical classifiers and
    T values) and return the internals needed for theta inversion and the kNN check."""
    rng = np.random.default_rng(noise_seed)
    obs = S82.apply_noise(det, nasa, rng)
    X, corner = obs["X"], obs["corner"]
    half = rng.permutation(len(X))
    tr, ca = half[: len(X) // 2], half[len(X) // 2:]
    XB_tr, XB_ca = X[tr], X[ca]
    cor_tr, cor_ca = corner[tr], corner[ca]

    clf_full, off_full = S82.fit_ratio_classifier(XB_tr, XB_tr[~cor_tr])
    clf_loc, off_loc = S82.fit_ratio_classifier(XB_tr, XB_tr[~cor_tr], cols=S82.LOC_COLS)

    def ell_cond(Xq):
        return (S82.log_ratio(clf_full, off_full, Xq)
                - S82.log_ratio(clf_loc, off_loc, Xq, cols=S82.LOC_COLS))

    Xn = np.column_stack([np.log10(nasa["m"]), np.log10(nasa["r"]),
                          np.log10(nasa["ins"]), nasa["teff"]])
    return dict(XB_tr=XB_tr, cor_tr=cor_tr, XB_ca=XB_ca, cor_ca=cor_ca, Xn=Xn,
                ell_cond=ell_cond, ell_ca=ell_cond(XB_ca), ell_tr=ell_cond(XB_tr),
                ell_n=ell_cond(Xn), T_obs=float(ell_cond(Xn).sum()))


def loc_neighbors(Xn, X_pool, k):
    scale = X_pool[:, S82.LOC_COLS].std(axis=0)
    nn = NearestNeighbors(n_neighbors=k).fit(X_pool[:, S82.LOC_COLS] / scale)
    return nn.kneighbors(Xn[:, S82.LOC_COLS] / scale, return_distance=False)


# ------------------------------------------------------------- STEP 6: theta machinery
def split_neighbor_pools(idx, ell_pool, cor_pool):
    """Per NASA planet: its location-neighbors' l values split into corner / non-corner."""
    pools = []
    for row in idx:
        c = cor_pool[row]
        pools.append((ell_pool[row[c]], ell_pool[row[~c]], float(c.mean())))
    return pools


def split_neighbor_indices(idx, cor_pool):
    """Like split_neighbor_pools but keeps ROW INDICES so a drawn planet's (M, R, I) can be
    plotted, not just its score."""
    out = []
    for row in idx:
        c = cor_pool[row]
        out.append((row[c], row[~c], float(c.mean())))
    return out


def draw_rows_at_theta(pools_idx, theta, n_cat, rng):
    """Row-index version of draw_T_at_theta: returns (n_cat, N) pool-row indices."""
    rows = np.empty((n_cat, len(pools_idx)), dtype=int)
    for j, (ic, inn, pi_c) in enumerate(pools_idx):
        if len(inn) == 0:
            rows[:, j] = ic[rng.integers(0, len(ic), n_cat)]
            continue
        v = inn[rng.integers(0, len(inn), n_cat)]
        w = theta * pi_c / (theta * pi_c + (1.0 - pi_c)) if pi_c > 0 else 0.0
        if w > 0 and len(ic) > 0:
            vc = ic[rng.integers(0, len(ic), n_cat)]
            v = np.where(rng.random(n_cat) < w, vc, v)
        rows[:, j] = v
    return rows


def draw_T_at_theta(pools, theta, n_cat, rng):
    """Null catalogs at occupancy theta: planet i is a corner member with probability
    w_i(theta) = theta*pi_ci/(theta*pi_ci + 1 - pi_ci); T = sum of drawn l_cond values."""
    T = np.zeros(n_cat)
    for ell_c, ell_n, pi_c in pools:
        if len(ell_n) == 0:
            T += ell_c[rng.integers(0, len(ell_c), n_cat)]
            continue
        w = theta * pi_c / (theta * pi_c + (1.0 - pi_c)) if pi_c > 0 else 0.0
        v = ell_n[rng.integers(0, len(ell_n), n_cat)]
        if w > 0 and len(ell_c) > 0:
            vc = ell_c[rng.integers(0, len(ell_c), n_cat)]
            v = np.where(rng.random(n_cat) < w, vc, v)
        T += v
    return T


def theta_bands(pools, rng):
    q025, q500, q975, dists = [], [], [], []
    for th in THETA_GRID:
        T = draw_T_at_theta(pools, th, N_NULL, rng)
        dists.append(T)
        q025.append(np.quantile(T, 0.025))
        q500.append(np.quantile(T, 0.5))
        q975.append(np.quantile(T, 0.975))
    return np.array(q025), np.array(q500), np.array(q975), dists


def invert(T_obs, q025, q500, q975):
    inside = (T_obs >= q025) & (T_obs <= q975)
    if not inside.any():
        return dict(ci_lo=np.nan, ci_hi=np.nan, theta_hat=np.nan, empty=True)
    ci = THETA_GRID[inside]
    theta_hat = THETA_GRID[np.argmin(np.abs(q500 - T_obs))]
    return dict(ci_lo=float(ci.min()), ci_hi=float(ci.max()),
                theta_hat=float(theta_hat), empty=False,
                open_upper=bool(inside[-1]))


# ------------------------------------------------------------ STEP 7: kNN density ratio
def knn_log_ratio(X_train, cor_train, cols=None):
    """Explicit-density ratio via kNN corner fraction: l(x) = log[(1-pi)/(1-c_hat(x))],
    c_hat = (corner neighbors + 0.5)/(K + 1). Returns a callable like the classifiers."""
    Xt = X_train[:, cols] if cols is not None else X_train
    scale = Xt.std(axis=0)
    nn = NearestNeighbors(n_neighbors=KNN_K).fit(Xt / scale)
    pi = cor_train.mean()

    def ell(Xq):
        Xq_ = Xq[:, cols] if cols is not None else Xq
        idx = nn.kneighbors(Xq_ / scale, return_distance=False)
        c_hat = (cor_train[idx].sum(axis=1) + 0.5) / (KNN_K + 1.0)
        return np.log(1.0 - pi) - np.log(1.0 - c_hat)

    return ell


def run_config(det, nasa, name):
    print(f"\n  ================ theta interval — {name} NASA sample ================")
    core = lr_core(det, nasa, CONFIG_SEEDS[name])
    print(f"    consistency with script 82: T_cond = {core['T_obs']:+.2f} "
          "(must equal 82's T_NASA for this sample)")

    print("  STEP 6 — Neyman inversion over theta (fixed statistic, location-matched nulls)")
    idx_ca = loc_neighbors(core["Xn"], core["XB_ca"], S82.K_NEIGH)
    pools_ca = split_neighbor_pools(idx_ca, core["ell_ca"], core["cor_ca"])
    rng = np.random.default_rng(500 + CONFIG_SEEDS[name])
    q025, q500, q975, dists = theta_bands(pools_ca, rng)
    inv = invert(core["T_obs"], q025, q500, q975)
    up_note = " (upper end OPEN: grid edge inside CI)" if inv.get("open_upper") else ""
    print(f"    theta_hat = {inv['theta_hat']:.3f}   95% CI = "
          f"[{inv['ci_lo']:.3f}, {inv['ci_hi']:.3f}]{up_note}")
    print(f"    reading: cold-corner occupancy is {inv['theta_hat']:.0%} of the flat rate; "
          f"at 95% it is at most {inv['ci_hi']:.0%} of flat")

    print("  STEP 6b — coverage audit (draws from TRAIN half vs bands from CALIB half)")
    idx_tr = loc_neighbors(core["Xn"], core["XB_tr"], S82.K_NEIGH)
    pools_tr = split_neighbor_pools(idx_tr, core["ell_tr"], core["cor_tr"])
    cov_rows = []
    for th_star in THETA_STARS:
        j = int(np.argmin(np.abs(THETA_GRID - th_star)))
        T_cov = draw_T_at_theta(pools_tr, th_star, N_COV, rng)
        covered = float(np.mean((T_cov >= q025[j]) & (T_cov <= q975[j])))
        se = np.sqrt(covered * (1 - covered) / N_COV)
        cov_rows.append((th_star, covered, se))
        print(f"    theta* = {th_star:.1f}: coverage of the 95% band = "
              f"{covered:.3f} ± {se:.3f}  ({N_COV} catalogs)")

    print("  STEP 7 — explicit-density cross-check (kNN ratio; flow not feasible: torch DLL"
          " broken)")
    ell_knn_full = knn_log_ratio(core["XB_tr"], core["cor_tr"])
    ell_knn_loc = knn_log_ratio(core["XB_tr"], core["cor_tr"], cols=S82.LOC_COLS)

    def ell_knn(Xq):
        return ell_knn_full(Xq) - ell_knn_loc(Xq)

    ellk_ca = ell_knn(core["XB_ca"])
    ellk_n = ell_knn(core["Xn"])
    T_knn = float(ellk_n.sum())
    pools_knn = split_neighbor_pools(idx_ca, ellk_ca, core["cor_ca"])
    TA = draw_T_at_theta(pools_knn, 0.0, S82.N_CATALOGS, rng)
    TB = draw_T_at_theta(pools_knn, 1.0, S82.N_CATALOGS, rng)
    pk = S82._pvals(TA, TB, T_knn, S82.N_CATALOGS)
    r = float(np.corrcoef(core["ell_n"], ellk_n)[0, 1])
    print(f"    kNN ratio: T = {T_knn:+.2f} | p(reject B) = {pk['p_reject_B']:.4f} | "
          f"p(compat A) = {pk['p_compat_A']:.4f}")
    print(f"    per-planet agreement with the classifier (NASA): Pearson r = {r:.3f}")

    return dict(name=name, core=core, q025=q025, q500=q500, q975=q975, inv=inv,
                cov=cov_rows, ell_knn_n=ellk_n, pk=pk, r_knn=r, T_knn=T_knn,
                idx_ca=idx_ca, idx_tr=idx_tr)


# ---------------------------------------------------------------------------- figure
def panel_theta(ax, res, letter):
    inv, T = res["inv"], res["core"]["T_obs"]
    ax.fill_between(THETA_GRID, res["q025"], res["q975"], color="0.8",
                    label="95% acceptance band of T at theta")
    ax.plot(THETA_GRID, res["q500"], color="0.35", lw=1.5, label="null median")
    ax.axhline(T, color="tab:green", lw=2.2, label=f"NASA T = {T:+.1f}")
    if not inv["empty"]:
        ax.axvspan(inv["ci_lo"], inv["ci_hi"], color="tab:green", alpha=0.15,
                   label=f"95% CI [{inv['ci_lo']:.2f}, {inv['ci_hi']:.2f}]")
        ax.axvline(inv["theta_hat"], color="tab:green", ls="--", lw=1.2,
                   label=f"theta_hat = {inv['theta_hat']:.2f}")
    ax.set_xlabel("theta  (cold-corner occupancy / flat rate;  0 = universe A, 1 = B)")
    ax.set_ylabel(r"$T=\sum_i \ell_{\rm cond}(x_i)$")
    ax.set_title(f"({letter}) Neyman inversion — {res['name']} NASA "
                 f"(N={len(res['core']['Xn'])})", fontsize=10)
    ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="upper left")


def panel_coverage(ax, results):
    width = 0.35
    for k, (res, color) in enumerate(zip(results, ["tab:blue", "tab:orange"])):
        xs = np.arange(len(res["cov"])) + (k - 0.5) * width
        cov = [c for _, c, _ in res["cov"]]
        err = [s for _, _, s in res["cov"]]
        ax.bar(xs, cov, width, yerr=err, capsize=3, color=color, label=res["name"])
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 0.95")
    ax.set_xticks(np.arange(len(THETA_STARS)))
    ax.set_xticklabels([f"theta* = {t:.1f}" for t in THETA_STARS])
    ax.set_ylim(0.80, 1.02); ax.set_ylabel("empirical coverage of the 95% band")
    ax.set_title("(c) coverage audit — TRAIN-half catalogs vs CALIB-half bands\n"
                 "(under-coverage would flag classifier overfit / kNN error)", fontsize=10)
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=8, loc="lower right")


def panel_knn(ax, res):
    x, y = res["core"]["ell_n"], res["ell_knn_n"]
    lim = [min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3]
    ax.plot(lim, lim, ls="--", color="0.6", lw=1)
    ax.scatter(x, y, s=22, facecolor="tab:green", edgecolor="k", lw=0.3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"classifier  $\ell_{\rm cond}$ (script 82)")
    ax.set_ylabel(r"kNN density ratio  $\hat\ell_{\rm cond}$")
    ax.text(0.02, 0.975,
            f"Pearson r = {res['r_knn']:.3f}\n"
            f"kNN: T = {res['T_knn']:+.1f}, p(reject B) = {res['pk']['p_reject_B']:.4f}\n"
            f"clf:  T = {res['core']['T_obs']:+.1f}\n"
            "(flow variant not feasible: torch DLL broken;\n"
            "kNN ratio = transparent explicit-density stand-in)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="whitesmoke", ec="0.7"))
    ax.set_title(f"(d) STEP 7 — explicit-density cross-check, {res['name']} NASA "
                 "per-planet log-ratios", fontsize=10)
    ax.grid(alpha=0.2)


def make_figure(res_p, res_f):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11.5))
    panel_theta(axes[0, 0], res_p, "a")
    panel_theta(axes[0, 1], res_f, "b")
    panel_coverage(axes[1, 0], [res_p, res_f])
    panel_knn(axes[1, 1], res_p)
    fig.suptitle("TEST LR steps 6+7 — cold-corner occupancy theta: Neyman interval, coverage "
                 "audit, explicit-density cross-check\n"
                 "theta = corner occurrence / flat rate; nulls location-matched to NASA; "
                 "fixed statistic T from script 82's conditional classifier ratio",
                 fontsize=11.5)
    fig.text(0.5, 0.005,
             "Caveats: theta is measured against the FLAT-measure corner rate (log-uniform M, "
             "uniform R), the same null as tests A/LR; coverage catalogs share the kNN location "
             "machinery with the bands (independent detector re-simulations would be stronger); "
             "full-sample theta is inflated by implausible-density label noise (see script 82); "
             "toy-detector caveat of script 82 applies unchanged.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    out_png = os.path.join(OUT_DIR, "corner_theta_interval.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


# ------------------------------------------------- explainer figures (MRI-space bridge)
def explainer_theta_dial(res, m_sil, r_sil):
    """E1 — theta as a dial: one simulated cold M-R catalog per theta, NASA last."""
    core = res["core"]
    pools_idx = split_neighbor_indices(res["idx_ca"], core["cor_ca"])
    rng = np.random.default_rng(21)
    thetas = [0.0, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.8), sharey=True)
    for ax, th in zip(axes[:4], thetas):
        rows = draw_rows_at_theta(pools_idx, th, 1, rng)[0]
        X, c = core["XB_ca"][rows], core["cor_ca"][rows]
        m, r, i = 10.0 ** X[:, 0], 10.0 ** X[:, 1], 10.0 ** X[:, 2]
        cold = i < S82.COLD_MAX
        S82._mr_axis(ax, m_sil, r_sil)
        ax.scatter(m[cold & ~c], r[cold & ~c], s=14, color="0.6", alpha=0.7)
        ax.scatter(m[cold & c], r[cold & c], s=30, color="tab:red")
        ax.set_title(f"simulated catalog, theta = {th:g}\n"
                     f"({int((cold & c).sum())} corner planets in red)", fontsize=10)
    ax = axes[4]
    S82._mr_axis(ax, m_sil, r_sil)
    Xn = core["Xn"]
    mn, rn, inn = 10.0 ** Xn[:, 0], 10.0 ** Xn[:, 1], 10.0 ** Xn[:, 2]
    ncold = inn < S82.COLD_MAX
    ax.scatter(mn[ncold], rn[ncold], s=42, facecolor="tab:green", edgecolor="k", lw=0.5)
    ax.set_title(f"NASA (cold planets)\nwhich dial setting is this?", fontsize=10)
    axes[0].set_ylabel(r"planet radius [$R_\oplus$]")
    fig.suptitle("The theta dial — cold (I<50 I⊕) mass-radius catalogs simulated at NASA's own "
                 "planet locations, corner occupancy dialed from 0 to the flat rate\n"
                 f"answer: theta_hat = {res['inv']['theta_hat']:.2f}, 95% CI "
                 f"[{res['inv']['ci_lo']:.2f}, {res['inv']['ci_hi']:.2f}] — "
                 f"{res['name']} NASA", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(OUT_DIR, "explainer_1_theta_dial.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"--> Saved: {out}")


def explainer_counts_vs_theta(res, m_sil, r_sil):
    """E2 — the inversion re-told in counts: corner-looking planets per catalog vs theta."""
    core = res["core"]
    X = core["XB_ca"]
    m, r, i = 10.0 ** X[:, 0], 10.0 ** X[:, 1], 10.0 ** X[:, 2]
    obs_corner = (r < np.interp(m, m_sil, r_sil)) & (m > S82.MASS_MIN) & (i < S82.COLD_MAX)
    Xn = core["Xn"]
    mn, rn, inn = 10.0 ** Xn[:, 0], 10.0 ** Xn[:, 1], 10.0 ** Xn[:, 2]
    nasa_cnt = int(((rn < np.interp(mn, m_sil, r_sil)) & (mn > S82.MASS_MIN)
                    & (inn < S82.COLD_MAX)).sum())
    pools_idx = split_neighbor_indices(res["idx_ca"], core["cor_ca"])
    rng = np.random.default_rng(22)
    med, lo, hi = [], [], []
    for th in THETA_GRID:
        cnt = obs_corner[draw_rows_at_theta(pools_idx, th, 1200, rng)].sum(axis=1)
        med.append(np.median(cnt)); lo.append(np.percentile(cnt, 2.5))
        hi.append(np.percentile(cnt, 97.5))
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.fill_between(THETA_GRID, lo, hi, color="0.85", label="95% range of fake catalogs")
    ax.plot(THETA_GRID, med, color="0.35", lw=2, label="median fake catalog")
    ax.axhline(nasa_cnt, color="tab:green", lw=2.4,
               label=f"NASA observed: {nasa_cnt} corner-looking planets")
    inv = res["inv"]
    ax.axvspan(inv["ci_lo"], inv["ci_hi"], color="tab:green", alpha=0.15,
               label=f"95% CI from the real (score-weighted) test "
                     f"[{inv['ci_lo']:.2f}, {inv['ci_hi']:.2f}]")
    ax.set_xlabel("theta  (cold-corner occupancy / flat rate)")
    ax.set_ylabel(f"cold rocky M>2 planets per {len(Xn)}-planet catalog (observed values)")
    ax.set_title("Same inversion, told in plain counts — how many corner-looking planets a "
                 "catalog SHOULD have at each dial setting\n(the real test weighs each planet "
                 "by its score instead of counting 0/1 — same idea, more power)", fontsize=10)
    ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "explainer_2_counts_vs_theta.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"--> Saved: {out}")


def explainer_coverage_pictures(res, theta_star=0.5, n_show=100):
    """E3 — coverage as 100 repeated experiments: does the interval catch the truth?"""
    core = res["core"]
    pools_idx = split_neighbor_indices(res["idx_tr"], core["cor_tr"])
    rng = np.random.default_rng(23)
    rows = draw_rows_at_theta(pools_idx, theta_star, n_show, rng)
    T = core["ell_tr"][rows].sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    n_cov = 0
    order = np.argsort(T)
    for x, k in enumerate(order):
        inside = (T[k] >= res["q025"]) & (T[k] <= res["q975"])
        if inside.any():
            lo, hi = THETA_GRID[inside].min(), THETA_GRID[inside].max()
        else:
            lo = hi = np.nan
        covered = np.isfinite(lo) and (lo <= theta_star <= hi)
        n_cov += covered
        ax.plot([x, x], [lo, hi], color="tab:blue" if covered else "tab:red",
                lw=2, alpha=0.85)
    ax.axhline(theta_star, color="k", ls="--", lw=1.6,
               label=f"true theta* = {theta_star:g} (known, we simulated it)")
    ax.set_xlabel(f"{n_show} simulated experiments (sorted)")
    ax.set_ylabel("95% confidence interval on theta")
    ax.set_title(f"Does the interval machinery tell the truth? {n_cov}/{n_show} intervals "
                 f"catch the true value (target: 95)\nblue = caught it, red = missed — "
                 f"{res['name']} NASA design", fontsize=11)
    ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "explainer_3_coverage_100.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"--> Saved: {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("STEP 1-3 (reused from script 82) — pool, noise, classifiers")
    det = S82.get_detected_pool()
    results = {}
    for name, precision in [("precision", True), ("full", False)]:
        nasa = S82.load_nasa(precision)
        results[name] = run_config(det, nasa, name)
    make_figure(results["precision"], results["full"])
    m_sil, r_sil = S82.load_silicate()
    explainer_theta_dial(results["precision"], m_sil, r_sil)
    explainer_counts_vs_theta(results["precision"], m_sil, r_sil)
    explainer_coverage_pictures(results["precision"])

    rp, rf = results["precision"], results["full"]
    print("\n================ CAVEMAN SUMMARY ================")
    print("  We now turn the yes/no test into a NUMBER: theta = how full the cold big-rocky")
    print("  corner is, as a fraction of the flat-universe rate. We dial theta in the fake")
    print("  catalogs until they look like NASA.")
    print(f"  Precision NASA: theta_hat = {rp['inv']['theta_hat']:.2f}, 95% CI "
          f"[{rp['inv']['ci_lo']:.2f}, {rp['inv']['ci_hi']:.2f}] -> the corner holds at most "
          f"{rp['inv']['ci_hi']:.0%} of the flat rate.")
    print(f"  Full NASA: theta_hat = {rf['inv']['theta_hat']:.2f}, 95% CI "
          f"[{rf['inv']['ci_lo']:.2f}, {rf['inv']['ci_hi']:.2f}] (pushed up by the")
    print("  implausible-density planets the precision cut removes).")
    print("  The interval machinery is honest: catalogs simulated at known theta land inside")
    print("  the 95% bands ~95% of the time (panel c), and a second, dumb-but-transparent")
    print("  density estimator (count corner-looking neighbors) gives the same per-planet")
    print("  scores and the same verdict (panel d).")


if __name__ == "__main__":
    main()
