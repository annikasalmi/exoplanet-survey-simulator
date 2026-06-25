"""
universe_metrics.py — population-comparison metrics for the rocky/puffy universe test.

Pure numpy / scipy / sklearn (NO detector or Qt imports), so it is safe to import anywhere.
All metrics take planet mass M [M_earth] and radius R [R_earth] arrays (+ the silicate curve
for the curve-dependent ones).

Groups:
  curve-DEPENDENT scalars   : puffy_fraction, mean_distance_to_curve
  curve-FREE scalars        : median_density, mr_scatter (slope/intercept/sigma_int),
                              logdensity_scatter, density_bimodality (BC + GMM dBIC)
  distribution distances    : energy_distance (+ permutation p-value)  [2-sample, in logM-logR]
  likelihood / Bayes        : kde_mean_loglik (mean log-likelihood of data under a model KDE)
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None
try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    GaussianMixture = None

RHO_EARTH = 5.513  # g/cc


# ----------------------------- curve-dependent scalars -----------------------------
def puffy_fraction(M, R, m_sil, r_sil) -> float:
    R = np.asarray(R, float); M = np.asarray(M, float)
    return float((R > np.interp(M, m_sil, r_sil)).mean())


def mean_distance_to_curve(M, R, m_sil, r_sil) -> float:
    R = np.asarray(R, float); M = np.asarray(M, float)
    return float((R - np.interp(M, m_sil, r_sil)).mean())


# ----------------------------- curve-free scalars -----------------------------
def median_density(M, R) -> float:
    M = np.asarray(M, float); R = np.asarray(R, float)
    return float(np.median(RHO_EARTH * M / R ** 3))


def mr_scatter(M, R):
    """Fit log10 R = a + b log10 M. Returns (slope b, intercept a, intrinsic scatter sigma_int [dex])."""
    x = np.log10(np.asarray(M, float)); y = np.log10(np.asarray(R, float))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return float(b), float(a), float(resid.std(ddof=1))


def logdensity_scatter(M, R) -> float:
    d = np.log10(RHO_EARTH * np.asarray(M, float) / np.asarray(R, float) ** 3)
    d = d[np.isfinite(d)]
    return float(d.std(ddof=1)) if d.size > 1 else np.nan


def _skew(x):
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0


def _exkurt(x):
    s = x.std()
    return float(((x - x.mean()) ** 4).mean() / s ** 4 - 3) if s > 0 else 0.0


def density_bimodality(M, R):
    """log10(density) bimodality. Returns (Sarle BC, GMM dBIC = BIC(1)-BIC(2)).
    BC > 0.555 hints bimodal (uniform=0.555, normal=0.333); dBIC > 0 favours 2 components."""
    d = np.log10(RHO_EARTH * np.asarray(M, float) / np.asarray(R, float) ** 3)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 20:
        return np.nan, np.nan
    g, k = _skew(d), _exkurt(d)
    bc = (g ** 2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    dbic = np.nan
    if GaussianMixture is not None:
        dd = d.reshape(-1, 1)
        try:
            # init_params="random" avoids the internal KMeans (and its MKL memory-leak
            # warning on Windows); fine here since we only need BIC(1) vs BIC(2).
            b1 = GaussianMixture(1, covariance_type="full", random_state=0, n_init=1,
                                 init_params="random").fit(dd).bic(dd)
            b2 = GaussianMixture(2, covariance_type="full", random_state=0, n_init=2,
                                 init_params="random").fit(dd).bic(dd)
            dbic = float(b1 - b2)
        except Exception:
            pass
    return float(bc), dbic


# ----------------------------- 2-sample distribution distance -----------------------------
def _logMR(M, R):
    P = np.column_stack([np.log10(np.asarray(M, float)), np.log10(np.asarray(R, float))])
    return P[np.isfinite(P).all(1)]


def _mean_pair_dist(X, Y):
    d = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    return d.mean()


def energy_distance_std(A, B):
    """Energy distance between two (n,2)/(m,2) point clouds, standardised by the pooled scale."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    P = np.vstack([A, B]); mu = P.mean(0); sd = P.std(0); sd[sd == 0] = 1
    A, B = (A - mu) / sd, (B - mu) / sd
    return float(2 * _mean_pair_dist(A, B) - _mean_pair_dist(A, A) - _mean_pair_dist(B, B))


def energy_distance(M1, R1, M2, R2, n_perm=200, rng=None, max_n=300):
    """Energy distance in logM-logR between sample 1 and sample 2, plus a permutation p-value.
    Smaller distance = more similar. p small = samples differ significantly."""
    rng = rng or np.random.default_rng(0)
    A, B = _logMR(M1, R1), _logMR(M2, R2)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    obs = energy_distance_std(A, B)
    if n_perm <= 0:
        return obs, np.nan
    pool = np.vstack([A, B]); n = len(A)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pool))
        if energy_distance_std(pool[idx[:n]], pool[idx[n:]]) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


# ----------------------------- KDE likelihood / Bayes -----------------------------
def kde_mean_loglik(model_M, model_R, data_M, data_R, bw=None):
    """Mean log-likelihood of the DATA planets under a Gaussian KDE of the MODEL planets,
    in logM-logR. Higher = data look more like they were drawn from this model.
    Bayes factor between two models = exp(N_data * (loglik_1 - loglik_2))."""
    if gaussian_kde is None:
        return np.nan
    m = _logMR(model_M, model_R); d = _logMR(data_M, data_R)
    if len(m) < 10 or len(d) < 1:
        return np.nan
    try:
        kde = gaussian_kde(m.T, bw_method=bw)
        return float(np.mean(kde.logpdf(d.T)))
    except Exception:
        return np.nan
