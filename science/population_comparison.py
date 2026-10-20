"""Reproducible, conditional-on-N population comparison shared by paper outputs.

Likelihood is a measurement-marginalized binomial score:
  L = sum_k q_observed(k) E_p[Binomial(k | N, p)].
q_observed is the count distribution induced by published independent asymmetric
measurement errors at fixed catalogue membership. This is an explicitly approximate
count likelihood (not a hierarchical fit to individual planet measurements).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from science.catalogs import COMPARISON_PARAMETER_BOX, load_measured_planets, read_nasa_csv
from science.physics import COMPARISON_MIN_RADIUS, SUPER_EARTH_MIN_MASS, is_volatile, load_mass_radius_curve
from science.statistics import NASA_MEASUREMENT_ERROR, perturb_asymmetric, perturb_fractional
from tools.paths import ANALYSIS_DIR, PSCOMPPARS_CSV, SILICON_CURVE

BINS = (("I < 10", .01, 10.), ("I < 50", .01, 50.), ("I ≥ 50", 50., 10000.))
POSTERIOR_BINS = ("I < 50", "I ≥ 50")
LABELS = {"rocky_formation": "Sub-Neptunes and super-Earths", "escape_only": "Sub-Neptunes"}
CACHE_DIR = ANALYSIS_DIR / "population_comparison"


@dataclass(frozen=True)
class ComparisonConfig:
    draws: int = 5000
    seed: int = 0
    pool_seed: int = 0
    pool_size: int = 4_000_000
    chunk_size: int = 2_000_000
    mass_min: float = SUPER_EARTH_MIN_MASS
    radius_min: float = COMPARISON_MIN_RADIUS
    mission: str = "kepler"
    rv_mag_target: float | None = 12.0
    error_model: str = "published_asymmetric"

    def __post_init__(self):
        if self.draws < 1 or self.pool_size < 1 or self.chunk_size < 1:
            raise ValueError("Draw and pool sizes must be positive")
        if self.error_model not in {"published_asymmetric", "matched_lognormal"}:
            raise ValueError("Unknown measurement-error model")


def rng_for(config, *labels):
    """Named streams keep results invariant under plot/model traversal order."""
    digest = hashlib.sha256("/".join(map(str, labels)).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([config.seed, *np.frombuffer(digest[:16], dtype='<u4').tolist()]))


def load_sample():
    """One strict transiting, measured-mass, two-sided-uncertainty sample."""
    sample = load_measured_planets(PSCOMPPARS_CSV, max_relative_error=NASA_MEASUREMENT_ERROR)
    accepted = sample.transit_flag.eq(1) & sample.mass_limit_flag.eq(0) & sample.radius_limit_flag.eq(0)
    columns = [f"{q}_error_{side}" for q in ("mass", "radius") for side in ("plus", "minus")]
    errors = sample[columns].to_numpy(float)
    accepted &= np.isfinite(errors).all(axis=1) & (errors > 0).all(axis=1)
    audit = {"catalogue_rows": len(read_nasa_csv(PSCOMPPARS_CSV)),
             "measured_box_precision_rows": len(sample),
             "strict_sample_rows": int(accepted.sum()),
             "rejected_after_box_precision": sample.loc[~accepted, "planet_name"].tolist(),
             "strict_sample_names": sample.loc[accepted, "planet_name"].tolist()}
    return sample.loc[accepted].reset_index(drop=True), audit


def zone_mask(frame, lo, hi, config):
    return (frame.insolation.ge(lo) & frame.insolation.lt(hi)
            & frame.mass.gt(config.mass_min) & frame.radius.gt(config.radius_min))


def observed_counts(sample, config, label, lo, hi):
    zone = sample.loc[zone_mask(sample, lo, hi, config)]
    n = len(zone)
    if n == 0:
        raise ValueError(f"No observed planets in {label}")
    m, r = zone.mass.to_numpy(), zone.radius.to_numpy()
    rng = rng_for(config, label, "observed")
    if config.error_model == "published_asymmetric":
        mm = perturb_asymmetric(np.broadcast_to(m, (config.draws, n)), zone.mass_error_plus.to_numpy(), zone.mass_error_minus.to_numpy(), rng)
        rr = perturb_asymmetric(np.broadcast_to(r, (config.draws, n)), zone.radius_error_plus.to_numpy(), zone.radius_error_minus.to_numpy(), rng)
    else:
        mm, rr = perturb_fractional(np.broadcast_to(m, (config.draws, n)), np.broadcast_to(r, (config.draws, n)), rng, NASA_MEASUREMENT_ERROR)
    curve = load_mass_radius_curve()
    classes = is_volatile(mm, rr, *curve)
    counts = classes.sum(axis=1)
    # Convolve independent per-planet class probabilities. Unlike a histogram of
    # 5,000 total counts, this retains rare combinations needed by likelihood tails.
    count_weights = np.array([1.0])
    for probability in classes.mean(axis=0):
        count_weights = np.convolve(count_weights, [1-probability, probability])
    return zone, counts, int(is_volatile(m, r, *curve).sum()), count_weights


def model_draws(population, error_sample, config, label, lo, hi, model, n):
    """Noise pool, select measured box, and sample exactly N, without replacement."""
    pool = population.loc[population.joint_detected & population.insolation.ge(lo) & population.insolation.lt(hi)]
    m, r = pool.mass.to_numpy(), pool.radius.to_numpy()
    if len(pool) < n:
        raise ValueError(f"{model}/{label}: only {len(pool)} detected objects for N={n}")
    rng = rng_for(config, label, model)
    curve = load_mass_radius_curve()
    fractions, probabilities = np.empty(config.draws), np.empty(config.draws)
    # Jointly resample the four fractional errors, retaining their empirical pairing.
    errors = np.column_stack([error_sample[f'{q}_error_{side}'].to_numpy() / error_sample[q].to_numpy()
                              for q in ('mass', 'radius') for side in ('plus', 'minus')])
    b = COMPARISON_PARAMETER_BOX
    eligible_counts = np.empty(config.draws, dtype=int)
    for i in range(config.draws):
        if config.error_model == 'published_asymmetric':
            e = errors[rng.integers(len(errors), size=len(m))]
            mm = perturb_asymmetric(m, m*e[:,0], m*e[:,1], rng)
            rr = perturb_asymmetric(r, r*e[:,2], r*e[:,3], rng)
        else:
            mm, rr = perturb_fractional(m, r, rng, NASA_MEASUREMENT_ERROR)
        keep = ((mm > config.mass_min) & (rr > config.radius_min)
                & (mm >= b['m_lo']) & (mm <= b['m_hi']) & (rr >= b['r_lo']) & (rr <= b['r_hi']))
        idx = np.flatnonzero(keep)
        eligible_counts[i] = len(idx)
        if len(idx) < n:
            raise ValueError(f"{model}/{label}: insufficient eligible pool in draw {i}; increase pool_size")
        volatile = is_volatile(mm[idx], rr[idx], *curve)
        # Jeffreys smoothing represents finite-pool probability uncertainty and avoids 0/1.
        probabilities[i] = rng.beta(volatile.sum()+.5, len(idx)-volatile.sum()+.5)
        chosen = rng.choice(len(idx), n, replace=False)
        fractions[i] = volatile[chosen].mean()
    return fractions, probabilities, int(eligible_counts.min())


def count_log_likelihood(observed_count_draws, probabilities, n, *, count_weights=None):
    """Marginalize both latent observed count and finite-pool probability draws."""
    q = (np.bincount(np.asarray(observed_count_draws, dtype=int), minlength=n+1).astype(float)
         if count_weights is None else np.asarray(count_weights, dtype=float).copy())
    q /= q.sum()
    support = np.flatnonzero(q)
    terms = binom.logpmf(support[:,None], n, np.asarray(probabilities)[None,:])
    terms += np.log(q[support])[:,None]
    return float(logsumexp(terms) - np.log(len(probabilities)))


def posterior_from_logs(log_likelihoods):
    values = np.array(list(log_likelihoods.values()))
    weights = np.exp(values-logsumexp(values))
    return dict(zip(log_likelihoods, map(float, weights)))


def source_digest():
    h = hashlib.sha256()
    for path in sorted((Path(__file__).parent).rglob('*.py')):
        h.update(str(path.relative_to(Path(__file__).parent)).encode()); h.update(path.read_bytes())
    for path in (PSCOMPPARS_CSV, SILICON_CURVE):
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def pool_digest(populations):
    h = hashlib.sha256()
    for name, pool in sorted(populations.items()):
        h.update(name.encode())
        h.update(pool[['mass','radius','insolation','joint_detected']].to_numpy(float).tobytes())
    return h.hexdigest()


def run_comparison(config=ComparisonConfig(), *, populations=None, scenario='primary', bins=BINS):
    """Return cached shared summary and named arrays; save full provenance and draws."""
    sample, selection_audit = load_sample()
    provenance = {"config": asdict(config), "source_sha256": source_digest(), "scenario": scenario,
                  "bins": list(bins), "custom_pool_sha256": pool_digest(populations) if populations is not None else None}
    token = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:20]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = CACHE_DIR / f'{scenario}_{token}'
    metadata, draws_file = Path(str(prefix)+'.json'), Path(str(prefix)+'.npz')
    if metadata.exists() and draws_file.exists():
        cached = json.loads(metadata.read_text())
        if cached['provenance'] == json.loads(json.dumps(provenance)):
            with np.load(draws_file) as archive:
                arrays = {k: archive[k] for k in archive.files}
            return cached, arrays
    if populations is None:
        from science.telescopes.detection import make_detected_pool, split_universes
        pool = make_detected_pool('superearths_supneptunes', pool_size=config.pool_size, chunk_size=config.chunk_size,
                                  cache_dir=CACHE_DIR, mission=config.mission, seed=config.pool_seed,
                                  box=COMPARISON_PARAMETER_BOX, rv_mag_target=config.rv_mag_target)
        populations = split_universes(pool)
    arrays, summaries = {}, {}
    for index, (label, lo, hi) in enumerate(bins):
        zone, k_draws, k, q = observed_counts(sample, config, label, lo, hi)
        n = len(zone)
        arrays[f'b{index}_observed'] = k_draws / n
        arrays[f'b{index}_observed_count_probabilities'] = q
        item = {'lo':lo, 'hi':hi, 'n_observed':n, 'k_observed':k, 'planet_names':zone.planet_name.tolist(), 'models':{}}
        for model, population in populations.items():
            fractions, probabilities, eligible = model_draws(population, zone, config, label, lo, hi, model, n)
            logl = count_log_likelihood(k_draws, probabilities, n, count_weights=q)
            arrays[f'b{index}_{model}'] = fractions
            arrays[f'b{index}_{model}_probabilities'] = probabilities
            item['models'][model] = {'mean_fraction':float(fractions.mean()), 'sd_fraction':float(fractions.std()),
                'log_likelihood':logl, 'deviance':-2*logl, 'valid_draws':len(fractions), 'minimum_eligible_pool':eligible}
        item['posterior'] = posterior_from_logs({m:v['log_likelihood'] for m,v in item['models'].items()})
        summaries[label] = item
    included = [label for label in POSTERIOR_BINS if label in summaries]
    total = {model:sum(summaries[label]['models'][model]['log_likelihood'] for label in included) for model in populations}
    summary = {'provenance':provenance, 'selection_audit':selection_audit,
               'assumptions':'Fixed nominal catalogue membership; independent asymmetric measurement errors; empirical zone-matched synthetic errors; measurement-marginalized binomial count approximation; Jeffreys finite-pool probability smoothing; equal model priors; disjoint posterior bins.',
               'posterior_bins':included, 'bins':summaries, 'total_log_likelihood':total,
               'posterior':posterior_from_logs(total), 'draws_file':str(draws_file)}
    np.savez_compressed(draws_file, **arrays)
    metadata.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    return summary, arrays
