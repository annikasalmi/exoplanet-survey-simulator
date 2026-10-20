# Mass–radius update audit — 20 October 2026

Subsequent approved changes: LHS 1478 b’s NASA mass/radius/insolation values and error bars and TOI-2427 b’s full current composite row were applied to both local PSCompPars files. All other candidates below remain unapplied. The comparisons below retain the original snapshot as their historical baseline.

The broader sample uses maximum one-sided fractional errors <=30% in mass and <=10% in radius, together with the same transiting/measured-mass/no-limit/positive-two-sided-error requirements and common parent box as the paper. This gives **154 planets** in the saved snapshot, including 28 in M>2.5, R>1.4, I<50. The active paper comparison remains the stricter 25%/8% sample.

The audit initially made no catalogue changes; the subsequent approved updates are recorded above. The original snapshot retrieval date is unknown; its first commit was 4 September 2026. “Updated” below means different from that snapshot, not necessarily published after that date.

## Live NASA catalogue comparison

Fetched the PSCompPars mass/radius fields, both error sides, limit flags, provenance, individual references, insolation, and transit flag through NASA TAP. All 154 original planets were matched. Compared values and both uncertainty sides numerically, with tolerance rtol=1e-8, atol=1e-10.

Only **LHS 1478 b** differs among those 154 existing planets:

| Quantity | Saved snapshot | Current NASA |
|---|---:|---:|
| Mass, Earth masses | 2.33 ± 0.20 | 2.27 ± 0.45 |
| Radius, Earth radii | 1.242 +0.051/-0.049 | 1.174 ± 0.055 |
| Insolation, Earth flux | 23.3722 | 20.9 |

NASA now adopts August et al. (2025), [Hot Rocks Survey I](https://arxiv.org/abs/2410.11048). This planet remains outside the final mass/radius box.

Two planets additionally become eligible when selecting the entire current catalogue:

| Planet | Current mass | Current radius | I | Eligibility |
|---|---:|---:|---:|---|
| TOI-2427 b | 5.69 +0.51/-0.50 | 1.64 +0.12/-0.11 | 289.2 | Passes both 30%/10% and 25%/8%. Saved mass was an M–R estimate, not a measured mass. |
| TOI-426 b | 6.7 ±1.8 | 2.19 ±0.09 | 1711 | Passes 30%/10%; fails 25% mass precision. Absent from saved snapshot. |

Sources: [Schmerling et al. (2025)](https://doi.org/10.1051/0004-6361/202452620), [Castro-González et al. (2026)](https://arxiv.org/abs/2609.05413). Current broader parent sample: **156**. No original broader-sample objects are lost.

A NASA-only refresh would leave the primary I<50 sample at **20 volatile / 22 total**, and change the primary I>=50 sample from **22/65 to 22/66**, owing to TOI-2427 b (rocky under the project's silicate criterion). The strict parent sample would increase from 128 to 129. These are nominal-selection counts, not a rerun posterior.

## Literature not yet reflected in adopted NASA values

Searched arXiv metadata by the sample's 125 host names, with submission dates from 1 January 2024 through 20 October 2026: seven API queries, **285 unique records**, no query truncated at its 200-result limit. Screened titles and abstracts for parameter work and inspected relevant full-text tables. This is not a guarantee of exhaustive coverage of aliases, measurements buried in unrelated papers, or papers without arXiv records.

### New RV and global fits

| Planet | Saved mass | Literature mass | Saved radius | Literature radius | Source |
|---|---:|---:|---:|---:|---|
| GJ 1132 b | 1.837 +0.191/-0.188 | 1.69 ±0.15 | 1.192 +0.038/-0.045 | 1.22 ±0.04 | [Weisserman et al. 2026, Table 4](https://arxiv.org/html/2604.07447v1) |
| GJ 1252 b | 1.32 ±0.28 | 1.54 ±0.18 | 1.193 ±0.074 | 1.19 ±0.05 | Same NIRPS paper |
| LTT 3780 b | 2.46 ±0.19 | 2.34 ±0.10 | 1.325 +0.057/-0.058 | 1.31 ±0.05 | Same NIRPS paper |
| HD 219134 b | 4.74 ±0.19 | 4.608 ±0.199 | 1.602 ±0.055 | 1.554 ±0.056 | [October 2026 study, Table 5](https://arxiv.org/html/2610.13200v1) |
| HD 219134 c | 4.36 ±0.22 | 4.244 ±0.226 | 1.511 ±0.047 | 1.465 ±0.048 | Same October 2026 study |

All masses/radii are in Earth units. The first three remain below the final box. GJ 1252 b nevertheless changes nominal parent-sample classification from above to below the project's silicate boundary; this affects a broad rocky-overlay sample but not the final mass/radius-selected comparison. HD 219134 b/c remain inside it and in the higher-insolation bin, but changed uncertainties should enter a literature-based refresh. The HD 219134 study explicitly retains only radius/minimum-mass limits for f and d; **HD 219134 f must remain excluded**.

The NIRPS paper also rescales several other small planets (GJ 357 b, HD 260655 b, L 98-59 b/c, LHS 1140 c, TOI-270 b); these are listed separately in the candidate CSV, and are not all new RV mass determinations. A separate earlier [2025 asteroseismic HD 219134 solution](https://arxiv.org/html/2502.00971v2) is included as an alternative, not combined with the 2026 solution.

### Competing fits that should be tested, not silently substituted

- **TOI-1266 c:** [Cloutier et al. 2024](https://arxiv.org/abs/2410.20282) report a combined RV mass of **2.4 ±0.54**, versus the adopted **3.17 ±0.76** from the TTV+RV analysis [Stefansson et al., updated 2025](https://arxiv.org/abs/2409.16374). Using 2.4 removes this planet from M>2.5. The TTV+RV preprint was revised after the RV paper, so arXiv identifier order alone is not an adoption rule. Run this as a named alternative-fit sensitivity.
- **LHS 3844 b:** an [alternative ESPRESSO fit](https://arxiv.org/abs/2603.28237) gives **2.27 ±0.23**, whereas the saved and current NASA mass is **2.37 ±0.25** from the [CRIRES+ plus ESPRESSO study](https://arxiv.org/abs/2603.28238). Both remain outside the final box.
- **Ross et al. 2025:** [homogeneous stellar reanalysis](https://arxiv.org/abs/2508.16421), with 24 planets overlapping the broader sample. Parsed their System Parameters table directly from the TeX source. All 24 overlap rows are in the candidate CSV. These are alternative stellar-rescaling solutions using existing transit/RV inputs; they do not necessarily supersede newer individual dynamical measurements.
- Under the project's operational silicate boundary, the Ross solutions move **55 Cnc e** and **WASP-47 e** from rocky to volatile, and **HD 136352 b** from volatile to rocky. Their nominal values are 8.10 +0.33/-0.32 and 1.99 ±0.02 for 55 Cnc e; 6.75 +0.56/-0.54 and 1.85 ±0.02 for WASP-47 e; 4.74 +0.39/-0.38 and 1.63 ±0.03 for HD 136352 b. These are calculated classification changes, not statements that their physical compositions are established.
- Do not automatically replace WASP-47 e: the Ross calculation reuses Bryant (2022) mass observables, while NASA adopts Nascimbeni et al. (2023). Likewise, their Kepler-105 c uses Hadden (2017) inputs while NASA adopts Householder et al. (2024). The Ross Kepler-105 c mass uncertainty also fails 30%.

## Already incorporated examples

The saved values already match the 2026 HD 3167 b global fit (4.844 +0.270/-0.254, 1.601 +0.048/-0.036; [Table A1](https://arxiv.org/html/2604.11911v2)), and the reported recent results for GJ 3090 b (4.52 ±0.47, 2.18 ±0.06), GJ 367 b (0.503 ±0.078, 0.736 ±0.035), the L 98-59 transiting planets, Kepler-10 b, LP 791-18 d, TOI-4336 A b to its published rounding, HD 80653 b, and LHS 1140 b. Merely finding a recent paper is not evidence that the saved measurement needs replacement.

## Saved evidence

In `results/catalogue_update_audit/`:

- `current_pscomppars.csv` and `query.txt`: fresh public NASA response and exact query URL.
- `local_30pct_10pct_sample.csv`: all 154 audit targets.
- `full_comparison.csv`: every matched existing target and old/current measurements.
- `changed_30pct_10pct_planets.csv`: the catalogue difference row.
- `literature_measurement_candidates.csv`: 41 review rows covering 37 existing planets, including one confirmed already-incorporated fit, with uncertainty sides, precision-cut checks, and nominal box/classification effects. Reused radii in mass-only alternatives are explicitly flagged.
- `arxiv_search.json` and `arxiv_batch_*.xml`: search queries and metadata.
- `arxiv_2508_16421_source/article.tex`: source table used for the homogeneous reanalysis comparison.
- `audit_summary.json`: compact machine-readable scope and results.

Recommended adoption policy: refresh the NASA snapshot with explicit provenance; treat individual newer RV/global fits as separately documented overrides only after choosing an adoption rule; retain conflicting fits and homogeneous rescalings as sensitivity cases. A refresh should update errors and reference fields together, and rerun selection and classification rather than assuming only the plotted coordinates change.
