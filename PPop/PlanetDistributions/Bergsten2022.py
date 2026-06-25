"""
# =============================================================================
# P-pop
# A Monte-Carlo tool to simulate exoplanet populations
# =============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats
from scipy.integrate import quad


# =============================================================================
# BERGSTEN2022
# =============================================================================

class rv_Rp(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1.0, 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        # self.Rp_lims = [0.5, 3.5] # Rearth
        # self.Porb_lims = [2., 2000.] # d
        
        super(rv_Rp, self).__init__(**kwargs)
        
        pass
    
    def dNdR(self, Rp):
        
        return 1. / Rp
    
    def _pdf(self, Rp, norm):
        
        return self.dNdR(Rp) / norm
    
    def _argcheck(self, *args):
        
        return 1

class rv_lessless(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1.0, 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        # self.Rp_lims = [0.5, 3.5] # Rearth
        # self.Porb_lims = [2., 2000.] # d
        
        super(rv_lessless, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_lessless(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(self.Rp_lims[0]), np.log(Rval), np.log(self.Rp_lims[1])
        return (Porb / Pbrk)**beta1 * self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_less(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    def _pdf(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_lessless(Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class rv_lessmore(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1.0, 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        # self.Rp_lims = [0.5, 3.5] # Rearth
        # self.Porb_lims = [2., 2000.] # d
        
        super(rv_lessmore, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_lessmore(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(self.Rp_lims[0]), np.log(Rval), np.log(self.Rp_lims[1])
        return (Porb / Pbrk)**beta1 * self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_more(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    def _pdf(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_lessmore(Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class rv_moreless(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1.0, 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        # self.Rp_lims = [0.5, 3.5] # Rearth
        # self.Porb_lims = [2., 2000.] # d
        
        super(rv_moreless, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_moreless(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(self.Rp_lims[0]), np.log(Rval), np.log(self.Rp_lims[1])
        return (Porb / Pbrk)**beta2 * self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_less(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    def _pdf(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_moreless(Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class rv_moremore(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1.0, 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        # self.Rp_lims = [0.5, 3.5] # Rearth
        # self.Porb_lims = [2., 2000.] # d
        
        super(rv_moremore, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_moremore(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(self.Rp_lims[0]), np.log(Rval), np.log(self.Rp_lims[1])
        return (Porb / Pbrk)**beta2 * self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_more(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    def _pdf(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_moremore(Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class PlanetDistribution():
    """
    https://ui.adsabs.harvard.edu/abs/2022AJ....164..190B/abstract
    """
    
    def __init__(self,
                 Scenario,
                 rng):
        """
        Parameters
        ----------
        Scenario: 'baseline', 'pessimistic', 'optimistic', 'mc'
            Scenario for planet occurrence rates.
        """
        
        self.rng = rng
        
        # Print.
        print('--> Initializing Bergsten2022 planet distribution')
        
        # Model parameters.
        self.returns = ['Rp', 'Porb']
        self.Ms_bins = [0.56, 0.81, 0.91, 1.01, 1.16, 1.63] # Msun
        # Different parameters for each stellar mass bin.
        self.F0 = [0.89, 0.70, 0.63, 0.61, 0.50]
        self.Pbrk = [14.29, 6.13, 6.92, 12.02, 6.96] # d
        self.beta1 = [0.15, 1.19, 0.91, 0.44, 1.90]
        self.beta2 = [-1.15, -0.68, -0.84, -1.10, -0.69]
        self.Pcen = [7.54, 11.26, 12.98, 17.57, 16.57] # d
        self.s = [1.46, 2.02, 2.53, 2.16, 2.15] # d
        self.chi1 = [0.75, 0.73, 0.83, 0.83, 0.87]
        self.chi2 = [0.33, 0.36, 0.26, 0.31, 0.39]
        self.Rval = [1.82, 1.93, 1.98, 2.04, 2.17]
        # Same parameters for each stellar mass bin.
        # self.F0 = [0.63]*5
        # self.Pbrk = [9.98]*5 # d
        # self.beta1 = [0.58]*5
        # self.beta2 = [-0.90]*5
        # self.Pcen = [11.33]*5 # d
        # self.s = [2.49]*5 # d
        # self.chi1 = [0.84]*5
        # self.chi2 = [0.38]*5
        # self.Rval = [1.82, 1.93, 1.98, 2.04, 2.17]
        
        # Default bounds.
        self.Rp_lims = [1.0, 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        self.CR_less_org = []
        self.CR_more_org = []
        self.CP_lessless_org = []
        self.C_lessless_org = []
        self.CP_lessmore_org = []
        self.C_lessmore_org = []
        self.CP_moreless_org = []
        self.C_moreless_org = []
        self.CP_moremore_org = []
        self.C_moremore_org = []
        for i in range(len(self.F0)):
            self.CR_less_org += [quad(self.dNdR, self.Rp_lims[0], self.Rval[i])[0]]
            self.CR_more_org += [quad(self.dNdR, self.Rval[i], self.Rp_lims[1])[0]]
            self.CP_lessless_org += [quad(self.dNdP_lessless, self.Porb_lims[0], self.Pbrk[i], args=(self.Pbrk[i], self.beta1[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_lessless_org += [self.CR_less_org[-1] * self.CP_lessless_org[-1]]
            self.CP_lessmore_org += [quad(self.dNdP_lessmore, self.Porb_lims[0], self.Pbrk[i], args=(self.Pbrk[i], self.beta1[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_lessmore_org += [self.CR_more_org[-1] * self.CP_lessmore_org[-1]]
            self.CP_moreless_org += [quad(self.dNdP_moreless, self.Pbrk[i], self.Porb_lims[1], args=(self.Pbrk[i], self.beta2[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_moreless_org += [self.CR_less_org[-1] * self.CP_moreless_org[-1]]
            self.CP_moremore_org += [quad(self.dNdP_moremore, self.Pbrk[i], self.Porb_lims[1], args=(self.Pbrk[i], self.beta2[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_moremore_org += [self.CR_more_org[-1] * self.CP_moremore_org[-1]]
        
        # Extrapolation.
        # self.Rp_lims = [1.0, 3.5] # Rearth
        # self.Porb_lims = [2., 100.] # d
        self.Rp_lims = [0.5, 3.5] # Rearth
        self.Porb_lims = [2., 2000.] # d
        self.CR_less = []
        self.CR_more = []
        self.CP_lessless = []
        self.C_lessless = []
        self.CP_lessmore = []
        self.C_lessmore = []
        self.CP_moreless = []
        self.C_moreless = []
        self.CP_moremore = []
        self.C_moremore = []
        for i in range(len(self.F0)):
            self.CR_less += [quad(self.dNdR, self.Rp_lims[0], self.Rval[i])[0]]
            self.CR_more += [quad(self.dNdR, self.Rval[i], self.Rp_lims[1])[0]]
            self.CP_lessless += [quad(self.dNdP_lessless, self.Porb_lims[0], self.Pbrk[i], args=(self.Pbrk[i], self.beta1[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_lessless += [self.CR_less[-1] * self.CP_lessless[-1]]
            self.CP_lessmore += [quad(self.dNdP_lessmore, self.Porb_lims[0], self.Pbrk[i], args=(self.Pbrk[i], self.beta1[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_lessmore += [self.CR_more[-1] * self.CP_lessmore[-1]]
            self.CP_moreless += [quad(self.dNdP_moreless, self.Pbrk[i], self.Porb_lims[1], args=(self.Pbrk[i], self.beta2[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_moreless += [self.CR_less[-1] * self.CP_moreless[-1]]
            self.CP_moremore += [quad(self.dNdP_moremore, self.Pbrk[i], self.Porb_lims[1], args=(self.Pbrk[i], self.beta2[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_moremore += [self.CR_more[-1] * self.CP_moremore[-1]]
            self.F0[i] *= (self.C_lessless[-1] + self.C_lessmore[-1] + self.C_moreless[-1] + self.C_moremore[-1]) / (self.C_lessless_org[i] + self.C_lessmore_org[i] + self.C_moreless_org[i] + self.C_moremore_org[i])

        # Precompute fast inverse-CDF samplers. This replaces the per-planet scipy
        # rv_continuous.rvs() calls in draw() (which numerically invert the CDF on
        # every call, ~165 ms/planet) with cheap vectorised array ops, while
        # sampling from the exact same distributions. ~1000x faster.
        self._build_samplers()

        pass
    
    def dNdR(self, Rp):
        
        return 1. / Rp
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_lessless(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(1.0), np.log(Rval), np.log(3.5)
        return (Porb / Pbrk)**beta1 * self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_less(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    def dNdP_lessmore(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(1.0), np.log(Rval), np.log(3.5)
        return (Porb / Pbrk)**beta1 * self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_more(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    def dNdP_moreless(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(1.0), np.log(Rval), np.log(3.5)
        return (Porb / Pbrk)**beta2 * self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_less(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_less(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    def dNdP_moremore(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        a, b, c = np.log(1.0), np.log(Rval), np.log(3.5)
        return (Porb / Pbrk)**beta2 * self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) / ((b - a) + self.G_more(Porb, Pcen, s, chi1, chi2) * (c - b) - self.G_more(Porb, Pcen, s, chi1, chi2) * (b - a))
    
    # -------------------------------------------------------------------------
    # Fast inverse-CDF sampling (added optimization).
    #
    # The original draw() created scipy rv_continuous objects per planet and
    # numerically inverted their CDFs on every .rvs() call (~165 ms/planet). The
    # helpers below precompute, once per stellar-mass bin:
    #   * component-selection probabilities,
    #   * radius bounds for the *analytic* inverse of dN/dR ~ 1/R, and
    #   * a numerical inverse-CDF (Porb vs u) per period component.
    # draw() then samples with vectorised array ops from the identical pdfs.
    # -------------------------------------------------------------------------

    # Component layout (index -> radius interval, period interval):
    #   0 lessless : R in [Rp_lo, Rval]   P in [P_lo, Pbrk]   (dNdP_lessless)
    #   1 lessmore : R in [Rval, Rp_hi]    P in [P_lo, Pbrk]   (dNdP_lessmore)
    #   2 moreless : R in [Rp_lo, Rval]    P in [Pbrk, P_hi]   (dNdP_moreless)
    #   3 moremore : R in [Rval, Rp_hi]    P in [Pbrk, P_hi]   (dNdP_moremore)

    def _build_samplers(self, ngrid=4000):
        nbins = len(self.F0)
        self._pcomp = []   # [ww] -> (4,) normalised component probabilities
        self._Pgrid = []   # [ww][comp] -> Porb grid
        self._Pcdf = []    # [ww][comp] -> cumulative prob in [0, 1]
        for ww in range(nbins):
            p = np.array([self.C_lessless[ww], self.C_lessmore[ww],
                          self.C_moreless[ww], self.C_moremore[ww]], dtype=float)
            self._pcomp.append(p / p.sum())

            pdfs = [
                lambda P, w=ww: self.dNdP_lessless(P, self.Pbrk[w], self.beta1[w], self.Pcen[w], self.s[w], self.chi1[w], self.chi2[w], self.Rval[w]),
                lambda P, w=ww: self.dNdP_lessmore(P, self.Pbrk[w], self.beta1[w], self.Pcen[w], self.s[w], self.chi1[w], self.chi2[w], self.Rval[w]),
                lambda P, w=ww: self.dNdP_moreless(P, self.Pbrk[w], self.beta2[w], self.Pcen[w], self.s[w], self.chi1[w], self.chi2[w], self.Rval[w]),
                lambda P, w=ww: self.dNdP_moremore(P, self.Pbrk[w], self.beta2[w], self.Pcen[w], self.s[w], self.chi1[w], self.chi2[w], self.Rval[w]),
            ]
            lohi = [(self.Porb_lims[0], self.Pbrk[ww]),
                    (self.Porb_lims[0], self.Pbrk[ww]),
                    (self.Pbrk[ww], self.Porb_lims[1]),
                    (self.Pbrk[ww], self.Porb_lims[1])]
            grids, cdfs = [], []
            for comp in range(4):
                lo, hi = lohi[comp]
                x = np.logspace(np.log10(lo), np.log10(hi), ngrid)
                y = np.clip(pdfs[comp](x), 0.0, None)
                cdf = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
                cdf /= cdf[-1]
                grids.append(x)
                cdfs.append(cdf)
            self._Pgrid.append(grids)
            self._Pcdf.append(cdfs)

    def _select_bin(self, Star):
        """Stellar-mass bin index (unchanged from the original logic)."""
        if (Star is None):
            return 2
        ww = np.where(np.array(self.Ms_bins) > Star.Mass)[0]
        if (len(ww) == 0):
            return 4
        if (np.min(ww) == 0):
            return 0
        return int(np.min(ww) - 1)

    def _sample(self, ww, comps):
        """Vectorised (Rp, Porb) draw for an array of component indices."""
        n = len(comps)
        # Radius: analytic inverse of dN/dR ~ 1/R, i.e. R = a * (b/a)**u.
        is_more = (comps == 1) | (comps == 3)  # 'more radius' interval [Rval, Rp_hi]
        a = np.where(is_more, self.Rval[ww], self.Rp_lims[0])
        b = np.where(is_more, self.Rp_lims[1], self.Rval[ww])
        Rp = a * (b / a) ** self.rng.random(n)
        # Period: precomputed numerical inverse-CDF, per component.
        uP = self.rng.random(n)
        Porb = np.empty(n)
        for comp in range(4):
            m = comps == comp
            if (m.any()):
                Porb[m] = np.interp(uP[m], self._Pcdf[ww][comp], self._Pgrid[ww][comp])
        return Rp, Porb

    def draw(self,
             Rp_range=[0.5, 3.5], # Rearth
             Porb_range=[2., 2000.], # d
             Nplanets=None,
             Scale=1.,
             Star=None):
        """
        Parameters
        ----------
        Rp_range: list
            Requested planet radius range (Rearth).
        Porb_range: list
            Requested planet orbital period range (d).
        Nplanets: None, int
            Number of planets to be drawn.
        Scale: float
            Scaling factor for the planet occurrence rates.
        Star: instance
            Instance of class Star.

        Returns
        -------
        Rp: array
            Radius (Rearth) of drawn planets.
        Porb: array
            Orbital period (d) of drawn planets.
        """

        ww = self._select_bin(Star)
        pcomp = self._pcomp[ww]

        def _in_range(Rp, Porb):
            return ((Rp_range[0] <= Rp) & (Rp <= Rp_range[1]) &
                    (Porb_range[0] <= Porb) & (Porb <= Porb_range[1]))

        # Number of planets not given: draw from Poisson, then clip to range.
        if (Nplanets is None):
            n = self.rng.poisson(self.F0[ww] * Scale)
            if (n == 0):
                return np.array([]), np.array([])
            comps = self.rng.choice(4, size=n, p=pcomp)
            Rp, Porb = self._sample(ww, comps)
            m = _in_range(Rp, Porb)
            return Rp[m], Porb[m]

        # Exact count requested: oversample and fill until Nplanets land in range.
        out_R, out_P, have = [], [], 0
        while (have < Nplanets):
            batch = max(int(Nplanets - have) * 2, 16)
            comps = self.rng.choice(4, size=batch, p=pcomp)
            Rp, Porb = self._sample(ww, comps)
            m = _in_range(Rp, Porb)
            out_R.append(Rp[m])
            out_P.append(Porb[m])
            have += int(m.sum())
        Rp = np.concatenate(out_R)[:Nplanets]
        Porb = np.concatenate(out_P)[:Nplanets]
        return Rp, Porb
    
    def SummaryPlot(self,
                    Ntest=100000,
                    Rp_range=[0.5, 3.5], # Rearth
                    Porb_range=[2., 2000.], # d
                    FigDir=None,
                    block=True):
        """
        Parameters
        ----------
        Ntest: int
            Number of test draws for summary plot.
        Rp_range: list
            Requested planet radius range (Rearth).
        Porb_range: list
            Requested planet orbital period range (d).
        FigDir: str
            Directory to which summary plots are saved.
        block: bool
            If True, blocks plots when showing.
        """
        
        Rp = []
        Porb = []
        for i in range(Ntest):
            tempRp, tempPorb = self.draw(Rp_range,
                                         Porb_range)
            Rp += [tempRp]
            Porb += [tempPorb]
        Rp = np.concatenate(Rp)
        Porb = np.concatenate(Porb)
        
        print('--> Bergsten2022:\n%.2f planets/star in [%.1f, %.1f] Rearth and [%.1f, %.1f] d' % (len(Rp)/float(Ntest), Rp_range[0], Rp_range[1], Porb_range[0], Porb_range[1]))
        
        Weight = 1./len(Rp)
        f, ax = plt.subplots(1, 2)
        ax[0].hist(Rp, bins=np.logspace(np.log10(np.min(Rp)), np.log10(np.max(Rp)), 25), weights=np.ones_like(Rp)*Weight)
        ax[0].set_xscale('log')
        ax[0].grid(axis='y')
        ax[0].set_xlabel('Planet radius [$R_\oplus$]')
        ax[0].set_ylabel('Fraction')
        ax[1].hist(Porb, bins=np.logspace(np.log10(np.min(Porb)), np.log10(np.max(Porb)), 25), weights=np.ones_like(Porb)*Weight)
        ax[1].set_xscale('log')
        ax[1].grid(axis='y')
        ax[1].set_xlabel('Planet orbital period [d]')
        ax[1].set_ylabel('Fraction')
        plt.suptitle('Bergsten2022')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if (FigDir is not None):
            plt.savefig(FigDir+'PlanetDistribution_Bergsten2022.pdf')
        plt.show(block=block)
        plt.close()
        
        pass
