import matplotlib.pyplot as plt
from assistant_functions import *
from dataclasses import dataclass
try:
    from scipy.interpolate import PchipInterpolator as _MonoSpline   # monotone & smooth
    _HAVE_PCHIP = True
except Exception:
    _HAVE_PCHIP = False

@dataclass
class LagModel:
    x_anchors: np.ndarray   # where lag was measured (on displacement x-axis)
    lags: np.ndarray        # measured local lags at anchors (Cp - Cd etc.)
    interp_kind: str = "pchip"

    def __call__(self, x):
        x = np.asarray(x, float)
        xa, ya = self.x_anchors, self.lags
        # strictly increasing domain for interpolation
        order = np.argsort(xa)
        xa, ya = xa[order], ya[order]
        # collapse duplicates by averaging
        uniq, idx = np.unique(xa, return_inverse=True)
        if len(uniq) != len(xa):
            ya = np.bincount(idx, weights=ya) / np.bincount(idx)
            xa = uniq
        # build interpolant
        if self.interp_kind == "pchip" and _HAVE_PCHIP and len(xa) >= 2:
            f = _MonoSpline(xa, ya, extrapolate=True)
            return f(x)
        # fallback: piecewise-linear
        return np.interp(x, xa, ya, left=ya[0], right=ya[-1])

def estimate_variable_lag(x_p, segs_p, x_d, segs_d, include_bounds=True,
                          weights="len", interp_kind="pchip"):
    """
    Build a lag function L(x) so that x_d + L(x_d) aligns to x_p.
    """
    # sort segments by time
    segs_p = sorted([np.asarray(s, int) for s in segs_p], key=lambda s: s[0])
    segs_d = sorted([np.asarray(s, int) for s in segs_d], key=lambda s: s[0])
    K = min(len(segs_p), len(segs_d))
    segs_p, segs_d = segs_p[:K], segs_d[:K]

    # centers and boundaries
    Cp, Bp, _ = _centers_and_bounds(x_p, segs_p)
    Cd, Bd, _ = _centers_and_bounds(x_d, segs_d)

    # anchor set: centers (and optionally boundaries)
    xA, lagA, wA = [], [], []
    for i in range(K):
        xA.append(Cd[i])
        lagA.append(Cp[i] - Cd[i])
        wA.append(len(segs_d[i]) if weights == "len" else 1.0)

    if include_bounds and K > 1:
        for i in range(K-1):
            xA.append(Bd[i])
            lagA.append(Bp[i] - Bd[i])
            wi = 0.5 * ((len(segs_d[i]) if weights == "len" else 1.0) +
                        (len(segs_d[i+1]) if weights == "len" else 1.0))
            wA.append(wi)

    xA = np.asarray(xA, float)
    lagA = np.asarray(lagA, float)
    wA = np.asarray(wA, float)

    # robustify anchors a bit (down-weight outliers with Huber-like weights)
    m = np.median(lagA)
    s = 1.4826 * np.median(np.abs(lagA - m)) + 1e-12
    r = (lagA - m) / s
    huber = np.minimum(1.0, 1.345 / (np.abs(r) + 1e-12))

    lagA = m + huber * (lagA - m)

    model = LagModel(x_anchors=xA, lags=lagA, interp_kind=interp_kind)
    return model, {"anchors_x": xA, "anchors_lag": lagA, "K": K, "interp": interp_kind}

def apply_variable_lag(x_d, lag_model):
    return np.asarray(x_d, float) + lag_model(x_d)

def moving_avg(x, k=7):
    k = max(1, int(k) // 2 * 2 + 1)  # force odd
    w = np.ones(k) / k
    return np.convolve(x, w, mode='same')

def rolling_median(x, k):
    k = int(max(1, k))
    if k % 2 == 0: k += 1
    # pad at ends so length stays the same
    pad = k // 2
    xp = np.pad(x, pad_width=pad, mode='edge')
    # naive median filter (fast enough for 1D)
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i+k])
    return out

def _runs_from_mask(mask):
    if not np.any(mask):
        return []
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0]
    if mask[0]:  starts = np.r_[0, starts]
    if mask[-1]: ends   = np.r_[ends, len(mask)-1]
    return list(zip(starts, ends))

def find_vertical_regions(x, y, slope_thresh=None, min_len=8, gap_tol=3, bounds=None, smooth_window=9, vis=False, title=""):
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    assert x.shape == y.shape and x.ndim == 1
    if not np.all(np.diff(x) >= 0):
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
    else:
        idx = None

    # smoothen curve
    y_sm = rolling_median(y, smooth_window)

    # slope dy/dx
    dx = np.gradient(x)
    slope = np.gradient(y_sm) / np.maximum(dx, np.finfo(float).eps)

    # Ensure bounds are enforced
    in_bounds = np.ones_like(y, dtype=bool)
    if bounds is not None:
        lo, hi = bounds
        in_bounds &= (y >= lo) & (y <= hi)

    # select slope threshold if not defined
    if slope_thresh is None:
        slope_thresh = np.percentile(np.abs(slope[in_bounds]), 75)

    vert_mask = (np.abs(slope) >= slope_thresh) & in_bounds

    if gap_tol > 0 and np.any(vert_mask):
        runs = _runs_from_mask(vert_mask)
        merged = []
        cur_s, cur_e = runs[0]
        for s, e in runs[1:]:
            if s - cur_e - 1 <= gap_tol:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        runs = merged
    else:
        runs = _runs_from_mask(vert_mask)

    # 5) drop short runs and return index arrays
    regions = []
    for s, e in runs:
        if (e - s + 1) >= min_len:
            regions.append(np.arange(s, e + 1))

    # Visualization
    if vis:
        plt.figure()
        plt.scatter(x, y, s=4, alpha=0.6, label="Data")
        for r in regions:
            plt.scatter(x[r], y[r], s=8, label="Vertical")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Optional diagnostic: slope & threshold
        fig, ax1 = plt.subplots()
        ax1.plot(x, y_sm, lw=1, label="y (smoothed)")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")
        ax2 = ax1.twinx()
        ax2.plot(x, slope, lw=1, alpha=0.7, label="dy/dx", color="tab:red")
        ax2.axhline(+slope_thresh, ls="--", color="tab:red")
        ax2.axhline(-slope_thresh, ls="--", color="tab:red")
        ax2.set_ylabel("slope")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc="best")
        plt.tight_layout()
        plt.show()

    # If we sorted, map back to original indices
    if idx is not None:
        inv = np.empty_like(idx)
        inv[idx] = np.arange(len(idx))
        regions = [inv[r] for r in regions]

    return regions, {
        "slope_thresh": float(slope_thresh),
        "min_len": int(min_len),
        "gap_tol": int(gap_tol),
        "bounds": None if bounds is None else tuple(map(float, bounds)),
        "smooth_window": int(smooth_window),
    }

def _center_idx(seg):  # seg is an index array
    return (seg[0] + seg[-1]) // 2

def _centers_and_bounds(x, segs):
    # centers of each plateau core
    C = np.array([x[_center_idx(s)] for s in segs], float)
    # boundaries between consecutive plateaus (midpoint between runs)
    B = np.array([0.5*(x[segs[i][-1]] + x[segs[i+1][0]]) for i in range(len(segs)-1)], float)
    # intervals for diagnostics
    intervals = [(float(x[s[0]]), float(x[s[-1]])) for s in segs]
    return C, B, intervals

def estimate_constant_lag(x_p, segs_p, x_d, segs_d,
                          method="median", include_bounds=True, weights="len"):
    """
    Return a single lag L so that x_d + L best aligns with x_p.
    segs_* are lists of index arrays (your 4 flat regions, time-sorted).
    """
    # sort by time just in case
    segs_p = sorted([np.asarray(s, int) for s in segs_p], key=lambda s: s[0])
    segs_d = sorted([np.asarray(s, int) for s in segs_d], key=lambda s: s[0])

    K = min(len(segs_p), len(segs_d))
    segs_p, segs_d = segs_p[:K], segs_d[:K]

    Cp, Bp, _ = _centers_and_bounds(x_p, segs_p)
    Cd, Bd, _ = _centers_and_bounds(x_d, segs_d)

    # shifts from plateau centers
    shifts = [Cp[i] - Cd[i] for i in range(K)]
    w = [len(segs_d[i]) if weights=="len" else 1.0 for i in range(K)]

    # optionally include boundaries (align the vertical jumps too)
    if include_bounds and K > 1:
        for i in range(K-1):
            shifts.append(Bp[i] - Bd[i])
            # weight: average size of the two plateaus around the boundary
            wi = 0.5*((len(segs_d[i]) if weights=="len" else 1.0) +
                      (len(segs_d[i+1]) if weights=="len" else 1.0))
            w.append(wi)

    shifts = np.asarray(shifts, float)
    w = np.asarray(w, float)

    if method == "median":
        L = float(np.median(shifts))
    elif method == "wmean":
        L = float(np.average(shifts, weights=w))
    else:
        # simple Huber-like reweighting for robustness
        m = np.median(shifts); s = 1.4826*np.median(np.abs(shifts - m)) + 1e-9
        r = (shifts - m)/s
        huber_w = np.minimum(1.0, 1.345/np.abs(r))
        L = float(np.average(shifts, weights=w*huber_w))

    return L

def apply_constant_lag(x_d, L):
    return np.asarray(x_d, float) + float(L)
