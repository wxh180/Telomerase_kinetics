"""
First-pass sequential-state model for reproducing Kintek-style telomerase kinetics fits.

This model is intentionally closer to the manuscript direction than the older
repeat-level processivity model. It treats the observed species S1..Sn as a
sequential chain of extension states that can advance and optionally dissociate.

Model:
    S1 --k1--> S2 --k2--> S3 --k3--> ... --k(n-1)--> Sn
     |         |         |                    |
    koff1     koff2     koff3               koffn
     v         v         v                    v
    D1        D2        D3                  Dn

This is still a reduced model, but it is much closer to what the manuscript draft
appears to require than the original repeat-level model.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass
class SequentialFitResult:
    n_states: int
    k_forward: list
    k_off: list
    sse: float
    success: bool
    message: str


def load_kintek_txt(path: str | Path) -> dict:
    path = Path(path)
    with path.open(newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)
    header = [h.strip() for h in rows[0]]
    cols = {h: [] for h in header}
    for row in rows[1:]:
        if not row or all(not x.strip() for x in row):
            continue
        for h, val in zip(header, row):
            cols[h].append(float(val))
    return cols


def sequential_rhs(t: float, y: np.ndarray, kf: np.ndarray, koff: np.ndarray) -> np.ndarray:
    n = len(kf) + 1
    s = y[:n]
    d = y[n:]
    ds = np.zeros_like(s)
    dd = np.zeros_like(d)

    # State 0 (S1)
    ds[0] = -kf[0] * s[0] - koff[0] * s[0]
    dd[0] = koff[0] * s[0]

    # Intermediate states
    for i in range(1, n - 1):
        ds[i] = kf[i - 1] * s[i - 1] - kf[i] * s[i] - koff[i] * s[i]
        dd[i] = koff[i] * s[i]

    # Final state Sn has no forward transition in this reduced model
    ds[n - 1] = kf[n - 2] * s[n - 2] - koff[n - 1] * s[n - 1]
    dd[n - 1] = koff[n - 1] * s[n - 1]

    return np.concatenate([ds, dd])


def simulate(times: np.ndarray, kf: np.ndarray, koff: np.ndarray, s0: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    n = len(kf) + 1
    if s0 is None:
        s0 = np.zeros(n)
        s0[0] = 1.0
    d0 = np.zeros(n)
    y0 = np.concatenate([s0, d0])

    sol = solve_ivp(
        fun=lambda t, y: sequential_rhs(t, y, kf, koff),
        t_span=(float(times.min()), float(times.max())),
        y0=y0,
        t_eval=np.array(times, dtype=float),
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    y = sol.y.T
    return y[:, :n], y[:, n:]


def fit_dataset(df: dict, use_columns: list[str] | None = None, shared_koff: bool = False) -> SequentialFitResult:
    if use_columns is None:
        use_columns = [c for c in df.keys() if c != "Time"]
    times = np.array(df["Time"], dtype=float)
    data = np.column_stack([np.array(df[c], dtype=float) for c in use_columns])
    n = data.shape[1]

    # normalize to total at each timepoint if desired? Here we keep raw imported values.
    # Initialize from simple heuristics
    kf0 = np.full(n - 1, 0.2)
    if shared_koff:
        x0 = np.concatenate([kf0, np.array([0.02])])
        lb = np.concatenate([np.full(n - 1, 1e-6), np.array([0.0])])
        ub = np.concatenate([np.full(n - 1, 10.0), np.array([10.0])])
    else:
        koff0 = np.full(n, 0.02)
        x0 = np.concatenate([kf0, koff0])
        lb = np.concatenate([np.full(n - 1, 1e-6), np.zeros(n)])
        ub = np.concatenate([np.full(n - 1, 10.0), np.full(n, 10.0)])

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kf = x[: n - 1]
        if shared_koff:
            koff = np.full(n, x[-1])
        else:
            koff = x[n - 1 :]
        return kf, koff

    def residuals(x: np.ndarray) -> np.ndarray:
        kf, koff = unpack(x)
        s, _ = simulate(times, kf, koff)
        return (s - data).ravel()

    res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=5000)
    kf, koff = unpack(res.x)
    sse = float(np.sum(res.fun ** 2))
    return SequentialFitResult(
        n_states=n,
        k_forward=kf.tolist(),
        k_off=koff.tolist(),
        sse=sse,
        success=bool(res.success),
        message=str(res.message),
    )


def save_fit(result: SequentialFitResult, out_json: str | Path) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(asdict(result), indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fit first-pass sequential-state model to Kintek-style data.")
    parser.add_argument("data_file", help="Path to tab-delimited Kintek txt file")
    parser.add_argument("--shared-koff", action="store_true", help="Use one shared koff across all states")
    parser.add_argument("--out", default="fit_results/sequential_fit_summary.json", help="Output JSON path")
    args = parser.parse_args()

    df = load_kintek_txt(args.data_file)
    result = fit_dataset(df, shared_koff=args.shared_koff)
    save_fit(result, args.out)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
