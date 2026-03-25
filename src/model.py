#!/usr/bin/env python3
"""Simple telomerase repeat-addition ladder simulator.

Model:
- Bound state B_r: enzyme remains engaged after completing r repeats
- Dissociated state D_r: product released after r repeats
- Competing rates from each bound state:
    B_r --k_trans--> B_{r+1}
    B_r --k_off-->   D_r

This is a coarse-grained processivity model suited for predicting product
ladder intensities over time.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple


def simulate_ladder(
    k_trans: float = 0.2,
    k_off: float = 0.05,
    n_repeats: int = 12,
    t_end: float = 60.0,
    dt: float = 0.1,
    initial_bound: float = 1.0,
) -> Tuple[List[float], List[List[float]], List[List[float]]]:
    if k_trans < 0 or k_off < 0:
        raise ValueError("Rates must be non-negative")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if t_end <= 0 or dt <= 0:
        raise ValueError("t_end and dt must be > 0")

    steps = int(math.ceil(t_end / dt))
    times: List[float] = [0.0]
    bound_hist: List[List[float]] = [[0.0] * (n_repeats + 1)]
    diss_hist: List[List[float]] = [[0.0] * (n_repeats + 1)]
    bound_hist[0][0] = initial_bound

    for step in range(1, steps + 1):
        prev_b = bound_hist[-1]
        prev_d = diss_hist[-1]
        next_b = prev_b.copy()
        next_d = prev_d.copy()

        for r in range(n_repeats + 1):
            amount = prev_b[r]
            if amount <= 0:
                continue

            trans_flux = min(amount, k_trans * amount * dt)
            remaining = amount - trans_flux
            off_flux = min(remaining, k_off * amount * dt)

            next_b[r] -= (trans_flux + off_flux)
            next_d[r] += off_flux

            if r < n_repeats:
                next_b[r + 1] += trans_flux
            else:
                next_d[r] += trans_flux

        times.append(step * dt)
        bound_hist.append(next_b)
        diss_hist.append(next_d)

    return times, bound_hist, diss_hist


def write_csv(path: Path, times: List[float], rows: List[List[float]], prefix: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_min"] + [f"{prefix}_{i}" for i in range(len(rows[0]))])
        for t, vals in zip(times, rows):
            writer.writerow([f"{t:.4f}"] + [f"{v:.8f}" for v in vals])


def summarize_final(diss_hist: List[List[float]], k_trans: float, k_off: float) -> str:
    final = diss_hist[-1]
    total = sum(final)
    rap = k_trans / (k_trans + k_off) if (k_trans + k_off) > 0 else 0.0
    lines = [
        f"Estimated repeat-addition processivity p = {rap:.4f}",
        f"Final released product total = {total:.4f}",
        "Final released ladder fractions:",
    ]
    if total > 0:
        for i, v in enumerate(final):
            frac = v / total
            lines.append(f"  repeats={i:2d}: amount={v:.6f}, fraction={frac:.4f}")
    else:
        lines.append("  no released product yet")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate telomerase product ladder intensities")
    parser.add_argument("--k-trans", type=float, default=0.2, help="translocation/advance rate per min")
    parser.add_argument("--k-off", type=float, default=0.05, help="dissociation rate per min")
    parser.add_argument("--n-repeats", type=int, default=12, help="max repeat classes tracked")
    parser.add_argument("--t-end", type=float, default=60.0, help="simulation end time in min")
    parser.add_argument("--dt", type=float, default=0.1, help="time step in min")
    parser.add_argument("--initial-bound", type=float, default=1.0, help="initial bound enzyme-substrate fraction")
    parser.add_argument("--outdir", type=Path, default=Path("results"), help="output directory")
    args = parser.parse_args()

    times, bound_hist, diss_hist = simulate_ladder(
        k_trans=args.k_trans,
        k_off=args.k_off,
        n_repeats=args.n_repeats,
        t_end=args.t_end,
        dt=args.dt,
        initial_bound=args.initial_bound,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "bound_states.csv", times, bound_hist, "B")
    write_csv(args.outdir / "released_products.csv", times, diss_hist, "D")

    summary = summarize_final(diss_hist, args.k_trans, args.k_off)
    print(summary)
    (args.outdir / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
