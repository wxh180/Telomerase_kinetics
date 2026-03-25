#!/usr/bin/env python3
"""Fit coarse-grained telomerase ladder model to experimental data.

Input CSV format:
    time_min,repeat_0,repeat_1,repeat_2,...
    0, ...
    5, ...
    10, ...

Values can be raw intensities or already-normalized values. By default each lane is
normalized to unit sum before fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from model import simulate_ladder


def read_experimental_csv(path: Path) -> Tuple[List[float], List[List[float]], List[str]]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("Expected time column plus at least one repeat column")
        times: List[float] = []
        rows: List[List[float]] = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            times.append(float(row[0]))
            rows.append([float(x) for x in row[1:]])
    return times, rows, header[1:]


def normalize_lane(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [v / total for v in values]


def normalize_dataset(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    return [normalize_lane(r) for r in rows]


def simulate_at_observation_times(
    obs_times: Sequence[float],
    k_trans: float,
    k_off: float,
    n_repeats: int,
    dt: float,
    initial_bound: float,
) -> List[List[float]]:
    t_end = max(obs_times)
    sim_times, _bound, diss = simulate_ladder(
        k_trans=k_trans,
        k_off=k_off,
        n_repeats=n_repeats,
        t_end=t_end,
        dt=dt,
        initial_bound=initial_bound,
    )
    out: List[List[float]] = []
    for t in obs_times:
        idx = min(range(len(sim_times)), key=lambda i: abs(sim_times[i] - t))
        out.append(diss[idx][:])
    return out


def sse(obs: Sequence[Sequence[float]], pred: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for o_row, p_row in zip(obs, pred):
        for o, p in zip(o_row, p_row):
            d = o - p
            total += d * d
    return total


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    x = start
    while x <= stop + 1e-12:
        yield round(x, 12)
        x += step


def grid_search_fit(
    obs_times: Sequence[float],
    obs_rows: Sequence[Sequence[float]],
    n_repeats: int,
    dt: float,
    initial_bound: float,
    k_trans_min: float,
    k_trans_max: float,
    k_trans_step: float,
    k_off_min: float,
    k_off_max: float,
    k_off_step: float,
) -> dict:
    best = None
    for k_trans in frange(k_trans_min, k_trans_max, k_trans_step):
        for k_off in frange(k_off_min, k_off_max, k_off_step):
            pred = simulate_at_observation_times(
                obs_times=obs_times,
                k_trans=k_trans,
                k_off=k_off,
                n_repeats=n_repeats,
                dt=dt,
                initial_bound=initial_bound,
            )
            pred_norm = normalize_dataset(pred)
            score = sse(obs_rows, pred_norm)
            if best is None or score < best["sse"]:
                best = {
                    "k_trans": k_trans,
                    "k_off": k_off,
                    "sse": score,
                    "predicted_normalized": pred_norm,
                    "processivity": k_trans / (k_trans + k_off) if (k_trans + k_off) > 0 else 0.0,
                }
    if best is None:
        raise RuntimeError("Grid search failed")
    return best


def write_predicted_csv(path: Path, times: Sequence[float], rows: Sequence[Sequence[float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_min"] + [f"repeat_{i}" for i in range(len(rows[0]))])
        for t, row in zip(times, rows):
            writer.writerow([t] + list(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit telomerase ladder model to experimental data")
    parser.add_argument("data", type=Path, help="experimental CSV file")
    parser.add_argument("--dt", type=float, default=0.1, help="simulation time step in minutes")
    parser.add_argument("--initial-bound", type=float, default=1.0, help="initial bound fraction")
    parser.add_argument("--k-trans-min", type=float, default=0.01)
    parser.add_argument("--k-trans-max", type=float, default=1.00)
    parser.add_argument("--k-trans-step", type=float, default=0.01)
    parser.add_argument("--k-off-min", type=float, default=0.01)
    parser.add_argument("--k-off-max", type=float, default=0.50)
    parser.add_argument("--k-off-step", type=float, default=0.01)
    parser.add_argument("--outdir", type=Path, default=Path("fit_results"))
    args = parser.parse_args()

    times, raw_rows, labels = read_experimental_csv(args.data)
    obs_rows = normalize_dataset(raw_rows)
    n_repeats = len(labels) - 1

    fit = grid_search_fit(
        obs_times=times,
        obs_rows=obs_rows,
        n_repeats=n_repeats,
        dt=args.dt,
        initial_bound=args.initial_bound,
        k_trans_min=args.k_trans_min,
        k_trans_max=args.k_trans_max,
        k_trans_step=args.k_trans_step,
        k_off_min=args.k_off_min,
        k_off_max=args.k_off_max,
        k_off_step=args.k_off_step,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_file": str(args.data),
        "best_fit": {
            "k_trans": fit["k_trans"],
            "k_off": fit["k_off"],
            "processivity": fit["processivity"],
            "sse": fit["sse"],
        },
        "grid": {
            "k_trans": [args.k_trans_min, args.k_trans_max, args.k_trans_step],
            "k_off": [args.k_off_min, args.k_off_max, args.k_off_step],
        },
    }
    (args.outdir / "fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_predicted_csv(args.outdir / "predicted_normalized.csv", times, fit["predicted_normalized"])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
