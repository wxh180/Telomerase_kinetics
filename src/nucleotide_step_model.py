"""
Second-generation nucleotide-step telomerase kinetics model skeleton.

Goal:
Move beyond the simple sequential S1->S2->... chain and closer to the manuscript's
intended biology: individual nucleotide addition steps within each telomeric repeat,
followed by a translocation step to the next repeat.

This is an explicit state model skeleton meant to be the bridge to reproducing
Kintek-style fits in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import csv
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass
class NucleotideStepModelConfig:
    n_repeats: int = 4
    n_nt_per_repeat: int = 6
    shared_koff: bool = True
    shared_repeat_pattern: bool = True


@dataclass
class NucleotideStepFitResult:
    n_repeats: int
    n_nt_per_repeat: int
    k_nt: list
    k_trans: float
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


def state_index(r: int, i: int, n_nt: int) -> int:
    """State B_(r,i), where i=0..n_nt and r=0..n_repeats-1."""
    return r * (n_nt + 1) + i


def build_state_labels(n_repeats: int, n_nt: int) -> List[str]:
    labels = []
    for r in range(n_repeats):
        for i in range(n_nt + 1):
            labels.append(f"B_{r}_{i}")
    return labels


def rhs(t: float, y: np.ndarray, k_nt: np.ndarray, k_trans: float, koff: np.ndarray, n_repeats: int, n_nt: int) -> np.ndarray:
    n_states = n_repeats * (n_nt + 1)
    B = y[:n_states]
    D = y[n_states:]
    dB = np.zeros_like(B)
    dD = np.zeros_like(D)

    for r in range(n_repeats):
        for i in range(n_nt + 1):
            idx = state_index(r, i, n_nt)
            loss = koff[idx] * B[idx]
            dD[idx] += loss
            dB[idx] -= loss

            # nucleotide addition steps within repeat
            if i < n_nt:
                rate = k_nt[i]
                dB[idx] -= rate * B[idx]
                dB[state_index(r, i + 1, n_nt)] += rate * B[idx]
            else:
                # translocation from completed repeat state B_(r,n_nt)
                if r < n_repeats - 1:
                    dB[idx] -= k_trans * B[idx]
                    dB[state_index(r + 1, 0, n_nt)] += k_trans * B[idx]
                # if final repeat, material stays in final completed state unless dissociated

    return np.concatenate([dB, dD])


def simulate(times: np.ndarray, k_nt: np.ndarray, k_trans: float, koff: np.ndarray, n_repeats: int, n_nt: int) -> Tuple[np.ndarray, np.ndarray]:
    n_states = n_repeats * (n_nt + 1)
    y0 = np.zeros(2 * n_states)
    y0[state_index(0, 0, n_nt)] = 1.0
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, k_nt, k_trans, koff, n_repeats, n_nt),
        t_span=(float(times.min()), float(times.max())),
        y0=y0,
        t_eval=np.array(times, dtype=float),
        method='LSODA',
        rtol=1e-8,
        atol=1e-10,
    )
    y = sol.y.T
    return y[:, :n_states], y[:, n_states:]


def observable_mapping(B: np.ndarray, D: np.ndarray, n_repeats: int, n_nt: int, n_obs: int) -> np.ndarray:
    """
    Placeholder mapping from biochemical states to observables S1..Sn.

    Current provisional assumption:
    each observable corresponds to one extension-position-like state in order,
    prioritizing dissociated species from successive states.

    This function MUST be refined once we map Kintek observables more precisely.
    """
    total_states = B.shape[1]
    candidate = D[:, :min(n_obs, total_states)]
    if candidate.shape[1] < n_obs:
        pad = np.zeros((candidate.shape[0], n_obs - candidate.shape[1]))
        candidate = np.hstack([candidate, pad])
    return candidate


def fit_dataset(df: dict, config: NucleotideStepModelConfig) -> NucleotideStepFitResult:
    use_columns = [c for c in df.keys() if c != 'Time']
    times = np.array(df['Time'], dtype=float)
    data = np.column_stack([np.array(df[c], dtype=float) for c in use_columns])
    n_obs = data.shape[1]

    n_states = config.n_repeats * (config.n_nt_per_repeat + 1)
    k_nt0 = np.full(config.n_nt_per_repeat, 0.2)
    k_trans0 = np.array([0.1])

    if config.shared_koff:
        x0 = np.concatenate([k_nt0, k_trans0, np.array([0.02])])
        lb = np.concatenate([np.full(config.n_nt_per_repeat, 1e-6), np.array([1e-6, 0.0])])
        ub = np.concatenate([np.full(config.n_nt_per_repeat, 10.0), np.array([10.0, 10.0])])
    else:
        koff0 = np.full(n_states, 0.02)
        x0 = np.concatenate([k_nt0, k_trans0, koff0])
        lb = np.concatenate([np.full(config.n_nt_per_repeat, 1e-6), np.array([1e-6]), np.zeros(n_states)])
        ub = np.concatenate([np.full(config.n_nt_per_repeat, 10.0), np.array([10.0]), np.full(n_states, 10.0)])

    def unpack(x: np.ndarray):
        k_nt = x[:config.n_nt_per_repeat]
        k_trans = x[config.n_nt_per_repeat]
        if config.shared_koff:
            koff = np.full(n_states, x[config.n_nt_per_repeat + 1])
        else:
            koff = x[config.n_nt_per_repeat + 1:]
        return k_nt, float(k_trans), koff

    def residuals(x: np.ndarray) -> np.ndarray:
        k_nt, k_trans, koff = unpack(x)
        B, D = simulate(times, k_nt, k_trans, koff, config.n_repeats, config.n_nt_per_repeat)
        pred = observable_mapping(B, D, config.n_repeats, config.n_nt_per_repeat, n_obs)
        return (pred - data).ravel()

    res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=5000)
    k_nt, k_trans, koff = unpack(res.x)
    sse = float(np.sum(res.fun ** 2))
    return NucleotideStepFitResult(
        n_repeats=config.n_repeats,
        n_nt_per_repeat=config.n_nt_per_repeat,
        k_nt=k_nt.tolist(),
        k_trans=k_trans,
        k_off=koff.tolist(),
        sse=sse,
        success=bool(res.success),
        message=str(res.message),
    )


def save_fit(result: NucleotideStepFitResult, out_json: str | Path) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.__dict__, indent=2))


if __name__ == '__main__':
    # simple placeholder demo on one dataset
    data_file = Path('/home/wei/.openclaw/workspace/Telomerase_kinetics/manuscript/data_dropbox/extracted/2019-6-24 PC 1 dATP/Rep 1/6-24-2019 dATP + control 1uM.txt')
    df = load_kintek_txt(data_file)
    config = NucleotideStepModelConfig(n_repeats=4, n_nt_per_repeat=6, shared_koff=True)
    result = fit_dataset(df, config)
    save_fit(result, '/home/wei/.openclaw/workspace/Telomerase_kinetics/fit_results/nucleotide_step_fit_skeleton.json')
    print(json.dumps(result.__dict__, indent=2))
