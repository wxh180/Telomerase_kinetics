#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from fit_experimental_data import normalize_dataset, simulate_at_observation_times

ROOT = Path('/home/wei/.openclaw/workspace/Telomerase_kinetics')
RAW_ROOT = ROOT / 'manuscript' / 'data_dropbox'
SOURCE_SUMMARY = ROOT / 'batch_fit_results' / 'master_summary.tsv'
OUTROOT = ROOT / 'batch_fit_results_refined'
FITROOT = OUTROOT / 'fits'

K_TRANS_BOUNDS = (1e-3, 5.0)
K_OFF_BOUNDS = (1e-4, 2.0)
LOCAL_STARTS = 10
BOUND_TOL_FRAC = 0.01
DT = 0.1
INITIAL_BOUND = 1.0


@dataclass
class FitResult:
    k_trans: float
    k_off: float
    processivity: float
    sse: float
    rmse: float
    r2_flat: float
    aic: float
    bic: float
    n_obs: int
    hit_k_trans_bound: bool
    hit_k_off_bound: bool
    hit_any_bound: bool
    optimizer_success: bool
    optimizer_message: str
    coarse_seed_rank: int



def parse_timecourse_txt(path: Path):
    with path.open() as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        times = []
        rows = []
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            times.append(float(row[0]))
            rows.append([float(x) for x in row[1:] if x != ''])
    if not times or not rows:
        raise ValueError('no numeric rows')
    ncols = len(rows[0])
    if any(len(r) != ncols for r in rows):
        raise ValueError('ragged rows')
    return header, times, rows


def infer_condition(path: Path) -> str:
    rel = path.relative_to(RAW_ROOT)
    parts = rel.parts
    for p in parts[::-1]:
        if p.lower().endswith('.txt'):
            continue
        if 'Kintek fitting' in p:
            continue
        return p
    return parts[0]


def infer_replicate(path: Path) -> str:
    m = re.search(r'(?:^|\b)(Rep(?:licate)?\s*\d+)', str(path), re.I)
    if m:
        return m.group(1)
    return ''


def flatten(rows: Sequence[Sequence[float]]) -> List[float]:
    return [x for row in rows for x in row]


def score_dataset(obs_rows: Sequence[Sequence[float]], pred_rows: Sequence[Sequence[float]]) -> Dict[str, float]:
    flat_obs = flatten(obs_rows)
    flat_pred = flatten(pred_rows)
    n = len(flat_obs)
    residuals = [a - b for a, b in zip(flat_obs, flat_pred)]
    sse = sum(r * r for r in residuals)
    rmse = math.sqrt(sse / n)
    mean_obs = sum(flat_obs) / n
    ss_tot = sum((x - mean_obs) ** 2 for x in flat_obs)
    r2 = 1 - (sse / ss_tot) if ss_tot > 0 else float('nan')
    # Gaussian error AIC/BIC proxy using RSS/n; safe epsilon for perfect fits.
    sigma2 = max(sse / n, 1e-12)
    k = 2
    aic = n * math.log(sigma2) + 2 * k
    bic = n * math.log(sigma2) + k * math.log(n)
    return {
        'sse': sse,
        'rmse': rmse,
        'r2_flat': r2,
        'aic': aic,
        'bic': bic,
        'n_obs': n,
    }


def simulate_norm(times: Sequence[float], n_repeats: int, k_trans: float, k_off: float) -> List[List[float]]:
    pred = simulate_at_observation_times(
        obs_times=times,
        k_trans=k_trans,
        k_off=k_off,
        n_repeats=n_repeats,
        dt=DT,
        initial_bound=INITIAL_BOUND,
    )
    return normalize_dataset(pred)


def objective_log_params(log_params: np.ndarray, times: Sequence[float], obs_rows: Sequence[Sequence[float]], n_repeats: int) -> float:
    k_trans = float(np.exp(log_params[0]))
    k_off = float(np.exp(log_params[1]))
    pred = simulate_norm(times, n_repeats, k_trans, k_off)
    return score_dataset(obs_rows, pred)['sse']


def adaptive_seed_grid(seed_kt: float, seed_ko: float) -> List[Tuple[float, float]]:
    points = set()
    # broad + progressively tighter windows in log-space around first-pass seed
    for span in [2.5, 1.5, 0.8, 0.4, 0.2]:
        kt_vals = np.exp(np.linspace(math.log(max(K_TRANS_BOUNDS[0], seed_kt / math.exp(span))), math.log(min(K_TRANS_BOUNDS[1], seed_kt * math.exp(span))), 7))
        ko_vals = np.exp(np.linspace(math.log(max(K_OFF_BOUNDS[0], seed_ko / math.exp(span))), math.log(min(K_OFF_BOUNDS[1], seed_ko * math.exp(span))), 7))
        for kt in kt_vals:
            for ko in ko_vals:
                points.add((round(float(kt), 10), round(float(ko), 10)))
    # explicit boundary checks in case first pass sat on the original ceiling
    for kt in [K_TRANS_BOUNDS[0], 0.01, 0.1, 0.5, 1.0, 2.0, K_TRANS_BOUNDS[1]]:
        for ko in [K_OFF_BOUNDS[0], 0.01, 0.05, 0.1, 0.5, 1.0, K_OFF_BOUNDS[1]]:
            if K_TRANS_BOUNDS[0] <= kt <= K_TRANS_BOUNDS[1] and K_OFF_BOUNDS[0] <= ko <= K_OFF_BOUNDS[1]:
                points.add((float(kt), float(ko)))
    return sorted(points)


def refine_dataset(times: Sequence[float], obs_rows: Sequence[Sequence[float]], n_repeats: int, seed_points: List[Tuple[float, float]]):
    coarse_candidates = []
    for seed_kt, seed_ko in seed_points:
        for k_trans, k_off in adaptive_seed_grid(seed_kt, seed_ko):
            pred = simulate_norm(times, n_repeats, float(k_trans), float(k_off))
            sse = score_dataset(obs_rows, pred)['sse']
            coarse_candidates.append((sse, float(k_trans), float(k_off)))
    coarse_candidates.sort(key=lambda x: x[0])

    starts = []
    seen = set()
    for _, kt, ko in coarse_candidates[:LOCAL_STARTS]:
        key = (round(kt, 10), round(ko, 10))
        if key not in seen:
            starts.append((kt, ko))
            seen.add(key)
    for kt, ko in seed_points:
        key = (round(kt, 10), round(ko, 10))
        if key not in seen:
            starts.append((kt, ko))
            seen.add(key)

    best = None
    best_pred = None
    best_meta = None
    bounds_log = [(math.log(K_TRANS_BOUNDS[0]), math.log(K_TRANS_BOUNDS[1])), (math.log(K_OFF_BOUNDS[0]), math.log(K_OFF_BOUNDS[1]))]

    for rank, (seed_kt, seed_ko) in enumerate(starts, start=1):
        res = minimize(
            objective_log_params,
            x0=np.array([math.log(seed_kt), math.log(seed_ko)]),
            args=(times, obs_rows, n_repeats),
            method='L-BFGS-B',
            bounds=bounds_log,
        )
        k_trans = float(np.exp(res.x[0]))
        k_off = float(np.exp(res.x[1]))
        pred = simulate_norm(times, n_repeats, k_trans, k_off)
        metrics = score_dataset(obs_rows, pred)
        candidate = FitResult(
            k_trans=k_trans,
            k_off=k_off,
            processivity=k_trans / (k_trans + k_off) if (k_trans + k_off) > 0 else 0.0,
            sse=metrics['sse'],
            rmse=metrics['rmse'],
            r2_flat=metrics['r2_flat'],
            aic=metrics['aic'],
            bic=metrics['bic'],
            n_obs=metrics['n_obs'],
            hit_k_trans_bound=min(
                abs(k_trans - K_TRANS_BOUNDS[0]) / K_TRANS_BOUNDS[0],
                abs(k_trans - K_TRANS_BOUNDS[1]) / K_TRANS_BOUNDS[1],
            ) <= BOUND_TOL_FRAC,
            hit_k_off_bound=min(
                abs(k_off - K_OFF_BOUNDS[0]) / K_OFF_BOUNDS[0],
                abs(k_off - K_OFF_BOUNDS[1]) / K_OFF_BOUNDS[1],
            ) <= BOUND_TOL_FRAC,
            hit_any_bound=False,
            optimizer_success=bool(res.success),
            optimizer_message=str(res.message),
            coarse_seed_rank=rank,
        )
        candidate.hit_any_bound = candidate.hit_k_trans_bound or candidate.hit_k_off_bound
        if best is None or candidate.sse < best.sse:
            best = candidate
            best_pred = pred
            best_meta = {
                'start_k_trans': seed_kt,
                'start_k_off': seed_ko,
                'optimizer': {
                    'success': bool(res.success),
                    'message': str(res.message),
                    'nit': int(getattr(res, 'nit', -1)),
                    'nfev': int(getattr(res, 'nfev', -1)),
                },
            }
    if best is None:
        raise RuntimeError('refinement produced no candidate result')
    return coarse_candidates[:LOCAL_STARTS], best, best_pred, best_meta


def write_csv(path: Path, header: Sequence[str], times: Sequence[float], rows: Sequence[Sequence[float]]) -> None:
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Time'] + list(header[1:]))
        for t, row in zip(times, rows):
            w.writerow([t] + list(row))


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    FITROOT.mkdir(parents=True, exist_ok=True)
    with SOURCE_SUMMARY.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    targets = [r for r in rows if r['fit_status'] == 'fit']

    records = []
    for src in targets:
        rel_path = src['file_path']
        raw_path = ROOT / rel_path
        rec = {
            'dataset': rel_path,
            'condition': src['condition'],
            'replicate': src['replicate'],
            'model': 'coarse_grained_sequential_repeat_addition_refined',
            'refined_fit_status': 'failed',
            'k_trans': '',
            'k_off': '',
            'processivity': '',
            'sse': '',
            'rmse': '',
            'r2_flat': '',
            'aic': '',
            'bic': '',
            'notes': '',
            'hit_bounds': '',
            'hit_k_trans_bound': '',
            'hit_k_off_bound': '',
        }
        try:
            header, times, raw_rows = parse_timecourse_txt(raw_path)
            obs_rows = normalize_dataset(raw_rows)
            n_repeats = len(header) - 2
            seed_points = [
                (float(src['k_trans']), float(src['k_off'])),
            ]
            coarse_top, best, pred, best_meta = refine_dataset(times, obs_rows, n_repeats, seed_points)
            first_pred = simulate_norm(times, n_repeats, float(src['k_trans']), float(src['k_off']))
            first_metrics = score_dataset(obs_rows, first_pred)
            reverted_to_first_pass = False
            if first_metrics['sse'] < best.sse:
                reverted_to_first_pass = True
                best = FitResult(
                    k_trans=float(src['k_trans']),
                    k_off=float(src['k_off']),
                    processivity=float(src['processivity']),
                    sse=float(src['sse']),
                    rmse=float(src['rmse']),
                    r2_flat=float(src['r2_flat']),
                    aic=first_metrics['aic'],
                    bic=first_metrics['bic'],
                    n_obs=first_metrics['n_obs'],
                    hit_k_trans_bound=min(
                        abs(float(src['k_trans']) - K_TRANS_BOUNDS[0]) / K_TRANS_BOUNDS[0],
                        abs(float(src['k_trans']) - K_TRANS_BOUNDS[1]) / K_TRANS_BOUNDS[1],
                    ) <= BOUND_TOL_FRAC,
                    hit_k_off_bound=min(
                        abs(float(src['k_off']) - K_OFF_BOUNDS[0]) / K_OFF_BOUNDS[0],
                        abs(float(src['k_off']) - K_OFF_BOUNDS[1]) / K_OFF_BOUNDS[1],
                    ) <= BOUND_TOL_FRAC,
                    hit_any_bound=False,
                    optimizer_success=True,
                    optimizer_message='reverted to original first-pass optimum because refinement increased SSE',
                    coarse_seed_rank=0,
                )
                best.hit_any_bound = best.hit_k_trans_bound or best.hit_k_off_bound
                pred = first_pred
                best_meta = {
                    'reverted_to_first_pass': True,
                    'first_pass_sse': first_metrics['sse'],
                }
            outdir = FITROOT / raw_path.relative_to(RAW_ROOT).with_suffix('')
            outdir.mkdir(parents=True, exist_ok=True)
            write_csv(outdir / 'observed_normalized.csv', header, times, obs_rows)
            write_csv(outdir / 'predicted_normalized.csv', header, times, pred)
            summary = {
                'input_file': str(raw_path),
                'condition': infer_condition(raw_path),
                'replicate': infer_replicate(raw_path),
                'model': 'coarse_grained_sequential_repeat_addition',
                'refinement_method': {
                    'global_search': {
                        'k_trans_bounds': K_TRANS_BOUNDS,
                        'k_off_bounds': K_OFF_BOUNDS,
                        'adaptive_seed_grid_method': 'multi-scale log-space windows around first-pass seed plus explicit boundary probes',
                        'top_candidates_used_for_local_optimization': LOCAL_STARTS,
                    },
                    'local_optimization': 'scipy.optimize.minimize(method=L-BFGS-B) in log-parameter space',
                    'simulation_dt_min': DT,
                    'initial_bound': INITIAL_BOUND,
                },
                'first_pass_seed': {
                    'k_trans': float(src['k_trans']),
                    'k_off': float(src['k_off']),
                    'processivity': float(src['processivity']),
                    'sse': float(src['sse']),
                    'rmse': float(src['rmse']),
                    'r2_flat': float(src['r2_flat']),
                },
                'coarse_top_candidates': [
                    {'rank': i + 1, 'sse': sse, 'k_trans': kt, 'k_off': ko}
                    for i, (sse, kt, ko) in enumerate(coarse_top)
                ],
                'best_fit': asdict(best),
                'best_run_metadata': best_meta,
            }
            (outdir / 'fit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            rec.update({
                'refined_fit_status': 'fit',
                'k_trans': best.k_trans,
                'k_off': best.k_off,
                'processivity': best.processivity,
                'sse': best.sse,
                'rmse': best.rmse,
                'r2_flat': best.r2_flat,
                'aic': best.aic,
                'bic': best.bic,
                'notes': f"Refined from first-pass seed k_trans={src['k_trans']}, k_off={src['k_off']}; optimizer_success={best.optimizer_success}; message={best.optimizer_message}",
                'hit_bounds': str(best.hit_any_bound),
                'hit_k_trans_bound': str(best.hit_k_trans_bound),
                'hit_k_off_bound': str(best.hit_k_off_bound),
            })
        except Exception as e:
            rec['notes'] = f'refinement failed: {e}'
        records.append(rec)

    fields = [
        'dataset', 'condition', 'replicate', 'model', 'refined_fit_status',
        'k_trans', 'k_off', 'processivity', 'sse', 'rmse', 'r2_flat', 'aic', 'bic',
        'notes', 'hit_bounds', 'hit_k_trans_bound', 'hit_k_off_bound'
    ]
    with (OUTROOT / 'refined_master_summary.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        w.writerows(records)
    with (OUTROOT / 'refined_master_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    fit_count = sum(r['refined_fit_status'] == 'fit' for r in records)
    hit_count = sum(r['hit_bounds'] == 'True' for r in records)
    lines = [
        '# Refined batch fit summary',
        '',
        f'- Source successful fits refined: {len(records)}',
        f'- Refined successfully: {fit_count}',
        f'- Failures: {len(records) - fit_count}',
        f'- Fits at/near parameter bounds: {hit_count}',
        f'- Parameter bounds: k_trans in [{K_TRANS_BOUNDS[0]}, {K_TRANS_BOUNDS[1]}], k_off in [{K_OFF_BOUNDS[0]}, {K_OFF_BOUNDS[1]}]',
        f'- Refinement method: broad log-spaced grid screening followed by L-BFGS-B local optimization in log-parameter space.',
        '',
        '## Dataset summaries',
        '',
    ]
    for r in records:
        lines.append(
            f"- `{r['dataset']}` | status={r['refined_fit_status']} | k_trans={r['k_trans']} | k_off={r['k_off']} | processivity={r['processivity']} | rmse={r['rmse']} | r2={r['r2_flat']} | hit_bounds={r['hit_bounds']}"
        )
    (OUTROOT / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
