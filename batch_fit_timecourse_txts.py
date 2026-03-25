#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import List

from fit_experimental_data import normalize_dataset, grid_search_fit

ROOT = Path('/home/wei/.openclaw/workspace/Telomerase_kinetics')
RAW_ROOT = ROOT / 'manuscript' / 'data_dropbox'
OUTROOT = ROOT / 'batch_fit_results'
FITROOT = OUTROOT / 'fits'


def is_candidate_raw(path: Path) -> bool:
    parts = path.parts
    if 'extracted' in parts or 'presentations' in parts:
        return False
    return path.suffix.lower() in {'.txt', '.xls', '.xlsx', '.csv', '.tsv'}


def sniff_txt_type(path: Path):
    try:
        with path.open('r', errors='ignore') as f:
            lines = [next(f).rstrip('\n\r') for _ in range(3)]
    except (StopIteration, FileNotFoundError):
        pass
    except Exception as e:
        return 'unreadable', str(e)
    if not lines:
        return 'empty', 'empty file'
    first = lines[0].strip()
    if re.match(r'^Time\tS\d+(\tS\d+)+\s*$', first):
        return 'timecourse_txt', 'Time + S1..Sn tab-delimited matrix'
    if re.match(r'^Time[,\t]', first):
        return 'maybe_timecourse', 'Time-delimited table with unexpected headers'
    if re.match(r'^[\d.Ee+\-]+(\s+|\t)[\d.Ee+\-]+', first):
        return 'numeric_matrix_no_header', 'numeric matrix without clear header or condition mapping'
    return 'other_txt', first[:120]


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


def fit_one(path: Path):
    header, times, raw_rows = parse_timecourse_txt(path)
    obs_rows = normalize_dataset(raw_rows)
    n_repeats = len(header) - 2
    fit = grid_search_fit(
        obs_times=times,
        obs_rows=obs_rows,
        n_repeats=n_repeats,
        dt=0.1,
        initial_bound=1.0,
        k_trans_min=0.01,
        k_trans_max=1.0,
        k_trans_step=0.01,
        k_off_min=0.01,
        k_off_max=0.5,
        k_off_step=0.01,
    )
    flat_obs = [x for row in obs_rows for x in row]
    flat_pred = [x for row in fit['predicted_normalized'] for x in row]
    n = len(flat_obs)
    rmse = math.sqrt(sum((a-b)**2 for a,b in zip(flat_obs, flat_pred))/n)
    mean_obs = sum(flat_obs) / n
    ss_tot = sum((x - mean_obs) ** 2 for x in flat_obs)
    r2 = 1 - (fit['sse'] / ss_tot) if ss_tot > 0 else float('nan')
    outdir = FITROOT / path.relative_to(RAW_ROOT).with_suffix('')
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'observed_normalized.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Time'] + header[1:])
        for t, row in zip(times, obs_rows):
            w.writerow([t] + row)
    with (outdir / 'predicted_normalized.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Time'] + header[1:])
        for t, row in zip(times, fit['predicted_normalized']):
            w.writerow([t] + row)
    summary = {
        'input_file': str(path),
        'condition': infer_condition(path),
        'replicate': infer_replicate(path),
        'model': 'coarse_grained_sequential_repeat_addition',
        'best_fit': {
            'k_trans': fit['k_trans'],
            'k_off': fit['k_off'],
            'processivity': fit['processivity'],
            'sse': fit['sse'],
            'rmse': rmse,
            'r2_flat': r2,
        },
        'n_times': len(times),
        'n_species': len(header) - 1,
    }
    (outdir / 'fit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    FITROOT.mkdir(parents=True, exist_ok=True)
    records = []
    raw_files = sorted([p for p in RAW_ROOT.rglob('*') if p.is_file() and is_candidate_raw(p)])
    for path in raw_files:
        rec = {
            'file_path': str(path.relative_to(ROOT)),
            'condition': infer_condition(path),
            'replicate': infer_replicate(path),
            'model_used': '',
            'fit_status': 'not_fit',
            'k_trans': '',
            'k_off': '',
            'processivity': '',
            'sse': '',
            'rmse': '',
            'r2_flat': '',
            'fit_quality': '',
            'notes_errors': '',
        }
        suf = path.suffix.lower()
        if suf in {'.xls', '.xlsx'}:
            rec['notes_errors'] = 'Spreadsheet file not parsed in this pass; mapping sheets/lanes to model inputs is ambiguous.'
        elif suf in {'.csv', '.tsv'}:
            rec['notes_errors'] = 'No raw fit-ready CSV/TSV assay inputs identified in archive; file left unfit.'
        elif suf == '.txt':
            kind, note = sniff_txt_type(path)
            if kind == 'timecourse_txt':
                try:
                    summary = fit_one(path)
                    rec.update({
                        'model_used': summary['model'],
                        'fit_status': 'fit',
                        'k_trans': summary['best_fit']['k_trans'],
                        'k_off': summary['best_fit']['k_off'],
                        'processivity': summary['best_fit']['processivity'],
                        'sse': summary['best_fit']['sse'],
                        'rmse': summary['best_fit']['rmse'],
                        'r2_flat': summary['best_fit']['r2_flat'],
                        'fit_quality': f"rmse={summary['best_fit']['rmse']:.4g}; r2={summary['best_fit']['r2_flat']:.4g}",
                        'notes_errors': 'Parsed as tab-delimited Time/S1..Sn time course and fit successfully.',
                    })
                except Exception as e:
                    rec['model_used'] = 'coarse_grained_sequential_repeat_addition'
                    rec['notes_errors'] = f'timecourse parse/fit failed: {e}'
            else:
                rec['notes_errors'] = f'Not confidently fit-ready: {note}'
        records.append(rec)

    tsv_path = OUTROOT / 'master_summary.tsv'
    csv_path = OUTROOT / 'master_summary.csv'
    fields = list(records[0].keys()) if records else []
    for outp, delim in [(tsv_path, '\t'), (csv_path, ',')]:
        with outp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter=delim)
            w.writeheader()
            w.writerows(records)

    fit_count = sum(r['fit_status'] == 'fit' for r in records)
    md = [
        '# Batch fit summary',
        '',
        f'- Total candidate raw files reviewed: {len(records)}',
        f'- Successfully fit: {fit_count}',
        f'- Not fit: {len(records) - fit_count}',
        '- Inclusion rule for fitting: only plain-text files confidently recognized as tab-delimited `Time` + `S1..Sn` time-course matrices.',
        '- Spreadsheets and ambiguous text files were left unfit on purpose.',
        '',
        '## Successfully fit datasets',
        '',
    ]
    for r in records:
        if r['fit_status'] == 'fit':
            md.append(f"- `{r['file_path']}` | condition={r['condition']} | replicate={r['replicate'] or 'NA'} | k_trans={r['k_trans']} | k_off={r['k_off']} | processivity={r['processivity']} | {r['fit_quality']}")
    (OUTROOT / 'README.md').write_text('\n'.join(md) + '\n')

if __name__ == '__main__':
    main()
