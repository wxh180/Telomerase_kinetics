# deposition_scripts

Script-only deposition package for **Mathematical modeling of telomerase repeat-addition kinetics**.

## Included files

### Core modeling and fitting
- `src/model.py` — coarse repeat-level telomerase kinetics model
- `src/fit_experimental_data.py` — fitting workflow for experimental ladder/time-course data

### Extended models
- `src/sequential_state_model.py` — reduced sequential-state observable model
- `src/validate_sequential_model.py` — validation/diagnostic workflow for sequential fits
- `src/nucleotide_step_model.py` — nucleotide-step mechanistic model scaffold

### Archive-scale fitting scripts
- `batch_fit_timecourse_txts.py` — conservative first-pass batch fitting of clearly parseable time-course text datasets
- `refine_batch_fits.py` — refined fitting pass for previously successful datasets

## Notes
- This package includes **scripts only**.
- Raw data, fitted outputs, manuscript files, and supplementary figures are intentionally excluded from this deposition subset.
- The scripts were prepared from the `Telomerase_kinetics` project workspace for code deposition and reuse.
