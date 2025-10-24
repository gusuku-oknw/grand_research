# SIS Image Sharing Package

This repository packages the original SIS demos into an installable Python module named `sis_image`. It bundles reusable implementations of the following components:

- 64-bit perceptual hashing helpers.
- Shamir secret sharing for hash bytes and full RGB images.
- Search indices that use banded HMAC tokens to preselect candidates before reconstructing secrets.
- A persistent image store capable of reconstructing full-resolution assets from any `k` of `n` servers.

## Installation

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

For lightweight usage (e.g. running the experiment scripts directly), the core dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Command Line Demos

Once installed, run the CLI to access the demos:

```bash
sis-image selective-demo --images_dir data/
sis-image search-demo --images_dir data/ --reconstruct_top 2
sis-image image-store-demo --images_dir data/
sis-image secure-demo --images_dir data/
```

Each original script under `SIS_image/` now delegates to these entry points, keeping the legacy workflow intact while allowing production usage via the package.

## Stage-A/B/C Pipeline Overview

The searchable SIS workflow narrows candidates in three stages before reconstructing images:

- **Stage-A - Band Token Preselection**: During ingestion the index splits the 64-bit pHash into `bands` chunks, issues HMAC tokens per server, and stores image IDs per token bucket (`pHR_SIS/index.py`). At query time `preselect_candidates` tallies token matches and keeps IDs meeting `min_band_votes`, drastically shrinking the candidate set.
- **Stage-B - Partial Share Filtering**: `stage_b_filter` (`scripts/run_search_experiments.py`) pulls only a few bytes from each server's Shamir share, reconstructs an approximate hash, and rejects candidates whose Hamming distance exceeds `max_hamming + margin`, tracking both latency and communication.
- **Stage-C - Selective Reconstruction and Ranking**: Remaining IDs undergo full share recovery via `rank_candidates` (or MPC-based `rank_candidates_secure`) in `pHR_SIS/index.py`, with `SearchableSISWithImageStore` (`pHR_SIS/workflow.py`) optionally rebuilding the top `reconstruct_top` images to deliver final matches.

`scripts/run_search_experiments.py` records timing, candidate counts, and byte usage for each stage so the generated figures (`candidate_reduction.png`, `time_breakdown.png`) and metrics feed directly into the reporting template under `reports/2025_selective_reconstruction_report_template.md`.

## Reproducible Experiments (Local or Colab)

1. **Clone the repository**
   ```bash
   git clone https://github.com/<org>/Grand_Research.git
   cd Grand_Research
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare COCO derivatives**
   ```
   python scripts/prepare_coco.py \
       --coco_dir data/coco2017/val2017 \
       --output_dir data/coco2017_derivatives/val2017 \
       --mapping_json data/coco2017_derivatives/derivative_mapping.json \
       --profile medium \
       --variant_scope all \
       --max_images 5000
   ```
   - Existing derivatives and mappings are reused automatically; add `--force` to rebuild from scratch.
   - Switch difficulty with `--profile`, and use `--variant_scope original_only` or `--include_transforms watermark_timestamp` when you want a lighter validation set.
   - Combine `--exclude_transforms rotate_plus30_black` and `--list_transforms` to prune or inspect the catalog quickly.
   - Install `pillow-avif-plugin` for AVIF variants; pass `--no_progress` to silence logs, and lower `--max_images` on Colab to save runtime and storage.

4. **Run staged SIS experiments**
   ```
   PYTHONPATH=. python scripts/run_search_experiments.py \
       --mapping_json data/coco2017_derivatives/derivative_mapping.json \
       --output_dir evaluation/results/coco_val2017_stageABC \
       --work_dir evaluation/artifacts/coco_val2017_stageABC
   ```
   - If `metrics.csv` already exists, rerun with `--force` to refresh the results.

5. **Produce Matplotlib figures from the metrics**
   ```
   python -m evaluation.plotting \
       evaluation/results/coco_val2017_stageABC/metrics.csv \
       --output_dir evaluation/figures/coco_val2017_stageABC
   ```

### Running on Google Colab

- Upload or mount the COCO `val2017` images and annotations under `/content/data/coco2017/val2017`.
- Install dependencies:
  ```python
  !pip install -r requirements.txt
  import sys, pathlib
  sys.path.append(str(pathlib.Path('.').resolve()))
  ```
- Execute the same commands as above (use `!python ...`), optionally lowering `--max_images` / `--max_queries` to fit Colab’s storage and time limits.
- Persist large outputs (e.g. `derivative_mapping.json`, figure PNGs) by copying them to Google Drive before ending the session.
