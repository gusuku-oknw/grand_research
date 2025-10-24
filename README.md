# SIS Image Sharing Package

This repository packages the original **Searchable Image Sharing (SIS)** demos into an installable Python module named **`sis_image`**.
It provides modular, reproducible research components for privacy-preserving image search and reconstruction, using perceptual hashing (`pHash`) and Shamir secret sharing.

---

## 🔧 Features

The `sis_image` package implements reusable building blocks:

* **64-bit perceptual hashing (`pHash`)** — DCT-based perceptual similarity hashing.
* **Shamir secret sharing** — secure split & recovery for hash bytes and RGB images.
* **Searchable SIS index** — banded HMAC tokens for secure candidate preselection.
* **Persistent image store** — reconstructs full images from any `k` of `n` servers.
* **Modular experiment runners (`sis_modes/`)** — independently testable Stage-A/B/C pipelines.

---

## 🧩 Installation

```bash
python -m venv .venv
. .venv/bin/activate      # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

For quick experimentation:

```bash
pip install -r requirements.txt
```

---

## 🧪 Command-Line Demos

Once installed, the package provides a unified CLI entry point:

```bash
sis-image selective-demo --images_dir data/
sis-image search-demo --images_dir data/ --reconstruct_top 2
sis-image image-store-demo --images_dir data/
sis-image secure-demo --images_dir data/
```

Each original demo under `SIS_image/` now delegates to these commands for reproducibility.

---

## ⚙️ Modular Experiment Architecture

Experiment scripts under `scripts/` are now **mode-based** and follow a shared interface.

```
scripts/
  run_search_experiments.py      # orchestrator (lightweight)
sis_modes/
  base.py                        # abstract ModeRunner, PhaseTimer, ByteMeter
  plain.py                       # baseline pHash ranking
  sis_naive.py                   # full reconstruction baseline
  sis_selective.py               # Stage-B selective reconstruction
  sis_staged.py                  # staged refinement pipeline
  sis_mpc.py                     # MPC-style fully private variant
  sis_common.py                  # shared helpers (filtering, ranking)
```

### 🧠 Stage-A/B/C Overview

| Stage       | Description                                                                           | Module / Function                                  | Primary Metrics                                   |
| :---------- | :------------------------------------------------------------------------------------ | :------------------------------------------------- | :------------------------------------------------ |
| **Stage-A** | Band-token preselection via HMAC buckets per server. Filters 1-2 orders of magnitude. | `index.preselect_candidates`                       | Candidate count (`n_cand_f1`), bytes (`bytes_f1`) |
| **Stage-B** | Partial share recovery for approximate Hamming filtering.                             | `sis_common.stage_b_filter`                        | Time, communication, candidate reduction          |
| **Stage-C** | Final reconstruction & secure ranking (selective or MPC).                             | `index.rank_candidates` / `rank_candidates_secure` | Precision, recall, latency                        |

All modes conform to the same `ModeRunner` interface in `sis_modes/base.py`, making experiments interchangeable and their results comparable.

---

## 🧬 Reproducible Experiments

### 1. Clone and Setup

```bash
git clone https://github.com/<org>/Grand_Research.git
cd Grand_Research
pip install -r requirements.txt
```

### 2. Prepare COCO Derivatives

```bash
python scripts/prepare_coco.py \
    --coco_dir data/coco2017/val2017 \
    --output_dir data/coco2017_derivatives/val2017 \
    --mapping_json data/coco2017_derivatives/derivative_mapping.json \
    --profile medium \
    --variant_scope all \
    --max_images 5000
```

* Reuse existing derivatives automatically; add `--force` to rebuild.
* Use `--profile light` or lower `--max_images` for Colab or limited machines.
* `--list_transforms` shows available augmentations (JPEG noise, rotation, color shifts, etc.).

### 3. Run Modular SIS Experiments

```bash
PYTHONPATH=. python scripts/run_search_experiments.py \
    --mapping_json data/coco2017_derivatives/derivative_mapping.json \
    --output_dir evaluation/results/coco_val2017_modular \
    --work_dir evaluation/artifacts/coco_val2017_modular \
    --modes plain sis_naive sis_selective sis_staged sis_mpc \
    --max_queries 500 \
    --bands 8 --k 3 --n 5 \
    --force
```

* Each mode logs Stage-wise latency, bytes, and precision metrics.
* Output:

  * `metrics.csv` — consolidated per-query results
  * `security_summary.json` — entropy & leakage analysis
  * `evaluation/figures/*.png` — candidate reduction & latency graphs

### 4. Generate Figures

```bash
python -m evaluation.plotting \
    evaluation/results/coco_val2017_modular/metrics.csv \
    --output_dir evaluation/figures/coco_val2017_modular
```

Produces:

* `candidate_reduction.png`
* `communication_breakdown.png`
* `precision_latency.png`
* `reconstruction_ratio.png`
* `tau_sensitivity.png`

---

## ☁️ Running on Google Colab

1. Mount or upload the COCO `val2017` images under `/content/data/coco2017/val2017`.

2. Install dependencies:

   ```python
   !pip install -r requirements.txt
   import sys, pathlib
   sys.path.append(str(pathlib.Path('.').resolve()))
   ```

3. Run the same commands as above with smaller datasets:

   ```bash
   !python scripts/run_search_experiments.py \
       --max_images 1000 --max_queries 100 \
       --modes sis_selective sis_mpc
   ```

4. Copy results to Google Drive before session ends.

---

## 📊 Reporting

All experiment outputs feed into the Markdown report template:

```
reports/
  2025_selective_reconstruction_report_template.md
```

This includes automatic embedding of:

* `metrics.csv` summaries
* ROC/PR curves
* Stage-wise communication/latency breakdowns

---

## 🧱 Package Internals

```
sis_image/
  phash.py              # perceptual hashing (64-bit DCT)
  shamir.py             # GF(257) secret sharing core
  index.py              # banded HMAC token index
  workflow.py           # SIS + image store orchestration
  utils.py              # common math and I/O helpers
evaluation/
  dataset.py            # COCO derivative mapping
  plotting.py           # report figure generator
```

---

## 🧠 Related Papers / Concepts

* **Shamir, Adi.** *How to share a secret.* Communications of the ACM, 1979.
* **Zauner, C.** *Implementation and benchmarking of perceptual image hash functions.* 2010.
* **VeilHash / Searchable SIS (2024)** — pHash × SIS framework enabling privacy-preserving perceptual search.

---

## 🧾 License

MIT License © 2025 Grand Research Lab.
You may reuse, modify, or cite this package for academic and research purposes with attribution.

---

Would you like me to also generate a **Japanese-translated version** of this updated README (技術要約を含む) for documentation or publication use?
