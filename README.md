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
* **Modular experiment runners (`sis_modes/`)** — independently testable Stage-1/Stage-2 pipelines.

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

The repository also includes a visualization demo:

```bash
python demo_k_sweep.py --share_strategy phash-fusion --output figures/k_sweep.png
```

This script sweeps over the number of responding servers, plots the top-ranked Hamming distance, and highlights when the fusion fallback activates (k < required). The generated `k_sweep.png` makes it easy to see how rankings/reconstruction degrade as servers drop out.

Another visual helper is `demo_k_vs_strategy.py`:

```bash
python demo_k_vs_strategy.py --share_strategy phash-fusion --output figures/k_strategy_compare.png
```

It runs both the pure Shamir pipeline and the fusion hash pipeline for every `k` from 1 to `n`, then draws the top-ranked Hamming distance and reconstruction counts per strategy so you can directly compare how the result “looks” as servers become available.

To see the **actual images** the system would pick/reconstruct as `k` increases, use:

```bash
python demo_k_recovery_gallery.py --output figures/k_recovery_gallery.png
```

Fusion-mode cells blend the candidate with a progressively blurred version based on the Hamming distance, so you can see how pHash融合型 returns “fuzzy” approximations when `k` is short while Shamir simply waits until enough shares exist.

It arranges the recovered/candidate image per `k` for both strategies so you can visually trace how Shamir waits until `k` is reached while the fusion variant continually surfaces the nearest perceptual match.

If you don't have a curated dataset, generate synthetic noise images with:

```bash
python experiments/scripts/generate_noise_images.py --count 20 --size 128 128 --output_dir data/noise
```

These PNGs provide a quick visual workload for the demo scripts so you can inspect how noise versus real images behave as `k` changes.

---

## ⚙️ Modular Experiment Architecture

Experiment scripts under `experiments/scripts/` are now **mode-based** and follow a shared interface.

```
experiments/scripts/
  run_search_experiments.py      # orchestrator (lightweight)
experiments/modes/
  base_runner.py                 # abstract ModeRunner and helpers
  plain.py                       # baseline pHash ranking
  sis_naive.py                   # full reconstruction baseline
  sis_selective.py               # Stage-2 selective reconstruction
  sis_staged.py                  # staged refinement pipeline
  sis_mpc.py                     # MPC-style fully private variant
```

### Design Philosophy and Architecture

The core of this research is to **achieve both computational efficiency and confidentiality by first narrowing down candidates with pHash, and then calculating distances using MPC (Multi-Party Computation) while the data remains in its secret-shared (SIS) state.**

The conventional, naive approach of "reconstruct all images, then search in plaintext" is computationally expensive and carries privacy risks. This system completes the process within the framework of secret sharing and MPC, without ever converting the original images or their pHashes back to plaintext.

A note on terminology: Unlike cryptographic "decryption," SIS involves gathering shares to restore the original information. Therefore, this document uses the term "**reconstruction**."

### Two-Stage Search Pipeline

The search is executed in a two-stage pipeline to keep the flow aligned with the new reporting terminology.

- **Stage-1 (Index-based Candidate Reduction)**  
  HMAC tokens derived from the query pHash are matched against each server's index to emit an initial candidate set. This stage limits the search space to a very small fraction of the dataset.

- **Stage-2 (Reconstruction + Secure Distance)**  
  Candidates that survive Stage-1 are moved into Stage-2, where their SIS shares may be partially reconstructed (e.g., in `sis_selective`) and their distances are evaluated securely, typically via MPC without fully exposing the pHash values. Stage-2 therefore encompasses the previous reconstruction and distance evaluation phases (formerly Stage-B and Stage-C).

The final image **reconstruction** remains separate from the search pipeline and is executed only when the original image is actually needed, using the K-of-N threshold reconstruction strategy, usually at the client-side.

### Why It's Fast and Secure

- **Reduced Computational Complexity**: The workload is significantly reduced by limiting comparisons from O(N) for the entire dataset to O(|candidates|) for a small subset.
- **No Plaintext Exposure**: High confidentiality is maintained because neither the original images nor their pHashes are reconstructed into plaintext during the search process.
- **Scalability**: The most computationally expensive operation, MPC, is reserved for the small number of final candidates, ensuring the system remains scalable.

### Considerations and Future Work

- **Access Pattern Leakage**: Information about which HMAC tokens matched the index (the access pattern) can be leaked. This could be mitigated by introducing dummy queries or by batching queries to anonymize them.
- **Key Management**: It is recommended to use separate HMAC keys for each band and to rotate them periodically. The future use of an OPRF (Oblivious Pseudorandom Function) is desirable.
- **MPC Implementation**: The current implementation is a simulated MPC that performs "reconstruction -> distance calculation" for demonstration purposes. For practical applications, this should be replaced with a true MPC protocol or a TEE (Trusted Execution Environment).

### Mode-specific Stage Usage

| Mode | Stage-1 | Stage-2 |
| :--- | :------ | :------ |
| `plain` | ❌ (Scans all with pHash distance) | ✅ Sorts by distance with `compute_plain_distances` |
| `sis_naive` | ❌ (No candidate reduction) | ✅ Reconstructs and evaluates all candidates with `rank_candidates` |
| `sis_selective` | ✅ Reduces candidates with HMAC voting | ✅ `stage_b_filter` + reconstruction for Top-K |
| `sis_staged` | ✅ | ✅ |
| `sis_mpc` | ✅ HMAC voting | ✅ MPC ranking with `rank_candidates_secure` (no reconstruction) |

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
python experiments/scripts/prepare_coco.py \
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
PYTHONPATH=. python experiments/scripts/run_search_experiments.py \
    --mapping_json data/coco2017_derivatives/derivative_mapping.json \
    --output_dir output/results/coco_val2017_modular \
    --work_dir output/artifacts/coco_val2017_modular \
    --modes plain sis_naive sis_selective sis_staged sis_mpc \
    --max_queries 500 \
    --bands 8 --k 3 --n 5 \
    --force
```

* Each mode logs Stage-wise latency, bytes, and precision metrics.
* Install `tqdm` to see per-query progress bars: `pip install tqdm`.
* Output:

  * `metrics.csv` — consolidated per-query results
  * `security_summary.json` — entropy & leakage analysis
  * `output/figures/*.png` — candidate reduction & latency graphs

### 4. Generate Figures

```bash
python -m experiments.common.plotting \
    output/results/coco_val2017_modular/metrics.csv \
    --output_dir output/figures/coco_val2017_modular
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
   !python experiments/scripts/run_search_experiments.py \
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
  common/               # shared helpers (dataset parsing, metrics, plotting, tokens)
  dealer_based/         # dealer-present index/store/workflow/CLI
  dealer_free/          # dealer-free simulator + MPC helpers
experiments/
  common/               # moved dataset/metrics/plotting logic
  scripts/              # run_search_experiments.py + helpers (COCO prep, plots)
  modes/                # mode-specific runners for plain/sis_naive/selective/staged/mpc
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
