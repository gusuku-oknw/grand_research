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
       --max_images 5000
   ```
   Adjust `--max_images` if you need a smaller subset (helpful when storage or runtime is limited).

4. **Run staged SIS experiments**
   ```
   PYTHONPATH=. python scripts/run_search_experiments.py \
       --mapping_json data/coco2017_derivatives/derivative_mapping.json \
       --output_dir evaluation/results/coco_val2017_stageABC \
       --work_dir evaluation/artifacts/coco_val2017_stageABC
   ```

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
