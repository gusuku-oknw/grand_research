# Project Overview

This repository, named "SIS Image Sharing Package," is a Python module designed for secure image sharing and searchable image retrieval. It implements several key components:

*   **Perceptual Hashing:** Utilizes 64-bit perceptual hashing for image identification.
*   **Shamir Secret Sharing:** Employs Shamir secret sharing for both hash bytes and full RGB images, enabling secure distribution and reconstruction.
*   **Search Indices:** Incorporates search indices that use banded HMAC tokens to preselect candidates efficiently before reconstructing secrets.
*   **Persistent Image Store:** Features a persistent image store capable of reconstructing full-resolution assets from a specified number of servers (`k` of `n`).

The core functionality revolves around a three-stage (Stage-A/B/C) pipeline for searchable SIS workflow, which narrows down image candidates before full reconstruction.

# Building and Running

## Installation

To set up the project, create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
. .venv/bin/activate  # For Windows: .\\.venv\\Scripts\\activate
pip install --upgrade pip
pip install -e .
```

Alternatively, for lightweight usage (e.g., running experiment scripts directly), install core dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Command Line Demos

Once installed, the CLI provides access to various demos:

```bash
sis-image selective-demo --images_dir data/
sis-image search-demo --images_dir data/ --reconstruct_top 2
sis-image image-store-demo --images_dir data/
sis-image secure-demo --images_dir data/
```

## Reproducible Experiments

The project supports reproducible experiments, locally or on Google Colab, involving COCO dataset preparation and staged SIS experiments.

1.  **Prepare COCO derivatives:**
    ```bash
    python scripts/prepare_coco.py \
        --coco_dir data/coco2017/val2017 \
        --output_dir data/coco2017_derivatives/val2017 \
        --mapping_json data/coco2017_derivatives/derivative_mapping.json \
        --profile medium \
        --variant_scope all \
        --max_images 5000
    ```
2.  **Run staged SIS experiments:**
    ```bash
    PYTHONPATH=. python scripts/run_search_experiments.py \
        --mapping_json data/coco2017_derivatives/derivative_mapping.json \
        --output_dir evaluation/results/coco_val2017_stageABC \
        --work_dir evaluation/artifacts/coco_val2017_stageABC
    ```
3.  **Produce Matplotlib figures from the metrics:**
    ```bash
    python -m evaluation.plotting \
        evaluation/results/coco_val2017_stageABC/metrics.csv \
        --output_dir evaluation/figures/coco_val2017_stageABC
    ```

# Development Conventions

The project emphasizes reproducible research and evaluation, with dedicated scripts for preparing datasets, running experiments, and generating visual reports (Matplotlib figures) from collected metrics. The `scripts/run_search_experiments.py` plays a central role in recording timing, candidate counts, and byte usage for each stage of the SIS pipeline, feeding directly into reporting templates.
