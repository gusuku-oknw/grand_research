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

## Command Line Demos

Once installed, run the CLI to access the demos:

```bash
sis-image selective-demo --images_dir data/
sis-image search-demo --images_dir data/ --reconstruct_top 2
sis-image image-store-demo --images_dir data/
sis-image secure-demo --images_dir data/
```

Each original script under `SIS_image/` now delegates to these entry points, keeping the legacy workflow intact while allowing production usage via the package.
