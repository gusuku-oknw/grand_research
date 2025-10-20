# Repository Guidelines

## Project Structure & Module Organization
- `SIS_image/` contains the production-ready secret image sharing modules and demos; treat it as the primary source tree when adding reusable code or CLIs.
- `data/`, `image/`, and `out/` are input/output scratch spaces used by experiments—keep large artifacts out of version control and document reproducibility steps when updating them.
- `.venv/` is the local Python environment and should remain untracked; introduce requirements in a manifest instead of editing this directory directly.
- Store regenerated assets (e.g., `img_meta/`, `img_shares/`, `recon_out/`) under `SIS_image/` to keep checkpoints aligned with the code that produced them.

## Build, Test, and Development Commands
- `python -m venv .venv` then `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix) to create and enter the project environment.
- `pip install -r requirements.txt` to sync dependencies once the manifest is added or updated.
- `python -m SIS_image.searchable_sis_phash_image_sis_demo --help` verifies the package entry point wiring without running a full demo.
- `python SIS_image/searchable_sis_phash_image_sis_demo.py --images_dir images` remains the canonical smoke test until a CLI wrapper is published.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, lowercase_with_underscores for modules, and UpperCamelCase for classes; keep functions noun_verb for clarity (`add_image`, `preselect_candidates`).
- Prefer type hints and dataclasses as seen in existing modules; add concise comments only for non-obvious math (e.g., Shamir interpolation details).
- Run `ruff` (configured in the upcoming package metadata) before committing to catch formatting and complexity issues.

## Testing Guidelines
- Use `pytest` and place suites under `tests/`; mirror module names (e.g., `test_searchable_sis.py`) and group fixtures for synthetic images.
- Target high coverage on cryptographic primitives: add regression vectors for Shamir share/recovery, HMAC token emission, and pHash distance thresholds.
- For demos that touch the filesystem, rely on `tmp_path` fixtures and lightweight PNG fixtures committed under `tests/fixtures/`.

## Commit & Pull Request Guidelines
- Write commits in the imperative mood (`Add Shamir share benchmark`) and keep them scoped to one logical change; include CLI examples in the body when behavior shifts.
- Reference issues or task IDs in pull requests, summarize validation steps, and attach screenshots or hashes when reconstructions change.
- Request review whenever crypto logic or storage layout is touched; include backward-compatibility notes for any serialization change.

## Security & Configuration Tips
- Never commit generated shares, reconstructed images, or secrets; add `.gitignore` rules when introducing new cache directories.
- Rotate demo seeds (`seed=2025`) in production deployments and store real keys in environment variables or a secrets manager, not in code.
- Document any configuration flags added to demos so operational agents can reproduce search rankings across environments.
