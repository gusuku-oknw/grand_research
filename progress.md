# Thesis Progress Summary

Date: (please fill)  
Repo: Grand_Research

## What changed
- Updated `thesis.tex` title page: standalone page, centered title, removed decorative lines; set title to「秘密画像共有におけるpHashでのシェア収集を可能にする知覚暗号化の設計」; faculty set to「工学部 電気電子・情報工学科 情報コース」; name=玉城洵弥, student ID=1213033903.
- Expanded related work with formal citations (`\cite{}`) and added thebibliography (Barni2010, Xia2020, Zhang2024, Troncoso2017, Tian2024, Xia2021, Venkatesan2000). Added explicit contrast vs MPC/SE (heavy, no staged disclosure) and tied to low-cost pHash design.
- Clarified proposal: definitions, three-stage disclosure, dummy generation with margin reinforcement, safety discussion. Search pipeline description simplified to masked SIS use only (k1 dummy, k2 full).
- Implementation section rewritten to objective technical details (NumPy/Pillow, DCT/IDCT, dummy generation, TwoLevel Shamir).
- Experiments section rewritten: how derivatives/mapping.json were built (500 samples from COCO val2017, 20 deterministic variants), added definitions of Precision@k/Recall@k, and replaced all metrics with actual CSV/JSON averages.
  - Original only (`masked_phash_eval.csv`): P@1=1.00, P@5=0.20, P@10=0.10, Recall@10=1.00; latency 0.581 ms (plain) / 0.548 ms (dummy).
  - All variants (`masked_phash_eval_all.csv`): P@1=1.00, P@5=0.866, P@10=0.8482, Recall@10=0.4241; latency 0.527 ms (plain) / 0.494 ms (dummy).
  - Reconstruction (`phash_masked_sis_eval.json`): phash_dist_dummy mean 0, phash_dist_<k1 mean 20.8, PSNR dummy 10.76 dB, recover_dummy_ms 121 ms, recover_full_ms 10.2 s.
- Added dummy generation figure reference in text; P@k wording changed to sentences (no shorthand).

## Files touched
- `thesis.tex` (major rewrite as above)
- `progress.md` (this summary)

## Outstanding checks / next steps
- Fill advisor name and submit date placeholders.
- Re-run `platex thesis.tex` twice to settle references.
- If needed, tighten wording around safety/access-pattern discussion or adjust figure placement.***
