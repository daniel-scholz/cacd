# Reproducibility notes

This file logs any drift observed when reproducing the paper's reported
numbers from the post-refactor codebase. Each entry should record:

- which artifact was reproduced (e.g., `Figure 2(a)`, `Table 1 row "Ours"`),
- the wandb run / config / checkpoint used,
- the new value vs. the paper's reported value,
- a note on the suspected source of any difference.

## Eval-only reproduction (cached checkpoints)

When re-running `eval_harm_baselines.py` and the plotting scripts on the
post-refactor branch, compare the new artifacts against the committed
references with a numeric tolerance: `np.allclose(atol=1e-4, rtol=1e-3)`.
Float-level run-to-run drift at this tolerance is expected from
non-deterministic CUDA ops (DDIM sampling) and is not a regression.

```bash
# 1. CACD ("Ours")
uv run eval_harm_baselines.py --method DiffAE --wandb_id 1voovf9c --target_scanner GEM
# 2. Vanilla DiffAE baseline
uv run eval_harm_baselines.py --method DiffAE --wandb_id 72jaa0rm --target_scanner GEM
# 3. Other baselines (no wandb id needed)
uv run eval_harm_baselines.py --method HACA3 --target_scanner GEM
uv run eval_harm_baselines.py --method histogram_matching --target_scanner GEM
uv run eval_harm_baselines.py --method unharmonized --target_scanner GEM

# 4. Re-aggregate metrics + replot Figure 2
uv run scripts/plots/plot_paired_eval.py
uv run scripts/plots/vis_paired_eval.py

# 5. Table 1: scanner classification + age regression (needs pyradiomics)
uv run eval_scanner_classification.py
```

Diff the regenerated `results/baseline_comp/all_images.csv` and
`methods_average.csv` against the committed copies, and visually compare
`scripts/plots/figures/paired_eval.{pdf,png}` and
`vis_paired_eval.{pdf,png}` to the paper's Figure 2.

## Spot-check retrain (~1 hour on selene)

The full paper run is 2.5M steps on an A40. For a fast sanity check
that the post-refactor training pipeline still converges, train the
paper config for ~50–100k steps and verify against the wandb history of
`1voovf9c` at the same step count:

```bash
sbatch slurm_train.sh scanner_harm \
    --conf oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs
# Then `squeue -u $USER` and `tail -f slurm_out/im2im-<jobid>.out`.
```

Watch in wandb:

- `train_loss` should decrease (DDIM noise prediction MSE).
- `loss_anatomy` and `loss_contrast` should both decrease and stay
  bounded; runaway either way means contrastive collapse.
- `z_a_var` and `z_c_var` (mean per-dim variance) should stabilise to
  non-zero values.
- No NaNs in any logged metric.

If any of these diverge from the `1voovf9c` history at the same step
count, log the divergence here.

### Attempt 1 — 2026-04-27 (deferred, queue-blocked)

- Job id: `138414` on `selene`, partition `universe`, `gpu:1` (scheduled
  for `atlas`, A40).
- Submission cmd: `sbatch slurm_train.sh scanner_harm --conf spot_check_cacd_1h`
  (commit `bc920349` adds `conf/spot_check_cacd_1h.json` capping
  `total_steps=100000` on top of `oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs`).
- TimeLimit reduced from 11d (header default) to 2h via
  `scontrol update JobId=138414 TimeLimit=02:00:00` to make the job
  backfill-eligible.
- Earliest backfill slot offered by the scheduler: `2026-04-28T01:25` —
  ~9 h after submit, well past the 1 h spot-check session budget.
  Cluster was heavily loaded across `universe` (atlas/chameleon/helios/
  prometheus all `mix-`).
- Outcome: **deferred**. Job left queued; once it transitions to
  `RUNNING`, run `scancel 138414` after ~1 h of accrued runtime to keep
  the spot check in budget. SLURM log:
  `slurm_out/im2im-138414.{out,err}`. wandb run id will appear under
  project `med-image-translation` once the script starts logging.
- No verdict on regime-match yet — pending the actual run.

### Attempt 1 — 2026-04-28 (workstation, TITAN RTX 24 GB)

Reproduced Figure 2(a) end-to-end on the local workstation against
reference snapshot ``results/baseline_comp.ref-20260428/``. Per-row
drift summary on ``all_images.csv`` (960 rows, 5 methods, target=GEM):

| Method | rows | max ssim |Δ| | max psnr |Δ| | verdict |
| --- | ---: | ---: | ---: | --- |
| DiffAE (CACD `1voovf9c` + vanilla `72jaa0rm`) | 720 | 2.0e-5 | 2.9e-5 | ✅ within ``atol=1e-4`` |
| HACA3 | 80 | 2.8e-3 | 0.20 dB | ⚠️ ~28× ``atol``; baseline only |
| histogram_matching | 80 | 3.4e-2 | 6.5 dB | ❌ ~340× ``atol``; baseline only |
| unharmonized | 70 | 0 | 0 | ✅ identity |

**Verdict: paper claims (CACD vs vanilla DiffAE) reproduce within
documented tolerance.** Both DiffAE runs come in *two orders of magnitude
tighter* than the ``atol=1e-4`` budget — DDIM non-determinism is
well-bounded on this hardware.

Visual check: ``scripts/plots/figures/{paired_eval,vis_paired_eval}.png``
are pixel-equivalent to the committed references. Histogram-matching
does not appear in Figure 2 so its drift is cosmetic for the headline
artefact.

Two issues uncovered, neither blocks the verdict:

1. **Bug in ``eval_harm_baselines.py:calc_mean_metrics``** — the
   groupby keys dropped ``global_step``, so the second invocation
   crashed with ``KeyError: 'global_step'`` on the existence-check at
   line 214. Fixed in this attempt by adding ``global_step`` to the
   groupby (matches the reference CSV's column structure). The
   reference CSV must have been written by an older 4-key groupby.

2. **histogram_matching baseline shifted ~2% mean SSIM / 1.4 dB mean
   PSNR** (max 3.4e-2 SSIM, 6.5 dB PSNR). Current env: ``scikit-image
   0.26.0``. The wrapper in ``harm_model.py:362`` is unchanged in our
   git history, so the drift must come from skimage. **Hypothesis
   unverified** — would need to install an older skimage and rerun.
   ``README.md:126`` documents the histogram_matching command as a
   paper-reproduction step, so a public reader running it gets numbers
   that don't match the committed ``methods_average.csv`` row. Fix
   before release: pin skimage in ``pyproject.toml`` once we identify
   the offending version, or regenerate the reference rows on the
   current pin.

3. **HACA3 small-but-non-trivial drift** — max ssim |Δ|=2.8e-3
   (~28× the doc's ``atol=1e-4``), max psnr |Δ|=0.20 dB. Below the
   threshold at which the figure changes, but well above what we'd
   expect for a deterministic pretrained model on identical inputs.
   Worth a follow-up to check whether HACA3 inference has stochastic
   components (sampling, dropout-at-inference) before declaring the
   doc tolerance stable.

Recipe note: ``methods_average.csv`` and ``all_images.csv`` are
overwritten in-place each run, so snapshot the reference dir
(``cp -r results/baseline_comp results/baseline_comp.ref-$(date +%Y%m%d)``)
before running the recipe. The "results already exist" warning at line
214 only logs — it does not skip.

Inference cost on workstation: vanilla DiffAE ``72jaa0rm`` ~7 min,
CACD ``1voovf9c`` ~7 min, HACA3 < 1 min, histogram_matching + unharmonized
seconds. Total wall-clock ~20 min for the full Figure 2(a) regen.

## Known mismatches between paper and committed configs

| Field | Paper / wandb run `1voovf9c` | Pre-fix config | Post-fix config |
| --- | --- | --- | --- |
| `slices_around_middle` | `10` | `50` | `10` (corrected, see commit) |

Any further mismatches uncovered during reproduction should be added
above with the same structure.
