"""Compare regenerated baseline_comp CSVs against a reference snapshot.

Tolerances from docs/reproducibility_notes.md: atol=1e-4, rtol=1e-3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ATOL = 1e-4
RTOL = 1e-3

KEY_COLS = ["method", "method_specific_name", "wandb_id", "global_step"]


def _load(p: Path, kind: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    if kind == "all_images":
        return df.sort_values(
            ["subject", "session", "scanner", "target_scanner", "method", "wandb_id"]
        ).reset_index(drop=True)
    return df.sort_values(KEY_COLS).reset_index(drop=True)


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in KEY_COLS]


def diff(new: Path, ref: Path, kind: str) -> int:
    a = _load(new, kind)
    b = _load(ref, kind)
    print(f"\n=== {kind}: {new.name} vs {ref.name} ===")
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH: new={a.shape} ref={b.shape}")
        return 1
    cols_a, cols_b = set(a.columns), set(b.columns)
    if cols_a != cols_b:
        print(f"  COLUMN MISMATCH: only in new={cols_a - cols_b}, only in ref={cols_b - cols_a}")
        return 1
    bad = 0
    for c in _numeric_cols(a):
        x = a[c].to_numpy(dtype=float)
        y = b[c].to_numpy(dtype=float)
        mask = ~(np.isnan(x) & np.isnan(y))
        x_, y_ = x[mask], y[mask]
        if len(x_) == 0:
            continue
        ok = np.allclose(x_, y_, atol=ATOL, rtol=RTOL, equal_nan=True)
        if ok:
            continue
        absdiff = np.abs(x_ - y_)
        reldiff = absdiff / (np.abs(y_) + 1e-12)
        worst_i = int(np.argmax(absdiff))
        print(
            f"  ✘ {c}: max |Δ|={absdiff.max():.3e}, max relΔ={reldiff.max():.3e} "
            f"(at row {worst_i}: new={x_[worst_i]:.6g}, ref={y_[worst_i]:.6g})"
        )
        bad += 1
    if bad == 0:
        print(
            f"  OK: all {len(_numeric_cols(a))} numeric columns within tol "
            f"(atol={ATOL}, rtol={RTOL})"
        )
    return bad


def main():
    repo = Path(__file__).resolve().parent.parent
    new_dir = repo / "results" / "baseline_comp"
    ref_dirs = sorted((repo / "results").glob("baseline_comp.ref-*"))
    if not ref_dirs:
        print("no reference snapshot found", file=sys.stderr)
        sys.exit(2)
    ref_dir = ref_dirs[-1]
    print(f"reference: {ref_dir}")
    bad = 0
    bad += diff(new_dir / "methods_average.csv", ref_dir / "methods_average.csv", "methods_average")
    bad += diff(new_dir / "all_images.csv", ref_dir / "all_images.csv", "all_images")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
