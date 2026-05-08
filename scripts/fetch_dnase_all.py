#!/usr/bin/env python3
"""
Fetch DNase-seq signal (ENCODE/UCSC) for all 599 UCOE candidates.
Uses 11 ENCODE Tier 1/2 cell lines, same as the UCOE pipeline.
Saves data/dnase_all.npz: each key = "gene1_gene2", value = float32 array
  shape (n_cell_lines, region_length) — mean across cell lines also saved
  as "gene1_gene2_mean".
Run from the ucoe-atlas/ directory.
"""

import sys
import time
import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
SCORED_TSV = DATA_DIR / "scored_candidates.tsv"
OUT_NPZ    = DATA_DIR / "dnase_all.npz"

UCSC_BASE = "https://hgdownload.cse.ucsc.edu/gbdb/hg38/bbi/wgEncodeRegDnase"

# ENCODE Tier 1/2 cell lines matching those used in the UCOE pipeline
CELL_LINES = {
    "K562":     "wgEncodeRegDnaseUwK562Signal.bw",
    "GM12878":  "wgEncodeRegDnaseUwGm12878Signal.bw",
    "HepG2":    "wgEncodeRegDnaseUwHepg2Signal.bw",
    "H1-hESC":  "wgEncodeRegDnaseUwH7hescSignal.bw",
    "HUVEC":    "wgEncodeRegDnaseUwHuvecSignal.bw",
    "HMEC":     "wgEncodeRegDnaseUwHmecSignal.bw",
    "NHEK":     "wgEncodeRegDnaseUwNhekSignal.bw",
    "HSMM":     "wgEncodeRegDnaseUwHsmmSignal.bw",
    "NHLF":     "wgEncodeRegDnaseUwNhlfSignal.bw",
    "A549":     "wgEncodeRegDnaseUwA549Signal.bw",
}

scored  = pd.read_csv(SCORED_TSV, sep="\t")
n_total = len(scored)
print(f"Candidates: {n_total}")

# Open all BigWig handles
print("Opening DNase BigWig connections...")
bw_handles = {}
for cl, fname in CELL_LINES.items():
    url = f"{UCSC_BASE}/{fname}"
    try:
        bw = pyBigWig.open(url)
        bw_handles[cl] = bw
        print(f"  OK: {cl}")
    except Exception as e:
        print(f"  SKIP {cl}: {e}")

n_cl = len(bw_handles)
cl_names = list(bw_handles.keys())
print(f"\nUsing {n_cl} cell lines: {cl_names}")

results = {}
t0 = time.time()

for i, (_, row) in enumerate(scored.iterrows(), 1):
    key   = f"{row['gene1']}_{row['gene2']}"
    chrom = row["chrom"]
    start = int(row["start"])
    end   = int(row["end"])
    length = end - start

    mat = np.full((n_cl, length), np.nan, dtype=np.float32)
    for j, cl in enumerate(cl_names):
        try:
            vals = bw_handles[cl].values(chrom, start, end)
            mat[j] = np.array(vals, dtype=np.float32)
        except Exception:
            pass  # stays NaN

    mean_sig = np.nanmean(mat, axis=0)
    mean_sig = np.nan_to_num(mean_sig, nan=0.0)

    results[key]           = mat        # (n_cl, length) — per cell line
    results[f"{key}_mean"] = mean_sig   # (length,)      — mean across cell lines

    elapsed = time.time() - t0
    rate    = i / elapsed
    eta     = (n_total - i) / rate if rate > 0 else 0
    print(f"  [{i:3d}/{n_total}] {key}  mean_dnase={mean_sig.mean():.2f}  ETA {eta/60:.1f} min",
          flush=True)

for bw in bw_handles.values():
    bw.close()

# Save metadata alongside
results["__cell_lines__"] = np.array(cl_names)

np.savez_compressed(str(OUT_NPZ), **results)
total_time = (time.time() - t0) / 60
print(f"\nDone in {total_time:.1f} min")
print(f"Saved {len(scored)} candidates × {n_cl} cell lines → {OUT_NPZ}")
