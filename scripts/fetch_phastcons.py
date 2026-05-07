#!/usr/bin/env python3
"""
Fetch PhastCons 100-way scores for the top-50 UCOE candidates.
Requires: pyBigWig, pandas, numpy

Run from the ucoe-atlas/ directory:
    python3 scripts/fetch_phastcons.py
"""

import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
SCORED_TSV    = DATA_DIR / "scored_candidates.tsv"
OUT_NPZ       = DATA_DIR / "phastcons_top50.npz"
PHASTCONS_URL = ("https://hgdownload.cse.ucsc.edu/goldenpath/hg38/"
                 "phastCons100way/hg38.phastCons100way.bw")

scored = pd.read_csv(SCORED_TSV, sep="\t")
top50  = scored.sort_values("composite_rank").head(50)

print(f"Opening remote bigWig: {PHASTCONS_URL}")
bw = pyBigWig.open(PHASTCONS_URL)

results = {}
for _, row in top50.iterrows():
    key   = f"{row['gene1']}_{row['gene2']}"
    chrom = row["chrom"]
    start = int(row["start"])
    end   = int(row["end"])
    try:
        vals        = bw.values(chrom, start, end)
        arr         = np.array(vals, dtype=np.float32)
        arr         = np.nan_to_num(arr, nan=0.0)
        results[key] = arr
        print(f"  {key}: {len(arr)} bp  mean={arr.mean():.3f}")
    except Exception as ex:
        print(f"  SKIP {key}: {ex}")

bw.close()

np.savez_compressed(str(OUT_NPZ), **results)
print(f"\nSaved {len(results)} candidates → {OUT_NPZ}")
