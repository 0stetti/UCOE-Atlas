#!/usr/bin/env python3
"""
Fetch PhyloP 100-way and PhastCons 100-way scores for ALL candidates.
Saves data/conservation_all.npz and data/phastcons_all.npz.
Run from the ucoe-atlas/ directory.
"""

import sys
import time
import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCORED_TSV    = DATA_DIR / "scored_candidates.tsv"
OUT_PHYLOP    = DATA_DIR / "conservation_all.npz"
OUT_PHASTCONS = DATA_DIR / "phastcons_all.npz"

PHYLOP_URL    = ("https://hgdownload.cse.ucsc.edu/goldenpath/hg38/"
                 "phyloP100way/hg38.phyloP100way.bw")
PHASTCONS_URL = ("https://hgdownload.cse.ucsc.edu/goldenpath/hg38/"
                 "phastCons100way/hg38.phastCons100way.bw")

scored = pd.read_csv(SCORED_TSV, sep="\t")
n_total = len(scored)
print(f"Candidates: {n_total}")

print(f"Opening PhyloP BigWig: {PHYLOP_URL}")
bw_phylop = pyBigWig.open(PHYLOP_URL)
print(f"Opening PhastCons BigWig: {PHASTCONS_URL}")
bw_phastcons = pyBigWig.open(PHASTCONS_URL)

phylop_results    = {}
phastcons_results = {}
errors = []

t0 = time.time()
for i, (_, row) in enumerate(scored.iterrows(), 1):
    key   = f"{row['gene1']}_{row['gene2']}"
    chrom = row["chrom"]
    start = int(row["start"])
    end   = int(row["end"])

    try:
        pp = bw_phylop.values(chrom, start, end)
        arr_pp = np.array(pp, dtype=np.float32)
        phylop_results[key] = arr_pp
    except Exception as ex:
        errors.append(f"PHYLOP SKIP {key}: {ex}")
        print(f"  SKIP phylop {key}: {ex}")

    try:
        pc = bw_phastcons.values(chrom, start, end)
        arr_pc = np.array(pc, dtype=np.float32)
        arr_pc = np.nan_to_num(arr_pc, nan=0.0)
        phastcons_results[key] = arr_pc
    except Exception as ex:
        errors.append(f"PHASTCONS SKIP {key}: {ex}")
        print(f"  SKIP phastcons {key}: {ex}")

    elapsed = time.time() - t0
    rate    = i / elapsed
    eta     = (n_total - i) / rate if rate > 0 else 0
    print(f"  [{i:3d}/{n_total}] {key}  "
          f"pp_mean={arr_pp.mean():.3f}  "
          f"pc_mean={arr_pc.mean():.3f}  "
          f"ETA {eta/60:.1f} min",
          flush=True)

bw_phylop.close()
bw_phastcons.close()

np.savez_compressed(str(OUT_PHYLOP),    **phylop_results)
np.savez_compressed(str(OUT_PHASTCONS), **phastcons_results)

total_time = (time.time() - t0) / 60
print(f"\nDone in {total_time:.1f} min")
print(f"PhyloP:    {len(phylop_results)} entries → {OUT_PHYLOP}")
print(f"PhastCons: {len(phastcons_results)} entries → {OUT_PHASTCONS}")
if errors:
    print(f"\n{len(errors)} errors:")
    for e in errors:
        print(f"  {e}")
