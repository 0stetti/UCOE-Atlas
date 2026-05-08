#!/usr/bin/env python3
"""
Pre-compute ETS motif positions and CpG islands for all candidates.
Reads output/ucoe_sequences.fa from the UCOE pipeline.
Saves:
  data/ets_motifs_all.tsv   — one row per motif occurrence
  data/cpg_islands_all.tsv  — one row per CpG island
Run from the ucoe-atlas/ directory.
"""

import re
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FASTA    = Path(os.environ.get(
    "UCOE_FASTA",
    "/Users/eltonostetti/Documents/USP/Tese/Algoritmo UCOE/output/ucoe_sequences.fa"
))

ETS_FWD  = re.compile(r"CGGAA[GA]")
ETS_REV  = re.compile(r"[TC]TTCCG")

CpG_WIN  = 200
CpG_GC   = 0.50
CpG_OE   = 0.60
CpG_MIN  = 200


def read_all_fasta(path):
    seqs = {}
    header, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header:
                    seqs[header] = "".join(buf).upper()
                header, buf = line[1:], []
            else:
                buf.append(line)
    if header:
        seqs[header] = "".join(buf).upper()
    return seqs


def find_cpg_islands(seq, win=CpG_WIN, gc_thr=CpG_GC, oe_thr=CpG_OE, min_len=CpG_MIN):
    n = len(seq)
    flagged = np.zeros(n, dtype=bool)
    for i in range(n - win + 1):
        w  = seq[i:i + win]
        gc = (w.count("G") + w.count("C")) / win
        if gc < gc_thr:
            continue
        nc, ng, ncg = w.count("C"), w.count("G"), w.count("CG")
        if nc == 0 or ng == 0:
            continue
        if (ncg * win) / (nc * ng) >= oe_thr:
            flagged[i:i + win] = True
    islands, in_isl, s0 = [], False, 0
    for i, f in enumerate(flagged):
        if f and not in_isl:
            s0, in_isl = i, True
        elif not f and in_isl:
            if i - s0 >= min_len:
                islands.append((s0, i))
            in_isl = False
    if in_isl and n - s0 >= min_len:
        islands.append((s0, n))
    return islands


def parse_header(header):
    # rank001_SMIM27_TOPORS_score0.7869::chr9:32550561-32552586
    m = re.match(r"rank(\d+)_(.+)_score([\d.]+)::(\w+):(\d+)-(\d+)", header)
    if not m:
        return None
    rank = int(m.group(1))
    label_part = m.group(2)          # e.g. SMIM27_TOPORS
    score = float(m.group(3))
    chrom = m.group(4)
    start = int(m.group(5))
    end   = int(m.group(6))
    # split label into gene1/gene2 at last underscore that separates two gene names
    parts = label_part.split("_")
    if len(parts) >= 2:
        gene1, gene2 = parts[0], "_".join(parts[1:])
    else:
        gene1 = gene2 = label_part
    key = f"{gene1}_{gene2}"
    return rank, gene1, gene2, key, score, chrom, start, end


print(f"Reading FASTA: {FASTA}")
seqs = read_all_fasta(FASTA)
print(f"  {len(seqs)} sequences loaded")

ets_rows = []
cpg_rows = []

for header, seq in seqs.items():
    parsed = parse_header(header)
    if not parsed:
        print(f"  SKIP unparseable header: {header[:60]}")
        continue
    rank, gene1, gene2, key, score, chrom, start, end = parsed

    # ETS motifs
    fwd = [(m.start(), m.end(), "+", m.group()) for m in ETS_FWD.finditer(seq)]
    rev = [(m.start(), m.end(), "-", m.group()) for m in ETS_REV.finditer(seq)]
    for s, e, strand, motif in sorted(fwd + rev, key=lambda x: x[0]):
        ets_rows.append({
            "rank": rank, "gene1": gene1, "gene2": gene2, "key": key,
            "chrom": chrom, "locus_start": start, "locus_end": end,
            "rel_start": s, "rel_end": e,
            "abs_start": start + s, "abs_end": start + e,
            "strand": strand, "motif": motif,
        })

    # CpG islands
    for isl_s, isl_e in find_cpg_islands(seq):
        w = seq[isl_s:isl_e]
        gc = (w.count("G") + w.count("C")) / len(w)
        nc, ng, ncg = w.count("C"), w.count("G"), w.count("CG")
        oe = (ncg * len(w)) / (nc * ng) if nc and ng else 0.0
        cpg_rows.append({
            "rank": rank, "gene1": gene1, "gene2": gene2, "key": key,
            "chrom": chrom, "locus_start": start, "locus_end": end,
            "rel_start": isl_s, "rel_end": isl_e,
            "abs_start": start + isl_s, "abs_end": start + isl_e,
            "length_bp": isl_e - isl_s,
            "gc_pct": round(gc * 100, 2),
            "obs_exp_cpg": round(oe, 4),
            "n_cpg": ncg,
        })

ets_df = pd.DataFrame(ets_rows)
cpg_df = pd.DataFrame(cpg_rows)

ets_out = DATA_DIR / "ets_motifs_all.tsv"
cpg_out = DATA_DIR / "cpg_islands_all.tsv"
ets_df.to_csv(ets_out, sep="\t", index=False)
cpg_df.to_csv(cpg_out, sep="\t", index=False)

print(f"\nETS motifs: {len(ets_df)} occurrences across {ets_df['key'].nunique()} candidates → {ets_out}")
print(f"CpG islands: {len(cpg_df)} islands across {cpg_df['key'].nunique()} candidates → {cpg_out}")

# Summary stats
print(f"\nETS per candidate: mean={ets_df.groupby('key').size().mean():.1f}, "
      f"max={ets_df.groupby('key').size().max()}, "
      f"zero-motif candidates={ets_df['key'].nunique() - len(set(ets_df['key'].unique()))}")
zero_ets = set(f"{r['gene1']}_{r['gene2']}" for _, r in pd.read_csv(
    DATA_DIR / 'scored_candidates.tsv', sep='\t').iterrows()) - set(ets_df['key'].unique())
print(f"  Candidates with 0 ETS motifs: {len(zero_ets)}")
print(f"CpG islands per candidate: mean={cpg_df.groupby('key').size().mean():.1f}")
