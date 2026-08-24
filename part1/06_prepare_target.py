#!/usr/bin/env python3
"""Prepare target FASTAs and manifests from mutation boxes.

Creates per-box FASTA targets where positions outside the box are masked
with 'X', writes a box definition TSV and a JSON manifest for downstream
engines.

Usage:
  python 06_prepare_target.py --indir data --outdir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare targets from mutation boxes")
    parser.add_argument("--indir", type=Path, default=Path("data"))
    parser.add_argument("--outdir", type=Path, default=Path("data"))
    return parser.parse_args()


def load_fasta_one(path: Path) -> tuple[str, str]:
    with open(path, "r") as f:
        name = None
        seq_lines: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    break
                name = line[1:].split()[0]
            else:
                seq_lines.append(line)
    if name is None:
        raise FileNotFoundError(f"No FASTA record in {path}")
    return name, "".join(seq_lines)


def write_fasta(path: Path, header: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i : i + 60] + "\n")


def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load mutation boxes
    boxes_path = indir / "mutation_boxes.json"
    if not boxes_path.exists():
        raise SystemExit(f"mutation_boxes.json not found in {indir}")
    boxes = json.loads(boxes_path.read_text())
    mutation_boxes = boxes.get("mutation_boxes", [])

    # Load consensus / wuhan reference if present
    consensus_path = indir / "consensus_aa.npy"
    consensus = None
    if consensus_path.exists():
        consensus = np.load(consensus_path)
        consensus = "".join(list(consensus.astype(str)))

    wuhan_path = indir / "wuhan_aa.fasta"
    wuhan_name = None
    wuhan_seq = None
    if wuhan_path.exists():
        wuhan_name, wuhan_seq = load_fasta_one(wuhan_path)

    # Prefer consensus as base sequence, fall back to wuhan, else error
    if consensus is not None:
        base_seq = consensus
        base_name = "consensus"
    elif wuhan_seq is not None:
        base_seq = wuhan_seq
        base_name = wuhan_name
    else:
        raise SystemExit("Neither consensus_aa.npy nor wuhan_aa.fasta found in data dir")

    seq_len = len(base_seq)

    targets_dir = outdir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Write full-length base reference as a FASTA for downstream use
    write_fasta(targets_dir / f"{base_name}_full.fasta", base_name, base_seq)

    # Write box definitions TSV
    tsv_path = outdir / "box_definitions.tsv"
    with open(tsv_path, "w", newline="\n") as f:
        f.write("box_id\tstart_0based\tend_0based\tsize\torfs\n")
        for box in mutation_boxes:
            bid = box.get("id")
            start = int(box.get("start"))
            end = int(box.get("end"))
            size = int(box.get("size", end - start))
            orfs = ",".join(box.get("orfs", [])) if box.get("orfs") else ""
            f.write(f"{bid}\t{start}\t{end}\t{size}\t{orfs}\n")

    manifest = {"n_boxes": len(mutation_boxes), "boxes": []}

    # For each box, create a FASTA where positions outside the box are masked with 'X'
    for box in mutation_boxes:
        bid = box.get("id")
        start = int(box.get("start"))
        end = int(box.get("end"))
        seq_chars = list(base_seq)
        for i in range(seq_len):
            if not (start <= i < end):
                seq_chars[i] = "X"
        seq_masked = "".join(seq_chars)
        header = f"box_{bid}_masked_{base_name}"
        out_fa = targets_dir / f"target_box_{bid}.fasta"
        write_fasta(out_fa, header, seq_masked)
        manifest["boxes"].append({"id": bid, "start": start, "end": end, "fasta": str(out_fa.name)})

    # Save manifest
    manifest_path = outdir / "targets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Prepared targets:")
    print(f"  base: {base_name}, length={seq_len}")
    print(f"  boxes: {len(mutation_boxes)}")
    print(f"  targets dir: {targets_dir}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
