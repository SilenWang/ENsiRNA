#!/usr/bin/env python3
"""
ENsiRNA: Single siRNA-mRNA pair scoring

Predicts efficacy for one siRNA-mRNA pair using the original Rosetta PDB pipeline.

Usage:
  python score.py \\
    --sense CAUGCAUGCAUGCAUGCAU \\
    --mrna UCGCAUGCAUGCAUGCAUGCAUGC...
    --rosetta-dir /path/to/rosetta

Options:
  --sense      siRNA sense sequence (19-mer), required
  --antisense  siRNA antisense sequence (auto-computed if omitted)
  --mrna       Target mRNA sequence (local region around binding site)
  --position   Binding start position in mRNA (0-indexed, default: auto-detect)
  --id         Optional identifier (default: "custom")
  --gpu        GPU device ID (-1 for CPU)
  --outdir     Output directory (default: ./result)
  --rosetta-dir  Rosetta installation directory (or set ROSETTA_DIR env var)
  --quiet      Only print the score, nothing else
"""

import argparse, csv, os, sys, tempfile, warnings, shutil
import numpy as np
import pandas as pd
from Bio.Seq import Seq

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Score a single siRNA-mRNA pair")
    p.add_argument("--sense", required=True, help="siRNA sense sequence (19-mer)")
    p.add_argument("--antisense", default=None, help="siRNA antisense (auto if omitted)")
    p.add_argument("--mrna", required=True, help="Target mRNA sequence (local region)")
    p.add_argument("--position", type=int, default=None, help="Binding position in mRNA (0-indexed)")
    p.add_argument("--id", default="custom", help="Identifier (default: custom)")
    p.add_argument("--gpu", type=int, default=-1, help="GPU device (-1 for CPU)")
    p.add_argument("--outdir", default="result", help="Output directory")
    p.add_argument("--rosetta-dir", default=None, help="Rosetta installation directory")
    p.add_argument("--quiet", action="store_true", help="Only print the score")
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    sense = args.sense.upper().replace("T", "U")
    anti = (args.antisense or str(Seq(sense).reverse_complement())).upper().replace("T", "U")
    mrna = args.mrna.upper().replace("T", "U")
    sid = args.id

    pos = args.position
    if pos is None:
        pos = mrna.find(sense)
        if pos < 0:
            print("Error: sense sequence not found in mRNA" if not args.quiet else "",
                  file=sys.stderr)
            sys.exit(1)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="score_", dir=outdir)

    # Write single-row CSV for Data_Prepare
    csv_path = os.path.join(tmp, f"{sid}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["siRNA", "sense seq", "anti seq", "mRNA_seq", "position", "efficacy"])
        w.writerow([sid, sense, anti, mrna, pos, 0])

    # Generate PDB + JSON via original Rosetta pipeline
    from data.get_pdb import Data_Prepare
    pdb_dir = os.path.join(tmp, "pdbs")
    dp = Data_Prepare(csv_path, pdb_dir, rosetta_dir=args.rosetta_dir)
    if not args.quiet:
        print("Generating PDB via Rosetta pipeline...")
    dp.process()
    json_path = dp.json_dir

    # Run prediction
    ckpts = [f"pkl/checkpoint_{i}.ckpt" for i in range(1, 6)]
    for c in ckpts:
        if not os.path.exists(c):
            print(f"Error: checkpoint not found: {c}", file=sys.stderr)
            sys.exit(1)

    cmd = (f"python run.py --ckpt {' '.join(ckpts)} "
           f"--test_set {json_path} --save_dir {outdir} --gpu {args.gpu} --id {sid}")
    ret = os.system(cmd)
    if ret != 0:
        print("Error: prediction failed", file=sys.stderr)
        sys.exit(1)

    shutil.rmtree(tmp, ignore_errors=True)

    result_file = os.path.join(outdir, f"{sid}_result.xlsx")
    if not os.path.exists(result_file):
        print("Error: result file not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(result_file)
    score_cols = [c for c in df.columns if c.startswith("result_")]
    if score_cols:
        scores = df[score_cols].values[0]
        mean_score = float(np.mean(scores))
        if args.quiet:
            print(f"{mean_score:.6f}")
        else:
            print(f"\n=== Prediction Results ===")
            print(f"siRNA: {sid}")
            print(f"Sense: {sense}")
            print(f"Anti:  {anti}")
            print(f"mRNA binding pos: {pos}")
            for j, s in enumerate(scores):
                print(f"  Model {j+1}: {s:.6f}")
            print(f"  Mean:   {mean_score:.6f}")
            print(f"\nFull results: {result_file}")


if __name__ == "__main__":
    main()
