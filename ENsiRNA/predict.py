#!/usr/bin/env python3
"""
ENsiRNA Prediction CLI

Predict siRNA efficacy against a target mRNA.

Input: mRNA FASTA file — provides the target mRNA sequence context.
The model enumerates all possible 19-mer siRNAs targeting the mRNA, or you can
specify custom siRNA sequences with --sense / --antisense.

PDB generation (required by the model) uses Rosetta via the original pipeline.
Set ROSETTA_DIR or pass --rosetta-dir to point to your Rosetta installation.

Output: Excel file with prediction scores for each candidate siRNA.

Examples:
  # Predict all possible 19-mer siRNAs targeting an mRNA
  python predict.py -f mrna.fasta -o ./results --rosetta-dir /path/to/rosetta

  # Predict a specific siRNA against an mRNA
  python predict.py -f mrna.fasta --sense CAUGCAUGCAUGCAUGCAU -o ./results

  # Use GPU
  python predict.py -f mrna.fasta -o ./results --gpu 0
"""

import argparse
import csv
import os
import sys
import tempfile
import warnings

import pandas as pd
from Bio.Seq import Seq

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ENsiRNA: Predict siRNA efficacy against a target mRNA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-f", "--fasta", type=str, required=True,
        help="Input mRNA FASTA file (required — provides target sequence context)"
    )
    parser.add_argument(
        "--sense", type=str, default=None,
        help="Custom siRNA sense sequence (19-mer). If omitted, all 19-mers are enumerated."
    )
    parser.add_argument(
        "--antisense", type=str, default=None,
        help="Custom siRNA antisense sequence (19-mer). Auto-computed from sense if omitted."
    )
    parser.add_argument(
        "-s", "--sirna-id", type=str, default=None,
        help="Custom siRNA identifier (default: auto-generated)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="result",
        help="Output directory (default: 'result')"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="GPU device ID (-1 for CPU, default: -1)"
    )
    parser.add_argument(
        "--rosetta-dir", type=str, default=None,
        help="Rosetta installation directory (or set ROSETTA_DIR env var)"
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary files (CSV, PDB, JSON) after prediction"
    )
    return parser.parse_args()


def read_fasta(fasta_path):
    records = {}
    with open(fasta_path, "r") as f:
        seq_id = None
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records[seq_id] = "".join(seq)
                seq_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.upper())
        if seq_id is not None:
            records[seq_id] = "".join(seq)
    return records


def generate_sirnas_from_mrna(mrna_id, mrna_seq):
    mrna_seq = mrna_seq.replace("T", "U")
    candidates = []
    for i in range(len(mrna_seq) - 18):
        sense = mrna_seq[i: i + 19]
        anti = str(Seq(sense).reverse_complement())
        sid = f"{mrna_id}_{i}"
        candidates.append([sid, sense, anti, mrna_seq, i])
    return candidates


def run_prediction(json_path, output_dir, sirna_id, gpu):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_dir = os.path.join(script_dir, "pkl")
    ckpts = [
        os.path.join(pkl_dir, f"checkpoint_{i}.ckpt") for i in range(1, 6)
    ]
    for ckpt in ckpts:
        if not os.path.exists(ckpt):
            print(f"Error: Checkpoint not found: {ckpt}")
            sys.exit(1)

    cmd = (
        f"python {os.path.join(script_dir, 'run.py')} "
        f"--ckpt {' '.join(ckpts)} "
        f"--test_set {json_path} "
        f"--save_dir {output_dir} "
        f"--gpu {gpu} "
        f"--id {sirna_id}"
    )
    print(f"\nRunning prediction with {len(ckpts)} ensemble models...")
    ret = os.system(cmd)
    if ret != 0:
        print("Error: Prediction failed")
        sys.exit(1)

    result_file = os.path.join(output_dir, f"{sirna_id}_result.xlsx")
    if os.path.exists(result_file):
        return result_file
    return None


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading mRNA FASTA: {args.fasta}")
    records = read_fasta(args.fasta)
    if not records:
        print("Error: No sequences found in FASTA file")
        sys.exit(1)

    # Generate siRNA candidate list
    if args.sense:
        sense = args.sense.upper().replace("T", "U")
        anti = (args.antisense or str(Seq(sense).reverse_complement())).upper().replace("T", "U")
        sid = args.sirna_id or "custom_sirna"
        all_sirnas = []
        for mrna_id, mrna_seq in records.items():
            mrna_clean = mrna_seq.replace("\n", "").replace(" ", "").upper().replace("T", "U")
            pos = mrna_clean.find(sense)
            if pos >= 0:
                print(f"  Found siRNA binding site at position {pos} in {mrna_id}")
            else:
                print(f"  Warning: siRNA sense sequence not found in {mrna_id}")
                pos = 0
            all_sirnas.append([sid, sense, anti, mrna_clean, pos])
        if not all_sirnas:
            print("Error: Could not match siRNA to any mRNA record")
            sys.exit(1)
    else:
        sirna_id = args.sirna_id or os.path.splitext(os.path.basename(args.fasta))[0]
        all_sirnas = []
        for mrna_id, mrna_seq in records.items():
            mrna_clean = mrna_seq.replace("\n", "").replace(" ", "").upper().replace("T", "U")
            candidates = generate_sirnas_from_mrna(mrna_id, mrna_clean)
            all_sirnas.extend(candidates)
            print(f"  {mrna_id}: {len(mrna_clean)} nt → {len(candidates)} siRNA candidates")
        if not all_sirnas:
            print("Error: mRNA sequence too short (< 19 nt)")
            sys.exit(1)

    sirna_id = args.sirna_id or (args.sense[:10] if args.sense else
                                  os.path.splitext(os.path.basename(args.fasta))[0])
    print(f"\nTotal siRNAs to predict: {len(all_sirnas)}")

    # Write CSV for Data_Prepare
    temp_dir = tempfile.mkdtemp(prefix="ensirna_", dir=output_dir)
    csv_path = os.path.join(temp_dir, f"{sirna_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["siRNA", "sense seq", "anti seq", "mRNA_seq", "position", "efficacy"])
        for sid, sense, anti, mrna_seq, pos in all_sirnas:
            writer.writerow([sid, sense, anti, mrna_seq, pos, 0])
    print(f"CSV written: {csv_path}")

    # Generate PDBs and JSON via the original Rosetta pipeline
    from data.get_pdb import Data_Prepare
    pdb_dir = os.path.join(temp_dir, "pdbs")
    dp = Data_Prepare(csv_path, pdb_dir, rosetta_dir=args.rosetta_dir)
    print(f"\nGenerating PDB structures via Rosetta pipeline...")
    dp.process()
    json_path = dp.json_dir
    print(f"JSON data: {json_path}")

    # Run prediction
    result_file = run_prediction(json_path, output_dir, sirna_id, args.gpu)

    if not args.keep_temp:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleaned up temporary files")

    if result_file:
        df = pd.read_excel(result_file)
        print(f"\n{'='*65}")
        print(f"Prediction Results ({len(df)} siRNAs)")
        print(f"{'='*65}")
        score_cols = [c for c in df.columns if c.startswith("result_")]
        if score_cols:
            df["mean_score"] = df[score_cols].mean(axis=1)
            df = df.sort_values("mean_score", ascending=False)
            display_cols = ["ID", "sense seq", "anti seq"] + score_cols + ["mean_score"]
            print(df[display_cols].to_string(index=False, float_format="%.4f"))
        print(f"\nFull results: {result_file}")


if __name__ == "__main__":
    main()
