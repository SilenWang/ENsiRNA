#!/usr/bin/env python3
"""
ENsiRNA Prediction CLI

Predict siRNA efficacy against a target mRNA.

Input: mRNA FASTA file (required) — provides the target mRNA sequence context.
The model enumerates all possible 19-mer siRNAs targeting the mRNA, or you can
specify custom siRNA sequences with --sense / --antisense.

Output: Excel file with prediction scores for each candidate siRNA.

Examples:
  # Predict all possible 19-mer siRNAs targeting an mRNA
  python predict.py -f mrna.fasta -o ./results

  # Predict a specific siRNA against an mRNA
  python predict.py -f mrna.fasta --sense CAUGCAUGCAUGCAUGCAU -o ./results

  # Use GPU
  python predict.py -f mrna.fasta -o ./results --gpu 0
"""

import argparse
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm

warnings.filterwarnings("ignore")

RNA_BASES = {"A", "C", "G", "U", "T"}


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
        "--keep-temp", action="store_true",
        help="Keep temporary files (PDB, JSON) after prediction"
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


def generate_simple_pdb(sense_seq, anti_seq, pdb_path):
    sense_seq = sense_seq.upper()
    anti_seq = anti_seq.upper()
    rise = 2.8
    radius = 10.0
    twist = 32.7

    with open(pdb_path, "w") as f:
        atom_idx = 1
        res_idx = 1
        for i, base in enumerate(sense_seq):
            angle = np.radians(i * twist)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = i * rise
            for atom_name, offset in [("P", 0), ("C4'", 1.5)]:
                ox = x + offset * np.cos(angle + 0.5)
                oy = y + offset * np.sin(angle + 0.5)
                oz = z
                f.write(
                    f"ATOM  {atom_idx:5d}  {atom_name:<3s} {base:3s} A{res_idx:4d}    "
                    f"{ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                )
                atom_idx += 1
            sc_map = {"A": "N9", "C": "N1", "G": "N9", "U": "N1", "T": "N1", "N": "N1"}
            sc_atom = sc_map.get(base, "N1")
            sx = 1.2 * radius * np.cos(angle + np.radians(15))
            sy = 1.2 * radius * np.sin(angle + np.radians(15))
            sz = z
            f.write(
                f"ATOM  {atom_idx:5d}  {sc_atom:<3s} {base:3s} A{res_idx:4d}    "
                f"{sx:8.3f}{sy:8.3f}{sz:8.3f}  1.00  0.00           {sc_atom[0]:>2s}\n"
            )
            atom_idx += 1
            res_idx += 1
        anti_comp = {"A": "U", "U": "A", "C": "G", "G": "C", "T": "A", "N": "N"}
        for i, base in enumerate(anti_seq):
            comp_base = anti_comp.get(base, "N")
            angle = np.radians((len(sense_seq) - 1 - i) * twist + 180)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = (len(sense_seq) - 1 - i) * rise
            for atom_name, offset in [("P", 0), ("C4'", 1.5)]:
                ox = x + offset * np.cos(angle + 0.5)
                oy = y + offset * np.sin(angle + 0.5)
                oz = z
                f.write(
                    f"ATOM  {atom_idx:5d}  {atom_name:<3s} {comp_base:3s} B{res_idx:4d}    "
                    f"{ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                )
                atom_idx += 1
            sc_atom = sc_map.get(comp_base, "N1")
            sx = 1.2 * radius * np.cos(angle + np.radians(15))
            sy = 1.2 * radius * np.sin(angle + np.radians(15))
            sz = z
            f.write(
                f"ATOM  {atom_idx:5d}  {sc_atom:<3s} {comp_base:3s} B{res_idx:4d}    "
                f"{sx:8.3f}{sy:8.3f}{sz:8.3f}  1.00  0.00           {sc_atom[0]:>2s}\n"
            )
            atom_idx += 1
            res_idx += 1
        f.write("END\n")


def get_anti_start(sense_seq, anti_seq):
    padlen = int((61 - len(sense_seq)) / 2)
    sec_pos = [1000]
    chain = [0]
    for i in range(-padlen, len(sense_seq) + padlen):
        sec_pos.append(i)
        chain.append(1)
    sec_pos.append(2000)
    chain.append(2)
    for i in range(len(sense_seq)):
        sec_pos.append(i)
        chain.append(3)
    for i in range(len(anti_seq) - 1, -1, -1):
        sec_pos.append(i)
        chain.append(3)
    return sec_pos, chain


def prepare_data(sirnas, pdb_dir, json_path):
    rows = []
    for sid, sense, anti, mrna_seq, position in sirnas:
        pdb_path = os.path.join(pdb_dir, f"{sid}.pdb")
        if not os.path.exists(pdb_path):
            generate_simple_pdb(sense, anti, pdb_path)
        start, chain = get_anti_start(sense, anti)
        rows.append({
            "siRNA": sid,
            "mRNA_seq": mrna_seq,
            "position": position,
            "sense seq": sense,
            "anti seq": anti,
            "efficacy": 0,
            "pdb_data_path": os.path.abspath(pdb_path),
            "start": start,
            "chain": chain,
        })
    with open(json_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_prediction(json_path, output_dir, sirna_id, gpu):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_dir = os.path.join(script_dir, "pkl")
    ckpts = [
        os.path.join(pkl_dir, "checkpoint_1.ckpt"),
        os.path.join(pkl_dir, "checkpoint_2.ckpt"),
        os.path.join(pkl_dir, "checkpoint_3.ckpt"),
        os.path.join(pkl_dir, "checkpoint_4.ckpt"),
        os.path.join(pkl_dir, "checkpoint_5.ckpt"),
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

    if args.sense:
        sense = args.sense.upper().replace("T", "U")
        if args.antisense:
            anti = args.antisense.upper().replace("T", "U")
        else:
            anti = str(Seq(sense).reverse_complement())
        sid = args.sirna_id or "custom_sirna"
        # Match the siRNA against each mRNA record to find position
        all_sirnas = []
        for mrna_id, mrna_seq in records.items():
            mrna_seq_clean = mrna_seq.replace("\n", "").replace(" ", "").upper().replace("T", "U")
            pos = mrna_seq_clean.find(sense)
            if pos >= 0:
                all_sirnas.append([sid, sense, anti, mrna_seq_clean, pos])
                print(f"  Found siRNA binding site at position {pos} in {mrna_id}")
            else:
                print(f"  Warning: siRNA sense sequence not found in {mrna_id}")
                # Still predict with position 0
                all_sirnas.append([sid, sense, anti, mrna_seq_clean, 0])
        if not all_sirnas:
            print("Error: Could not match siRNA to any mRNA record")
            sys.exit(1)
    else:
        mrna_prefix = args.sirna_id or os.path.splitext(os.path.basename(args.fasta))[0]
        all_sirnas = []
        for mrna_id, mrna_seq in records.items():
            mrna_seq_clean = mrna_seq.replace("\n", "").replace(" ", "").upper().replace("T", "U")
            candidates = generate_sirnas_from_mrna(mrna_id, mrna_seq_clean)
            all_sirnas.extend(candidates)
            print(f"  {mrna_id}: {len(mrna_seq_clean)} nt → {len(candidates)} siRNA candidates")
        if not all_sirnas:
            print("Error: mRNA sequence too short (< 19 nt)")
            sys.exit(1)

    sirna_id = args.sirna_id or (args.sense[:10] if args.sense else
                                  os.path.splitext(os.path.basename(args.fasta))[0])
    print(f"\nTotal siRNAs to predict: {len(all_sirnas)}")

    temp_dir = tempfile.mkdtemp(prefix="ensirna_", dir=output_dir)
    pdb_dir = os.path.join(temp_dir, "pdbs")
    os.makedirs(pdb_dir, exist_ok=True)

    print(f"Generating PDB structures...")
    for sid, sense, anti, _mrna, _pos in tqdm(all_sirnas):
        pdb_path = os.path.join(pdb_dir, f"{sid}.pdb")
        if not os.path.exists(pdb_path):
            generate_simple_pdb(sense, anti, pdb_path)

    json_path = os.path.join(temp_dir, f"{sirna_id}.json")
    print(f"Preparing data...")
    prepare_data(all_sirnas, pdb_dir, json_path)

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
