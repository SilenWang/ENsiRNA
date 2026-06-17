#!/usr/bin/env python3
"""
ENsiRNA: Single siRNA-mRNA pair scoring

Usage:
  python score.py \\
    --sense CAUGCAUGCAUGCAUGCAU \\
    --mrna UCGCAUGCAUGCAUGCAUGCAUGC...

Options:
  --sense     siRNA sense sequence (19-mer), required
  --antisense siRNA antisense sequence (auto-computed if omitted)
  --mrna      Target mRNA sequence (local region around binding site, ~61 nt+)
  --position  Binding start position in mRNA (0-indexed, default: auto-detect)
  --id        Optional identifier for the siRNA (default: "custom")
  --gpu       GPU device ID (-1 for CPU)
  --outdir    Output directory (default: ./result)
  --quiet     Only print the score, nothing else
"""

import argparse, json, os, sys, tempfile, warnings, shutil
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
    p.add_argument("--quiet", action="store_true", help="Only print the score")
    return p.parse_args()


def gen_pdb(sense, anti, path):
    rise, radius, twist = 2.8, 10.0, 32.7
    sc_map = {"A": "N9", "C": "N1", "G": "N9", "U": "N1", "T": "N1", "N": "N1"}
    anti_comp = {"A": "U", "U": "A", "C": "G", "G": "C", "T": "A", "N": "N"}
    with open(path, "w") as f:
        aidx, ridx = 1, 1
        for i, base in enumerate(sense):
            ang = np.radians(i * twist)
            x, y, z = radius * np.cos(ang), radius * np.sin(ang), i * rise
            for atom, off in [("P", 0), ("C4'", 1.5)]:
                ox = x + off * np.cos(ang + 0.5); oy = y + off * np.sin(ang + 0.5)
                f.write(f"ATOM  {aidx:5d}  {atom:<3s} {base:3s} A{ridx:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom[0]:>2s}\n")
                aidx += 1
            sc = sc_map.get(base, "N1")
            sx, sy, sz = 1.2*radius*np.cos(ang+0.26), 1.2*radius*np.sin(ang+0.26), z
            f.write(f"ATOM  {aidx:5d}  {sc:<3s} {base:3s} A{ridx:4d}    {sx:8.3f}{sy:8.3f}{sz:8.3f}  1.00  0.00           {sc[0]:>2s}\n")
            aidx += 1; ridx += 1
        for i, base in enumerate(anti):
            comp = anti_comp.get(base, "N")
            ang = np.radians((len(sense)-1-i)*twist + 180)
            x, y, z = radius*np.cos(ang), radius*np.sin(ang), (len(sense)-1-i)*rise
            for atom, off in [("P", 0), ("C4'", 1.5)]:
                ox = x + off*np.cos(ang+0.5); oy = y + off*np.sin(ang+0.5)
                f.write(f"ATOM  {aidx:5d}  {atom:<3s} {comp:3s} B{ridx:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom[0]:>2s}\n")
                aidx += 1
            sc = sc_map.get(comp, "N1")
            sx, sy = 1.2*radius*np.cos(ang+0.26), 1.2*radius*np.sin(ang+0.26)
            f.write(f"ATOM  {aidx:5d}  {sc:<3s} {comp:3s} B{ridx:4d}    {sx:8.3f}{sy:8.3f}{sz:8.3f}  1.00  0.00           {sc[0]:>2s}\n")
            aidx += 1; ridx += 1
        f.write("END\n")


def get_chain_start(sense, anti):
    padlen = int((61 - len(sense)) / 2)
    sec, ch = [1000], [0]
    for i in range(-padlen, len(sense) + padlen):
        sec.append(i); ch.append(1)
    sec.append(2000); ch.append(2)
    for i in range(len(sense)):
        sec.append(i); ch.append(3)
    for i in range(len(anti)-1, -1, -1):
        sec.append(i); ch.append(3)
    return sec, ch


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
            print("Error: sense sequence not found in mRNA" if not args.quiet else "", file=sys.stderr)
            sys.exit(1)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="score_", dir=outdir)
    pdb_dir = os.path.join(tmp, "pdbs")
    os.makedirs(pdb_dir)

    pdb_path = os.path.join(pdb_dir, f"{sid}.pdb")
    gen_pdb(sense, anti, pdb_path)

    start, chain = get_chain_start(sense, anti)
    json_path = os.path.join(tmp, f"{sid}.json")
    with open(json_path, "w") as f:
        f.write(json.dumps({
            "siRNA": sid, "mRNA_seq": mrna, "position": pos,
            "sense seq": sense, "anti seq": anti, "efficacy": 0,
            "pdb_data_path": os.path.abspath(pdb_path),
            "start": start, "chain": chain,
        }) + "\n")

    ckpts = [f"pkl/checkpoint_{i}.ckpt" for i in range(1, 6)]
    for c in ckpts:
        if not os.path.exists(c):
            print(f"Error: checkpoint not found: {c}", file=sys.stderr)
            sys.exit(1)

    cmd = f"python run.py --ckpt {' '.join(ckpts)} --test_set {json_path} --save_dir {outdir} --gpu {args.gpu} --id {sid}"
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
