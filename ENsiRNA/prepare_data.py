#!/usr/bin/python
import os
import json
import csv
import argparse
import numpy as np
from tqdm import tqdm
import RNA

def raw_pre(seq):
    return seq.lower().replace(' + ', '').replace('d', '').replace('t', 'u').replace(' ', '')

def get_secondary_structure(seq1, seq2):
    """Get secondary structure using RNAplex"""
    import subprocess
    import re
    com = f"echo -e \"{seq1}\n{seq2}\n\" | RNAplex"
    raw_se = subprocess.run(com, shell=True, capture_output=True, text=True).stdout
    return raw_se

def generate_simple_pdb(sense_seq, anti_seq, pdb_path):
    """Generate a simple PDB file for siRNA (sense+anti duplex)"""
    sense_seq = sense_seq.upper()
    anti_seq = anti_seq.upper()
    
    # Simple A-form RNA helix parameters
    rise = 2.8  # Angstroms per residue
    radius = 10.0  # Helix radius
    twist = 32.7  # Degrees per residue
    
    with open(pdb_path, 'w') as f:
        atom_idx = 1
        res_idx = 1
        
        # Sense strand (chain A)
        for i, base in enumerate(sense_seq):
            angle = np.radians(i * twist)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = i * rise
            
            # Backbone atoms: P and C4'
            for atom_name, offset in [("P", 0), ("C4'", 1.5)]:
                ox = x + offset * np.cos(angle + 0.5)
                oy = y + offset * np.sin(angle + 0.5)
                oz = z
                f.write(f"ATOM  {atom_idx:5d}  {atom_name:<3s} {base:3s} A{res_idx:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n")
                atom_idx += 1
            
            # Sidechain: base-specific
            sidechain_atom = {"A": "N9", "C": "N1", "G": "N9", "U": "N1", "T": "N1", "N": "N1"}
            sc_atom = sidechain_atom.get(base, "N1")
            sx = 1.2 * radius * np.cos(angle + np.radians(15))
            sy = 1.2 * radius * np.sin(angle + np.radians(15))
            sz = z
            f.write(f"ATOM  {atom_idx:5d}  {sc_atom:<3s} {base:3s} A{res_idx:4d}    {sx:8.3f}{sy:8.3f}{sz:8.3f}  1.00  0.00           {sc_atom[0]:>2s}\n")
            atom_idx += 1
            res_idx += 1
        
        # Anti strand (chain B) - complementary
        anti_complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        for i, base in enumerate(anti_seq):
            comp_base = anti_complement.get(base, 'N')
            angle = np.radians((len(sense_seq) - 1 - i) * twist + 180)  # opposite side
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = (len(sense_seq) - 1 - i) * rise
            
            for atom_name, offset in [("P", 0), ("C4'", 1.5)]:
                ox = x + offset * np.cos(angle + 0.5)
                oy = y + offset * np.sin(angle + 0.5)
                oz = z
                f.write(f"ATOM  {atom_idx:5d}  {atom_name:<3s} {comp_base:3s} B{res_idx:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n")
                atom_idx += 1
            
            sidechain_atom = {"A": "N9", "C": "N1", "G": "N9", "U": "N1", "T": "N1", "N": "N1"}
            sc_atom = sidechain_atom.get(comp_base, "N1")
            sx = 1.2 * radius * np.cos(angle + np.radians(15))
            sy = 1.2 * radius * np.sin(angle + np.radians(15))
            sz = z
            f.write(f"ATOM  {atom_idx:5d}  {sc_atom:<3s} {comp_base:3s} B{res_idx:4d}    {sx:8.3f}{sy:8.3f}{sz:8.3f}  1.00  0.00           {sc_atom[0]:>2s}\n")
            atom_idx += 1
            res_idx += 1
        
        f.write("END\n")

def generate_chain_and_start(sense_seq, anti_seq, position, mrna_seq_len):
    """Generate chain and start position arrays matching the dataset format"""
    seq1 = sense_seq
    padlen = int((61 - len(seq1)) / 2)
    
    sec_pos = [1000]
    chain = [0]
    for i in range(-padlen, len(seq1) + padlen):
        sec_pos.append(i)
        chain.append(1)
    sec_pos.append(2000)
    chain.append(2)
    for i in range(len(seq1)):
        sec_pos.append(i)
        chain.append(3)
    for i in range(len(seq1) - 1, -1, -1):
        sec_pos.append(i)
        chain.append(3)
    
    return sec_pos, chain

def csv_to_json(csv_file, pdb_dir, json_file):
    """Convert CSV to JSON lines format"""
    items = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            sirna_id = row['siRNA']
            sense_seq = raw_pre(row['sense seq'])
            anti_seq = raw_pre(row['anti seq'])
            pdb_path = os.path.join(pdb_dir, f"{sirna_id}.pdb")
            
            start, chain = generate_chain_and_start(
                sense_seq, anti_seq, 
                int(row['position']), 
                len(row['mRNA_seq'])
            )
            
            item = {
                'siRNA': sirna_id,
                'mRNA_seq': row['mRNA_seq'],
                'position': int(row['position']),
                'sense seq': sense_seq,
                'anti seq': anti_seq,
                'efficacy': float(row['efficacy']),
                'pdb_data_path': pdb_path,
                'start': start,
                'chain': chain
            }
            items.append(item)
    
    with open(json_file, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
    
    return items

def main():
    parser = argparse.ArgumentParser(description='Prepare data for ENsiRNA training')
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory with CSV files')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory to save PDB files')
    parser.add_argument('--json_dir', type=str, default=None, help='Directory to save JSON files')
    args = parser.parse_args()
    
    if args.json_dir is None:
        args.json_dir = args.csv_dir
    
    os.makedirs(args.pdb_dir, exist_ok=True)
    os.makedirs(args.json_dir, exist_ok=True)
    
    # Process training and validation files
    for prefix in ['train', 'valid']:
        for i in range(1, 6):
            csv_file = os.path.join(args.csv_dir, f'{prefix}_{i}.csv')
            json_file = os.path.join(args.json_dir, f'{prefix}_{i}.json')
            
            if not os.path.exists(csv_file):
                print(f'Skipping {csv_file}, not found')
                continue
            
            print(f'Processing {csv_file} -> {json_file}')
            items = csv_to_json(csv_file, args.pdb_dir, json_file)
            
            # Generate PDB files
            for item in tqdm(items, desc=f'Generating PDBs for {prefix}_{i}'):
                pdb_path = item['pdb_data_path']
                if not os.path.exists(pdb_path):
                    try:
                        generate_simple_pdb(item['sense seq'], item['anti seq'], pdb_path)
                    except Exception as e:
                        print(f'Failed to generate PDB for {item["siRNA"]}: {e}')

if __name__ == '__main__':
    main()
