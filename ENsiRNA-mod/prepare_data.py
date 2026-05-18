#!/usr/bin/python
import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.mod_utils import MOD_VOCAB, MOGANRdkit_VOCAB

def raw_pre(seq):
    return seq.lower().replace(' + ', '').replace('d', '').replace('y', 'g').replace('x', 't').replace(' ', '')

def seq_pre(seq):
    return seq.replace('t', 'u')

def get_atommod(item):
    VOCAB_MAX_ATOM_NUMBER = 3
    atom_mask = [[0 for _ in range(VOCAB_MAX_ATOM_NUMBER)] for _ in range(1 + len(item['sense seq']) + len(item['anti seq']))]
    
    smode = None
    amode = None
    if item['sense mod'] != 'none' and item['sense mod'] != 0 and not pd.isna(item['sense mod']):
        smode = str(item['sense mod']).split('* ')
        spose = str(item['sense pos']).replace(' ', '').split('*')
    if item['anti mod'] != 'none' and item['anti mod'] != 0 and not pd.isna(item['anti mod']):    
        amode = str(item['anti mod']).split('* ')
        apose = str(item['anti pos']).replace(' ', '').split('*')

    offset = len(item['sense seq'])
    base_map = {'a': 'Standard adenine', 'c': 'Standard cytosine', 'g': 'Standard guanine', 
                'u': 'Standard uracil', 't': 'Standard thymine'}

    for pos in range(1, 1 + len(item['sense seq'])):
        atom_mask[pos][0] = MOD_VOCAB.mod2index.get('Standard sugar', 0)
        atom_mask[pos][1] = MOD_VOCAB.mod2index.get('Standard phosphate', 0)
        base = item['sense raw seq'][pos - 1]
        if base in base_map:
            atom_mask[pos][2] = MOD_VOCAB.mod2index.get(base_map[base], 0)
        if base == 't':
            atom_mask[pos][0] = MOD_VOCAB.mod2index.get('2-Deoxyribonucleotide', 0)

    if smode is not None:
        for i in range(len(smode)):
            smode_i = smode[i].strip()
            if smode_i in MOD_VOCAB.base_mod:
                for pos in str(spose[i]).split(','):
                    atom_mask[int(pos)][2] = MOD_VOCAB.mod2index.get(smode_i, 0)
            if smode_i in MOD_VOCAB.phosphate_mod:
                for pos in str(spose[i]).split(','):
                    atom_mask[int(pos)][1] = MOD_VOCAB.mod2index.get(smode_i, 0)
            if smode_i in MOD_VOCAB.sugar_mod:
                for pos in str(spose[i]).split(','):
                    atom_mask[int(pos)][0] = MOD_VOCAB.mod2index.get(smode_i, 0)

    _mask = [[0 for _ in range(VOCAB_MAX_ATOM_NUMBER)] for _ in range(len(item['anti seq']))]
    for pos in range(len(item['anti seq'])):
        base = item['anti raw seq'][pos]
        if base in base_map:
            _mask[pos][0] = MOD_VOCAB.mod2index.get('Standard sugar', 0)
            _mask[pos][1] = MOD_VOCAB.mod2index.get('Standard phosphate', 0)
            _mask[pos][2] = MOD_VOCAB.mod2index.get(base_map[base], 0)
        if base == 't':
            _mask[pos][0] = MOD_VOCAB.mod2index.get('2-Deoxyribonucleotide', 0)

    if amode is not None:
        for i in range(len(amode)):
            amode_i = amode[i].strip()
            if amode_i in MOD_VOCAB.base_mod:
                for pos in str(apose[i]).split(','):
                    _mask[int(pos) - 1][2] = MOD_VOCAB.mod2index.get(amode_i, 0)
            if amode_i in MOD_VOCAB.phosphate_mod:
                for pos in str(apose[i]).split(','):
                    _mask[int(pos) - 1][1] = MOD_VOCAB.mod2index.get(amode_i, 0)
            if amode_i in MOD_VOCAB.sugar_mod:
                for pos in str(apose[i]).split(','):
                    _mask[int(pos) - 1][0] = MOD_VOCAB.mod2index.get(amode_i, 0)

    atom_mask[offset + 1:] = _mask
    return atom_mask

def get_smask(item):
    sp = []
    ap = []
    sense_pos = str(item['sense pos']).replace(' ', '') if not pd.isna(item['sense pos']) else ''
    anti_pos = str(item['anti pos']).replace(' ', '') if not pd.isna(item['anti pos']) else ''
    
    for i in sense_pos.split('*'):
        sp += i.split(',')
    for i in anti_pos.split('*'):
        ap += i.split(',')
    
    smask = [int(i) for i in set(sp) if i] + [int(i) + len(item['sense seq']) for i in set(ap) if i]
    return smask

def generate_simple_pdb(sense_seq, anti_seq, pdb_path):
    import math
    sense_seq = sense_seq.upper()
    anti_seq = anti_seq.upper()
    rise = 2.8
    radius = 10.0
    twist = 32.7
    
    with open(pdb_path, 'w') as f:
        atom_idx = 1
        res_idx = 1
        for i, base in enumerate(sense_seq):
            angle = math.radians(i * twist)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = i * rise
            for atom_name, ox, oy, oz in [
                ("P", x, y, z), 
                ("C4'", x + 1.5 * math.cos(angle + 0.5), y + 1.5 * math.sin(angle + 0.5), z)
            ]:
                f.write(f"ATOM  {atom_idx:5d}  {atom_name:<3s} {base:3s} A{res_idx:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n")
                atom_idx += 1
            sc_map = {"A": "N9", "C": "N1", "G": "N9", "U": "N1", "T": "N1"}
            sc = sc_map.get(base, "N1")
            sx = 1.2 * radius * math.cos(angle + math.radians(15))
            sy = 1.2 * radius * math.sin(angle + math.radians(15))
            f.write(f"ATOM  {atom_idx:5d}  {sc:<3s} {base:3s} A{res_idx:4d}    {sx:8.3f}{sy:8.3f}{z:8.3f}  1.00  0.00           {sc[0]:>2s}\n")
            atom_idx += 1
            res_idx += 1
        comp = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C', 'T': 'A'}
        for i, base in enumerate(anti_seq):
            cb = comp.get(base, 'N')
            ai = len(sense_seq) - 1 - i
            angle = math.radians(ai * twist + 180)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = ai * rise
            for atom_name, ox, oy, oz in [
                ("P", x, y, z),
                ("C4'", x + 1.5 * math.cos(angle + 0.5), y + 1.5 * math.sin(angle + 0.5), z)
            ]:
                f.write(f"ATOM  {atom_idx:5d}  {atom_name:<3s} {cb:3s} B{res_idx:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n")
                atom_idx += 1
            sc_map = {"A": "N9", "C": "N1", "G": "N9", "U": "N1", "T": "N1"}
            sc = sc_map.get(cb, "N1")
            sx = 1.2 * radius * math.cos(angle + math.radians(15))
            sy = 1.2 * radius * math.sin(angle + math.radians(15))
            f.write(f"ATOM  {atom_idx:5d}  {sc:<3s} {cb:3s} B{res_idx:4d}    {sx:8.3f}{sy:8.3f}{z:8.3f}  1.00  0.00           {sc[0]:>2s}\n")
            atom_idx += 1
            res_idx += 1
        f.write("END\n")

def get_start(item):
    seq1 = item['sense seq']
    seq2 = item['anti seq']
    sec_pos = [1000]
    for i in range(len(seq1)):
        sec_pos.append(i)
    for i in range(len(seq2) - 1, -1, -1):
        sec_pos.append(i)
    return sec_pos

def main():
    parser = argparse.ArgumentParser(description='Prepare ENsiRNA-mod data')
    parser.add_argument('--xlsx_dir', type=str, required=True)
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--json_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.json_dir is None:
        args.json_dir = args.xlsx_dir
    os.makedirs(args.pdb_dir, exist_ok=True)
    os.makedirs(args.json_dir, exist_ok=True)

    for prefix in ['train_88', 'valid_88']:
        for i in range(1, 6):
            xlsx_file = os.path.join(args.xlsx_dir, f'{prefix}_{i}.xlsx')
            json_file = os.path.join(args.json_dir, f'{prefix}_{i}.json')
            if not os.path.exists(xlsx_file):
                continue
            
            print(f'Processing {xlsx_file} -> {json_file}')
            df = pd.read_excel(xlsx_file)
            items = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                sense_raw = raw_pre(row['sense raw seq'])
                anti_raw = raw_pre(row['anti raw seq'])
                sense_seq = seq_pre(sense_raw)
                anti_seq = seq_pre(anti_raw)
                entry_id = row['ID']
                pdb_path = os.path.join(args.pdb_dir, f"{entry_id}.pdb")
                item = {
                    'ID': entry_id,
                    'sense seq': sense_seq,
                    'sense raw seq': sense_raw,
                    'anti seq': anti_seq,
                    'anti raw seq': anti_raw,
                    'PCT': float(row['PCT']),
                    'sense mod': row.get('sense mod', 'none'),
                'sense pos': str(row['sense pos']) if not pd.isna(row.get('sense pos')) else '0',
                'anti mod': row.get('anti mod', 'none'),
                'anti pos': str(row['anti pos']) if not pd.isna(row.get('anti pos')) else '0',
                    'cc': float(row['cc']),
                    'pdb_data_path': pdb_path,
                    'start': [],
                    'smask': [],
                    'atom_mask': [],
                }
                item['start'] = get_start(item)
                item['smask'] = get_smask(item)
                item['atom_mask'] = get_atommod(item)
                items.append(item)

            with open(json_file, 'w') as f:
                for item in items:
                    f.write(json.dumps(item) + '\n')

            for item in tqdm(items, desc=f'Generating PDBs for {prefix}_{i}'):
                pdb_path = item['pdb_data_path']
                if not os.path.exists(pdb_path):
                    generate_simple_pdb(item['sense seq'], item['anti seq'], pdb_path)

if __name__ == '__main__':
    main()
