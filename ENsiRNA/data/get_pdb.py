import pandas as pd

import json
from utils.rna_utils import VOCAB
import argparse
import os

import torch
import RNA
import re
import subprocess
import multiprocessing


# Default Rosetta paths (can be overridden via env vars or constructor args)
#   ROSETTA_DIR       — root directory of the Rosetta installation
#   RNA_DENOVO_EXEC   — path to rna_denovo executable
#   EXTRACT_DECOYS    — path to extract_lowscore_decoys.py
_DEFAULT_RF = os.environ.get("ROSETTA_DIR", "")
_DEFAULT_FF = os.environ.get(
    "RNA_DENOVO_EXEC",
    os.path.join(_DEFAULT_RF, "main", "source", "bin",
                 "rna_denovo.static.linuxgccrelease"),
)
_DEFAULT_EX = os.environ.get(
    "EXTRACT_DECOYS",
    os.path.join(_DEFAULT_RF, "main", "tools", "rna_tools",
                 "silent_util", "extract_lowscore_decoys.py"),
)


class Data_Prepare:
    def __init__(self, excel_dir, pdb_dir,
                 rosetta_dir=None, rna_denovo_exec=None, extract_decoys=None):
        self.excel_dir = excel_dir
        self.pdb_dir = os.path.abspath(pdb_dir)

        self.rf = rosetta_dir or _DEFAULT_RF
        self.ff = rna_denovo_exec or _DEFAULT_FF
        self.ex = extract_decoys or _DEFAULT_EX

        self.json_dir = excel_dir[:-4] + ".json"
        self.secondary_structure = True
        self.chunk_size = None
        self.num_cores = multiprocessing.cpu_count()
        self.json = True

        if self.secondary_structure == False:
            self.json_dir = excel_dir[:-4] + "no2.json"

    def get_path(self, siRNA):
        return f"{self.pdb_dir}/{siRNA}.pdb"

    def raw_pre(self, seq):
        return (
            seq.lower()
            .replace(" + ", "")
            .replace("d", "")
            .replace("t", "u")
            .replace(" ", "")
        )

    def get_rpos(self, item):
        pos = [0]
        for i in range(1, 1 + len(item["sense seq"])):
            pos.append(i)
        for i in range(30, 30 + len(item["anti seq"])):
            pos.append(i)
        return pos

    def chunk_dataframe(self, df, chunk_size):
        chunks = []
        chunk_count = len(df) // chunk_size
        for i in range(chunk_count):
            chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
        if len(df) % chunk_size != 0:
            chunks.append(df[chunk_count * chunk_size :])
        return chunks

    def get_anti_start(self, data):
        seq1 = data["sense seq"]
        seq2 = data["anti seq"]
        sec_pos = [1000]
        padlen = int((61 - len(seq1)) / 2)
        chain = [0]
        for i in range(-padlen, len(seq1) + padlen):
            sec_pos.append(i)
            chain.append(1)
        sec_pos.append(2000)
        chain.append(2)
        for i in range(len(seq1)):
            sec_pos.append(i)
            chain.append(3)
        for i in range(len(seq2) - 1, -1, -1):
            sec_pos.append(i)
            chain.append(3)
        return sec_pos, chain

    def process(self):
        os.makedirs(self.pdb_dir, exist_ok=True)
        df = pd.read_csv(self.excel_dir)
        df["sense seq"] = df["sense seq"].apply(self.raw_pre)
        df["anti seq"] = df["anti seq"].apply(self.raw_pre)

        df[["start", "chain"]] = df.apply(
            self.get_anti_start, axis=1, result_type="expand"
        )

        total_rows = len(df)
        self.chunk_size = max(1, total_rows // self.num_cores)
        chunks = self.chunk_dataframe(df, self.chunk_size)
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            drops = pool.map(self.get_data, chunks)

        # print(drops)

        # df=df.drop(index=drops)
        if self.json == True:
            dfj = df[
                [
                    "siRNA",
                    "mRNA_seq",
                    "position",
                    "sense seq",
                    "anti seq",
                    "efficacy",
                    "start",
                    "chain",
                ]
            ]  # delect marker

            dfj["pdb_data_path"] = dfj["siRNA"].apply(self.get_path)
            dfj["efficacy"] = dfj["efficacy"].apply(lambda x: x if x > 0 else 0)
            dfj = dfj.dropna()
            dfj.to_json(self.json_dir, orient="records", lines=True)

    def get_data(self, df):

        drops = []

        for index, row in df.iterrows():
            if os.path.exists(f"{self.pdb_dir}/{row['siRNA']}.pdb") == True:
                continue
            if os.path.exists(f"{self.pdb_dir}/{row['siRNA']}") == True:
                continue
            if self.secondary_structure == True:
                if self.get_secondary_structure(row) == False:
                    print(f"drop{self.pdb_dir}/{row['siRNA']}.pdb")
                    drops.append(index)

            else:
                if self.get_structure(row) == False:
                    print(f"drop{self.pdb_dir}/{row['siRNA']}.pdb")
                    drops.append(index)

        return drops

    def get_structure(self, data):

        seq1 = data["sense seq"]
        seq2 = data["anti seq"]

        pose = assembler.build_init_pose(seq1, seq2)
        pose.dump_pdb(f"{self.pdb_dir}/{data['siRNA']}.pdb")
        if os.path.exists(f"{self.pdb_dir}/{data['siRNA']}.pdb"):
            return True
        else:
            return False

    def get_secondary_structure(self, data):
        seq1 = data["sense seq"]
        seq2 = data["anti seq"]
        seq = seq1 + " " + seq2

        duplex = RNA.duplexfold(seq1, seq2)
        secondary_seq = duplex.structure.replace("&", " ")

        os.makedirs(f"{self.pdb_dir}/{data['siRNA']}", exist_ok=True)
        os.chdir(f"{self.pdb_dir}/{data['siRNA']}")

        subprocess.run(
            [self.ff, "-sequence", seq, "-secstruct", secondary_seq, "-minimize_rna"]
        )
        subprocess.run(["python", self.ex, "default.out", "-rosetta_folder", self.rf, "1"])
        subprocess.run(
            ["cp", "default.out.1.pdb", f"{self.pdb_dir}/{data['siRNA']}.pdb"]
        )

        subprocess.run(["rm", "-r", f"{self.pdb_dir}/{data['siRNA']}"])

        if os.path.exists(f"{self.pdb_dir}/{data['siRNA']}.pdb"):
            return True
        else:
            return False


def parse():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("-f", "--filenames", nargs="+", help="train/valsiRNA/test set")
    parser.add_argument(
        "-p", "--pdb_dir", type=str, default=None, help="Path to save processed data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    for filename in args.filenames:
        Data_Prepare(filename, args.pdb_dir).process()
