#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from data.dataset import E2EDataset
from utils.rna_utils import VOCAB
from utils.logger import print_log
from utils.random_seed import setup_seed
from utils.logger import print_log
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def generate(args):
    # load model
    model = torch.load(args.ckpt, map_location="cpu")
    device = torch.device("cpu" if args.gpu == -1 else f"cuda:{args.gpu}")
    model.to(device)
    model.eval()
    for i in args.test_set:
        # load test set
        test_set = E2EDataset(i)
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=E2EDataset.collate_fn,
        )

        # create save dir
        if args.save_dir is None:
            save_dir = ".".join(args.ckpt.split(".")[:-1]) + "_results"
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ID = []
        with open(i, "r") as fin:
            lines = fin.read().strip().split("\n")
            for line in tqdm(lines):
                item = json.loads(line)

                ID.append(item["siRNA"])

        df_out = pd.DataFrame()
        summary_items = []
        probs = []
        pcts = []

        for batch in tqdm(test_loader):
            pcts += batch["pct"].tolist()
            with torch.no_grad():
                # move data
                for k in batch:
                    if hasattr(batch[k], "to"):
                        batch[k] = batch[k].to(device)

                prob, _, _, _ = model.test(**batch)
                probs += prob.cpu().tolist()

        pcts_arr = np.array(pcts)
        probs_arr = np.array(probs)
        pcc, pcc_p = pearsonr(pcts_arr, probs_arr)
        sp, sp_p = spearmanr(pcts_arr, probs_arr)
        print(f"PCC: {pcc:.4f} (p-value: {pcc_p:.4e})")
        print(f"Spearman: {sp:.4f} (p-value: {sp_p:.4e})")

        df_out["ID"] = ID
        df_out["result"] = probs
        df_out["ground_truth"] = pcts
        df_out.to_excel(f"{save_dir}/{args.id}_result.xlsx")

        with open(f"{save_dir}/{args.id}_metrics.txt", "w") as f:
            f.write(f"PCC: {pcc:.6f}\n")
            f.write(f"PCC p-value: {pcc_p:.6e}\n")
            f.write(f"Spearman: {sp:.6f}\n")
            f.write(f"Spearman p-value: {sp_p:.6e}\n")
            f.write(f"Samples: {len(pcts_arr)}\n")


def parse():
    parser = argparse.ArgumentParser(description="Generate antibodies given epitopes")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--test_set", type=str, required=True, nargs="+", help="Path to test set"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save generated antibodies",
    )

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers to use"
    )

    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use, -1 for cpu")
    parser.add_argument("--id", type=str, default="sirna", help="ID of the design")
    return parser.parse_args()


if __name__ == "__main__":
    setup_seed(12)
    generate(parse())
