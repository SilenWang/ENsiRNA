#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
ENsiRNA 训练代码 —— 带详细中文注释版
=========================================

本文件是对 ENsiRNA 项目训练部分的完整解析与拆分，
包含从数据加载、预处理、模型构建到训练循环的每一步说明。

原始项目地址: https://github.com/SilenWang/ENsiRNA
"""

# ============================================================
# 第一部分：依赖导入与全局设置
# ============================================================

import os
import json
import re
import argparse
import pickle
from typing import List
from math import cos, pi, log, exp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

import RNA  # ViennaRNA 包，用于计算 RNA 热力学能
import fm  # RNA-FM 预训练模型，用于提取 RNA 序列特征

# ---------- 工具函数 ----------
from utils.logger import print_log
from utils.random_seed import setup_seed, SEED, seed_worker
from utils.rna_utils import SIRNA, VOCAB, RNAFeature
from utils.nn_utils import (
    SeparatedNucleicAcidFeature,
    NucleicAcidEmbedding,
    EdgeConstructor,
    GMEdgeConstructor,
    _knn_edges,
)
from utils.singleton import singleton
from model.am_egnn3 import AMEGNN, AM_E_GCL, RollerPooling, coord2radial

# ============================================================
# 第二部分：随机种子设置
# ============================================================
# 保证实验可重复性：固定 PyTorch / NumPy / Python 随机数生成器
setup_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)


# ============================================================
# 第三部分：RNA-FM 预训练模型初始化
# ============================================================
# RNA-FM 是一个基于 BERT 的 RNA 语言模型，
# 用于从 RNA 序列中提取 640 维的上下文特征表示。
# 这些特征作为模型的输入之一，提供序列层面的语义信息。

if torch.cuda.is_available():
    fm_device = "cuda"
elif torch.backends.mps.is_available():
    fm_device = "mps"
else:
    fm_device = "cpu"
fm_model, fm_alphabet = fm.pretrained.rna_fm_t12()
fm_batch_converter = fm_alphabet.get_batch_converter()
fm_model.eval()
fm_model.to(device=fm_device)


# ============================================================
# 第四部分：二核苷酸热力学参数表
# ============================================================
# 来自实验测定的 RNA 二核苷酸（最近邻模型）自由能参数（kcal/mol），
# 用于计算每个碱基位置的局部热力学稳定性。
# 来源：Turner 规则 (1999)

EN_DICT = {
    "AA": -0.93, "UU": -0.93,
    "AU": -1.10, "UA": -1.33,
    "CU": -2.08, "AG": -2.08,
    "CA": -2.11, "UG": -2.11,
    "GU": -2.24, "AC": -2.24,
    "GA": -2.35, "UC": -2.35,
    "CG": -2.36, "GG": -3.26,
    "CC": -3.26, "GC": -3.42,
}


# ============================================================
# 第五部分：能量特征计算函数
# ============================================================
# 使用 ViennaRNA 库计算多种热力学能量特征，
# 包括：单链折叠自由能、双链结合自由能、
# 以及每个碱基对总自由能的贡献（通过 in silico 突变扫描）。


def single_energy(sequence, left_padlen=0, right_padlen=0):
    """
    计算 RNA 单链的折叠自由能及每个碱基的贡献。

    思路：
    1. 用 RNA.fold_compound 计算整个序列的最小自由能 (MFE)
    2. 逐个突变每个碱基为 'X'，重新计算 MFE
    3. 突变前后 MFE 的差值 = 该碱基对总自由能的贡献

    返回：
        [总MFE, 碱基1贡献, 碱基2贡献, ...]
    """
    sequence = sequence.replace("\n", "").replace("T", "U")
    fc = RNA.fold_compound(sequence)
    (ss, mfe) = fc.mfe()
    base_contributions = [mfe]
    if left_padlen > 0:
        base_contributions += [0] * left_padlen
    for i in range(len(sequence)):
        mutated_seq = sequence[:i] + "X" + sequence[i + 1:]
        fc_mut = RNA.fold_compound(mutated_seq)
        (_, mfe_mut) = fc_mut.mfe()
        contribution = mfe_mut - mfe
        base_contributions.append(-contribution)
    if right_padlen > 0:
        base_contributions += [0] * right_padlen
    return base_contributions


def calculate_duplex_energy(seq1, seq2):
    """计算两条 RNA 链的双链结合自由能"""
    duplexes = RNA.duplexfold(seq1, seq2)
    return duplexes.energy


def duplex_energy(seq1, seq2, left_padlen=0, right_padlen=0):
    """
    计算双链 RNA 的结合自由能及每个碱基的贡献。
    同样通过逐个突变来估算每个位置对结合自由能的贡献。
    """
    total_energy = calculate_duplex_energy(seq1, seq2)
    contributions = [total_energy]
    if left_padlen > 0:
        contributions += [0] * left_padlen
    for i in range(len(seq1)):
        mutated_seq1 = seq1[:i] + "X" + seq1[i + 1:]
        mutated_energy = calculate_duplex_energy(mutated_seq1, seq2)
        contribution = mutated_energy - total_energy
        contributions.append(-contribution)
    if right_padlen > 0:
        contributions += [0] * right_padlen
    for i in range(len(seq2)):
        mutated_seq2 = seq2[:i] + "X" + seq2[i + 1:]
        mutated_energy = calculate_duplex_energy(seq1, mutated_seq2)
        contribution = mutated_energy - total_energy
        contributions.append(-contribution)
    return contributions


def _generate_chain_data(residues, start):
    """
    从 PDB 解析的残基对象生成坐标和序列数据。

    步骤：
    1. 为全局节点（BOS）添加中心坐标
    2. 遍历每个残基，提取骨架原子 (C4', P) 和侧链碱基原子坐标
    3. 每个残基填充到 MAX_ATOM_NUMBER 个原子位（不足的补 0）

    返回：
        {'X': 坐标数组 [N, MAX_ATOM_NUMBER, 3],
         'S': 碱基类型索引列表 [N]}
    """
    backbone_atoms = VOCAB.backbone_atoms
    X, S, res_pos = [], [], []

    # 全局节点（BOS token），坐标初始化为 (0,0,0)
    X.append([[0, 0, 0] for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    S.append(VOCAB.symbol_to_idx(VOCAB.BOS))
    res_pos.append(0)

    # 遍历每个碱基残基
    for residue in residues:
        x = [residue.get_backbone_coord_map().get("P", [0, 0, 0])
             for _ in range(VOCAB.MAX_ATOM_NUMBER)]

        i = 0
        bb_atom_coord = residue.get_backbone_coord_map()
        sc_atom_coord = residue.get_sidechain_coord_map()
        for atom in backbone_atoms:
            if atom in bb_atom_coord:
                x[i] = bb_atom_coord[atom]
            i += 1
        for atom in residue.sidechain:
            if atom in sc_atom_coord:
                x[i] = sc_atom_coord[atom]
            i += 1

        X.append(x)
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        res_pos.append(residue.get_id()[0])

    X = np.array(X)
    center = np.mean(X[1:].reshape(-1, 3), axis=0)
    X[0] = center  # 全局节点坐标设为所有残基的中心

    return {"X": X, "S": S}


# ============================================================
# 第六部分：数据集类 (E2EDataset)
# ============================================================
# 这是一个端到端的数据集，负责：
# 1. 首次运行时：读取 JSON lines → 解析 PDB → 提取 RNA-FM 特征 →
#    计算能量特征 → 缓存为 pickle 文件
# 2. 后续运行时：直接从缓存加载预处理结果
# 3. 支持分批缓存，避免大数据集占用过多内存


class E2EDataset(Dataset):
    """
    siRNA 效能预测的端到端数据集。

    输入 JSON lines 格式（每行一条记录）：
        {
            "siRNA": "...",
            "mRNA_seq": "...",
            "position": 123,
            "sense seq": "...",
            "anti seq": "...",
            "efficacy": 0.85,
            "start": 0,
            "chain": "sirna",
            "pdb_data_path": "/path/to/pdb"
        }

    预处理输出（每个样本）：
        - cplx: SIRNA 对象（包含 3D 结构）
        - PCT: 效能值 (float)
        - sec_pos: 起始位置
        - rna_pos: 位置编码
        - single_embeddings: 单链 RNA-FM 嵌入
        - duplex_embeddings: 双链 RNA-FM 嵌入
        - energys: 能量特征
        - mseq: mRNA 序列
        - chain: 链类型标识
    """

    def __init__(self, file_path, save_dir=None, num_entry_per_file=-1, random=False):
        super().__init__()
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if "." in prefix:
                prefix = prefix.split(".")[0]
            save_dir = os.path.join(save_dir, f"{prefix}_processed")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, "_metainfo")
        self.data = []
        need_process = False
        try:
            with open(metainfo_file, "r") as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo["num_entry"]
                self.file_names = metainfo["file_names"]
                self.file_num_entries = metainfo["file_num_entries"]
        except FileNotFoundError:
            print_log("No meta-info file found, start processing", level="INFO")
            need_process = True
        except Exception as e:
            print_log(f"Failed to load {metainfo_file}, error: {e}", level="WARN")
            need_process = True

        if need_process:
            # ---------- 首次运行：执行完整的预处理流程 ----------
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)
            metainfo = {
                "num_entry": self.num_entry,
                "file_names": self.file_names,
                "file_num_entries": self.file_num_entries,
            }
            with open(metainfo_file, "w") as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])
        self._load_part()
        self.idx_mapping = [i for i in range(self.num_entry)]

    def _save_part(self, save_dir, num_entry):
        """将当前 self.data 中的前 num_entry 个样本保存为 pickle 文件"""
        file_name = os.path.join(save_dir, f"part_{len(self.file_names)}.pkl")
        print_log(f"Saving {file_name} ...")
        file_name = os.path.abspath(file_name)
        end = min(num_entry, len(self.data)) if num_entry != -1 else len(self.data)
        with open(file_name, "wb") as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        """从 pickle 文件加载批量数据到内存"""
        f = self.file_names[self.cur_file_idx]
        print_log(
            f"Loading preprocessed file {f}, "
            f"{self.cur_file_idx + 1}/{len(self.file_names)}"
        )
        with open(f, "rb") as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        """根据索引切换当前加载的 pickle 分片"""
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx

    def __len__(self):
        return self.num_entry

    def preprocess(self, file_path, save_dir, num_entry_per_file):
        """
        ---------- 核心预处理流程 ----------
        对数据集中的每条记录执行：

        步骤 1: 读取 JSON lines 文件
        步骤 2: 解析 PDB → SIRNA 对象（含 3D 坐标）
        步骤 3: 构建 mRNA 序列（带左右填充）
        步骤 4: 计算 RNA-FM 嵌入（单链 & 双链）
        步骤 5: 计算能量特征（单链自由能、双链结合能）
        步骤 6: 构建位置编码
        步骤 7: 缓存到 pickle 文件
        """
        with open(file_path, "r") as fin:
            lines = fin.read().strip().split("\n")

        for line in tqdm(lines):
            item = json.loads(line)

            # ---- 步骤 2: 解析 PDB 3D 结构 ----
            try:
                cplx = SIRNA.from_pdb(item["pdb_data_path"])
            except AssertionError as e:
                print_log(e, level="ERROR")
                print_log(f"parse {item['pdb']} pdb failed, skip", level="ERROR")
                continue

            sec_pos = item["start"]
            chain = item["chain"]

            # ---- 步骤 3: 构建 mRNA 靶序列窗口 ----
            # 以 siRNA 的 anti 链结合位置为中心，取左右各 padlen 个碱基
            # 从而得到长度为 61 的 mRNA 窗口（含 padding）
            padlen = int((61 - len(item["anti seq"])) / 2)
            position = int(item["position"])
            mRNA_seq_len = len(item["mRNA_seq"])
            anti_seq_len = len(item["anti seq"])
            left_padlen = 0
            right_padlen = 0

            fm_mseq = ""
            if position - padlen < 0:
                left_padlen = padlen - position
                fm_mseq += "-" * left_padlen
                fm_mseq += item["mRNA_seq"][:position]
            else:
                fm_mseq += item["mRNA_seq"][position - padlen : position]

            fm_mseq += item["mRNA_seq"][position : position + anti_seq_len]

            if position + anti_seq_len + padlen > mRNA_seq_len:
                right_padlen = (
                    position + anti_seq_len + padlen - mRNA_seq_len
                )
                fm_mseq += item["mRNA_seq"][position + anti_seq_len :]
                fm_mseq += "-" * right_padlen
            else:
                fm_mseq += item["mRNA_seq"][
                    position + anti_seq_len : position + anti_seq_len + padlen
                ]

            real_mrna = (
                fm_mseq.upper().replace("T", "U").replace("N", "X").replace("-", "")
            )
            fm_mseq = fm_mseq.upper().replace("T", "U").replace("N", "X")
            mseq = fm_mseq.replace("-", "#")

            # ---- 步骤 4: RNA-FM 特征提取 ----
            # RNA-FM 是一个 12 层 Transformer 模型，输出每碱基 640 维表示
            # 我们提取最后第 12 层的表示作为特征

            # 4a: sense + anti 双链嵌入
            sense_anti_seq = [
                ("sense_anti", item["sense seq"].upper() + item["anti seq"].upper())
            ]
            batch_labels, batch_strs, batch_tokens = fm_batch_converter(sense_anti_seq)
            with torch.no_grad():
                results = fm_model(batch_tokens.to(device=fm_device), repr_layers=[12])
                sense_anti_embeddings = results["representations"][12][0][:-1].cpu()

            # 4b: mRNA + anti 双链嵌入
            mrna_anti_seq = [("mrna_anti", real_mrna + item["anti seq"].upper())]
            batch_labels, batch_strs, batch_tokens = fm_batch_converter(mrna_anti_seq)
            with torch.no_grad():
                results = fm_model(batch_tokens.to(device=fm_device), repr_layers=[12])
                mrna_anti_embeddings = results["representations"][12][0][:-1].cpu()
            if left_padlen > 0:
                mrna_anti_embeddings = torch.cat(
                    [
                        torch.zeros(
                            left_padlen, mrna_anti_embeddings.shape[1]
                        ),
                        mrna_anti_embeddings,
                    ],
                    dim=0,
                )
            if right_padlen > 0:
                mid = torch.cat(
                    [
                        mrna_anti_embeddings[
                            : 1 + len(real_mrna) + left_padlen
                        ],
                        torch.zeros(
                            right_padlen, mrna_anti_embeddings.shape[1]
                        ),
                    ],
                    dim=0,
                )
                mrna_anti_embeddings = torch.cat(
                    [
                        mid,
                        mrna_anti_embeddings[
                            1 + len(real_mrna) + left_padlen :
                        ],
                    ],
                    dim=0,
                )

            # 4c: mRNA 序列嵌入
            mrna_seq = [("mrna", real_mrna)]
            batch_labels, batch_strs, batch_tokens = fm_batch_converter(mrna_seq)
            with torch.no_grad():
                results = fm_model(batch_tokens.to(device=fm_device), repr_layers=[12])
                mrna_embeddings = results["representations"][12][0][:-1].cpu()
            if left_padlen > 0:
                mrna_embeddings = torch.cat(
                    [torch.zeros(left_padlen, mrna_embeddings.shape[1]),
                     mrna_embeddings], dim=0
                )
            if right_padlen > 0:
                mrna_embeddings = torch.cat(
                    [mrna_embeddings,
                     torch.zeros(right_padlen, mrna_embeddings.shape[1])], dim=0
                )

            # 4d: sense 链嵌入
            sense_seq = [("sense", item["sense seq"].upper())]
            batch_labels, batch_strs, batch_tokens = fm_batch_converter(sense_seq)
            with torch.no_grad():
                results = fm_model(batch_tokens.to(device=fm_device), repr_layers=[12])
                sense_embeddings = results["representations"][12][0][:-1].cpu()

            # 4e: anti 链嵌入
            anti_seq = [("anti", item["anti seq"].upper())]
            batch_labels, batch_strs, batch_tokens = fm_batch_converter(anti_seq)
            with torch.no_grad():
                results = fm_model(batch_tokens.to(device=fm_device), repr_layers=[12])
                anti_embeddings = results["representations"][12][0][:-1].cpu()

            # ---- 组合嵌入向量 ----
            # 拼接顺序：mRNA + 全局anti起点 + sense(去掉起点) + anti(去掉起点)
            single_embeddings = torch.cat(
                [
                    mrna_embeddings,
                    anti_embeddings[0].unsqueeze(0),
                    sense_embeddings[1:],
                    anti_embeddings[1:],
                ],
                dim=0,
            )
            duplex_embeddings = torch.cat(
                [
                    mrna_anti_embeddings[: 1 + len(fm_mseq)],
                    sense_anti_embeddings[: 1 + len(item["sense seq"])],
                    mrna_anti_embeddings[1 + len(fm_mseq) :],
                ],
                dim=0,
            )

            # ---- 步骤 5: 能量特征计算 ----
            # 使用 ViennaRNA 计算 3 种能量：
            # 1. 单链折叠自由能
            # 2. 双链结合自由能 (mRNA-anti + sense-anti)
            # 3. 双链结合自由能 (mRNA + sense-anti 双链)
            sense_anti_energy = duplex_energy(
                item["sense seq"].upper(), item["anti seq"].upper()
            )
            mrna_anti_energy = duplex_energy(
                real_mrna, item["anti seq"].upper(), left_padlen, right_padlen
            )
            mrna_sense_energy = duplex_energy(
                real_mrna, item["sense seq"].upper(), left_padlen, right_padlen
            )
            mrna_energys = single_energy(real_mrna, left_padlen, right_padlen)
            anti_energys = single_energy(item["anti seq"].upper())
            sense_energys = single_energy(item["sense seq"].upper())

            single_energys = (
                mrna_energys
                + [anti_energys[0]]
                + sense_energys[1:]
                + anti_energys[1:]
            )
            duplex1_energys = (
                mrna_anti_energy[: 1 + len(fm_mseq)]
                + sense_anti_energy[: 1 + len(item["sense seq"])]
                + mrna_anti_energy[1 + len(fm_mseq) :]
            )
            duplex2_energys = mrna_energys + sense_anti_energy
            energys = np.stack(
                [single_energys, duplex1_energys, duplex2_energys], axis=-1
            )

            # ---- 步骤 6: 位置编码 ----
            rna_pos = [0]
            for i in range(51, 51 + 61):
                rna_pos.append(i)
            rna_pos.append(1)
            for i in range(2, 2 + len(item["sense seq"])):
                rna_pos.append(i)
            for i in range(26, 26 + len(item["anti seq"])):
                rna_pos.append(i)

            self.data.append(
                [
                    cplx,
                    item["efficacy"],
                    sec_pos,
                    rna_pos,
                    single_embeddings,
                    duplex_embeddings,
                    energys,
                    mseq,
                    chain,
                ]
            )
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    def __getitem__(self, idx):
        """
        获取单个训练样本，在 __getitem__ 中完成：
        1. 从缓存加载原始数据
        2. 使用 _generate_chain_data 将 PDB 残基转换为模型输入张量
        3. 拼接 mRNA 序列的 token 表示到模型输入
        4. 构建 mask (smask) 标记 siRNA 起止位置
        """
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)

        item, PCT, sec_pos, rna_pos, single_emb, duplex_emb, energys, mseq, chain = (
            self.data[idx]
        )
        mseq = mseq.replace("#", "N").replace("R", "N").replace("X", "N")

        sense_rna = item.get_chain("A")
        anti_rna = item.get_chain("B")
        rna_acids = []
        for i in range(len(sense_rna)):
            rna_acids.append(sense_rna.get_residue(i))
        for i in range(len(anti_rna)):
            rna_acids.append(anti_rna.get_residue(i))

        rna_data = _generate_chain_data(rna_acids, VOCAB.BOS)

        # mRNA 序列的 token 表示（'+' 作为分隔符后接 mRNA 序列）
        rna_data["S"] = [VOCAB.symbol_to_idx(_) for _ in "-" + mseq] + rna_data["S"]
        rna_data["X"] = np.concatenate(
            [
                np.zeros((len(mseq) + 1, VOCAB.MAX_ATOM_NUMBER, 3)),
                np.array(rna_data["X"]),
            ]
        )

        # smask: 标记 sense 和 anti 链的起始位置（用于预测的位点）
        smask = [0 for _ in range(len(rna_data["S"]))]
        smask[0] = 1
        smask[len(mseq) + 1] = 1

        rna_data["energys"] = energys
        rna_data["smask"] = smask
        rna_data["rna_pos"] = rna_pos
        rna_data["sec_pos"] = sec_pos
        rna_data["chain"] = chain
        rna_data["pct"] = [float(PCT)]
        rna_data["single_embeddings"] = single_emb.tolist()
        rna_data["duplex_embeddings"] = duplex_emb.tolist()

        return rna_data

    @classmethod
    def collate_fn(cls, batch):
        """
        自定义批处理函数：
        将所有样本的图数据拼接为一个大图（batch graph），
        不同样本之间没有边连接（通过 batch_id 区分）。
        """
        keys = [
            "X", "S", "rna_pos", "sec_pos", "pct",
            "smask", "single_embeddings", "duplex_embeddings",
            "chain", "energys",
        ]
        types = [
            torch.float, torch.long, torch.long, torch.long, torch.float,
            torch.bool, torch.float, torch.float, torch.long, torch.float,
        ]

        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)

        lengths = [len(item["S"]) for item in batch]
        res["lengths"] = torch.tensor(lengths, dtype=torch.long)
        return res


# ============================================================
# 第七部分：模型定义
# ============================================================
# ENsiRNA 使用 RNAmaskModel，核心结构为：
#
# 1. SeparatedNucleicAcidFeature (RNA 特征提取器)
#    - 嵌入层：将碱基类型、位置、RNA-FM 特征、能量特征拼接为节点特征
#    - 原子嵌入：每个碱基的 C4'/P/碱基 坐标通过小嵌入层编码
#    - 图构建：在相同二级结构区域内构建 KNN 边 + 全局边 + 序列邻接边
#
# 2. AMEGNN (Adaptive Multi-Channel E(n) Equivariant GNN)
#    - 多通道等变图神经网络
#    - 每个原子位点视为一个"通道"，通道间有自适应池化
#    - 使用坐标信息进行等变消息传递
#
# 3. 预测头 (pred1/pred2/pred3)
#    - pred2 是实际使用的预测头
#    - 将 sense 和 anti 链的起始节点表示拼接后，
#      通过 MLP 映射到 [0,1] 的效能值


class RNAmaskModel(nn.Module):
    """
    ENsiRNA 核心模型：掩码式 siRNA 效能预测。

    输入：
        S: 碱基类型索引 [N]
        X: 原子坐标 [N, n_channel, 3]
        rna_pos: 碱基位置编码 [N]
        sec_pos: 二级结构位置 [N]
        lengths: 每个样本的节点数 [B]
        pct: 真实效能值 [B]
        smask: 掩码标记 [N] (标记 sense/anti 起始位置)
        single_embeddings: 单链 RNA-FM 特征 [N, 640]
        duplex_embeddings: 双链 RNA-FM 特征 [N, 640]
        chain: 链类型 [N]
        energys: 能量特征 [N, 3]

    输出:
        loss: Smooth L1 Loss（训练用）
        或 probs: 预测效能值（测试用）
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        n_channel,
        num_classes,
        mask_id=VOCAB.get_mask_idx(),
        k_neighbors=9,
        n_layers=3,
        dropout=0.1,
        MLM=False,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        atom_embed_size = 16

        # ---- RNA 特征提取器 ----
        # 将原始序列/结构信息转换为节点嵌入和图结构
        self.rna_feature = SeparatedNucleicAcidFeature(
            embed_size,
            atom_embed_size,
            relative_position=False,
            fix_atom_weights=False,
            edge_constructor=GMEdgeConstructor,
            feature="MOGAN",
            atommod=False,
        )

        self.num_classes = num_classes

        # ---- 自适应多通道等变 GNN ----
        self.gnn = AMEGNN(
            embed_size + 64 + 3,  # 输入维度 = 初始嵌入 + RNA-FM (64) + 能量 (3)
            hidden_size,
            hidden_size,
            n_channel,
            channel_nf=atom_embed_size,
            radial_nf=hidden_size,
            in_edge_nf=0,
            n_layers=n_layers,
            residual=True,
            dropout=0.1,
            dense=True,
        )

        # ---- 3 个预测头（仅 pred2 被使用）----
        self.pred1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.pred2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, 1),
        )
        self.pred3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        S,
        X,
        rna_pos,
        sec_pos,
        lengths,
        pct,
        smask,
        single_embeddings,
        duplex_embeddings,
        chain,
        energys,
    ):
        """
        前向传播（训练模式）：
        1. 构建 batch_id（区分不同样本的节点）
        2. RNA 特征提取 → 节点嵌入 H_0 + 图边 (siRNA 边 / mRNA 边) + 原子嵌入
        3. AMEGNN 消息传递 → 更新节点表示
        4. 从 smask 位置取 sense/anti 起始节点的表示，拼接后预测效能
        5. 计算 Smooth L1 Loss
        """
        batch_id = torch.zeros_like(S)
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)

        # 特征提取与图构建
        H_0, (sirna_edges, mrna_edges), (atom_embeddings, atom_weights) = (
            self.rna_feature(
                X, S, batch_id, self.k_neighbors,
                rna_pos, sec_pos, chain, energys,
                single_embeddings, duplex_embeddings,
            )
        )

        # GNN 消息传递
        mod_h, x, _ = self.gnn(
            H_0, X, sirna_edges, mrna_edges,
            channel_attr=atom_embeddings,
            channel_weights=atom_weights,
        )

        # 取 sense 链起点 (smask[0::2]) 和 anti 链起点 (smask[1::2]) 拼接
        h_ = torch.cat([mod_h[smask][0::2, :], mod_h[smask][1::2, :]], dim=1)
        logits2 = self.pred2(h_).squeeze()
        probs2 = torch.sigmoid(logits2)
        if probs2.dim() == 0:
            probs2 = probs2.unsqueeze(0)
        loss = F.smooth_l1_loss(probs2, pct)
        return loss

    def test(
        self,
        S,
        X,
        rna_pos,
        sec_pos,
        lengths,
        pct,
        smask,
        single_embeddings,
        duplex_embeddings,
        chain,
        energys,
    ):
        """测试/推理模式，返回预测概率、隐藏状态等"""
        batch_id = torch.zeros_like(S)
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)

        H_0, (sirna_edges, mrna_edges), (atom_embeddings, atom_weights) = (
            self.rna_feature(
                X, S, batch_id, self.k_neighbors,
                rna_pos, sec_pos, chain, energys,
                single_embeddings, duplex_embeddings,
            )
        )
        mod_h, x, _ = self.gnn(
            H_0, X, sirna_edges, mrna_edges,
            channel_attr=atom_embeddings,
            channel_weights=atom_weights,
        )
        h_ = torch.cat([mod_h[smask][0::2, :], mod_h[smask][1::2, :]], dim=1)
        logits2 = self.pred2(h_).squeeze()
        probs2 = torch.sigmoid(logits2)
        if probs2.dim() == 0:
            probs2 = probs2.unsqueeze(0)
        return probs2, h_, mod_h[smask][0::2, :], mod_h[smask][1::2, :]


# ============================================================
# 第八部分：训练配置类 (TrainConfig)
# ============================================================


class TrainConfig:
    """
    训练配置容器，存储所有训练超参数。

    关键参数：
        lr: 初始学习率
        max_epoch: 最大训练轮数
        warmup: 学习率预热步数
        patience: 早停耐心值（验证指标不改善的容忍步数）
        grad_clip: 梯度裁剪阈值
        save_topk: 保存最优 K 个 checkpoint
        metric_min_better: 验证指标是否越小越好
    """

    def __init__(
        self,
        save_dir,
        lr,
        max_epoch,
        warmup=0,
        metric_min_better=True,
        patience=3,
        grad_clip=None,
        save_topk=-1,
        **kwargs,
    ):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.warmup = warmup
        self.metric_min_better = metric_min_better
        self.patience = patience
        self.grad_clip = grad_clip
        self.save_topk = save_topk
        self.__dict__.update(kwargs)

    def add_parameter(self, **kwargs):
        self.__dict__.update(kwargs)


# ============================================================
# 第九部分：训练器 (Trainer / RNAmaskModelTrainer)
# ============================================================
# RNAmaskModelTrainer 继承自 Trainer，实现了：
#   - 指数学习率衰减 (从 initial_lr → final_lr)
#   - ReduceLROnPlateau 调度器作为实际使用的学习率策略
#   - 训练/验证步中的日志记录


class RNAmaskModelTrainer:
    """
    siRNA 效能预测模型的训练器。

    训练流程（每轮）：
    1. 遍历训练 DataLoader，执行 train_step
    2. 计算损失，反向传播，梯度裁剪，优化器更新
    3. 遍历验证 DataLoader，计算验证损失
    4. 根据验证指标判断是否保存 checkpoint
    5. 早停判断
    """

    def __init__(self, model, train_loader, valid_loader, config):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # 学习率衰减参数
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        # 对数空间线性衰减：log(final_lr) = log(lr) + log_alpha * max_step
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        self.MLM = False

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr
        )

        # 调度器：ReduceLROnPlateau（验证损失不改善时降学习率）
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=20, min_lr=0.00001
        )

        # 分布式训练
        self.local_rank = -1

        # 日志与版本管理
        self.version = self._get_version()
        self.config.save_dir = os.path.join(
            self.config.save_dir, f"version_{self.version}"
        )
        self.model_dir = os.path.join(self.config.save_dir, "checkpoint")
        self.writer = None
        self.writer_buffer = {}

        # 训练状态
        self.valid_global_step = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []
        self.patience = self.config.patience

    @classmethod
    def to_device(cls, data, device):
        """递归地将数据移到指定设备"""
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, (list, tuple)):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, "to"):
            data = data.to(device)
        return data

    def _is_main_proc(self):
        """判断是否是主进程（单卡或分布式 rank 0）"""
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        """自动递增版本号"""
        version, pattern = -1, r"version_(\d+)"
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return version + 1

    def _train_epoch(self, device):
        """
        执行一个训练轮次：
        1. 遍历训练数据加载器
        2. 对每个 batch 执行 train_step → 计算损失
        3. 反向传播 + 梯度裁剪 + 参数更新
        4. 记录训练损失到 TensorBoard
        """
        t_iter = (
            tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        )
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss = self.train_step(batch, self.global_step)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.optimizer.step()
            if hasattr(t_iter, "set_postfix"):
                t_iter.set_postfix(loss=loss.item(), version=self.version)
            self.global_step += 1

    def _valid_epoch(self, device):
        """
        执行一个验证轮次：
        1. 计算验证集上的平均损失
        2. 根据验证指标判断是否优于历史最佳
        3. 保存 checkpoint（如适用）
        4. 更新学习率调度器
        5. 早停判断
        """
        self.model.eval()
        metric_arr = []
        with torch.no_grad():
            t_iter = (
                tqdm(self.valid_loader)
                if self._is_main_proc()
                else self.valid_loader
            )
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric.cpu().item())
            self.valid_global_step += 1
        valid_metric = np.mean(metric_arr)

        self.model.train()
        self.scheduler.step(valid_metric)

        # 判断验证指标是否改善，保存最优 checkpoint
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
            if self._is_main_proc():
                save_path = os.path.join(
                    self.model_dir, f"epoch{self.epoch}_step{self.global_step}.ckpt"
                )
                module_to_save = (
                    self.model.module if self.local_rank == 0 else self.model
                )
                torch.save(module_to_save, save_path)
                self._maintain_topk_checkpoint(valid_metric, save_path)
        else:
            self.patience -= 1
        self.last_valid_metric = valid_metric

        # 写入验证指标到 TensorBoard
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}
        return valid_metric

    def _metric_better(self, new):
        """判断新指标是否优于历史最佳"""
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        """
        维护 Top-K 最优 checkpoint：
        按验证指标排序，只保留最优的 K 个
        """
        topk = self.config.save_topk
        if self.config.metric_min_better:
            better = lambda a, b: a < b
        else:
            better = lambda a, b: a > b
        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break
        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))
        if topk > 0:
            while len(self.topk_ckpt_map) > topk:
                last_ckpt_path = self.topk_ckpt_map[-1][1]
                os.remove(last_ckpt_path)
                self.topk_ckpt_map.pop()
        # 保存排序记录
        topk_map_path = os.path.join(self.model_dir, "topk_map.txt")
        with open(topk_map_path, "w") as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f"{metric}: {path}\n")

    def train(self, device_ids, local_rank):
        """
        完整训练入口：
        1. 初始化日志和保存目录
        2. 自动检测设备（CUDA / MPS / CPU）
        3. 循环执行 train_epoch + valid_epoch
        4. 早停判断
        """
        self.local_rank = local_rank
        best_loss = float("inf")

        # ---- 初始化 TensorBoard Writer 和保存目录 ----
        if self._is_main_proc():
            self.writer = SummaryWriter(self.config.save_dir)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(
                os.path.join(self.config.save_dir, "namespace.json"), "w"
            ) as fout:
                json.dump(self.config.__dict__, fout, indent=2)

        # ---- 设备检测 ----
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        if main_device_id == -1:
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{main_device_id}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.model.to(device)

        # ---- 分布式训练设置 ----
        if local_rank != -1:
            print_log(
                f"Using data parallel, local rank {local_rank}, all {device_ids}"
            )
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            print_log(f"training on {device_ids}")

        # ---- 主训练循环 ----
        for _ in range(self.config.max_epoch):
            print_log(f"epoch{self.epoch} starts")
            self._train_epoch(device)
            print_log("validating ...")
            eloss = self._valid_epoch(device)
            if eloss < best_loss:
                best_loss = eloss
            self.epoch += 1
            if self.patience <= 0:
                print_log(f"Early stopping at epoch {self.epoch}")
                return best_loss
        return best_loss

    def log(self, name, value, step, val=False):
        """写入 TensorBoard 日志"""
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if val:
                if name not in self.writer_buffer:
                    self.writer_buffer[name] = []
                self.writer_buffer[name].append(value)
            else:
                self.writer.add_scalar(name, value, step)

    def train_step(self, batch, batch_idx):
        """单步训练：前向传播 → 返回损失"""
        loss = self.model(**batch)
        self.log("Loss/train", loss, batch_idx)
        return loss

    def valid_step(self, batch, batch_idx):
        """单步验证：前向传播 → 返回损失"""
        loss = self.model(**batch)
        self.log("Loss/validation", loss, batch_idx, val=True)
        return loss


# ============================================================
# 第十部分：训练入口 (main)
# ============================================================
# 整合所有组件，执行完整的训练流程


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ENsiRNA 训练入口")

    # ---------- 数据参数 ----------
    parser.add_argument(
        "--train_set", type=str, required=True, help="训练集 JSON 路径"
    )
    parser.add_argument(
        "--valid_set", type=str, required=True, help="验证集 JSON 路径"
    )

    # ---------- 训练参数 ----------
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument(
        "--final_lr", type=float, default=1e-5, help="最终学习率（指数衰减目标）"
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="线性学习率预热步数"
    )
    parser.add_argument("--max_epoch", type=int, default=100, help="最大训练轮数")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="梯度裁剪阈值"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="模型与日志保存目录"
    )
    parser.add_argument("--batch_size", type=int, required=True, help="批大小")
    parser.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="早停耐心值（设为较大值可关闭早停）",
    )
    parser.add_argument(
        "--save_topk",
        type=int,
        default=10,
        help="保存最优 K 个 checkpoint，-1 表示全部保存",
    )
    parser.add_argument("--shuffle", action="store_true", help="是否打乱训练数据")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作进程数")

    # ---------- 设备参数 ----------
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        required=True,
        help="使用的 GPU 编号，-1 表示 CPU",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="分布式训练 local rank",
    )

    # ---------- 模型参数 ----------
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["RNAModel", "RNAmaskModel", "RNAGNNModel"],
        help="模型类型",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="碱基嵌入维度",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="隐藏层维度",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=9,
        help="KNN 图邻居数",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="EGNN 层数",
    )

    return parser.parse_args()


def main(args):
    """
    训练主函数：组装所有组件并启动训练。

    完整训练流程：
    Step 1: 加载训练集和验证集（首次运行会自动预处理）
    Step 2: 配置训练参数
    Step 3: 构建 RNAmaskModel
    Step 4: 创建 DataLoader
    Step 5: 初始化训练器
    Step 6: 启动训练循环
    Step 7: 返回最佳损失
    """
    print_log("=" * 60)
    print_log("ENsiRNA 训练开始")
    print_log("=" * 60)

    # ---- 设置自动梯度检测（便于调试 NaN）----
    torch.autograd.set_detect_anomaly(True)
    print_log(f"Arguments: {args}")

    # ---- Step 1: 加载数据集 ----
    # E2EDataset 首次运行会执行完整的预处理流程（PDB 解析 + RNA-FM + 能量计算）
    # 预处理结果会缓存到 {prefix}_processed/ 目录下
    print_log("Loading training set...")
    train_set = E2EDataset(args.train_set)
    print_log(f"Training set size: {len(train_set)}")

    print_log("Loading validation set...")
    valid_set = E2EDataset(args.valid_set)
    print_log(f"Validation set size: {len(valid_set)}")

    # ---- Step 2: 训练配置 ----
    config = TrainConfig(**vars(args))

    # ---- Step 3: 构建模型 ----
    # RNAmaskModel 是 ENsiRNA 的核心模型：
    #   - 使用 SeparatedNucleicAcidFeature 提取 RNA 特征（序列 + 结构 + 能量 + RNA-FM）
    #   - 使用 AMEGNN 进行等变图消息传递
    #   - 通过 pred2 头预测 siRNA 效能
    if args.model_type == "RNAmaskModel":
        model = RNAmaskModel(
            args.embed_dim,
            args.hidden_size,
            VOCAB.MAX_ATOM_NUMBER,
            VOCAB.get_num_amino_acid_type() + 1,
            VOCAB.get_mask_idx(),
            args.k_neighbors,
            n_layers=args.n_layers,
        )
    else:
        raise NotImplementedError(f"Model {args.model_type} not implemented")

    # ---- Step 4: 计算每轮步数 ----
    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)
    print_log(f"Step per epoch: {step_per_epoch}")

    # ---- Step 5: 分布式训练设置 ----
    train_sampler = None
    if len(args.gpus) > 1:
        # 多 GPU 分布式训练
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", world_size=len(args.gpus)
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, shuffle=args.shuffle
        )
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f"Batch size on single GPU: {args.batch_size}")
    else:
        args.local_rank = -1
    config.local_rank = args.local_rank

    # ---- Step 6: 创建 DataLoader ----
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(args.shuffle and train_sampler is None),
        sampler=train_sampler,
        collate_fn=train_set.collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_set.collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # ---- Step 7: 训练 ----
    trainer = RNAmaskModelTrainer(model, train_loader, valid_loader, config)
    best_loss = trainer.train(args.gpus, args.local_rank)

    print_log(f"Training complete. Best validation loss: {best_loss:.6f}")
    return best_loss


# ============================================================
# 第十一部分：程序入口
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    main(args)
