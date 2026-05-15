# ENsiRNA-mod 训练说明

本文档详细说明如何从零开始配置环境、准备数据并完成 ENsiRNA-mod（修饰 siRNA 功效预测模型）的训练。

> ENsiRNA 基础模型的训练说明见 [ENsiRNA/TRAINING.md](../ENsiRNA/TRAINING.md)，两模型共享同一环境。

---

## 1. 环境配置

### 1.1 安装 Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 1.2 创建环境

在项目根目录创建 Pixi 环境（与 ENsiRNA 共用）：

```bash
pixi init ensirna_env
cd ensirna_env
```

编辑 `pixi.toml`：

```toml
[workspace]
channels = ["conda-forge", "bioconda"]
name = "ensirna_env"
platforms = ["linux-64"]
version = "0.1.0"

[dependencies]
viennarna = "2.6.4.*"
python = "3.10.*"
pip = "*"
biopython = ">=1.87,<2"
numpy = ">=2.2.6,<3"
pandas = ">=2.3.3,<3"
scipy = ">=1.15.2,<2"
tensorboard = ">=2.20.0,<3"
tqdm = ">=4.67.3,<5"
openpyxl = ">=3.1.5,<4"
scikit-learn = ">=1.7.2,<2"
rdkit = ">=2026.3.2,<2027"
xgboost = ">=3.2.0,<4"
```

安装：

```bash
pixi install
```

### 1.3 PyTorch 和 CUDA

```bash
pixi run pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

验证：

```bash
pixi run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 1.4 其他 Python 依赖

```bash
pixi run pip install rna-fm torch-geometric
```

### 1.5 RNA-FM 预训练权重

RNA-FM 模型权重（约 1.14 GB），从 HuggingFace 下载：

```bash
curl -L -o RNA-FM_pretrained.pth \
  "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
mkdir -p ~/.cache/torch/hub/checkpoints/
cp RNA-FM_pretrained.pth ~/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth
```

也可放置在工作目录下供训练脚本自动定位（`run_training.py` 会依次搜索 `../ENsiRNA/RNA-FM_pretrained.pth`、`ENsiRNA-mod/RNA-FM_pretrained.pth` 和 torch hub cache）。

---

## 2. 数据准备

### 2.1 原始数据格式

训练数据位于 `dataset/` 下，为 xlsx（Excel）格式：

- `train_88_1.xlsx` ~ `train_88_5.xlsx`：训练集
- `valid_88_1.xlsx` ~ `valid_88_5.xlsx`：验证集
- `test_88.xlsx`：测试集

每行包含以下字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `ID` | siRNA 标识符 | SM3958 |
| `source` | 来源专利 | US20120088815A1 |
| `cc` | 置信度评分 | 13.0 |
| `sense raw seq` | sense 链原始序列 | cuagucgaaagaaaacgggtt |
| `sense mod` | sense 链修饰类型 | 2-O-Methyl |
| `sense pos` | sense 链修饰位置 | 2 |
| `anti raw seq` | anti 链原始序列 | cccguuuucuuucgacuagtt |
| `anti mod` | anti 链修饰类型 | 2-O-Methyl |
| `anti pos` | anti 链修饰位置 | 17 |
| `PCT` | 修饰功效百分比 | 56.0 |
| `anti length` | anti 链长度 | 21 |
| `sense length` | sense 链长度 | 21 |

### 2.2 修饰类型编码

`data/mod_utils.py` 定义了 120+ 种 RNA 化学修饰的 SMILES 表示，分为三类：

- **sugar_mod**（糖修饰）：2-O-Methyl、Locked nucleic acid (LNA)、2-Fluoro、2-Deoxy 等
- **phosphate_mod**（磷酸修饰）：Phosphorothioate、Boranophosphate 等
- **base_mod**（碱基修饰）：5-Methylcytosine、Pseudouridine、2-Thiouridine 等

每种修饰通过 RDKit Morgan Fingerprint（512 bit）编码为特征向量。

### 2.3 获取 PDB 结构文件

#### 方式 A：下载预折叠 PDB 文件

```bash
# Google Drive（如可访问）
pixi run pip install gdown
pixi run gdown 1F7cNJXMNPSjFb0UvDkDRHTkt9Tt4EGWe
unzip pdb_files.zip -d pdb_data/
```

#### 方式 B：使用 Rosetta 生成 PDB 结构（原始项目方法）

原始项目使用 Rosetta 的 `rna_denovo` 进行 RNA 3D 结构预测。需要先安装 Rosetta 并配置 `data/get_pdb.py`：

1. **获取 Rosetta**：从 [https://www.rosettacommons.org](https://www.rosettacommons.org) 下载 Rosetta（需学术/商业许可）
2. **配置 `data/get_pdb.py` 中的 Rosetta 路径**：编辑文件顶部三个路径变量，指向你的 Rosetta 安装目录：
   ```python
   RF='/path/to/rosetta.binary.linux.release-371'
   FF='/path/to/rosetta/main/source/bin/rna_denovo.static.linuxgccrelease'
   EX='/path/to/rosetta/main/tools/rna_tools/silent_util/extract_lowscore_decoys.py'
   ```
3. **运行脚本**（需在 `ENsiRNA-mod/` 目录下执行）：
   ```bash
   pixi run python -m data.get_pdb \
     -f dataset/train_88_1.xlsx dataset/valid_88_1.xlsx \
     -p pdb_data
   ```
   脚本会：
   - 使用 ViennaRNA 的 RNAplex 预测 sense/anti 链的二级结构
   - 调用 Rosetta `rna_denovo.static.linuxgccrelease` 进行 RNA 3D 结构折叠
   - 使用 `extract_lowscore_decoys.py` 提取最优构象为 PDB
   - 并行计算 `atom_mask`（每个残基的糖/磷酸/碱基修饰编码）和 `smask`（修饰位置标记）
   - 输出 JSON lines 文件，包含：ID, sense seq, sense raw seq, anti seq, anti raw seq, PCT, sense mod, sense pos, anti mod, anti pos, start, cc, pdb_data_path, atom_mask, smask
   - 使用 multiprocessing 并行处理多个样本
   
   > 注意：`get_pdb.py` 的 `secondary_structure` 默认开启，基于二级结构进行 Rosetta 折叠。修饰数据相关逻辑（`atom_mask`、`smask` 等）由 `get_atommod` 和 `get_smask` 方法在 `process()` 阶段自动处理。

#### 方式 C：生成近似 PDB 结构（无需 Rosetta）

无法获取原始 PDB 时，可使用脚本生成：

```bash
pixi run python prepare_data.py \
  --xlsx_dir dataset \
  --pdb_dir pdb_data
```

脚本功能：
1. 读取 xlsx 文件生成 JSON lines（含 ID, sense seq, sense raw seq, anti seq, anti raw seq, PCT, sense mod, sense pos, anti mod, anti pos, start, cc, pdb_data_path, atom_mask, smask）
2. 计算 `atom_mask`：为每个残基的 3 个原子位点（糖/磷酸/碱基）编码对应的修饰类型索引
3. 生成 `smask`：标记被修饰的位置
4. 生成简单 A-型螺旋 PDB 结构

### 2.4 数据预处理说明

`E2EDataset` 首次加载 JSON 时会自动预处理：

1. **PDB 解析**：使用 `Bio.PDB` 读取 PDB 中的 sense（A 链）和 anti（B 链）结构
2. **RNA-FM 特征提取**：计算 sense+anti 序列的 RNA-FM embedding（640 维）
3. **修饰指纹嵌入**：根据 `atom_mask` 中的修饰索引，从预计算的 MOGAN RDKit 字典 (`MOGANRdkit_VOCAB`) 中查找对应指纹
4. **理化特征**：计算 one-hot 编码和 10 维理化性质特征
5. **缓存**：处理后数据保存为 `{prefix}_processed/` 下的 pickle 文件，后续直接加载

预处理速度约 3 样本/秒（RTX 3060），train_88_1（2528 样本）约需 14 分钟。

---

## 3. 模型训练

### 3.1 训练脚本

```bash
pixi run python run_training.py \
  --train_set dataset/train_88_1.json \
  --valid_set dataset/valid_88_1.json \
  --lr 5e-4 \
  --final_lr 1e-4 \
  --max_epoch 200 \
  --save_dir model_pkl \
  --batch_size 16 \
  --save_topk 10 \
  --shuffle \
  --num_workers 0 \
  --gpus 0 \
  --model_type RNAmaskModel \
  --embed_dim 128 \
  --hidden_size 256 \
  --k_neighbors 9 \
  --n_layers 2
```

### 3.2 模型架构差异（与 ENsiRNA 对比）

| 特性 | ENsiRNA | ENsiRNA-mod |
|------|---------|-------------|
| 数据格式 | CSV（mRNA序列为中心） | xlsx（修饰 siRNA 为中心） |
| 学习率 | 1e-4 → 1e-5 | 5e-4 → 1e-4 |
| 最大 epoch | 100 | 200 |
| 额外特征 | — | atom_mask, mod_mask, rna_raw, FM, cc |
| 位置编码 | relative_position=False | relative_position=True |
| 坐标归一化 | 无 | CoordNormalizer |
| 特征维度 | embed_size | embed_size/2 + 额外融合 |

### 3.3 模型输入特征说明

| 参数 | 形状 | 说明 |
|------|------|------|
| `S` | [N] | 序列 token 索引 |
| `X` | [N, 3, 3] | 3D 坐标（全局节点 + backbone + sidechain） |
| `rna_pos` | [N] | RNA 残基位置编号 |
| `sec_pos` | [N] | 二级结构位置 |
| `pct` | [B] | 目标 efficacy（/100 归一化） |
| `marker` | [B] | 标记值（/100 归一化） |
| `smask` | [N] | BOS 特殊标记 |
| `atom_mask` | [N, 3] | 每残基 3 个原子位的修饰索引 |
| `mod_mask` | [N] | 修饰位置掩码 |
| `rna_raw` | [N, 6] | one-hot 编码 |
| `chain_id` | [N] | 链标识 |
| `mod` | [N, 3, 512] | RDKit Morgan 指纹嵌入 |
| `cc` | [B] | 置信度评分 |
| `FM` | [N, 640] | RNA-FM embedding |

### 3.4 使用 config.json 训练

```json
{
  "train_set": "train_88_1.json",
  "valid_set": "valid_88_1.json",
  "save_dir": "model_pkl",
  "max_epoch": 200,
  "save_topk": 10,
  "batch_size": 16,
  "shuffle": true,
  "model_type": "RNAmaskModel",
  "embed_dim": 128,
  "hidden_size": 256,
  "k_neighbors": 9,
  "n_layers": 2,
  "lr": 5e-4,
  "final_lr": 1e-4
}
```

```bash
GPU=0 bash train.sh config.json
```

### 3.5 训练日志

```
Namespace(lr=0.0005, final_lr=0.0001, max_epoch=200, ...)
step per epoch: 158
training on [0]
epoch0 starts
158it [00:17,  9.17it/s, loss=0.0301]
validating ...
epoch1 starts
158it [00:17,  9.17it/s, loss=0.0296]
validating ...
```

---

## 4. 训练结果

### 4.1 输出文件

```
model_pkl/
└── version_X/
    └── checkpoint/
        ├── epoch0_step158.ckpt   # 每 epoch 的 checkpoint（~9MB）
        ├── epoch1_step316.ckpt
        └── topk_map.txt          # 最优 checkpoint 记录
```

### 4.2 模型推理

```bash
pixi run python test.py \
  --ckpt model_pkl/version_0/checkpoint/epoch1_step316.ckpt \
  --test_set dataset/valid_88_1.json \
  --batch_size 16 \
  --gpu 0
```

输出为 Excel 文件，包含 ID 和预测结果。

### 4.3 easy_run.py

也可使用 `easy_run.py` 进行交互式预测：

```bash
pixi run python easy_run.py \
  -o result \
  --model pkl/checkpoint_1.ckpt
```

---

## 5. 常见问题

### Q1: 找不到 RNA-FM 权重

```bash
curl -L -o ~/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth \
  "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
```

### Q2: RDKit Morgan Fingerprint 警告

RDKit 提示 `DEPRECATION WARNING: please use MorganGenerator` 属于正常现象，不影响训练。

### Q3: 修饰未被识别

检查 `data/mod_utils.py` 中是否包含该修饰名称。若缺失，需添加其 SMILES 并归入相应类别（sugar_mod / phosphate_mod / base_mod）。

### Q4: OOM（显存不足）

降低 batch_size：

```bash
--batch_size 8
```

### Q5: 没有 GPU

```bash
--gpus -1
```
