# ENsiRNA 训练说明

本文档详细说明如何从零开始配置环境、准备数据并完成 ENsiRNA 模型的训练。

---

## 1. 环境配置

### 1.1 安装 Pixi

Pixi 是 conda/pip 的替代工具，用于创建可复现的 Python 环境。

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 1.2 创建训练环境

在项目根目录创建并初始化 Pixi 环境：

```bash
# 进入项目目录
cd ENsiRNA

# 初始化 pixi 环境
pixi init ensirna_env
cd ensirna_env
```

编辑 `pixi.toml`，写入以下内容：

```toml
[workspace]
channels = ["conda-forge", "bioconda"]
name = "ensirna_env"
platforms = ["linux-64"]
version = "0.1.0"

[tasks]

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

安装 conda 依赖：

```bash
pixi install
```

### 1.3 安装 PyTorch 和 CUDA 支持

```bash
# 安装 PyTorch 2.5.1 with CUDA 12.4
pixi run pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

验证 CUDA 可用性：

```bash
pixi run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 1.4 安装其他 Python 依赖

```bash
pixi run pip install rna-fm torch-geometric
```

### 1.5 下载 RNA-FM 预训练权重

RNA-FM 模型权重需要从 HuggingFace 下载（约 1.14 GB）：

```bash
# 下载权重文件
curl -L -o RNA-FM_pretrained.pth \
  "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"

# 复制到 torch hub 缓存，避免重复下载
mkdir -p ~/.cache/torch/hub/checkpoints/
cp RNA-FM_pretrained.pth ~/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth
```

---

## 2. 数据准备

### 2.1 原始数据格式

训练数据位于 `dataset/` 目录下，为 CSV 格式：

- `train_1.csv` ~ `train_5.csv`：训练集
- `valid_1.csv` ~ `valid_5.csv`：验证集

每行包含字段：`siRNA, anti seq, sense seq, mRNA, mRNA_seq, position, efficacy, anti_seq_len, group`

### 2.2 获取 PDB 结构文件

模型训练需要 siRNA 的 3D 结构（PDB 格式）。有两种方式获取：

#### 方式 A：下载预折叠 PDB 文件（推荐）

```bash
# 从 Google Drive 下载（如可访问）
pixi run pip install gdown
pixi run gdown 1XHuFuqW7s93lBmCrZH70jN-41hmsF071
# 解压到 pdb_data/ 目录
unzip pdb_files.zip -d pdb_data/
```

#### 方式 B：生成近似 PDB 结构（无需 Rosetta）

如果无法获取原始 PDB 文件，可以使用数据准备脚本基于序列生成近似结构：

```bash
pixi run python prepare_data.py \
  --csv_dir dataset \
  --pdb_dir pdb_data
```

此脚本会：
1. 读取 CSV 文件，生成 JSON lines 格式（含 siRNA、mRNA_seq、position、sense seq、anti seq、efficacy 等字段）
2. 为每个 RNA 序列生简单的 A-型螺旋 PDB 结构

### 2.3 数据预处理说明

训练时，`E2EDataset` 会自动对 JSON 数据进行预处理（仅首次运行）：

1. **PDB 解析**：使用 `Bio.PDB` 读取 PDB 文件，提取残基坐标
2. **RNA-FM 特征提取**：对每对 sense/anti/mRNA 序列计算 RNA-FM embedding（640维）
3. **能量特征计算**：使用 ViennaRNA 计算单链折叠能和双链结合能
4. **缓存预处理结果**：处理后数据保存为 `{prefix}_processed/` 目录下的 pickle 文件

处理速度约 3 样本/秒（RTX 3060），train_1 + valid_1 约需 15 分钟。

---

## 3. 模型训练

### 3.1 训练脚本

使用 `run_training.py` 启动训练：

```bash
pixi run python run_training.py \
  --train_set dataset/train_1.json \
  --valid_set dataset/valid_1.json \
  --lr 1e-4 \
  --final_lr 1e-5 \
  --max_epoch 100 \
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

### 3.2 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_set` | 必填 | 训练集 JSON 路径 |
| `--valid_set` | 必填 | 验证集 JSON 路径 |
| `--lr` | 1e-4 | 初始学习率 |
| `--final_lr` | 1e-5 | 最终学习率（指数衰减） |
| `--max_epoch` | 100 | 最大训练轮数 |
| `--batch_size` | 16 | 批大小 |
| `--save_dir` | model_pkl | 模型保存目录 |
| `--save_topk` | 10 | 保存最优 k 个 checkpoint |
| `--gpus` | 0 | 使用的 GPU 编号，-1 表示 CPU |
| `--model_type` | RNAmaskModel | 模型类型 |
| `--embed_dim` | 128 | 残基嵌入维度 |
| `--hidden_size` | 256 | 隐藏层维度 |
| `--k_neighbors` | 9 | KNN 图邻居数 |
| `--n_layers` | 2 | EGNN 层数 |

### 3.3 训练日志

训练过程中输出格式示例：

```
2026-05-09 02:18:22::INFO::step per epoch: 141
2026-05-09 02:18:22::INFO::training on [0]
2026-05-09 02:18:22::INFO::epoch0 starts
141it [02:16,  1.02it/s, loss=0.0296]
2026-05-09 02:20:38::INFO::validating ...
```

- `loss`：当前 batch 的 Smooth L1 Loss
- `step per epoch`：每轮步数（`ceil(样本数 / batch_size)`）
- `validating`：每轮结束后执行验证

### 3.4 使用 config.json

也可使用原始 `config.json` 配合 `train.sh` 训练：

```json
{
  "train_set": "train_1.json",
  "valid_set": "valid_1.json",
  "save_dir": "model_pkl",
  "max_epoch": 100,
  "save_topk": 10,
  "batch_size": 16,
  "shuffle": true,
  "model_type": "RNAmaskModel",
  "embed_dim": 128,
  "hidden_size": 256,
  "k_neighbors": 9,
  "n_layers": 2,
  "lr": 1e-4,
  "final_lr": 1e-5
}
```

```bash
GPU=0 bash train.sh config.json
```

### 3.5 多 GPU 训练

```bash
GPU="0,1" ADDR=localhost PORT=9901 bash train.sh config.json
```

---

## 4. 训练结果

### 4.1 输出文件

训练完成后，在 `save_dir`（默认 `model_pkl/`）下生成：

```
model_pkl/
└── version_X/                    # 每次运行递增
    └── checkpoint/
        ├── epoch0_step141.ckpt   # 每个 epoch 的 checkpoint
        ├── epoch1_step282.ckpt
        └── topk_map.txt          # 最优 checkpoint 记录
```

### 4.2 模型推理

使用预训练 checkpoint 进行预测：

```bash
pixi run python test.py \
  --ckpt model_pkl/version_2/checkpoint/epoch1_step282.ckpt \
  --test_set dataset/valid_1.json \
  --batch_size 16 \
  --gpu 0
```

结果输出为 Excel 文件，包含 siRNA ID 和预测 efficacy。

---

## 5. 常见问题

### Q1: 找不到 RNA-FM 权重

从 HuggingFace 手动下载并放置到 torch hub cache：
```bash
curl -L -o ~/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth \
  "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
```

### Q2: 训练时提示 OOM（显存不足）

降低 batch size：
```bash
--batch_size 8
```

### Q3: 数据预处理速度过慢

- 首次预处理后数据会被缓存，后续运行直接从缓存加载
- 可设置 `num_workers > 0`（但注意 RNA-FM 在多 worker 下可能重复加载）

### Q4: 没有 GPU

使用 CPU 训练：
```bash
--gpus -1
```
注意 CPU 训练速度会慢 5-10 倍。
