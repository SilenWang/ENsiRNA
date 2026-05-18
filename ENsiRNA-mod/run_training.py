#!/usr/bin/python
import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# Override RNA-FM loading to use local file
RNA_FM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ENsiRNA', 'RNA-FM_pretrained.pth')
if not os.path.exists(RNA_FM_PATH):
    RNA_FM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ENsiRNA-mod', 'RNA-FM_pretrained.pth')
import fm.pretrained
original_fn = fm.pretrained.rna_fm_t12
def rna_fm_t12_local(model_location=None):
    return original_fn(model_location=RNA_FM_PATH) if os.path.exists(RNA_FM_PATH) else original_fn()
fm.pretrained.rna_fm_t12 = rna_fm_t12_local

import argparse
from torch.utils.data import DataLoader
from data.dataset import E2EDataset, VOCAB
from trainer import TrainConfig
from utils.logger import print_log
from utils.random_seed import setup_seed, SEED, seed_worker
setup_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)

def parse():
    parser = argparse.ArgumentParser(description='ENsiRNA-mod training')
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, required=True)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--final_lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--save_topk', type=int, default=10)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--model_type', type=str, required=True, choices=['RNAmaskModel'])
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--n_layers', type=int, default=3)
    return parser.parse_args()

def main(args):
    torch.autograd.set_detect_anomaly(True)
    print_log(args)
    
    train_set = E2EDataset(args.train_set)
    valid_set = E2EDataset(args.valid_set)
    collate_fn = train_set.collate_fn
    config = TrainConfig(**vars(args))
    
    if args.model_type == 'RNAmaskModel':
        from trainer import RNAmaskModelTrainer as Trainer
        from model import RNAmaskModel
        model = RNAmaskModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type()+1, VOCAB.get_mask_idx(),
                   args.k_neighbors, n_layers=args.n_layers)
    else:
        raise NotImplementedError(f'model {args.model_type} not implemented')
    
    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)
    
    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None
    config.local_rank = args.local_rank
    
    if args.local_rank == 0 or args.local_rank == -1:
        print_log(f'step per epoch: {step_per_epoch}')
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn,
                              worker_init_fn=seed_worker,
                              generator=g)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn,
                              worker_init_fn=seed_worker,
                              generator=g)
    
    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)

if __name__ == '__main__':
    args = parse()
    main(args)
