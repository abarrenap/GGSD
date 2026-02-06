import os
import glob
import argparse
import torch
import random
import pytorch_lightning as pl

from dataset.load_data_generated import LaplacianDatasetNX
from torch.utils.data import DataLoader
from models.diffusion import SpectralDiffusion
from models.predictor import Predictor
from utils.misc import seed_all
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, shuffle=-1):
        self.datasets = datasets
        self.shuffle = [False]*len(datasets)
        if shuffle is not None:
            self.shuffle[shuffle] = True

    def __getitem__(self, i):
        return tuple(d[random.randint(0,len(d)-1)] if s else d[i] 
                     for d,s in zip(self.datasets,self.shuffle))

    def __len__(self):
        return min(len(d) for d in self.datasets)

def get_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--diffusion_model', type=str, required=True)
    p.add_argument('--diffusion_ckpt', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)

    p.add_argument('--n_graphs_train', type=int, default=64)
    p.add_argument('--n_graphs_test', type=int, default=32)
    p.add_argument('--sampling_steps', type=int, default=20)

    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--max_epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--wandb', type=eval, default=False, choices=[True, False])
    return p

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    seed_all(args.seed)

    # Load diffusion model + hparams
    model = SpectralDiffusion.load_from_checkpoint(args.diffusion_ckpt)
    model.hparams.update(args.__dict__)
    args = model.hparams

    print("Loaded dataset from diffusion ckpt:", args.dataset)

    # Load real dataset (NO stacking!)
    graphs_train_set = LaplacianDatasetNX(
        args.dataset,
        'data/'+args.dataset,
        point_dim=args.k,
        smallest=args.smallest,
        split='train',
        nodefeatures=args.dataset[:3] in ["qm9"]
    )

    graphs_val_set = LaplacianDatasetNX(
        args.dataset,
        'data/'+args.dataset,
        point_dim=args.k,
        smallest=args.smallest,
        split='test',
        nodefeatures=args.dataset[:3] in ["qm9"]
    )

    # Generate synthetic graphs (small + chunked)
    model.to("cpu")
    n_total = args.n_graphs_train + args.n_graphs_test
    max_gen = 128

    gens_x, gens_y = [], []
    for i in range(0, n_total, max_gen):
        n = min(max_gen, n_total - i)
        n_nodes = list(graphs_train_set.sample_n_nodes(n-1)) + [graphs_train_set.n_max]
        gx, gy = model.sample_eigs(
            max_nodes=n_nodes,
            num_eigs=args.k+args.feature_size,
            scale_xy=graphs_train_set.scale_xy,
            unscale_xy=graphs_train_set.unscale_xy,
            device="cpu",
            num_graphs=n,
            reproject=True,
            sampling_steps=args.sampling_steps
        )
        gens_x.append(gx.cpu())
        gens_y.append(gy.cpu())
        del gx, gy

    gens_x = torch.cat(gens_x,0)
    gens_y = torch.cat(gens_y,0)

    gen_train = torch.utils.data.TensorDataset(gens_x[:args.n_graphs_train], gens_y[:args.n_graphs_train])
    gen_val   = torch.utils.data.TensorDataset(gens_x[args.n_graphs_train:], gens_y[args.n_graphs_train:])

    train_loader = DataLoader(
        ConcatDataset(graphs_train_set, gen_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        ConcatDataset(graphs_val_set, gen_val, shuffle=None),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    args.n_max = graphs_train_set.n_max
    ref = Predictor(args)

    checkpoint_cb = ModelCheckpoint(save_last=True, monitor='avg_degrad', mode='min')
    early_stop_cb = EarlyStopping(monitor='avg_degrad', patience=200, mode='min')

    logger = WandbLogger(project="graph_diffusion_predictor", offline=True) if args.wandb else None

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        max_epochs=args.max_epochs
    )

    trainer.fit(ref, train_loader, val_loader)
    trainer.save_checkpoint(f"predictor_final_{args.dataset}.ckpt")