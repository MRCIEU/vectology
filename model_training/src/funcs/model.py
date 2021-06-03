import argparse
import math
from typing import Dict

import pytorch_lightning as pl
from loguru import logger

import wandb


def get_num_steps_per_epoch(
    data_module: pl.LightningDataModule, args: argparse.Namespace
):
    train_dataloader = data_module.train_dataloader()
    num_batches = len(train_dataloader)
    num_gpus: int = args.gpus
    num_steps_per_epoch = math.ceil(num_batches / num_gpus)
    return num_steps_per_epoch


def perform_lr_find(
    trainer: pl.Trainer,
    data_module: pl.LightningDataModule,
    model: pl.LightningModule,
    num_steps: int = 100,
    min_lr: float = 1e-08,
    max_lr: float = 0.1,
) -> Dict:
    wandb.init(job_type="lr_find")
    lr_finder = trainer.tuner.lr_find(
        model,
        datamodule=data_module,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_steps,
    )
    suggested_lr = lr_finder.suggestion()
    logger.info(f"Suggested learning rate: {suggested_lr}")
    lr_plot = lr_finder.plot()
    lr_results = lr_finder.results
    lr_list = lr_results["lr"]
    lr_scientific_list = [f"{_:.4e}" for _ in lr_list]
    loss_list = lr_results["loss"]
    lr_table = wandb.Table(
        data=[
            [lr, lr_scientific, loss]
            for (lr, lr_scientific, loss) in zip(
                lr_list, lr_scientific_list, loss_list
            )
        ],
        columns=["lr", "lr_scientific", "loss"],
    )
    wandb.log({"lr_data": lr_table})
    wandb.log({"lr_finder_plot": wandb.Image(lr_plot)})
    # NOTE: plotly relies on d3 format which takes liberty in
    #       reformatting floats with prefixes, fuck
    #       https://github.com/d3/d3-format
    wandb.log({"lr_finder_plotly": lr_plot})
    res = {
        "suggested_lr": suggested_lr,
        "lr_list": lr_list,
        "loss_list": loss_list,
        "lr_plot": lr_plot,
    }
    return res
