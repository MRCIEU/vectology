import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from loguru import logger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

import settings

LOGGER_STEP = settings.LOGGER_STEP


def configure_loggers_and_callbacks(
    cache_dir: Path, use_lr_monitor: bool = False
) -> Dict:
    save_dir = cache_dir / "model_runs"
    save_dir.mkdir(parents=True, exist_ok=True)
    # tensorboard_logger
    tensorboard_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(save_dir), name=None
    )
    tensorboard_logger_version = tensorboard_logger.version
    # csv logger
    csv_logger = pl_loggers.csv_logs.CSVLogger(
        save_dir=str(save_dir), name=None, version=tensorboard_logger_version
    )
    wandb_logger = pl_loggers.WandbLogger()
    trainer_logger = [tensorboard_logger, csv_logger, wandb_logger]
    # checkpoint callback
    checkpoint_file_path = (
        save_dir
        / f"version_{tensorboard_logger_version}"
        / "{epoch:02d}-{val_loss:.2f}"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", filepath=str(checkpoint_file_path)
    )
    # other custom callbacks
    early_stopping = EarlyStopping(monitor="val_loss")
    step_logging = StepLogging()
    pred_target_logger = PredTargetLogger(save_dir=save_dir)
    callbacks = [
        checkpoint_callback,
        early_stopping,
        step_logging,
        pred_target_logger,
    ]
    if use_lr_monitor:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    # res
    res = {
        "trainer_logger": trainer_logger,
        "callbacks": callbacks,
    }
    return res


class StepLogging(Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % LOGGER_STEP == 0:
            logger.info(f"stage: training step: #{batch_idx}")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % LOGGER_STEP == 0:
            logs = trainer.logger_connector.callback_metrics
            loss = logs.get("train_loss")
            logger.info(f"training loss: {loss}")

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % LOGGER_STEP == 0:
            logger.info(f"stage: validation step: #{batch_idx}")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % LOGGER_STEP == 0:
            logs = trainer.logger_connector.callback_metrics
            loss = logs.get("val_loss")
            logger.info(f"validation loss: {loss}")


class PredTargetLogger(Callback):
    def __init__(self, save_dir: Path):
        # save_dir is actually a top level, log_dir is the directory
        # to write things to
        self.log_dir = get_log_dir(save_dir)
        logger.info(f"log_dir: {self.log_dir}")
        self.result_path = self.log_dir / "validation_result_pred_target.csv"
        self.val_pred_list: List[torch.Tensor] = []
        self.val_target_list: List[torch.Tensor] = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        pred = outputs["pred"]
        target = outputs["target"]
        self.val_pred_list.append(pred)
        self.val_target_list.append(target)

    def on_validation_epoch_end(self, trainer, pl_module):
        pred = torch.cat(self.val_pred_list, dim=0).cpu().numpy()
        target = torch.cat(self.val_target_list, dim=0).cpu().numpy()
        res_df = pd.DataFrame({"target": target, "pred": pred})
        print("Result df")
        print(res_df.info())
        print(res_df.head())
        logger.info(f"Write to file {self.result_path}")
        res_df.to_csv(self.result_path, index=False)


def get_log_dir(save_dir: Path) -> Path:
    # I don't use `name` arg
    existing_versions = []
    for d in os.listdir(save_dir):
        if os.path.isdir(os.path.join(save_dir, d)) and d.startswith(
            "version_"
        ):
            existing_versions.append(int(d.split("_")[1]))
    if len(existing_versions) == 0:
        next_version = 0
    else:
        next_version = max(existing_versions) + 1
    log_dir = save_dir / f"version_{next_version}"
    return log_dir
