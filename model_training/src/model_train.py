"""
Train EFO.
"""
import argparse

import pytorch_lightning as pl
from loguru import logger

from funcs.callbacks import configure_loggers_and_callbacks
from funcs.efo.efo_mk1_data_module import DataModule
from funcs.efo.efo_mk1_model import Model
from funcs.hyperparams import TrainerHparams, create_trainer_parser
from funcs.model_utils import get_num_steps_per_epoch
from funcs.utils import find_project_root

ROOT = find_project_root()
CACHE_DIR = ROOT / "cache" / "efo_mk1"
DATA_CACHE_DIR = CACHE_DIR / "data"
USE_LR_MONITOR = True


def create_parser() -> argparse.ArgumentParser:
    parent_parser = create_trainer_parser(
        DataModule=DataModule, ModelModule=Model
    )
    parser = argparse.ArgumentParser(
        parents=[parent_parser], conflict_handler="resolve"
    )
    parser.set_defaults(use_lr_scheduler=True)
    return parser


def main() -> None:
    # parse arg
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # init
    trainer_hparams = TrainerHparams(**vars(args))
    pl.seed_everything(trainer_hparams.seed)
    loggers_and_callbacks = configure_loggers_and_callbacks(
        cache_dir=CACHE_DIR, use_lr_monitor=USE_LR_MONITOR
    )
    data_module = DataModule(args=args)
    data_module.setup("fit")
    args.num_steps_per_epoch = get_num_steps_per_epoch(
        data_module=data_module, args=args
    )
    model = Model(args)
    precision = 32 if not trainer_hparams.fp16 else 16
    trainer_flags = {
        "gpus": trainer_hparams.gpus,
        "accelerator": trainer_hparams.accelerator,
        "max_epochs": trainer_hparams.num_train_epochs,
        "callbacks": loggers_and_callbacks["callbacks"],
        "logger": loggers_and_callbacks["trainer_logger"],
        "precision": precision,
    }
    logger.info(f"trainer flags: {trainer_flags}")
    trainer = pl.Trainer(**trainer_flags)  # type: ignore
    if args.dry_run:
        logger.info("Dry run.")
        quit()

    logger.info("Start training.")
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
