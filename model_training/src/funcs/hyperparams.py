import argparse
from dataclasses import dataclass, fields
from typing import Optional

import pytorch_lightning as pl
from loguru import logger

import settings

DATA_STAGES = ["train", "validation"]


@dataclass
class ModelHparams:
    # fine tuner hparams
    model_name: str = settings.MODEL_NAME
    use_lr_scheduler: bool = False
    learning_rate: float = settings.LEARNING_RATE
    weight_decay: float = settings.WEIGHT_DECAY
    adam_epsilon: float = settings.ADAM_EPSILON
    init_learning_rate: Optional[float] = None
    num_steps_per_epoch: Optional[int] = None
    # data hparams
    train_data_limit: Optional[int] = None
    max_tokenization_length: int = settings.MAX_TOKENIZATION_LENGTH
    # trainer hparams
    num_train_epochs: int = 1
    fp16: bool = False

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)

    def __post_init__(self):
        if self.use_lr_scheduler and self.num_steps_per_epoch is None:
            logger.warning(
                "Need to specify `num_steps_per_epoch` when `use_lr_schduler`"
            )
            quit()


@dataclass
class LMDataModuleHparams:
    train_data_limit: Optional[int] = None
    max_tokenization_length: int = settings.MAX_TOKENIZATION_LENGTH
    train_batch_size: int = settings.BATCH_SIZE_LM
    eval_batch_size: int = settings.BATCH_SIZE_LM
    mlm_probability: float = settings.MLM_PROBABILITY
    num_workers: int = settings.NUM_WORKERS

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


@dataclass
class DataModuleHparams:
    train_data_limit: Optional[int] = None
    model_name: str = settings.MODEL_NAME
    max_tokenization_length: int = settings.MAX_TOKENIZATION_LENGTH
    train_batch_size: int = settings.BATCH_SIZE_GENERAL
    eval_batch_size: int = settings.BATCH_SIZE_GENERAL
    train_sample: float = 1.0
    val_sample: float = 1.0
    num_workers: int = settings.NUM_WORKERS

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


@dataclass
class LRFinderHparams:
    lr_find_steps: int = 100
    lr_find_min_lr: float = 1e-08
    lr_find_max_lr: float = 0.1

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


@dataclass
class TrainerHparams:
    num_train_epochs: int = 1
    seed: int = settings.SEED
    gpus: int = 1
    overwrite: bool = False
    fp16: bool = False
    accelerator: Optional[str] = None

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


def create_trainer_parser(
    DataModule: Optional[pl.LightningDataModule] = None,
    ModelModule: Optional[pl.LightningModule] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, conflict_handler="resolve"
    )
    if DataModule is not None:
        parser = DataModule.add_module_specific_args(parser)  # type: ignore
    if ModelModule is not None:
        parser = ModelModule.add_module_specific_args(parser)  # type: ignore
    parser.add_argument("-n", "--dry-run", action="store_true", help="dry run")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite cache"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--gpus", help="num GPUs", default=1, type=int)
    parser.add_argument(
        "--seed", default=settings.SEED, type=int, help="Random seed",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use 16-bit precision training."
    )
    parser.add_argument(
        "--accelerator", type=str, default=None, help="Multi-gpu processing."
    )
    return parser


def add_data_module_specific_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False, conflict_handler="resolve"
    )
    parser.add_argument("--train_data_limit", default=None, type=int)
    parser.add_argument(
        "--max_tokenization_length",
        default=settings.MAX_TOKENIZATION_LENGTH,
        type=int,
    )
    parser.add_argument(
        "--train_batch_size", default=settings.BATCH_SIZE_GENERAL, type=int
    )
    parser.add_argument(
        "--eval_batch_size", default=settings.BATCH_SIZE_GENERAL, type=int
    )
    parser.add_argument("--train_sample", default=1.0, type=float)
    parser.add_argument("--val_sample", default=1.0, type=float)
    parser.add_argument(
        "-j", "--num_workers", default=settings.NUM_WORKERS, type=int
    )
    return parser


def add_lm_data_module_specific_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False, conflict_handler="resolve"
    )
    parser.add_argument("--train_data_limit", default=None, type=int)
    parser.add_argument(
        "--max_tokenization_length",
        default=settings.MAX_TOKENIZATION_LENGTH,
        type=int,
    )
    parser.add_argument(
        "--train_batch_size", default=settings.BATCH_SIZE_LM, type=int
    )
    parser.add_argument(
        "--eval_batch_size", default=settings.BATCH_SIZE_LM, type=int
    )
    parser.add_argument(
        "--mlm_probability", default=settings.MLM_PROBABILITY, type=float
    )
    parser.add_argument(
        "-j", "--num_workers", default=settings.NUM_WORKERS, type=int
    )
    return parser


def add_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False, conflict_handler="resolve"
    )
    parser.add_argument(
        "--model_name",
        default=settings.MODEL_NAME,
        type=str,
        help="bert model name.",
    )
    lr_scheduler_parser = parser.add_mutually_exclusive_group(required=False)
    lr_scheduler_parser.add_argument(
        "--use_lr_scheduler", dest="use_lr_scheduler", action="store_true"
    )
    lr_scheduler_parser.add_argument(
        "--no_lr_scheduler", dest="use_lr_scheduler", action="store_false"
    )
    parser.set_defaults(use_lr_scheduler=False)
    parser.add_argument(
        "--learning_rate",
        default=settings.LEARNING_RATE,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--init_learning_rate",
        default=None,
        type=float,
        help="When using a learning rate scheduler, this is the initial learning rate and not the OPTIMAL learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        default=settings.WEIGHT_DECAY,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=settings.ADAM_EPSILON,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    return parser


def add_lr_finder_specific_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False, conflict_handler="resolve"
    )
    parser.add_argument(
        "--lr_find_steps", default=100, type=int,
    )
    parser.add_argument(
        "--lr_find_min_lr", default=1e-08, type=float,
    )
    parser.add_argument(
        "--lr_find_max_lr", default=0.1, type=float,
    )
    return parser
