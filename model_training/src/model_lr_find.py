"""
Train EFO.
"""
import pytorch_lightning as pl
from loguru import logger

from funcs.efo.efo_mk1_data_module import DataModule
from funcs.efo.efo_mk1_model import Model
from funcs.hyperparams import (
    LRFinderHparams,
    TrainerHparams,
    add_lr_finder_specific_args,
    create_trainer_parser,
)
from funcs.model_utils import perform_lr_find


def main() -> None:
    # parse arg
    parser = create_trainer_parser(DataModule=DataModule, ModelModule=Model)
    parser = add_lr_finder_specific_args(parser)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # init
    trainer_hparams = TrainerHparams(**vars(args))
    lr_finder_hparams = LRFinderHparams(**vars(args))
    pl.seed_everything(trainer_hparams.seed)
    data_module = DataModule(args=args)
    data_module.setup("fit")
    args.use_lr_scheduler = False
    model = Model(args)
    precision = 32 if not trainer_hparams.fp16 else 16
    trainer_flags = {
        "gpus": trainer_hparams.gpus,
        "precision": precision,
    }
    logger.info(f"trainer flags: {trainer_flags}")
    trainer = pl.Trainer(**trainer_flags)  # type: ignore
    if args.dry_run:
        logger.info("Dry run.")
        quit()

    perform_lr_find(
        trainer=trainer,
        data_module=data_module,
        model=model,
        num_steps=lr_finder_hparams.lr_find_steps,
        min_lr=lr_finder_hparams.lr_find_min_lr,
        max_lr=lr_finder_hparams.lr_find_max_lr,
    )


if __name__ == "__main__":
    main()
