import argparse
from dataclasses import asdict, fields
from typing import Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.regression import (
    ExplainedVariance,
    MeanAbsoluteError,
    MeanSquaredError,
)
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from funcs.hyperparams import ModelHparams, add_model_specific_args
from funcs.text_utils import get_batch_inputs

EFO_MODEL_NAME = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"


class EfoBertModelHparams(ModelHparams):
    model_name: str = EFO_MODEL_NAME

    def __init__(self, **kwargs):
        super().__init__()
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


class Model(pl.LightningModule):
    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__()
        num_labels = 1  # regression
        if args is not None:
            self.hparams = asdict(EfoBertModelHparams(**vars(args)))  # type: ignore
        else:
            self.hparams = asdict(EfoBertModelHparams())  # type: ignore
        logger.info(f"model hparams: {self.hparams}")
        model_name: str = self.hparams["model_name"]  # type: ignore
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.bert_config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True,
            return_dict=True,
        )
        self.bert_model = BertForSequenceClassification.from_pretrained(
            model_name, config=self.bert_config
        )
        self.train_exp_var = ExplainedVariance()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_exp_var = ExplainedVariance()

    def configure_optimizers(self):
        if not self.hparams["use_lr_scheduler"]:
            optimizer = AdamW(
                self.bert_model.parameters(),
                lr=self.hparams["learning_rate"],
                eps=self.hparams["adam_epsilon"],
            )
            res = {
                "optimizer": optimizer,
            }
        else:
            optimizer = AdamW(
                self.bert_model.parameters(),
                lr=self.hparams["init_learning_rate"],
                eps=self.hparams["adam_epsilon"],
            )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams["learning_rate"],
                epochs=self.hparams["num_train_epochs"],
                steps_per_epoch=self.hparams["num_steps_per_epoch"],
            )
            res = {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return res

    @auto_move_data
    def forward(self, **inputs):
        return self.bert_model(**inputs)

    @auto_move_data
    def training_step(self, batch, batch_idx):
        inputs = get_batch_inputs(batch)
        outputs = self(**inputs)
        loss = outputs["loss"]
        pred = outputs["logits"].detach().reshape(-1)
        target = inputs["labels"].detach().reshape(-1)
        self.train_exp_var(pred, target)
        # do not log "loss" as it has been done by lightning core
        log_metrics = {
            "train_loss": loss.detach(),
            "train_exp_var_step": self.train_exp_var,
        }
        self.log_dict(log_metrics)
        res = {"loss": loss, "pred": pred, "target": target}
        return res

    @auto_move_data
    def train_epoch_end(self, outputs):
        loss = torch.stack([_["loss"] for _ in outputs]).mean().item()
        log_metrics = {
            "train_loss_epoch": loss,
            "train_exp_var_epoch": self.train_exp_var.compute().item(),
        }
        logger.info(f"epoch_end metrics: {log_metrics}")
        self.log_dict(log_metrics)

    @auto_move_data
    def validation_step(self, batch, batch_idx):
        inputs = get_batch_inputs(batch)
        outputs = self(**inputs)
        loss = outputs["loss"].detach()
        pred = outputs["logits"].detach().reshape(-1)
        target = inputs["labels"].detach().reshape(-1)
        self.val_mse(pred, target)
        self.val_mae(pred, target)
        self.val_exp_var(pred, target)
        res = {
            "val_loss": loss,
            "pred": pred,
            "target": target,
        }
        return res

    @auto_move_data
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([_["val_loss"] for _ in outputs]).mean().item()
        log_metrics = {
            # NOTE: better keep val_loss metric which is used by
            #       early stopping
            "val_loss": val_loss,
            "val_mse_epoch": self.val_mse.compute().item(),
            "val_mae_epoch": self.val_mae.compute().item(),
            "val_exp_var_epoch": self.val_exp_var.compute().item(),
        }
        logger.info(f"epoch_end metrics: {log_metrics}")
        self.log_dict(log_metrics)

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = add_model_specific_args(parent_parser)
        parser.add_argument(
            "--model_name",
            default=EFO_MODEL_NAME,
            type=str,
            help="bert model name.",
        )
        return parser
