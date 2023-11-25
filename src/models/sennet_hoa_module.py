from typing import Any, List

import torch
from lightning import LightningModule 
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric

class SenNetHOAModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        self.criterion = criterion

        self.train_metric_1 = JaccardIndex(task="binary", num_classes=2)
        self.val_metric_1 = JaccardIndex(task="binary", num_classes=2)
        self.test_metric_1 = JaccardIndex(task="binary", num_classes=2)

        self.train_metric_2 = Dice()
        self.val_metric_2 = Dice()
        self.test_metric_2 = Dice()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best_1 = MaxMetric()
        self.val_metric_best_2 = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric_1.reset()
        self.val_metric_2.reset()
        self.val_metric_best_1.reset()
        self.val_metric_best_2.reset()

    def model_step(self, batch: Any):
        x, y = batch

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        return loss, y_hat, y 

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_metric_1(preds, targets)
        self.train_metric_2(preds, targets.int())

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/jaccard", self.train_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_metric_2, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_metric_1(preds, targets)
        self.val_metric_2(preds, targets.int())

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/jaccard",
            self.val_metric_1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/dice",
            self.val_metric_2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # get current val acc
        acc1 = self.val_metric_1.compute()
        acc2 = self.val_metric_2.compute()
        # update best so far val acc
        self.val_metric_best_1(acc1)
        self.val_metric_best_2(acc2)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/jaccard_best", self.val_metric_best_1.compute(), prog_bar=True)
        self.log("val/dice_best", self.val_metric_best_2.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        # update and log metrics
        self.test_loss(loss)
        self.test_metric_1(preds, targets)
        self.test_metric_2(preds, targets.int())

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/jaccard", self.test_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_metric_2, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}