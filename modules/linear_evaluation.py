import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Tuple
from tqdm import tqdm



def accuracy_f(output, target, topk=(1,2,5)):
    maxk = max(topk)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    ret = []
    for k in topk:
        correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
        ret.append( (correct.sum() / target.sum()).item() )
    return ret[0]


class LinearEvaluation(LightningModule):
    def __init__(self, args, encoder: nn.Module, hidden_dim: int, output_dim: int):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        self.criterion = self.configure_criterion()

#         self.average_precision = torchmetrics.AveragePrecision(pos_label=1)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        preds = self._forward_representations(x, y)
        loss = self.criterion(preds, y)
        return loss, preds

    def _forward_representations(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Perform a forward pass using either the representations, or the input data (that we still)
        need to extract the represenations from using our encoder.
        """
        if x.shape[-1] == self.hidden_dim:
            h0 = x
        else:
            with torch.no_grad():
                h0 = self.encoder(x)
        return self.model(h0)

    def training_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        y_bool = y > 0.5
        self.log("Valid/topk1", accuracy_f(preds, y_bool,(1,)))
        self.log("Valid/topk2", accuracy_f(preds, y_bool,(2,)))
        self.log("Valid/topk5", accuracy_f(preds, y_bool,(5,)))
        # self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        y_bool = y > 0.5
        self.log("Valid/topk1", accuracy_f(preds, y_bool,(1,)))
        self.log("Valid/topk2", accuracy_f(preds, y_bool,(2,)))
        self.log("Valid/topk5", accuracy_f(preds, y_bool,(5,)))
        # self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.classifier_learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}

    def extract_representations(self, dataloader: DataLoader) -> Dataset:
        representations = []
        ys = []
        for x, y in tqdm(dataloader):
            with torch.no_grad():
                representations.append(self.encoder(x))
                ys.append(y)
        if len(representations) > 1:
            representations = torch.cat(representations, dim=0)
            ys = torch.cat(ys, dim=0)
        else:
            representations = representations[0]
            ys = ys[0]

        tensor_dataset = TensorDataset(representations, ys)
        return tensor_dataset
