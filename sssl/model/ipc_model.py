import collections

import torch
from sssl import utils
from sssl.config import Config
from sssl.data.ipc import IPCBatch
from sssl.model.backbone_model import CNNOutput
from torch import nn


class IPCOutput:
    def __init__(self, logits: torch.FloatTensor):
        self.logits = logits

    def detach(self):
        self.logits = self.logits.clone().detach()
        return self


class LossOutput:
    def __init__(self, loss: torch.FloatTensor, logits: torch.FloatTensor):
        self.loss = loss
        self.logits = logits

    def detach(self):
        self.loss = self.loss.clone().detach()
        self.logits = self.logits.clone().detach()
        return self

    def to_dict(self):
        return {
            "loss": self.loss,
            "logits": tuple(s.clone().detach() for s in self.logits),
        }


class IPCClassifier(nn.Module):
    def forward(self, cnn_output: CNNOutput) -> IPCOutput:
        logits = self.head(cnn_output.features.squeeze(1))
        return IPCOutput(logits)

    @staticmethod
    def build_classifier(cfg: Config) -> "IPCClassifier":
        if cfg.finetune.clf_head == "mlp":
            return IPCClassifierMLP(cfg)
        else:
            assert cfg.finetune.clf_head == "linear"
            return IPCClassifierLinear(cfg)


class IPCClassifierMLP(IPCClassifier):
    def __init__(self, cfg: Config):
        super().__init__()
        nb_ipc = 2 if cfg.finetune.binarize_ipc else utils.Constants.NB_IPC_SCORES
        self.head = nn.Sequential(
            collections.OrderedDict(
                [
                    ("linear1", nn.Linear(cfg.feature_size, 256)),
                    ("bn1", nn.BatchNorm1d(256)),
                    ("relu", nn.LeakyReLU()),
                    ("linear2", nn.Linear(256, nb_ipc)),
                ]
            )
        )


class IPCClassifierLinear(IPCClassifier):
    def __init__(self, cfg: Config):
        super().__init__()
        nb_ipc = 2 if cfg.finetune.binarize_ipc else utils.Constants.NB_IPC_SCORES
        self.head = nn.Linear(cfg.feature_size, nb_ipc)


class IPCLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        weights = None
        if cfg.finetune.weight_classes:
            weights = (
                utils.Constants.IPC_CLASS_WEIGHTS
                if not cfg.finetune.binarize_ipc
                else utils.Constants.IPC_CLASS_WEIGHTS_BIN
            )
        if cfg.finetune.loss == "xent":
            self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            raise NotImplementedError()

    def forward(self, ipc_output: IPCOutput, batch: IPCBatch) -> LossOutput:
        loss = self.loss(ipc_output.logits, batch.ipcs)
        return LossOutput(loss, ipc_output.logits)
