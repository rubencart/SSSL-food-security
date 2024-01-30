import collections
import copy
import logging
import math
from typing import List, Tuple

import torch
import torchvision
from sssl import utils
from sssl.config import Config
from sssl.data.landsat8 import Batch
from torch import FloatTensor, IntTensor, Tensor, nn
from torchvision.models import ResNet18_Weights

logger = logging.getLogger("pytorch_lightning")


class CNNOutput:
    def __init__(self, features: FloatTensor):
        self.bs, self.K, self.ft = features.shape
        self.features = features


class LossOutput:
    def __init__(
        self, loss: FloatTensor, scores: Tuple[FloatTensor], correct: List[IntTensor]
    ):
        self.loss = loss
        self.scores = scores
        self.correct = correct

    def to_dict(self):
        return {
            "loss": self.loss,
            "scores": copy.deepcopy(tuple(s.detach() for s in self.scores)),
            **copy.deepcopy(
                {
                    f"correct_{i}": self.correct[i].detach().float().mean()
                    for i in range(len(self.correct))
                }
            ),
        }


class Conv4(torch.nn.Module):
    """A simple 4 layers CNN.
    Used as backbone.
    """

    def __init__(self, cfg: Config):
        super(Conv4, self).__init__()
        self.feature_size = cfg.conv4_feature_size
        self.name = "conv4"

        self.in_c = 3 if cfg.landsat8_bands == "RGB" else 7
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_c, 8, kernel_size=3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                32,
                cfg.conv4_feature_size,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(cfg.conv4_feature_size),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        self.flatten = torch.nn.Flatten()

        # self.linear = torch.nn.Sequential(
        #   torch.nn.Linear(64, 1),
        #   torch.nn.Sigmoid()
        # )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.flatten(h)
        # h = self.linear(h)
        return h


class CNNModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.cnn_type == "conv4":
            self.cnn = Conv4(cfg)
        else:
            assert cfg.cnn_type == "resnet18"
            if (
                not cfg.do_pretrain
                and cfg.do_downstream
                and cfg.finetune.pretrained_on == "ImageNet"
            ):
                logger.info("Using pretrained resnet18 ImageNet weights")
                weights = ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.cnn = torchvision.models.resnet18(weights=weights, progress=False)
            if cfg.landsat8_bands != "RGB":
                logger.info(
                    "Setting first resnet18 conv layer to a new one with 7 channels"
                )
                c1 = self.cnn.conv1
                new_conv1 = nn.Conv2d(
                    in_channels=7,
                    out_channels=c1.out_channels,
                    kernel_size=c1.kernel_size,
                    stride=c1.stride,
                    padding=c1.padding,
                    bias=c1.bias is not None,
                )

                if (
                    not cfg.do_pretrain
                    and cfg.do_downstream
                    and cfg.finetune.pretrained_on == "ImageNet"
                ):
                    # https://discuss.pytorch.org/t/how-to-transfer-the-pretrained-weights-for-a-standard-resnet50-to-a-4-channel/52252
                    with torch.no_grad():
                        new_conv1.weight[
                            :, utils.Constants.RGB_BANDS
                        ] = c1.weight.clone()
                        if c1.bias is not None:
                            new_conv1.bias = c1.bias.clone()

                self.cnn.conv1 = new_conv1

    def forward(self, tiles: Tensor) -> Tensor:
        bs, K, c, h, w = tiles.shape
        flat_tiles = tiles.view(bs * K, c, h, w)
        out = self.cnn(flat_tiles)
        return out.view(bs, K, out.shape[-1])

    def batch_forward(self, batch: Batch) -> CNNOutput:
        tiles = batch.tiles
        bs, K, c, h, w = tiles.shape
        flat_tiles = tiles.view(bs * K, c, h, w)
        out = self.cnn(flat_tiles)
        return CNNOutput(out.view(bs, K, out.shape[-1]))


class PretrainLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.pretrain.loss_type in ("sssl", "relreas"):
            logger.info("Using Relational Reasoning objective")
            self.loss = RelationalReasoning(cfg)
        else:
            if not cfg.pretrain.loss_type == "tile2vec":
                raise NotImplementedError
            logger.info("Using tile2vec")
            self.loss = Tile2Vec(cfg)

    def forward(self, cnn_output: CNNOutput) -> LossOutput:
        loss, scores, correct = self.loss(cnn_output.features)
        return LossOutput(loss, scores, correct)


class Tile2Vec(nn.Module):
    """
    Based on https://github.com/ermongroup/tile2vec/blob/master/src/tilenet.py
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.margin = cfg.tile2vec.margin
        self.l2 = cfg.tile2vec.l2_weight_decay

    def forward(
        self, features: FloatTensor
    ) -> Tuple[FloatTensor, Tuple[FloatTensor], List[IntTensor]]:
        # K should be 2
        bs, K, ft = features.shape
        z_p, z_n = features[:, 0], features[:, 1]
        z_d = features.roll(shifts=1, dims=0)[:, 1]
        l_n_sq = ((z_p - z_n) ** 2).sum(dim=1)
        l_d_sq = ((z_p - z_d) ** 2).sum(dim=1)
        l_n = torch.sqrt(l_n_sq + 1e-6)
        l_d = torch.sqrt(l_d_sq + 1e-6)
        loss = nn.functional.relu(l_n - l_d + self.margin)
        correct = [l_n - l_d + self.margin < 0]
        # l_n = torch.mean(l_n)
        # l_d = torch.mean(l_d)
        # l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if self.l2 != 0:
            reg = self.l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
            loss += reg
        return loss, (l_n, l_d), correct


class RelationalReasoning(nn.Module):
    """
    Based on https://github.com/mpatacchiola/self-supervised-relational-reasoning/blob/master/essential_script.py
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.pretrain.focal_loss:
            logger.info("Using focal loss")
            self.loss = FocalLoss(gamma=cfg.pretrain.focal_gamma)
        else:
            logger.info("Using BCE loss")
            self.loss = nn.BCEWithLogitsLoss()
        self.relation_head = nn.Sequential(
            collections.OrderedDict(
                [
                    ("linear1", nn.Linear(cfg.feature_size * 2, 256)),
                    ("bn1", nn.BatchNorm1d(256)),
                    ("relu", nn.LeakyReLU()),
                    ("linear2", nn.Linear(256, 1)),
                ]
            )
        )

    def forward(
        self, cnn_output: FloatTensor
    ) -> Tuple[FloatTensor, Tuple[FloatTensor], List[IntTensor]]:
        pairs, tgt = self.aggregate_cat(cnn_output)
        scores = self.relation_head(pairs).squeeze(-1)
        loss = self.loss(scores, tgt)
        correct = []
        for t in range(1, 10):
            correct.append((scores > t / 10).float() == tgt)
        return (
            loss,
            tuple(
                scores,
            ),
            correct,
        )

    def aggregate_cat(self, cnn_output: FloatTensor) -> Tuple[FloatTensor, IntTensor]:
        bs, K, ft = cnn_output.shape
        pos_pairs, neg_pairs = [], []
        shifts = 1
        for i in range(K):
            for j in range(i + 1, K):
                pp = torch.cat((cnn_output[:, i], cnn_output[:, j]), 1)
                pos_pairs.append(pp)
                # roll along batch size to match left aug with aug of different sample in batch
                # increase shifts to not always match with negative of same other sample in batch
                np = torch.cat(
                    (cnn_output[:, i], cnn_output.roll(shifts, dims=0)[:, j]), 1
                )
                neg_pairs.append(np)
                shifts += 1
                if shifts >= bs:
                    shifts = 1
        ppt, npt = torch.cat(pos_pairs, dim=0), torch.cat(neg_pairs, dim=0)
        all_pairs = torch.cat((ppt, npt), dim=0)
        tgt = torch.cat(
            (torch.ones(ppt.shape[:1]), torch.zeros(npt.shape[:1])), dim=0
        ).to(cnn_output)
        return all_pairs, tgt


class FocalLoss(torch.nn.Module):
    """
    From https://github.com/mpatacchiola/self-supervised-relational-reasoning/blob/master/methods/relationnet.py
    Sigmoid focal cross entropy loss.
    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=None):
        """Constructor.
        Args:
          gamma: exponent of the modulating factor (1 - p_t)^gamma.
          alpha: optional alpha weighting factor to balance positives vs negatives,
               with alpha in [0, 1] for class 1 and 1-alpha for class 0.
               In practice alpha may be set by inverse class frequency,
               so that for a low number of positives, its weight is high.
        """
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self.BCEWithLogits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, prediction_tensor, target_tensor):
        """Compute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets.
        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = self.BCEWithLogits(prediction_tensor, target_tensor)
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = (target_tensor * prediction_probabilities) + (  # positives probs
            (1 - target_tensor) * (1 - prediction_probabilities)
        )  # negatives probs
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(
                1.0 - p_t, self._gamma
            )  # the lowest the probability the highest the weight
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (
                1 - self._alpha
            )
        focal_cross_entropy_loss = (
            modulating_factor * alpha_weight_factor * per_entry_cross_ent
        )
        return torch.mean(focal_cross_entropy_loss)
