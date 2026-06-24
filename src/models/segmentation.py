"""
Segmentation model wrapper using segmentation-models-pytorch.

Supports U-Net and U-Net++ with pretrained encoder backbones.
Input is adapted to accept arbitrary numbers of Sentinel-2 bands.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU


class CropSegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for crop semantic segmentation.

    Args:
        architecture: One of "unet", "unetplusplus".
        encoder_name: Encoder backbone (e.g., "resnet34", "efficientnet-b4").
        in_channels: Number of input spectral bands.
        num_classes: Number of output crop classes.
        lr: Learning rate.
        encoder_weights: Pretrained weights for encoder ("imagenet" or None).
    """

    def __init__(
        self,
        architecture: str = "unet",
        encoder_name: str = "resnet34",
        in_channels: int = 12,
        num_classes: int = 8,
        lr: float = 1e-4,
        encoder_weights: str | None = "imagenet",
    ):
        super().__init__()
        self.save_hyperparameters()

        arch_map = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
        }
        assert architecture in arch_map, f"Unknown architecture: {architecture}"

        self.model = arch_map[architecture](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        # Patch the first conv layer if in_channels != 3 (imagenet pretrained expects 3)
        if encoder_weights == "imagenet" and in_channels != 3:
            self._adapt_first_conv(in_channels)

        self.loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.aux_loss_fn = nn.CrossEntropyLoss()

        metrics = MetricCollection({
            "miou": MeanIoU(num_classes=num_classes),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def _adapt_first_conv(self, in_channels: int) -> None:
        """Average imagenet weights across channels to support N-band input."""
        old_conv = self.model.encoder.conv1 if hasattr(self.model.encoder, "conv1") else None
        if old_conv is None:
            return
        weight = old_conv.weight.data  # (out, 3, H, W)
        new_weight = weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        new_conv.weight.data = new_weight
        self.model.encoder.conv1 = new_conv

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: dict, stage: str) -> Tensor:
        images, masks = batch["image"], batch["mask"]
        logits = self(images)
        loss = self.loss_fn(logits, masks) + 0.5 * self.aux_loss_fn(logits, masks)
        preds = logits.argmax(dim=1)
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(preds, masks)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
