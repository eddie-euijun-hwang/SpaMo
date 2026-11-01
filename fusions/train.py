import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from spamo.clip_loss import clip_loss


class PJMTranslator(pl.LightningModule):
    def __init__(
        self,
        fusion_model: torch.Module,
        t5_generator: torch.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        monitor: str = "val/loss",
        use_contrastive: bool = False,
        combined_loss: bool = False,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.fusion_module = fusion_model
        self.t5_generator = t5_generator
        self.lr = lr
        self.weight_decay = weight_decay
        self.monitor = monitor
        self.use_contrastive = use_contrastive
        self.combined_loss = combined_loss
        self.alpha = alpha

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        if self.combined_loss and not self.use_contrastive:
            raise ValueError("combined_loss=True requires use_contrastive=True.")

        self.save_hyperparameters(ignore=["fusion_module", "t5_generator"])

    def _get_t5_encoder(self) -> torch.nn.Module:
        model = getattr(self.t5_generator, "model", self.t5_generator)
        if not hasattr(model, "encoder"):
            raise ValueError("t5_generator must expose a T5 model with an encoder.")
        return model.encoder

    def _get_t5_tokenizer(self):
        tokenizer = getattr(self.t5_generator, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "t5_generator must expose a tokenizer for contrastive loss."
            )
        return tokenizer

    def _lengths_to_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        return torch.arange(max_len, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)

    def _mean_pool_by_mask(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(features.dtype)
        return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _contrastive_loss(
        self, pred_embeddings: torch.Tensor, pred_lengths: torch.Tensor, texts
    ):
        pred_mask = self._lengths_to_mask(pred_lengths, pred_embeddings.shape[1])
        pred_pooled = self._mean_pool_by_mask(pred_embeddings, pred_mask)

        tokenizer = self._get_t5_tokenizer()
        tokens = tokenizer(
            texts,
            padding="longest",
            return_tensors="pt",
        ).to(pred_embeddings.device)

        with torch.no_grad():
            text_embeds = self._get_t5_encoder().embed_tokens(tokens["input_ids"])

        text_pooled = self._mean_pool_by_mask(text_embeds, tokens["attention_mask"])

        pred_pooled = F.normalize(pred_pooled, dim=-1)
        text_pooled = F.normalize(text_pooled, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_pooled, pred_pooled.t()) * logit_scale

        return clip_loss(logits_per_text)

    def shared_step(self, batch, split: str):
        visual_features, motion_features, true_text = batch
        pred_embeddings, pred_lengths = self.fusion_module(
            visual_features, motion_features
        )

        log_dict = {}

        if self.combined_loss:
            t5_output = self.t5_generator((pred_embeddings, pred_lengths, true_text))
            if t5_output.loss is None:
                raise ValueError(
                    "t5_generator returned no loss for combined loss mode."
                )
            t5_loss = t5_output.loss
            cont_loss = self._contrastive_loss(pred_embeddings, pred_lengths, true_text)
            loss = t5_loss + self.alpha * cont_loss
            log_dict[f"{split}/t5_loss"] = t5_loss
            log_dict[f"{split}/contra_loss"] = cont_loss
            log_dict[f"{split}/loss"] = loss
        elif self.use_contrastive:
            cont_loss = self._contrastive_loss(pred_embeddings, pred_lengths, true_text)
            loss = cont_loss
            log_dict[f"{split}/contra_loss"] = cont_loss
            log_dict[f"{split}/loss"] = loss
        else:
            t5_output = self.t5_generator((pred_embeddings, pred_lengths, true_text))
            if t5_output.loss is None:
                raise ValueError("t5_generator returned no loss for T5-only mode.")
            loss = t5_output.loss
            log_dict[f"{split}/loss"] = loss

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, "train")
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, "val")
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def build_callbacks(
    monitor: str = "val/loss",
    mode: str = "min",
    checkpoint_dir: str = "checkpoints",
    checkpoint_filename: str = "epoch={epoch:03d}-val_loss={val/loss:.4f}",
    patience: int = 5,
    save_top_k: int = 1,
):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        verbose=True,
    )

    return [checkpoint_callback, early_stopping]


def build_wandb_logger(
    project: str,
    run_name: str,
    save_dir: str = "logs",
):
    return WandbLogger(
        project=project,
        name=run_name,
        save_dir=save_dir,
        log_model=True,
    )
