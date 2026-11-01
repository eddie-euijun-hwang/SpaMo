import pytorch_lightning as pl
import torch
from utils.helpers import instantiate_from_config, create_mask
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from spamo.mm_projector import build_vision_projector
from torch import nn
from spamo.tconv import TemporalConv
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from spamo.clip_loss import clip_loss


class FusionModel(nn.Module):
    def __init__(
        self,
        vision_input_dim: int = 2048,
        motion_input_dim: int = 1024,
        hidden_dim: int = 768,
        target_size: int = 2048,
        device: str | None = None,
    ):
        super().__init__()
        self.vision_projector = build_vision_projector(
            "linear", vision_input_dim, hidden_dim, device=device
        )
        self.motion_projector = build_vision_projector(
            "linear", motion_input_dim, hidden_dim, device=device
        )
        self.fusion_projector = build_vision_projector(
            "mlp2x_gelu", hidden_dim, target_size, device=device
        )
        self.temporal_encoder = TemporalConv(hidden_dim, hidden_dim).to(device)
        self.device = device

    def forward(self, batch):
        vision_outputs = self.vision_projector(batch["vision_features"])
        motion_outputs = self.motion_projector(batch["motion_features"])

        vision_seq_lengths = batch["vision_features_seq_lengths"]
        motion_seq_lengths = batch["motion_features_seq_lengths"]
        batch_size = vision_outputs.shape[0]
        joint_outputs = []
        new_lengths = []
        for i in range(batch_size):
            v_seq_length = vision_seq_lengths[i]
            m_seq_length = motion_seq_lengths[i]
            unpadded_vision_output = vision_outputs[i, :v_seq_length, :]
            unpadded_motion_output = motion_outputs[i, :m_seq_length, :]
            joint_output = torch.cat(
                (unpadded_vision_output, unpadded_motion_output), dim=0
            )
            joint_outputs.append(joint_output)
            new_lengths.append(v_seq_length + m_seq_length)

        joint_outputs = pad_sequence(joint_outputs, batch_first=True)

        visual_conv_outputs = self.temporal_encoder(
            joint_outputs.permute(0, 2, 1),
            torch.tensor(new_lengths, device=self.device),
        )

        vision_outputs = visual_conv_outputs["visual_feat"].permute(1, 0, 2)
        vision_masks = create_mask(
            seq_lengths=visual_conv_outputs["feat_len"].to(torch.int).tolist(),
            device=self.device,
        )

        vision_outputs = self.fusion_projector(vision_outputs)
        return vision_outputs, vision_masks


class LightningFusion(pl.LightningModule):
    def __init__(
        self,
        vision_input_dim: int = 2048,
        motion_input_dim: int = 1024,
        hidden_dim: int = 768,
        target_size: int = 2048,
        device: str | None = None,
        t5_checkpoint: str = "google/flan-t5-xl",
        lr: float = 0.0001,
        scheduler_config: dict | None = None,
        target_embedding_weights_path: str = "weights/flan-t5-xl-embeddings.pt",
        max_txt_len: int = 64,
    ):
        super().__init__()
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_checkpoint,
            max_length=max_txt_len,
        )
        self.fusion_model = FusionModel(
            vision_input_dim,
            motion_input_dim,
            hidden_dim,
            target_size,
            device,
        )

        weights = torch.load(target_embedding_weights_path)
        self.target_embedding = nn.Embedding.from_pretrained(weights, freeze=True)

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.scheduler_config = scheduler_config
        self.lr = lr

    def shared_step(self, batch):
        output_embeds, _ = self.fusion_model(batch)

        target_tokens = self.t5_tokenizer(
            batch["texts"],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        target_embeds = self.target_embedding(target_tokens.input_ids)

        output_embeds = output_embeds.mean(dim=1)
        target_embeds = target_embeds.mean(dim=1)

        output_embeds = F.normalize(output_embeds, dim=-1)
        target_embeds = F.normalize(target_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = (target_embeds @ output_embeds.t()) * logit_scale

        loss = clip_loss(logits)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a training step."""
        loss = self.shared_step(batch)
        self.log_dict(
            {"train_loss": loss},
            batch_size=len(batch["texts"]),
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a validation step."""
        loss = self.shared_step(batch)
        self.log_dict(
            {"val_loss": loss},
            batch_size=len(batch["texts"]),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx: int) -> None:
        """Perform a testing step."""
        loss = self.shared_step(batch)
        self.log_dict(
            {"test_loss": loss}, batch_size=len(batch["texts"]), sync_dist=True
        )

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8)

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            lr_scheduler = {
                "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]  # type: ignore
        return optimizer
