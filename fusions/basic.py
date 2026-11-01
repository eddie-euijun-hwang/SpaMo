import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
from fusions.utils import concat_outputs

VISUAL_INPUT_SIZE = 2048
MOTION_INPUT_SIZE = 1024
INTER_HIDDEN = 768
OUTPUT_SIZE = 2048


class BasicFusion(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.visual_fc = nn.Linear(VISUAL_INPUT_SIZE, INTER_HIDDEN)
        self.motion_fc = nn.Linear(MOTION_INPUT_SIZE, INTER_HIDDEN)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=INTER_HIDDEN,
                out_channels=INTER_HIDDEN,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
            nn.Conv1d(
                in_channels=INTER_HIDDEN,
                out_channels=INTER_HIDDEN,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
        )
        self.temporal_norm = nn.LayerNorm(INTER_HIDDEN)
        self.temporal_act = nn.ReLU(inplace=True)

        self.output_decoder = nn.Sequential(
            nn.Linear(INTER_HIDDEN, OUTPUT_SIZE),
            nn.GELU(),
            nn.Linear(OUTPUT_SIZE, OUTPUT_SIZE),
            nn.GELU(),
            nn.Linear(OUTPUT_SIZE, OUTPUT_SIZE),
        )

        self.device = device

    def forward(
        self,
        packed_visual_features: PackedSequence,
        packed_motion_features: PackedSequence,
    ):
        visual_features, visual_lens = pad_packed_sequence(
            packed_visual_features, batch_first=True
        )
        motion_features, motion_lens = pad_packed_sequence(
            packed_motion_features, batch_first=True
        )

        visual_outputs: torch.Tensor = self.visual_fc(visual_features)
        motion_outputs: torch.Tensor = self.motion_fc(motion_features)
        (joint_outputs, joint_lengths) = concat_outputs(
            first_feature=visual_outputs,
            second_feature=motion_outputs,
            first_feature_lengths=visual_lens,
            second_feature_lengths=motion_lens,
            device=self.device,
        )

        joint_outputs = self.temporal_conv(joint_outputs.permute(0, 2, 1))
        joint_outputs = joint_outputs.permute(0, 2, 1)
        joint_lengths = self._update_lengths(joint_lengths)
        joint_outputs = self.temporal_act(self.temporal_norm(joint_outputs))

        output = self.output_decoder(joint_outputs)

        return output, joint_lengths

    def _update_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        new_lengths = lengths - 4
        new_lengths = torch.div(new_lengths, 2, rounding_mode="floor")
        new_lengths = new_lengths - 4
        new_lengths = torch.div(new_lengths, 2, rounding_mode="floor")

        return new_lengths.clamp_min(0)


if __name__ == "__main__":
    from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

    visual = [torch.rand((20, 2048)), torch.rand((10, 2048)), torch.rand((30, 2048))]

    motion = [torch.rand((20, 1024)), torch.rand((10, 1024)), torch.rand((30, 1024))]

    packed_visual = pack_sequence(visual, enforce_sorted=False)
    packed_motion = pack_sequence(motion, enforce_sorted=False)

    fusion = BasicFusion("cpu")

    res = fusion(packed_visual, packed_motion)
