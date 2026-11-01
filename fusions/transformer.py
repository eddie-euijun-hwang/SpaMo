import sys

sys.path.append("./")
import torch

from transformers import T5ForConditionalGeneration, AutoTokenizer

from fusions.utils import concat_outputs

# Any value -100 is ignored by the T5 loss.
IGNORED_LABEL = -100


class T5Transformer(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "google/flan-t5-xl",
        initial_prompt: str = "Translate the given sentence into English",
        dtype: torch.dtype = torch.bfloat16,
        device="cpu",
        evaluation_mode=True,
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, dtype=dtype)

        self.device = device
        self.dtype = dtype
        self.model.to(self.device)
        if evaluation_mode:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokens = self.tokenizer(initial_prompt, return_tensors="pt")
        self.initial_prompt_tokens = tokens.input_ids.to(self.device)
        self.initial_prompt_length = tokens.attention_mask.sum(-1).item()

    def forward(self, batch):
        (features, feature_length, output_text) = batch
        batch_size = features.shape[0]

        prompt_embeds = self.model.encoder.embed_tokens(self.initial_prompt_tokens)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        prompt_lengths = (
            torch.tensor(
                self.initial_prompt_length, device=self.device, dtype=torch.long
            )
            .unsqueeze(0)
            .repeat(batch_size)
        )

        input_embeds, input_lengths = concat_outputs(
            first_feature=prompt_embeds,
            second_feature=features.to(device=self.device, dtype=self.dtype),
            first_feature_lengths=prompt_lengths,
            second_feature_lengths=feature_length.to(
                device=self.device, dtype=torch.long
            ),
            device=self.device,
        )
        input_lengths = input_lengths.to(device=self.device, dtype=torch.long)
        input_mask = torch.arange(input_lengths.max(), device=self.device).unsqueeze(
            0
        ) < input_lengths.unsqueeze(-1)

        output_tokens = self.tokenizer(
            output_text,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        target_attention_mask = output_tokens.attention_mask
        target_labels = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.tokenizer.pad_token_id, IGNORED_LABEL
        )

        output = self.model(
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            labels=target_labels,
            decoder_attention_mask=target_attention_mask,
            return_dict=True,
        )

        return output


if __name__ == "__main__":
    batch_input = torch.rand(32, 12, 768)
    batch_lengths = (torch.rand(32) * 12).to(dtype=torch.int)
    texts = ["This is test text"] * 32

    t5 = T5Transformer(
        model_name="google/flan-t5-base", dtype=torch.float32, evaluation_mode=True
    )
    batch_input.requires_grad_(True)
    out = t5((batch_input, batch_lengths, texts))
    out.loss.backward()
    print("loss:", out.loss.item())
    print("input_grad_norm:", batch_input.grad.norm().item())
