import os
import torch
import random
import math
from typing import Optional, Tuple, Any

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from utils.helpers import create_mask, derangement, instantiate_from_config
from utils.evaluate import evaluate_results
from spamo.asb import AbstractSLT
from torch.optim.lr_scheduler import LambdaLR


os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.set_float32_matmul_precision("high")


class T5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model.
    Assumes visual features are already processed and compatible with T5 embedding dimension.
    """

    def __init__(
        self,
        tuning_type: str = "lora",
        model_name: Optional[str] = None,
        frame_sample_rate: int = 1,
        prompt: str = "",
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.prepare_models(model_name)

        if tuning_type == "freeze":
            self._freeze_model()
        elif tuning_type == "lora":
            self._apply_lora()

        self.set_container()

    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the T5 model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)

    def _freeze_model(self) -> None:
        """Freeze the T5 model parameters."""
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, t5_model: str) -> None:
        """
        Prepare the textual model.

        Args:
            t5_model: Name or path of the T5 model to use
        """

        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model,
            cache_dir=self.cache_dir,
            dtype=torch.bfloat16,
        )

        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model,
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

    def prepare_inputs(
        self,
        visual_outputs: torch.Tensor,
        visual_mask: torch.Tensor,
        samples: dict,
        split: str,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        Prepare combined inputs for the T5 model.

        Args:
            visual_outputs: Visual features
            visual_mask: Mask for visual features
            samples: Input samples
            split: Current split (train, val, test)
            batch_idx: Current batch index

        Returns:
            Tuple of (joint_outputs, joint_mask, output_tokens, targets)
        """
        bs = visual_outputs.shape[0]

        prompts = [f"{self.prompt}"] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples["lang"])]

        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples["ex_lang_trans"])]

        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        visual_lengths = visual_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths

        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, : visual_lengths[i], :]
            prompt_embeds = input_embeds[i, : prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)

        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)

        output_tokens = self.t5_tokenizer(
            samples["text"],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs.

        Args:
            samples: Input samples containing visual features

        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        visual_outputs = pad_sequence(samples["pixel_values"], batch_first=True)
        visual_masks = create_mask(
            seq_lengths=samples["num_frames"], device=self.device
        )

        return visual_outputs, visual_masks

    def get_inputs(self, batch: list) -> dict:
        """
        Process batch inputs into a structured dictionary.

        Args:
            batch: Raw batch from dataloader

        Returns:
            Processed inputs dictionary
        """
        pixel_values, glor_values, ids = [], [], []
        texts = []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []

        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample["pixel_value"].shape[0] != 0:
                nframe = math.ceil(sample["num_frames"] / self.frame_sample_rate)
                pval = sample["pixel_value"][:: self.frame_sample_rate]

                ids.append(sample["id"])
                texts.append(sample["text"].lower())
                langs.append(sample["lang"])

                if self.use_in_context:
                    _ex_lang_trans = [
                        f"{sample['en_text']}={sample['text']}",
                        f"{sample['fr_text']}={sample['text']}",
                        f"{sample['es_text']}={sample['text']}",
                    ]
                    _ex_lang_trans = _ex_lang_trans[: self.num_in_context]
                    ex_lang_translations.append(" ".join(_ex_lang_trans))

                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index : start_index + max_frame_len]

                num_frames.append(nframe)
                pixel_values.append(pval)

                if sample["glor_value"] is not None:
                    if isinstance(sample["glor_value"], list):
                        glor_values.append(torch.cat(sample["glor_value"], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample["glor_value"]))
                    else:
                        glor_values.append(sample["glor_value"])
                        glor_lengths.append(len(sample["glor_value"]))

        ex_lang_translations = derangement(ex_lang_translations)

        return {
            "pixel_values": pixel_values,
            "glor_values": glor_values,
            "ids": ids,
            "text": texts,
            "ex_lang_trans": ex_lang_translations,
            "lang": langs,
            "num_frames": num_frames,
            "glor_lengths": glor_lengths,
        }

    def shared_step(
        self, inputs: dict, split: str, batch_idx: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Shared logic for training, validation and testing steps.

        Args:
            inputs: Input dictionary
            split: Current split (train, val, test)
            batch_idx: Current batch index

        Returns:
            Tuple of (loss, log_dict)
        """
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)

        log_dict = {}

        input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
            visual_outputs, visual_masks, inputs, split, batch_idx
        )

        outputs = self.t5_model(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            decoder_attention_mask=output_tokens.attention_mask,
            labels=targets,
            output_hidden_states=True,
            return_dict=True,
        )

        loss = outputs.loss
        log_dict[f"{split}/loss"] = loss

        if split != "train":
            input_embeds, input_masks, _, _ = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )

            generated = self.t5_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                num_beams=self.beam_size,
                max_length=self.max_txt_len,
                top_p=0.9,
                do_sample=True,
            )

            generated_strings = self.t5_tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            generated_strings = [gen.lower() for gen in generated_strings]

            reference_strings = self.t5_tokenizer.batch_decode(
                output_tokens.input_ids, skip_special_tokens=True
            )
            reference_strings = [ref.lower() for ref in reference_strings]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)

        return loss, log_dict

    def log_samples(self, split: str) -> None:
        """Log sample predictions to the logger."""
        if not self.generated or not self.references:
            return

        samples = list(zip(self.references[:5], self.generated[:5]))

        try:
            import wandb

            if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
                columns = ["Reference", "Generated"]
                self.logger.experiment.log(
                    {f"{split}/samples": wandb.Table(columns=columns, data=samples)}
                )
                return
        except (ImportError, AttributeError):
            pass

        text_log = f"### {split.capitalize()} Samples (Epoch {self.current_epoch})\n\n"
        text_log += "| Reference | Generated |\n|---|---|\n"
        for ref, gen in samples:
            text_log += f"| {ref} | {gen} |\n"

        try:
            if hasattr(self.logger, "experiment") and hasattr(
                self.logger.experiment, "add_text"
            ):
                self.logger.experiment.add_text(
                    f"{split}/samples", text_log, self.global_step
                )
        except (ImportError, AttributeError):
            pass

    def on_validation_epoch_end(self) -> None:
        self.log_samples("val")

        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split="val",
            # tokenizer='zh' if outputs['lang'][0] == 'Chinese' else '13a',
            device=self.device,
        )

        self.log_dict(eval_res, sync_dist=True)

        self.set_container()

    def on_test_epoch_end(self) -> None:
        self.log_samples("test")

        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split="test",
            device=self.device,
        )

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            eps=1e-8,
            weight_decay=0.01,
            betas=(0.9, 0.98),
        )

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            lr_scheduler = {
                "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]

        return optimizer
