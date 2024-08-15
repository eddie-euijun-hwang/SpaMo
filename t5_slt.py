import os
import torch
import torch.nn as nn
import random
import math
import pytorch_lightning as pl
import torch.nn.functional as F

from abc import ABC, abstractmethod
from utils.helpers import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
from utils.tconv import TemporalConv
from utils.helpers import create_mask
from utils.mm_projector import build_vision_projector
from abstract_slt import AbstractSLT
from peft import LoraConfig, get_peft_model, TaskType
from prompt import MTRANS_H2S, TRG_TRANS_H2S, MTRANS_P14T, TRG_TRANS_P14T


class AbstractSLT(pl.LightningModule, ABC):
    """
    Abstract Sign Language Translation (SLT) Module: An abstract PyTorch Lightning module that defines a common interface
    for translating sign language from video inputs to text. Specific visual and textual models are to be defined in subclasses.
    """
    def __init__(
        self,
        lr=0.0001,
        monitor=None,
        scheduler_config=None,
        max_length=128,
        beam_size=5,
    ):
        super().__init__()
        # Initialize module parameters
        self.lr = lr
        self.monitor = monitor
        self.scheduler_config = scheduler_config
        self.max_length = max_length
        self.beam_size = beam_size

    @abstractmethod
    def prepare_models(self):
        """
        Subclasses should implement this method to prepare the visual and textual models.
        """
        pass

    @abstractmethod
    def shared_step(self, inputs, split, batch_idx):
        """
        Implements the logic common to training, validation, and testing steps.
        This method can be overridden or extended by subclasses if necessary.
        """
        # Adjust the implementation to use abstract methods
        pass

    @abstractmethod
    def get_inputs(self, batch):
        """
        Prepares input data from a batch for processing. This method is common and does not need to be abstract.
        """
        pass
    
    def training_step(self, batch, batch_idx):
        # Perform a training step.
        inputs = self.get_inputs(batch)
        loss, log_dict = self.shared_step(inputs, "train", batch_idx)
        self.log_dict(log_dict, batch_size=len(inputs['text']))
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Perform a validation step.
        inputs = self.get_inputs(batch)
        _, log_dict = self.shared_step(inputs, "val", batch_idx)
        self.log_dict(log_dict, batch_size=len(inputs['text']))

    def test_step(self, batch, batch_idx):
        # Perform a testing step.
        inputs = self.get_inputs(batch)
        _, log_dict = self.shared_step(inputs, "test", batch_idx)
        self.log_dict(log_dict, batch_size=len(inputs['text']))

    def configure_optimizers(self):
        # Configure the optimizer and learning rate scheduler.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8)
        
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            lr_scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                            'interval': 'step',
                            'frequency': 1}
            return [optimizer], [lr_scheduler]
        return optimizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FlanT5SLT(AbstractSLT):
    def __init__(
        self, 
        tuning_type='lora', 
        textual_config=None, 
        frame_sample_rate=1, 
        prompt='',
        input_size=1024,
        fusion_mode='joint',
        inter_hidden=768,
        max_frame_len=1024,
        max_txt_len=64,
        lemmatized=False,
        cross_modal_align=False,
        align_version='v1',
        warm_up_steps=None,
        combined_loss=False,
        alpha=0.1,
        ic=False,
        num_ic=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.prompt = prompt
        self.textual_config = textual_config
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.lemmatized = lemmatized
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.align_version = align_version
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.ic = ic
        self.num_ic = num_ic

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        
        self.prepare_models(textual_config)

        if tuning_type == 'freeze':
            self.freeze_model()
        elif tuning_type == 'lora':
            self.apply_lora()
        elif tuning_type == 'finetuning':
            pass

        self.prepare_container()

    def prepare_container(self):
        self.vis_emb_list = []
        self.vis_word_list = []

        self.id_list = []
        self.gloss_list = []        
        self.vis_string_list = []
        self.generated_text_list = []
        self.reference_text_list = []

    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f'Checkpoint is loaded from {checkpoint_path}.')

    def apply_lora(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)

    def freeze_model(self):
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False

    def prepare_models(self, t5_model):
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model, torch_dtype=torch.bfloat16)
        
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')
        self.t5_tokenizer.add_tokens('[VIDEO]')
        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

    def prepare_inputs(self, visual_outputs, visual_mask, samples, split, batch_idx):
        bs = visual_outputs.shape[0]

        prompt = [f'[VIDEO]{self.prompt}'] * bs
        prompt = [p.format(l) for p, l in zip(prompt, samples['lang'])]
        
        if self.ic:
            if split == 'train':
                translation_pairs = [
                    random.choices([f'{ex}={target}' for ex in ex_lang_trans], k=self.num_ic)
                    for ex_lang_trans, target in zip(samples['ex_lang_trans'], samples['text'])
                ]
                translation_pairs = derangement(translation_pairs)
                prompt = [f'{p} {" ".join(t)}' for p, t in zip(prompt, translation_pairs)]
            else:
                if samples['lang'][0] == 'English':
                    ex_lang_trans = MTRANS_H2S[:bs]
                    trg_trans = TRG_TRANS_H2S[:bs]
                elif samples['lang'][0] == 'German':
                    ex_lang_trans = MTRANS_P14T[:bs]
                    trg_trans = TRG_TRANS_P14T[:bs]

                translation_pairs = [
                    random.choices([f'{ex}={target}' for ex in ex_lang_trans], k=self.num_ic)
                    for ex_lang_trans, target in zip(ex_lang_trans, trg_trans)
                ]
                prompt = [f'{p} {" ".join(t)}' for p, t in zip(prompt, ex_lang_trans)]
        
        input_tokens = self.t5_tokenizer(
            prompt,
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
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)
        
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples):
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length
            
            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
            )
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), device=self.device) 
            
        else:
            if spatial:
                spaital_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = spaital_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(seq_lengths=spaital_conv_outputs['feat_len'].to(torch.int).tolist(), device=self.device)
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError
        
        return visual_outputs, visual_masks

    def get_inputs(self, batch):
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        en_texts = []
        ex_lang_translations = []
        
        max_frame_len = self.max_frame_len
        lang_list = ['en', 'es', 'fr', 'de']

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]

                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                glosses.append(sample['gloss'])
                langs.append(sample['lang'])
                
                _ex_lang_trans = []
                for lang in lang_list:
                    if f'{lang}_text' in sample:
                        _ex_lang_trans.append(sample[f'{lang}_text'])
                ex_lang_translations.append(_ex_lang_trans)
                
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))
                
                if sample['bool_mask_pos'] is not None:
                    masks.append(torch.tensor(sample['bool_mask_pos'], dtype=torch.bool))
        
        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'bool_mask_pos': masks,
            'ids': ids,
            'text': texts,
            'ex_lang_trans': ex_lang_translations,
            'gloss': glosses,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
        }

    def visual_textual_align(self, visual_outputs, visual_masks, samples):        
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
        
        image_embeds = visual_outputs.mean(1)  # pooler_output
        text_embeds = text_embeds.mean(1)  # pooler_output
        
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        # Calculate clip loss for cross-modal similarities
        loss = clip_loss(logits_per_text)
        
        return loss

    def shared_step(self, inputs, split, batch_idx):
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        
        if self.cross_modal_align and self.warm_up_steps is None and not(self.combined_loss):
            with torch.no_grad():
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs=visual_outputs, 
                    visual_mask=visual_masks, 
                    samples=inputs,
                    split=split,
                    batch_idx=batch_idx
                )
            
            cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
            
            log_dict = {f"{split}/contra_loss": cont_loss}
            
            loss = cont_loss
        else:
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs=visual_outputs, 
                visual_mask=visual_masks, 
                samples=inputs,
                split=split,
                batch_idx=batch_idx
            )
            
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            
            loss = outputs.loss
            log_dict = {f"{split}/loss": loss}
            
            if (
                self.cross_modal_align 
                and self.warm_up_steps is not None 
                and self.global_step <= self.warm_up_steps
            ):
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)

                loss = cont_loss
            elif (self.cross_modal_align and self.combined_loss):
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)

                loss = loss + self.alpha * cont_loss

        if split != "train":
            input_embeds, input_masks, _, _ = self.prepare_inputs(
                visual_outputs=visual_outputs, 
                visual_mask=visual_masks, 
                samples=inputs,
                split=split,
                batch_idx=batch_idx
            )
            
            generated = self.t5_model.generate(
                inputs_embeds=input_embeds, 
                attention_mask=input_masks, 
                num_beams=5,
                max_length=self.max_txt_len,
                top_p=0.9,
                do_sample=True,
            )
            
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]
            
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            self.generated_text_list.extend(generated_strings)
            self.reference_text_list.extend(reference_strings)

            if inputs['gloss'][0] is not None:
                self.gloss_list.extend([gloss.lower() for gloss in inputs['gloss']])
            
            self.id_list.extend(inputs['ids'])
        
            if inputs['lang'][0] == 'Chinese': 
                tokenizer = 'zh' 
            else: 
                tokenizer = '13a'
            
            eval_res = evaluate_results(
                predictions=generated_strings, 
                references=reference_strings, 
                split=split, 
                tokenizer=tokenizer, 
                device=self.device
            )

            log_dict.update(eval_res)

        return loss, log_dict





