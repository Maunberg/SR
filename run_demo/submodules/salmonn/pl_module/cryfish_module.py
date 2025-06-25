# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import contextlib
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path
from transformers import LlamaTokenizer, StoppingCriteriaList, GenerationConfig

from peft import LoraConfig, TaskType, get_peft_model, PeftModel


from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric

from pytorch_lightning import LightningModule

from models.utils import StoppingCriteriaSub

# def zip_embeddings():
#     B, T, D = audiollm_embeds.shape
#        if audiollm_attention_mask is None:
#            audiollm_attention_mask = audiollm_embeds.ones((B, T), dtype=torch.bool)
#        if
#        speech_lengths = audiollm_attention_mask.sum(dim=1) if audiollm_attention_mask is not None else torch.full((B,), audiollm_embeds.shape[1], device=audiollm_embeds.device)
#        prompt_lengths_left = (
#            prompt_attention_mask[0].sum(dim=1)
#            if prompt_attention_mask is not None else
#            torch.full((B,), prompt_tokens_ids[0].shape[1], device=prompt_tokens_ids[0].device)
#        )
#        prompt_lengths_right = (
#            prompt_attention_mask[1].sum(dim=1)
#            if prompt_attention_mask is not None else
#            torch.full((B,), prompt_tokens_ids[1].shape[1], device=prompt_tokens_ids[1].device)
#        )
#
#        if text_tokens_ids is not None:
#            text_lengths = text_attention_mask.sum(dim=1) if text_attention_mask is not None else torch.full((B,), text_tokens_ids.shape[1], device=text_tokens_ids.device)
#        else:
#            text_lengths = 0
#        total_len = speech_lengths + prompt_lengths_left + prompt_lengths_right + text_lengths
#        max_len = total_len.max()
#        embs = audiollm_embeds.new_zeros((B, max_len, self.llm_token_embs.embedding_dim), requires_grad=True)


class CryFishDecoder(LightningModule):
    # enable loading from ckpt without LLM weights
    strict_loading = False

    @property
    def device(self):
        return list(self.parameters())[0].device

    def __init__(
        self,
        llm_model,
        connector_model,
        optimizers_config=None,
        llm_tokenizer=None,
        peft_config_or_path=None,
        debug_text_dir=None,
        llm_token_embs=None,
        save_text_every_n_step=1,
        DEBUG=False,
    ):
        super().__init__()
        self.DEBUG = DEBUG
        self.connector_model = connector_model
        if llm_token_embs is None:
            llm_token_embs = deepcopy(llm_model.get_input_embeddings())
            llm_model.set_input_embeddings(None)
            for p in llm_token_embs.parameters():
                p.requires_grad = False
        self.llm_token_embs = llm_token_embs

        if peft_config_or_path is not None:
            logging.info(
                "Peft config is found. Freeze llm parameters and initializing peft"
            )
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
            if isinstance(peft_config_or_path, str):
                logging.info(f"Loading peft from {peft_config_or_path}")
                self.llm_model = PeftModel.from_pretrained(
                    llm_model, peft_config_or_path
                )
            else:
                self.llm_model = get_peft_model(llm_model, peft_config_or_path)
        else:
            logging.info("Peft config is None. Finetune all model parameters")
            self.llm_model = llm_model

        self.llm_model.print_trainable_parameters()
        self.llm_tokenizer = llm_tokenizer
        self.train_loss = MeanMetric(nan_strategy="ignore")
        self.debug_text_dir = debug_text_dir
        self.save_text_every_n_step = save_text_every_n_step
        self.validation_step_outputs = []
        self.generate_cfg = None
        self.optimizers_config = optimizers_config

    def set_optimizers_config(self, optimizers_config):
        self.optimizers_config = optimizers_config

    def get_trainable_parameters(self, connector=True, llm=True, embs=False):
        parameters = []
        if connector:
            parameters.extend(
                [p for p in self.connector_model.parameters() if p.requires_grad]
            )
        if llm:
            parameters.extend(
                [p for p in self.llm_model.parameters() if p.requires_grad]
            )
        if embs:
            parameters.extend(
                [p for p in self.llm_token_embs.parameters() if p.requires_grad]
            )
        return parameters

    def configure_optimizers(self):
        return self.optimizers_config

    def _encode_auditory_feature(self, **kwargs):
        return self.connector_model(**kwargs)

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        raise DeprecationWarning("This method is deprecated")

    def concatenate_text_and_speech(
        self,
        audiollm_embeds,
        lprompt_tokens_ids,
        rprompt_tokens_ids,
        text_tokens_ids=None,
        audiollm_attention_mask=None,
        lprompt_attention_mask=None,
        rprompt_attention_mask=None,
        text_attention_mask=None,
    ):
        """
        audiollm_embeds is [B, T, D] tensor,
        prompt_tokens_ids (left_prompt[B, L1], right_pompt[B, L2]),
        audiollm_attention_mask is [B, T] bool tensor
        prompt_attention_mask is (left_prompt[B, L1], right_pompt[B, L2]) bool tensor
        """
        B, T, D = audiollm_embeds.shape
        logging.debug(
            f"{lprompt_tokens_ids.shape=}, {audiollm_embeds.shape=}, {rprompt_tokens_ids.shape=}"
        )
        if audiollm_attention_mask is None:
            audiollm_attention_mask = audiollm_embeds.ones((B, T), dtype=torch.bool)
        if lprompt_attention_mask is None:
            logging.debug("lprompt_attention_mask is None")
            lprompt_attention_mask = audiollm_embeds.new_ones(
                (B, lprompt_tokens_ids.shape[1]), dtype=torch.bool
            )
        if rprompt_attention_mask is None:
            logging.debug("rprompt_attention_mask is None")
            rprompt_attention_mask = audiollm_embeds.new_ones(
                (B, rprompt_tokens_ids.shape[1]), dtype=torch.bool
            )
        if text_tokens_ids is not None and text_attention_mask is None:
            logging.debug("text_attention_mask is None")
            text_attention_mask = audiollm_embeds.new_ones(
                (B, text_tokens_ids.shape[1]), dtype=torch.bool
            )

        prompt_embs = [
            self.llm_token_embs(p) for p in [lprompt_tokens_ids, rprompt_tokens_ids]
        ]
        if text_tokens_ids is not None:
            text_embs = self.llm_token_embs(text_tokens_ids)
            inputs_embeds = torch.cat(
                [prompt_embs[0], audiollm_embeds, prompt_embs[1], text_embs], dim=1
            )
            input_attention_mask = torch.cat(
                [
                    lprompt_attention_mask,
                    audiollm_attention_mask,
                    rprompt_attention_mask,
                    text_attention_mask,
                ],
                dim=1,
            )
            # labels = text_tokens_ids.new_fill(input_attention_mask.shape, self.llm_model.padding_idx)
            # labels[:, -text_tokens_ids.shape[1]: -1] = text_tokens_ids
        else:
            inputs_embeds = torch.cat(
                [prompt_embs[0], audiollm_embeds, prompt_embs[1]], dim=1
            )
            input_attention_mask = torch.cat(
                [
                    lprompt_attention_mask,
                    audiollm_attention_mask,
                    rprompt_attention_mask,
                ],
                dim=1,
            )
            # labels = None

        position_ids = (torch.cumsum(input_attention_mask, dim=1) - 1).clamp(
            min=0
        )  # skip all paddings
        return inputs_embeds, input_attention_mask, position_ids  # , labels

    def forward(self, batch):
        """
        batch = {
        "audio_feats.pth": [BxTxFeats] audio feats,
        "audio_feats_attention_mask.pth": [BxT] attention mask. 1 not mask, 0 - masking this element
        "lprompt_tokens_ids.pth" : [B x prefix], left-side paddings!
        "lprompt_attention_mask.pth": [B x prefix]
        "rprompt_tokens_ids.pth" : [B x prompt], left-side paddings!
        "rprompt_attention_mask.pth": [B x prompt]
        "text_tokens_ids.pth": [B x L]
        "text_attention_mask.pth": [B x L]
        }
        """
        audiollm_embeds, audiollm_attention_mask = self.connector_model(batch)
        #            audio_feats=batch["audio_feats.pth"],
        #            feats_attention_mask=batch.get("feats_attention_mask.pth", None),
        #        )
        labels = batch.get("text_tokens_ids.pth", None)
        inputs_embeds, attention_mask, position_ids = self.concatenate_text_and_speech(
            audiollm_embeds=audiollm_embeds,
            lprompt_tokens_ids=batch["lprompt_tokens_ids.pth"],
            rprompt_tokens_ids=batch["rprompt_tokens_ids.pth"],
            text_tokens_ids=labels,
            audiollm_attention_mask=audiollm_attention_mask,
            lprompt_attention_mask=batch.get("lprompt_attention_mask.pth", None),
            rprompt_attention_mask=batch.get("rprompt_attention_mask.pth", None),
            text_attention_mask=batch.get("text_attention_mask.pth", None),
        )
        if labels is not None:
            # added last rpromt token into labels
            labels = torch.cat([batch["rprompt_tokens_ids.pth"][:, -1:], labels], dim=1)
            # Default torch CE implementation ignores -100, not 0
            # assert (
            #    self.llm_token_embs.padding_idx == 0
            # ), "Carefully. Remove it if you are sure that everything is OK"
            labels = labels.masked_fill(labels == self.llm_token_embs.padding_idx, -100)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            labels=labels,  # carefull, labels are not shifted
            num_logits_to_keep=labels.shape[1] if labels is not None else 0,
        )
        logging.debug(f"{outputs.logits.shape=}, {outputs.loss=}")
        # DEBUG
        if self.DEBUG:
            torch.save(
                {
                    "batch": batch,
                    "labels": labels,
                    "audiollm_embeds": audiollm_embeds,
                    "audiollm_attention_mask": audiollm_attention_mask,
                    "outputs": outputs,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                f'tmp/{batch["__key__"][0]}.pth',
            )
        return outputs

    def training_step(self, batch, batch_idx):
        logging.debug(f"Process {batch_idx}, {batch['audio_feats.pth'].shape=}")
        outputs = self.forward(batch)
        loss = outputs.loss
        self.train_loss.update(loss)
        self.log(
            "loss_train",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            # batch_size=batch["audio_feats.pth"].shape[0],
        )
        logging.debug(f"loss_train for {batch_idx} is {loss.item()}")
        return {
            "loss": loss,
        }

    @torch.no_grad()
    @torch.inference_mode()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # logging.debug(f"valid started for {batch_idx}")
        outputs = self.forward(batch)
        self.log(
            "loss_valid",
            outputs.loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["audio_feats.pth"].shape[0],
        )
        if (
            self.debug_text_dir is not None
            and batch_idx % self.save_text_every_n_step == 0
        ):
            # remove last symbol, because  rprompt applied as a first item in labels in the forward
            # removes the last because the rprompt[-1] is used as the first element in labels in the forward
            preds = outputs.logits.argmax(dim=-1)[:, :-1]
            self.validation_step_outputs.append(
                (
                    batch_idx,
                    batch["text_tokens_ids.pth"],
                    preds.cpu(),
                    outputs.loss.cpu().item(),
                )
            )

    def on_validation_epoch_end(self):
        if self.debug_text_dir is not None:
            Path(self.debug_text_dir).mkdir(exist_ok=True)
            fname = str(time.strftime("%Y-%m-%d-%H:%M:%S.txt", time.gmtime()))
            with open(f"{self.debug_text_dir}/{fname}", "w") as f:
                for batch_idx, ref, hyp, loss in self.validation_step_outputs:
                    f.write(f"{batch_idx=} {loss=}\n")
                    for ref, hyp in zip(ref, hyp):
                        if self.llm_tokenizer is not None:
                            ref_txt = self.llm_tokenizer.decode(ref)
                            hyp_txt = self.llm_tokenizer.decode(hyp)
                            f.write(f"{ref_txt=}\n{hyp_txt=}\n")
                        ref_int = " ".join(f"{r.item()}" for r in ref)
                        hyp_int = " ".join(f"{h.item()}" for h in hyp)
                        f.write(f"{ref_int=}\n{hyp_int=}\n\n")
        self.validation_step_outputs.clear()  # free memory

    def prepare_for_generating(self, generate_cfg, llm_tokenizer=None, eos_token=None):
        """
        eos_token is a eos token or a list of EOS tokens
        If you know token ids you can pass it directly to the generate_cfg['eos_token_id']
        """
        eos_token_id = None
        if eos_token is not None:
            eos_token_id = self.llm_tokenizer.convert_tokens_to_ids(eos_token)
            logging.info(f"Stopping {eos_token=}, ids is {eos_token_id=}")
            assert isinstance(generate_cfg, dict) and not generate_cfg.get(
                "eos_token_id", None
            ), f"cannot reassign eos_token_id in {generate_cfg=}"
            generate_cfg["eos_token_id"] = eos_token_id
        if isinstance(generate_cfg, GenerationConfig):
            self.generate_cfg = generate_cfg
        else:
            self.generate_cfg = GenerationConfig(**generate_cfg)
        if llm_tokenizer is not None:
            self.llm_tokenizer = llm_tokenizer
        self.llm_model.set_input_embeddings(self.llm_token_embs)
        assert (
            self.llm_tokenizer is not None
        ), "Tokenizer must be specified in __init_ or in prepare_for_generating"

    @torch.no_grad()
    @torch.inference_mode()
    def predict_step(self, batch, batch_idx=None, dataloader_idx=0):
        assert self.llm_tokenizer is not None, "Cannot generate text without tokenizer!"
        assert (
            self.generate_cfg is not None
        ), "Initialize generation_config before predicting"
        audiollm_embeds, audiollm_attention_mask = self.connector_model(batch)
        # text_tokens_ids = batch['prompt_tokens_ids'][0].new_full((B, 1), self.llm_tokenizer.bos_token_id)
        inputs_embeds, attention_mask, position_ids = self.concatenate_text_and_speech(
            lprompt_tokens_ids=batch["lprompt_tokens_ids.pth"],
            audiollm_embeds=audiollm_embeds,
            rprompt_tokens_ids=batch["rprompt_tokens_ids.pth"],
            lprompt_attention_mask=batch.get("lprompt_attention_mask.pth", None),
            audiollm_attention_mask=audiollm_attention_mask,
            rprompt_attention_mask=batch.get("rprompt_attention_mask.pth", None),
        )
        logging.debug(
            "".join(
                f"{k}.shape={v.shape}"
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            )
        )
        logging.debug(
            f"{audiollm_embeds.shape=}, {audiollm_attention_mask.shape=}, {inputs_embeds.shape=}, {attention_mask.shape=}"
        )
        # TODO do we really need special stopping criteria ???
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            generation_config=self.generate_cfg,
            #stopping_criteria=self.stopping_criteria,
        )
        logging.debug(f"For batch {batch['__key__']}, {outputs.shape=}, {outputs}")
        text = self.llm_tokenizer.batch_decode(outputs, add_special_tokens=False)
        logging.debug(f"hyp: {text}")
        predicted = outputs.unsqueeze(0) if len(outputs.shape) == 1 else outputs
        return {
            "__key__": batch["__key__"],
            "prompt.txt": batch["prompt.txt"],
            "text.txt": batch["text.txt"],
            "wav_path.txt": batch["wav_path.txt"],
            "predicted.txt": text,
            "predicted.pth": predicted,
        }

    def on_save_checkpoint(self, checkpoint):
        state = checkpoint["state_dict"]
        param_grad_dic = {k: v.requires_grad for (k, v) in self.named_parameters()}
        for name in list(state.keys()):
            # saving only trainable parameters
            if not param_grad_dic[name]:
                state.pop(name)

    def inplace_load_from_checkpoint(self, ckpt_path):
        data = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missed, unexpected = self.load_state_dict(data, strict=self.strict_loading)
        logging.debug(f"Loaded model from {ckpt_path}\n{missed=}\n{unexpected=}")
        return self
