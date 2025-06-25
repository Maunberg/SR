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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import LlamaTokenizer, StoppingCriteriaList
from peft import LoraConfig, TaskType, get_peft_model

# from .Qformer import BertConfig, BertLMHeadModel
# from .modeling_llama import LlamaForCausalLM
from .modeling_whisper import WhisperModel

# from transformers import WhisperModel # doen't work
from .beats.BEATs import BEATsConfig, BEATs
from .utils import StoppingCriteriaSub


class AudioFeatsExtractor(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device

    def __init__(self, whisper_path="", beats_path="", feats_dtype=torch.float16):
        super().__init__()
        self.whisper_path = whisper_path
        self.beats_path = beats_path

        assert whisper_path
        logging.info("Loading Whisper Model")
        self.speech_encoder = WhisperModel.from_pretrained(self.whisper_path).encoder
        if self.beats_path:
            logging.info("Loading BEATs Model")
            beats_ckpt = torch.load(self.beats_path, map_location="cpu")
            beats_cfg = BEATsConfig(beats_ckpt["cfg"])
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt["model"])

    def forward(self, batch):
        # spec.pth is (B, T, F)
        # spectrogram is (B, F, T)
        spectrogram = batch["spec.pth"].to(self.device) #.permute(0, 2, 1)
        # attention_mask is (B, T)
        attention_mask = batch.get("spec_attention_mask.pth", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        raw_wav = batch.get("raw_wav.pth", None)
        raw_wav_attention_mask = batch.get("raw_wav_attention_mask.pth", None)

        print("spectrograms!!", spectrogram.shape)
        whisper_embeds = self.speech_encoder(
            spectrogram, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        # if not torch.all(attention_mask):
        '''
        attention_mask_pad = torch.nn.functional.pad(
            attention_mask, (0, 1), value=False
        )
        whisper_embeds_attention_mask = (
            attention_mask_pad.unfold(1, 2, 2).sum(axis=-1) > 0
        )'''
        whisper_embeds_attention_mask = None

        if self.beats_path:
            assert raw_wav is not None, f"raw_wav.pth not in {batch.keys()=}"
            padding_mask = (
                None
                if raw_wav_attention_mask is None
                else ~raw_wav_attention_mask.to(self.device)
            )
            beats_embeds, _ = self.beats.extract_features(
                raw_wav.to(self.device), padding_mask=padding_mask, feature_only=True
            )
            diff = whisper_embeds.shape[1] - beats_embeds.shape[1]
            if diff > 0:
                if diff > 5:
                    logging.warning(
                        f"Padding too much beats embs {beats_embeds.shape=}, {whisper_embeds.shape=}."
                    )
                beats_embeds = torch.nn.functional.pad(
                    beats_embeds, (0, 0, 0, diff), value=0.0
                )
            elif diff < 0:
                logging.warning(
                    f"{beats_embeds.shape=}, {whisper_embeds.shape=}. Something wrong"
                )
                beats_embeds = beats_embeds[:, : whisper_embeds.shape[1], :]
            return {
                "audio_feats.pth": torch.cat([whisper_embeds, beats_embeds], dim=-1),
                "audio_feats_attention_mask.pth": whisper_embeds_attention_mask,
            }
        else:
            return {
                "audio_feats.pth": whisper_embeds,
                "audio_feats_attention_mask.pth": whisper_embeds_attention_mask,
            }


class ChunkingAudioFeatsExtractor(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device

    def __init__(
        self,
        whisper_path="",
        beats_path="",
        feats_dtype=torch.float16,
        whisper_chunk_size=3000,
        whisper_chunk_step=3000,
        min_whisper_chunk=100,
    ):
        super().__init__()
        self.whisper_path = whisper_path
        self.beats_path = beats_path
        self.whisper_chunk_size = whisper_chunk_size
        self.whisper_chunk_step = whisper_chunk_step
        self.min_whisper_chunk = self.min_whisper_chunk

        assert whisper_path
        logging.info("Loading Whisper Model")
        self.speech_encoder = WhisperModel.from_pretrained(self.whisper_path).encoder
        if self.beats_path:
            logging.info("Loading BEATs Model")
            beats_ckpt = torch.load(self.beats_path, map_location="cpu")
            beats_cfg = BEATsConfig(beats_ckpt["cfg"])
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt["model"])

    def forward(self, batch):
        # spec.pth is (B, T, F)
        # spectrogram is (B, F, T)
        spectrogram = batch["spec.pth"].permute(0, 2, 1).to(self.device)
        # attention_mask is (B, T)
        spec_attention_mask = batch.get("spec_attention_mask.pth", None)
        if spec_attention_mask is not None:
            spec_attention_mask = spec_attention_mask.to(self.device)

        raw_wav = batch.get("raw_wav.pth", None)
        raw_wav_attention_mask = batch.get("raw_wav_attention_mask.pth", None)
        B, F, T = spectrogram.shape
        num_chunks = math.ceil(T / self.whisper_chunk_step)
        target_T = (num_chunks - 1) * self.whisper_chunk_step + self.whisper_chunk_size
        pad_right = target_T - T
        # numchunks=1 if T <= step, 0+ sz - T
        # numchunks=2 if step < T <= 2step, st + sz - T
        spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_right), value=0.0)
        spectrogram = torch.nn.functional.unfold(
            spectrogram.unsqueeze(dim=1),
            (F, self.whisper_chunk_size),
            stride=(F, self.whisper_chunk_step),
        )
        # torch.allclose(spectrogram2.view(B, F, whisper_chunk_size, L).transpose(2, 3).reshape(B, F, T), spectrogram) == True
        assert spectrogram.shape == (
            B,
            F * self.whisper_chunk_size,
            num_chunks,
        ), f"{spectrogram.shape=}, {B=}, {F=}, {T=}, {self.whisper_chunk_size=}, {self.whisper_chunk_step=}, {num_chunks=}, {pad_right=}"
        spectrogram = spectrogram.transpose(1, 2).view(
            B * num_chunks, F, self.whisper_chunk_size
        )
        logging.debug(
            f"Whisper chunker: {B=}, {T=}, {num_chunks=}, {spectrogram.shape=}"
        )
        whisper_embeds = self.speech_encoder(
            spectrogram, return_dict=True
        ).last_hidden_state
        NC, emb_chunk_size, emb_dim = whisper_embeds.shape
        whisper_embeds = whisper_embeds.view(
            B, num_chunks, emb_chunk_size * emb_dim
        ).transpose(1, 2)
        scale = emb_chunk_size / self.whisper_chunk_size
        emb_chunk_step = scale * self.whisper_chunk_step
        assert emb_chunk_step == int(
            emb_chunk_step
        ), "{whisper_embeds.shape=}, {spectrogram.shape=}, {self.whisper_chunk_size=}"
        emb_chunk_step = int(emb_chunk_step)
        emb_T = int(scale * target_T)
        emb_pad_right = int(scale * pad_right)
        whisper_embeds = torch.nn.functional.fold(
            whisper_embeds,
            (emb_T, emb_dim),
            kernel_size=(emb_chunk_size, emb_dim),
            stride=(emb_chunk_step, emb_dim),
        )
        r = int(1 / scale)
        attention_mask_pad = torch.nn.functional.pad(
            spec_attention_mask, (0, r - 1), value=False
        )
        whisper_embeds_attention_mask = (
            attention_mask_pad.unfold(1, r, r).sum(axis=-1) > 0
        )
        whisper_embeds = whisper_embeds[B, :-emb_pad_right]

        if self.beats_path:
            assert raw_wav is not None, f"raw_wav.pth not in {batch.keys()=}"
            padding_mask = (
                None
                if raw_wav_attention_mask is None
                else ~raw_wav_attention_mask.to(self.device)
            )
            beats_embeds, _ = self.beats.extract_features(
                raw_wav.to(self.device), padding_mask=padding_mask, feature_only=True
            )
            diff = whisper_embeds.shape[1] - beats_embeds.shape[1]
            if diff > 0:
                if diff > 5:
                    logging.warning(
                        f"Padding too much beats embs {beats_embeds.shape=}, {whisper_embeds.shape=}."
                    )
                beats_embeds = torch.nn.functional.pad(
                    beats_embeds, (0, 0, 0, diff), value=0.0
                )
            elif diff < 0:
                logging.warning(
                    f"{beats_embeds.shape=}, {whisper_embeds.shape=}. Something wrong"
                )
                beats_embeds = beats_embeds[:, : whisper_embeds.shape[1], :]
            return {
                "audio_feats.pth": torch.cat([whisper_embeds, beats_embeds], dim=-1),
                "audio_feats_attention_mask.pth": whisper_embeds_attention_mask,
            }
        else:
            return {
                "audio_feats.pth": whisper_embeds,
                "audio_feats_attention_mask.pth": whisper_embeds_attention_mask,
            }
