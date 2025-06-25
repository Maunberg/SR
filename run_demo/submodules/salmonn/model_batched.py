# Copyright (2023) Tsinghua University, Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
from beats.BEATs import BEATsConfig, BEATs
from qformer.Qformer import BertConfig, BertLMHeadModel

class SALMONN(nn.Module):
    def __init__(
        self,
        ckpt,
        whisper_path,
        beats_path,
        vicuna_path,
        encoder_device,
        llama_device_map,
        llama_dtype=torch.bfloat16,
        speech_qformer_token_num=1,
        speech_qformer_layer=2,
        lora=True,
        lora_alpha=32,
        lora_rank=8,
        lora_dropout=0.1,
        second_per_frame=0.333333,
        second_stride=0.333333,
        low_resource=False
    ):
        self.encoder_device = encoder_device
        self.llama_device_map = llama_device_map
        self.llama_first_device = torch.device(llama_device_map['model.embed_tokens'])
        super().__init__()

        # feature_extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)

        # whisper
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder.to(encoder_device)
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model).to(encoder_device)

        # beats
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats.to(encoder_device)
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim).to(encoder_device)
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

        # init speech Qformer
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            encoder_device,
            speech_qformer_layer
        )
        self.speech_Qformer = self.speech_Qformer.to(encoder_device)
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        
        # vicuna
        if not low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=llama_dtype,
                device_map=llama_device_map
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=llama_dtype,
                load_in_8bit=True,
                device_map=llama_device_map
            )

        # lora
        self.lora = lora
        if lora:
            target_modules = None
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=True, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)

        # proj
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size).to(encoder_device)

        # load ckpt
        ckpt_dict = torch.load(ckpt)['model']
        self.load_state_dict(ckpt_dict, strict=False)

    def encode(
        self,
        wav,
        prompt,
        prompt_pattern='USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:'
    ):
        # whisper
        spectrogram = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(self.encoder_device) # [1, 80, 3000]
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

        # beats
        raw_wav = torch.from_numpy(wav).to(self.encoder_device).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=self.encoder_device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames
        B, T, C = speech_embeds.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous().to(self.llama_first_device)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=self.llama_first_device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=self.llama_first_device,
            ) * self.llama_tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        
        return embeds, atts
        
    def batch_encode(
        self,
        wavs,
        prompts,
        prompt_pattern='USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:'
    ):
        # whisper
        spectrogram = self.feature_extractor(wavs, return_tensors="pt", sampling_rate=16000).input_features.to(self.encoder_device)
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

        # beats: non-batch because of beats' incorrect mask implementation
        audio_embeds = []
        for wav in wavs:
            raw_wav = torch.Tensor(wav).to(self.encoder_device).unsqueeze(0)
            audio_padding_mask = torch.zeros(raw_wav.shape, device=self.encoder_device).bool()
            audio_embeds_sub, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            audio_embeds_sub = F.pad(audio_embeds_sub, (0, 0, 0, speech_embeds.size(1) - audio_embeds_sub.size(1)))
            audio_embeds.append(audio_embeds_sub)
        audio_embeds = torch.cat(audio_embeds, dim = 0)

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames        
        B, T, C = speech_embeds.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous().to(self.llama_first_device)

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = [], []
        for prompt in prompts:
            prompt_left_, prompts_right_ = prompt_pattern.format(prompt).split('<SpeechHere>')
            prompt_left.append(prompt_left_)
            prompts_right.append(prompts_right_)
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.llama_tokenizer(
                prompts_right,
                add_special_tokens=False
            )
        prompt_right_tensors = []
        for elem in prompt_right_ids['input_ids']:
            prompt_right_tensors.append(embed_tokens(torch.Tensor(elem).int()))
        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [len(prompts), 1],
                dtype=torch.long,
                device=self.llama_first_device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [len(prompts), 1],
                dtype=torch.long,
                device=self.llama_first_device,
            ) * self.llama_tokenizer.bos_token_id
        )
        input_max_length = max([bos_embeds[i].shape[0] + prompt_left_embeds[i].shape[0] + 
                                speech_embeds[i].shape[0] + prompt_right_tensors[i].shape[0] for i in range(len(prompts))])
        tensor_list, attention_mask = [], []
        for i in range(len(prompts)):
            subtensor = torch.cat([bos_embeds[i], prompt_left_embeds[i], speech_embeds[i], prompt_right_tensors[i].to(self.llama_first_device)], dim=0)
            submask = torch.ones(subtensor.shape[0], device=self.llama_first_device)
            if input_max_length - subtensor.shape[0] > 0:
                submask = torch.cat([torch.zeros(input_max_length - subtensor.shape[0], device=self.llama_first_device), submask], dim=0)
                pad_embeds = self.llama_model.model.embed_tokens(
                    torch.ones(
                        [input_max_length - subtensor.shape[0]],
                        dtype=torch.long,
                        device=self.llama_first_device,
                    ) * self.llama_tokenizer.pad_token_id
                ) if not self.lora else self.llama_model.model.model.embed_tokens(
                    torch.ones(
                        [input_max_length - subtensor.shape[0]],
                        dtype=torch.long,
                        device=self.llama_first_device,
                    ) * self.llama_tokenizer.pad_token_id
                )
                subtensor = torch.cat([pad_embeds, subtensor], dim=0)
            tensor_list.append(subtensor)
            attention_mask.append(submask)

        embeds = torch.stack(tensor_list)
        atts = torch.stack(attention_mask).int()

        return embeds, atts

    def generate(
        self,
        wav,
        prompt,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        max_length=150,
        num_beams=4,
        do_sample=True,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ):
        if type(wav) is list:
            embeds, atts = self.batch_encode(wav, prompt, prompt_pattern)
        else:
            embeds, atts = self.encode(wav, prompt, prompt_pattern)

        # generate
        output = self.llama_model.generate(
            inputs_embeds=embeds,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            attention_mask=atts,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id
        )
        
        output_text = self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def init_speech_Qformer(self, num_query_token, speech_width, device, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size, device=device)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens