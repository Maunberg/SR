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
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
import librosa
from beats.BEATs import BEATsConfig, BEATs
from qformer.Qformer import BertConfig, BertLMHeadModel
import time
import psutil
gpus = torch.cuda.device_count()


def to_gb(num):
    return num/2**3**3/8
    
def print_mem(device_num):
    t = torch.cuda.get_device_properties(device_num).total_memory
    t = to_gb(t)
    r = torch.cuda.memory_reserved(device_num)
    r = to_gb(r)
    a = torch.cuda.memory_allocated(device_num)
    a = to_gb(a)
    f = r-a  # free inside reserved
    print(f"card cuda:{device_num}, total {t:.2f}G; reserved: {r:.2f}; allocated: {a:.2f}; 'free': {f:.2f}")

def print_sys_stats(gpus):
    print(f'RAM Used (GB):, {psutil.virtual_memory()[3]/1000000000:.2f}')
    for i in range(gpus):
        print_mem(i)

class SALMONN_mutigpu(nn.Module):
    def __init__(
        self,
        ckpt,
        whisper_path,
        beats_path,
        vicuna_path,
        speech_qformer_token_num=1,
        speech_qformer_layer=2,
        lora=True,
        lora_alpha=32,
        lora_rank=8,
        lora_dropout=0.1,
        second_per_frame=0.333333,
        second_stride=0.333333,
        low_resource=False, 
        gpu_switch_layer=18
    ):

        super().__init__()
        
        print_sys_stats(gpus)

        # feature_extractor
        self.whisper_device_in = "cuda:1"
        self.whisper_device_out = "cuda:2"
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path, device=self.whisper_device_in)
        print(f"WhisperFeatureExtractor loaded to {self.whisper_device_in}", flush=True)

        # whisper
        whisper = WhisperModel.from_pretrained(whisper_path)
        whisper.eval()
        print('loaded whisper to cpu', flush=True)
        self.speech_encoder = whisper.encoder.to(device=self.whisper_device_in)
        del(whisper.decoder)
        print('deleted decoder', flush=True)
        
        # splitting whisper to 2 gpus
        # gpu_switch_layer = 18
        def hook_on_tuple(_, inputs):
            return (inputs[0].to(self.whisper_device_out), None)
        for idx, encoder_layer in enumerate(self.speech_encoder.layers):
            if idx>=gpu_switch_layer:
                encoder_layer.to(self.whisper_device_out)
            if idx==gpu_switch_layer:
                self.hook_whisper_handle = encoder_layer.register_forward_pre_hook(hook_on_tuple)
        self.speech_encoder.layer_norm.to(self.whisper_device_out)
        
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model, device=self.whisper_device_out)
        
        print(f'sent encoder layers starting from {gpu_switch_layer} to {self.whisper_device_out}', flush=True)

        # beats
        self.beats_device = "cpu"
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location=self.beats_device)
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim, device=self.beats_device)
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

        # init speech Qformer
        self.qformer_device = "cuda:2"
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            speech_qformer_layer,
            device = self.qformer_device
        )
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        
        # vicuna
        self.llama_device = 'cuda:0'
        if not low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
            ) # .to(self.llama_device) can't load to 11g vram card anyway
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto' #{'': 0}
            )
        print(f"llama(vicuna) loaded to {self.llama_device}, probably...", flush=True)

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
            self.llama_model = get_peft_model(self.llama_model, self.peft_config) #.to(self.llama_device)
        print(f"lora applied to llama on {self.llama_device}", flush=True)

        # tokenizer
        self.llama_tokenizer_device = "cpu"
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        self.llama_tokenizer.padding_side = "right"
        print(f"llama tokenizer loaded to {self.llama_tokenizer_device}", flush=True)

        # proj
        self.llama_proj_device = self.llama_device
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size).to(self.llama_proj_device)

        # load ckpt
        ckpt_dict = torch.load(ckpt)['model']
        self.load_state_dict(ckpt_dict, strict=False)
        print_sys_stats(gpus)
        print(f"finished loading", flush=True)

    @torch.no_grad()
    def generate(
        self,
        wav_path,
        prompt,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
        max_length=200,
        num_beams=4,
        do_sample=True,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ):
        start_time = time.time()
        last_time = start_time
        # read wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        
        print(f"loaded wav in {time.time() - last_time:.2f} seconds", flush=True)
        print_sys_stats(gpus)
        last_time = time.time()
        
        # whisper
        spectrogram = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(self.whisper_device_in) # [1, 80, 3000]
        print(f"spectrogram done in {time.time() - last_time:.2f} seconds", flush=True)
        last_time = time.time()
        
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
        print(f"whisper done in {time.time() - last_time:.2f} seconds", flush=True)
        last_time = time.time()
       
        # beats
        raw_wav = torch.from_numpy(wav).to(self.beats_device).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=self.beats_device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
        print(f"beats done in {time.time() - last_time:.2f} seconds", flush=True)
        last_time = time.time()

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds).to(self.qformer_device)
        audio_embeds = self.ln_audio(audio_embeds).to(self.qformer_device)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)
        print(f"auditory embeds formed in {time.time() - last_time:.2f} seconds")
        last_time = time.time()

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
        print(f"split frames done in {time.time() - last_time:.2f} seconds", flush=True)
        last_time = time.time()

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1).to(self.qformer_device)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state.to(self.llama_proj_device))
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        print(f"Qformer done in {time.time() - last_time:.2f} seconds", flush=True)
        last_time = time.time()

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.llama_device).input_ids
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
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=self.llama_device,
            ) * self.llama_tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        print(f"Llama prep done in {time.time() - last_time:.2f} seconds", flush=True)
        print(f"Llama input shape {embeds.shape}")
        last_time = time.time()
        

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
        
        print(f"Llama gen done in {time.time() - last_time:.2f} seconds", flush=True)
        last_time = time.time()
        
        output_text = self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)
        
        print(f"Llama tokenization done in {time.time() - last_time:.2f} seconds", flush=True)
        print_sys_stats(gpus)
        print(f"Full infer done in {time.time() - start_time:.2f} seconds", flush=True)
        print("-----end of infer-----", flush=True)

        return output_text

    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2, device='cpu'):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        Qformer.to(device)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size, device=device)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
