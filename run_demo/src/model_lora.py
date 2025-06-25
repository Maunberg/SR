import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, PeftModel
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer
)
import librosa
from submodules.salmonn.models.Qformer import BertConfig, BertLMHeadModel
from submodules.salmonn.models.beats.BEATs import BEATsConfig, BEATs
from submodules.salmonn.models.whisper_beats_feats_extractor import AudioFeatsExtractor
# from .beats.BEATs import BEATsConfig, BEATs
# from .qformer.Qformer import BertConfig, BertLMHeadModel
import time
import psutil
import gc
import numpy as np
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional

debug_path = Path("/media/data/agafonov/repos/allm_service/debug/torch_model_tlite")

def to_gb(num):
    return num/2**3**3/8
    
def print_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    t = to_gb(t)
    r = torch.cuda.memory_reserved(0)
    r = to_gb(r)
    a = torch.cuda.memory_allocated(0)
    a = to_gb(a)
    f = r-a  # free inside reserved
    print(f"CUDA memory - total: {t:.2f}G; reserved: {r:.2f}; allocated: {a:.2f}; 'free': {f:.2f}")

def print_sys_stats():
    print(f'RAM Used (GB): {psutil.virtual_memory()[3]/1000000000:.2f}')
    print_mem()
        
class SALMONN_mutigpu(nn.Module):    
    def __init__(
        self,
        whisper_path,
        beats_path,
        vicuna_path,
        connector_path,
        lora_path,
        low_resource=False, 
    ):

        super().__init__()
        
        print_sys_stats()

        # spectogram
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path, device="cuda:0")
        
        # audio feats
        self.audio_feats_extractor_device = "cuda:0"
        self.audio_feats_extractor = AudioFeatsExtractor(whisper_path=whisper_path, beats_path=beats_path)
        self.audio_feats_extractor.to(self.audio_feats_extractor_device)
        
        # init connector
        self.connector_device = "cuda:0"
        self.connector = torch.jit.load(connector_path)
        self.connector.to(self.connector_device)
        print(f'loaded Qformer to {self.connector_device}', flush=True)
        print_sys_stats()
        
        # llm
        self.llama_device = 'cuda:0'
        if not low_resource:
            self.llm_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto'
            )
        print(f"llama(vicuna) loaded to {self.llama_device}", flush=True)
        print_sys_stats()

        # lora
        if lora_path is not None:
            self.lora = True
        else:
            self.lora = False
        if self.lora:
            target_modules = None
            self.llm_model = PeftModel.from_pretrained(self.llm_model, lora_path)
        print(f"lora applied to llama on {self.llama_device}", flush=True)
        print_sys_stats()
        
        self.embed_tokens = self.llm_model.model.model.embed_tokens if self.lora else self.llm_model.model.embed_tokens

        # tokenizer
        self.llama_tokenizer_device = "cuda:0"
        self.llama_tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        self.llama_tokenizer.padding_side = "right"
        print(f"llama tokenizer loaded to {self.llama_tokenizer_device}", flush=True)
        print_sys_stats()
        self.terminators = [
            self.llama_tokenizer.eos_token_id,
            self.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # load ckpt
        print(f"finished loading", flush=True)
        print_sys_stats()
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Done cleaning", flush=True)
        print_sys_stats()

    
    def _load_wavs(self, wav_paths):
        last_time = time.time()
        time_dict = dict()

        wavs = []
        srs = []
        for wav_path in wav_paths:
            if not wav_path.exists():
                print(f'NO FILE FOUND ON {wav_path} PATH!!!')
                continue
            wav, sr = sf.read(wav_path)
            if len(wav) < 100:
                print(f'EMPTY WAV ON {wav_path} PATH!!!')
                continue
            if len(wav.shape) == 2:
                wav = wav[:, 0]
            if len(wav) > 30 * sr:
                wav = wav[: 30 * sr]
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
            wavs.append(wav)
            srs.append(sr)
        time_dict['wav_load'] = time.time() - last_time
        return (wavs, srs, time_dict)

    def _generate_speech_embeds(self, audio_list, sr_list, time_dict=None):
        for i in sr_list:
            assert i == 16000, f'wrong sampling rate {i} != 16000'
        audio_mask_percent = [min((0.1+len(audio)/16000)/30, 1) for audio in audio_list]
        if time_dict is None:
            time_dict = {}
        last_time = time.time()

        # spectrogram
        print('spectrogram', audio_list[0].shape)
        spectrograms = self.feature_extractor(audio_list, return_tensors="pt", 
                                              sampling_rate=16000).input_features.to(self.audio_feats_extractor_device) # [1, 80, 3000]
        time_dict['spectrogram'] = time.time() - last_time
        
        last_time = time.time()
        
        # whisper
        with torch.cuda.stream(self.stream_whisper):
            speech_embeds = self.speech_encoder(spectrograms, return_dict=True).last_hidden_state
       
        # beats
        with torch.cuda.stream(self.stream_beats):
            max_len = max(len(arr) for arr in audio_list)
            batch_size = len(audio_list)

            raw_tensor = torch.zeros((batch_size, max_len), dtype=torch.float64, device=self.beats_device)
            padding_mask = torch.ones((batch_size, max_len), device=self.beats_device).bool()

            for i, wav_np in enumerate(audio_list):
                raw_tensor[i, :len(wav_np)] = torch.from_numpy(wav_np)
                padding_mask[i, :len(wav_np)] = 0
            
            audio_embeds, _ = self.beats.extract_features(raw_tensor, padding_mask=padding_mask, feature_only=True)
        
        # sync
        torch.cuda.synchronize(device="cuda:0")
        
        time_dict['whisper+beats'] = time.time() - last_time

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds).to(self.qformer_device)
        audio_embeds = self.ln_audio(audio_embeds).to(self.qformer_device)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)
        
        time_dict['auditory_embeds'] = time.time() - last_time
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
        
        time_dict['split_frames'] = time.time() - last_time
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
        
        time_dict['Qformer'] = time.time() - last_time
        last_time = time.time()

        return speech_embeds, time_dict, audio_mask_percent


    def _batch_from_prompt_embeds(
                        self,
                        embeds: List[torch.Tensor]|torch.Tensor, 
                        pre_prompts: List[str], 
                        post_prompts: List[str],
                        time_dict: dict = None,
                        audio_mask_percent=None,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if time_dict is None:
            time_dict = {}
        time_last = time.time()
        if isinstance(embeds, List):
            batch_size = len(embeds)
            embdeds_type = "list"
        else:
            batch_size = embeds.shape[0]
            embdeds_type = "tensor"

        if audio_mask_percent is None:
            audio_mask_percent = [1 for _ in range(batch_size)]

        # tokenize prompts
        pre_tokens = [self.llama_tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False).to(self.llama_device) for prompt in pre_prompts]
        
        post_tokens = [self.llama_tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False).to(self.llama_device) for prompt in post_prompts]
        
        # make list of embeddings from tokenized prompts
        pre_embeds = [self.embed_tokens(tokens.input_ids) for tokens in pre_tokens]
        post_embeds = [self.embed_tokens(tokens.input_ids) for tokens in post_tokens]

        max_len = 0
        for i in range(batch_size):
            max_len = max(max_len,pre_embeds[i].shape[1] + post_embeds[i].shape[1])
        max_len += embeds.shape[1]

        dim = embeds.shape[-1]
        batch_tensor = torch.zeros(batch_size, max_len, dim, dtype=torch.float16, device=self.llama_device)
        att_mask_tensor = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.llama_device)
        for batch_pos in range(batch_size):
            if embdeds_type == "list":
                audio_embeds_tmp = embeds[batch_pos]
            elif embdeds_type == "tensor":
                audio_embeds_tmp = embeds[batch_pos, :, :]
            # if it's 2d, add batch dim
            if len(audio_embeds_tmp.shape)==2:
                audio_embeds_tmp = audio_embeds_tmp.unsqueeze(0)
            print(audio_embeds_tmp.shape, audio_embeds_tmp.shape[1]*audio_mask_percent[batch_pos], audio_mask_percent[batch_pos])
            audio_embeds_tmp = audio_embeds_tmp[:, 0:int(audio_embeds_tmp.shape[1]*audio_mask_percent[batch_pos]), :]
            print(audio_embeds_tmp.shape, "________")
            embeds_insert = torch.cat([pre_embeds[batch_pos], audio_embeds_tmp ,post_embeds[batch_pos]], dim=1)
            att_mask_tmp = torch.ones(batch_size, embeds_insert.shape[1], dtype=torch.long, device=self.llama_device)
            emb_len = embeds_insert.shape[1]
            batch_tensor[batch_pos, -emb_len:,:] = embeds_insert
            att_mask_tensor[batch_pos, -emb_len:] = 1

        time_dict["prompt_wrap"] = time.time() - time_last

        return batch_tensor, att_mask_tensor, time_dict
        

    def generate(
            self,
            prompts: List[str],
            wav_paths: Optional[List[str]] = None,
            audios: Optional[List[np.ndarray]] = None,
            srs: Optional[List[int]] = None,
            max_new_tokens=200,
            num_beams=4,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.2,
            length_penalty=1.0,
            temperature=1.0,
            system_prompt=None,
            system_inject_type="system", 
            user_pattern = "<Speech><SpeechHere></Speech> {}"
    ):
        # cheching vaildiy of input
        if wav_paths is None and audios is None:
            raise ValueError("Either `wav_paths` or `audios` must be provided.")
        if wav_paths is not None and audios is not None:
            raise ValueError("Only one of `wav_paths` or `audios` can be provided.")
        if wav_paths is not None and len(prompts) != len(wav_paths):
            raise ValueError("The number of prompts must equal the number of audio paths.")
        if audios is not None and len(prompts) != len(audios):
            raise ValueError("The number of prompts must equal the number of audio arrays.")
        if srs is not None and len(srs) != len(prompts):
            raise ValueError("The number of prompts must equal the number of sample rates.")
        
        # print(str(audios)[:100])
        # wavs prepare
        if wav_paths is not None:
            wavs, srs, time_dict  = self._load_wavs(wav_paths)
        else:
            wavs = []
            for audio in audios:
                audio_np = np.asarray(audio, dtype=np.float64)
                if len(audio_np.shape) == 2:
                    audio_np = audio_np[:, 0]
                if len(audio_np) > 30 * 16000:
                    audio_np = audio_np[: 30 * 16000]
                wavs.append(audio_np)
            
            time_dict = dict()

        for wav in wavs:
            print(wav.shape)
        
        spectrograms = self.feature_extractor(wavs, return_tensors="pt", 
                                              sampling_rate=16000).input_features.to(self.audio_feats_extractor_device)
        
        print("spectrograms done", flush=True)
        
        batch = {}
        batch["spec.pth"] = spectrograms

        max_len = max(len(arr) for arr in wavs)
        raw_tensor = torch.zeros((1, max_len), dtype=torch.float64, device=self.audio_feats_extractor_device)
        raw_tensor[0, :] = torch.from_numpy(wavs[0])
        batch["raw_wav.pth"] = raw_tensor

        print("'batch' done", flush=True)
        
        print(batch["spec.pth"].shape, batch["raw_wav.pth"].shape, flush=True)
        batch_feats = self.audio_feats_extractor(batch) 
        batch_feats["audio_feats_attention_mask.pth"] = torch.ones(1, batch_feats["audio_feats.pth"].shape[1], dtype = torch.bool).to(self.connector_device)
        print("audio_feats_extractor done", flush=True)

        # print(batch_feats)
        batch_connector = self.connector(batch_feats)
        print("connector done", flush=True)
        
        speech_embeds = batch_connector[0].to("cuda:0")

        # batch_tensor
        pre_prompts, post_prompts = [], []
        for prompt in prompts:
            chat = []
            if system_prompt is not None:
                if system_inject_type == "user":
                    user_pattern = system_prompt + user_pattern
                elif system_inject_type == "system":
                    chat.append({"role": "system", "content": system_prompt})

            
            chat.append({"role": "user", "content": user_pattern.format(prompt.strip())})
            
            prompt_formatted = self.llama_tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            print(f"{prompt_formatted=}")
            prompt_left, prompts_right = prompt_formatted.split('<SpeechHere>')
            
            pre_prompts.append(prompt_left)
            post_prompts.append(prompts_right)
        input_embeds, attention_mask, time_dict = self._batch_from_prompt_embeds(
            embeds=speech_embeds,
            pre_prompts=pre_prompts,
            post_prompts=post_prompts,
            time_dict=time_dict, 
            audio_mask_percent=None
        )
        input_embeds = input_embeds.to(torch.bfloat16)

        # llm generation
        print(f"input_embeds: {input_embeds.shape}, attention_mask: {attention_mask.shape}")
        print(f"input_embeds: {input_embeds.dtype}, attention_mask: {attention_mask.dtype}")
        # print(attention_mask)
        last_time = time.time()
        output = self.llm_model.generate(
            inputs_embeds=input_embeds,
            # max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            attention_mask=attention_mask,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.terminators,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens
        )
        time_dict['llama_gen'] = time.time() - last_time
        output_text = self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)

        print(output_text)
        print(self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=False))
        print("finish!")

        return output_text, time_dict, None
