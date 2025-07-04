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

import argparse

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

#while True:
#    try:
#        print("=====================================")
#        wav_path = input("Your Wav Path:\n")
#        prompt = input("Your Prompt:\n")

#        samples = prepare_one_sample(wav_path, wav_processor)
#        prompt = [
#            cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
#        ]
#        print("Output:")
#        # for environment with cuda>=117
#        with torch.cuda.amp.autocast(dtype=torch.float16):
#            print(model.generate(samples, cfg.config.generate, prompts=prompt)[0])
#        # print(model.generate(samples, cfg.config.generate, prompts=prompt)[0])
#    except Exception as e:
#        print(e)
#        import pdb; pdb.set_trace()


annotations = {"annotation": []}
with open("/mnt/asr_hot/karelin/ALLM/train_ds/annotations_5_val.json", "r") as fin:
    data = json.load(fin)
    for q in tqdm(data["annotation"]):
        #print(q["path"])
        #print(q["Q"])
        #response = model.generate(q["path"], prompt=q["Q"])[0]
        try:
            samples = prepare_one_sample(q["path"], wav_processor)
            prompt = [
                cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + q["Q"].strip())
            ]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                response = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
            annotations["annotation"].append({
                "path": q["path"],
                "text": response,
                "task": "QA",
                "Q": q["Q"]
            })
        except Exception as e:
            print(e)
        #print(response)

json_data = json.dumps(annotations, ensure_ascii=False, indent=2)
with open('EVAL_annotations_6_val_base_salmon.json', 'w', encoding='utf-8') as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)
