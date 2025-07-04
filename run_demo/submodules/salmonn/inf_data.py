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

import os, glob
from tqdm import tqdm
import argparse
from model import SALMONN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0") #, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--whisper_path", type=str, default=None)
    parser.add_argument("--beats_path", type=str, default=None)
    parser.add_argument("--vicuna_path", type=str, default=None)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--low_resource", action='store_true', default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    model = SALMONN(
        ckpt=args.ckpt_path,
        whisper_path=args.whisper_path,
        beats_path=args.beats_path,
        vicuna_path=args.vicuna_path,
        lora_alpha=args.lora_alpha,
        low_resource=args.low_resource
    )
    model.to(args.device)
    model.eval()
#    while True:
#        print("=====================================")
#        wav_path = input("Your Wav Path:\n")
#        prompt = input("Your Prompt:\n")
#        try:
#            print("Output:")
#            print(model.generate(wav_path, prompt=prompt)[0])
#        except Exception as e:
#            print(e)
#            if args.debug:
#                import pdb; pdb.set_trace()
    for dir in glob.iglob('data/selected_salmon/**', recursive=False):
        print(dir + '/**')
        nf = len(os.listdir(dir))
        print(nf)
        for file in tqdm(glob.iglob(dir + '/**', recursive=False), total = nf):
            if os.path.isfile(file) and not os.path.isfile(dir + "_salmon/" + file.split("/")[-1]):
                print(file)
#                query = tokenizer.from_list_format([
#                    {'audio': file},
#                    {'text': "Recognize speech."},
#                ])
                try:
#                    response = model.generate(file, prompt="Recognize speech.")[0]
                    response = model.generate(file, prompt="Recognize russian speech.")[0]
                except Exception as e:
                    print(e)
                #print(response)
                os.makedirs(dir + "_salmon/", exist_ok=True)
                with open(dir + "_salmon/" + file.split("/")[-1], "w") as f:
                    f.write(response)

