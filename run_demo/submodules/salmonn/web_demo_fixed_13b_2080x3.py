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
import shutil
import os
import gradio as gr
import argparse
from pathlib import Path

from model_splitted_13b_2080x3 import SALMONN_mutigpu as SALMONN

os.makedirs("salmon_demo_out/wav", exist_ok=True)

class ff:
    def generate(self, wav_path, prompt, prompt_pattern, num_beams, temperature, top_p):
        print(f'wav_path: {wav_path}, prompt: {prompt}, temperature: {temperature}, num_beams: {num_beams}, top_p: {top_p}')
        return "I'm sorry, but I cannot answer that question as it is not clear what you are asking. Can you please provide more context or clarify your question?"

ckpt_path = Path("/mnt/asr_hot/karelin/ALLM/OUT/202404160514/checkpoint_4.pth")
# ckpt_path = Path("/mnt/asr_hot/karelin/ALLM/salmon/SALMONN-7B/salmonn_7b_v0.pth")
whisper_path = Path("/mnt/asr_hot/karelin/ALLM/salmon_old/whisper-large-v2/")
#whisper_path = Path("/mnt/asr_hot/karelin/ALLM/salmon/whisper-large-v2/")
beats_path = Path("/mnt/asr_hot/karelin/ALLM/salmon_old/beats.pt")
#beats_path = Path("/mnt/asr_hot/karelin/ALLM/salmon/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2 (2).pt")
vicuna_path = Path("/mnt/asr_hot/karelin/ALLM/salmon/SALMONN/vicuna-13b-v1.1")
#vicuna_path = Path("/mnt/asr_hot/karelin/ALLM/salmon/vicuna-7b-v1.5/")

lora_alpha = 32
low_resource = True
'''
class ff:
    def generate(self, wav_path, prompt, prompt_pattern='def', num_beams=0, temperature=0, top_p=0, device='cpu'):
        print(f'wav_path: {wav_path}, prompt: {prompt}, temperature: {temperature}, num_beams: {num_beams}, top_p: {top_p}, device: {device}', flush=True)
        return "Ii'm sorry, but I cannot answer that question as it is not clear what you are asking. Can you please provide more context or clarify your question?"

model = ff()
'''

model = SALMONN(
    ckpt=ckpt_path,
    whisper_path=whisper_path,
    beats_path=beats_path,
    vicuna_path=vicuna_path,
    lora_alpha=lora_alpha,
    low_resource=low_resource
)
model.eval()


print("loading done", flush=True)


# gradio
def gradio_reset(chat_state):

    chat_state = []
    return (None,
            gr.update(value=None, interactive=True),
            gr.update(interactive=False),
            gr.update(value="Upload & Start Chat", interactive=True),
            chat_state)

def upload_speech(gr_speech, text_input, chat_state):
    if gr_speech is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state.append(gr_speech)
    return (gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(value="Start Chatting", interactive=False),
            chat_state)

def gradio_ask(user_message, chatbot, chat_state):

    if len(user_message) == 0:
        return gr.update(interactive=True), chatbot, chat_state
    chat_state.append(user_message)
    chatbot.append([user_message, None])
    #
    return gr.update(interactive=False), chatbot, chat_state

def gradio_answer(chatbot, chat_state, num_beams, temperature, top_p):
    llm_message = model.generate(
        wav_path=chat_state[0],
        prompt=chat_state[1],
        num_beams=num_beams,
        temperature=temperature,
        top_p=top_p,
        device="cuda:0"
    )
    chatbot[-1][1] = llm_message[0]
    #os.makedirs("salmon_demo_out/wav", exist_ok=True)
    shutil.copyfile(chat_state[0], "salmon_demo_out/wav/" + chat_state[0].split("/")[-1])
    with open("salmon_demo_out/data.txt", "a") as f:
        f.write("wav/" + chat_state[0].split("/")[-1] + "|" + str(chat_state[1]) + "|" + str(llm_message[0]) + "\n")
    return chatbot, chat_state

title = """<h1 align="center">SALMONN: Speech Audio Language Music Open Neural Network</h1>"""
image_src = """<h1 align="center"><img src="https://raw.githubusercontent.com/bytedance/SALMONN/main/resource/salmon.png", alt="SALMONN" border="0" style="margin: 0 auto; height: 200px;" /></h1>"""
description = """<h3>This is the demo of SALMONN. Upload your audio and start chatting! Only first 30s and 1st channel will be processed. </h3>"""


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(image_src)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            speech = gr.Audio(label="Audio", type='filepath')
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True,
                label="top p",
            )

            temperature = gr.Slider(
                minimum=0.8,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=False,
                label="temperature",
            )

        with gr.Column():
            chat_state = gr.State([])

            chatbot = gr.Chatbot(label='SALMONN')
            text_input = gr.Textbox(label='User', placeholder='Please upload your speech first', interactive=False)

    with gr.Row():
        examples = gr.Examples(
            examples = [
                ["resource/audio_demo_new/gimn-rossii.wav", "Which country's national anthem is it?"],
                ["resource/audio_demo_new/kitten.wav", "Is that a cat or a dog?"],
                ["resource/audio_demo_new/kitten.wav", "What is that sound?"], 
                ["resource/audio_demo_new/twitch.wav", "Please shortly describe the topic of the audio."], 
                ["resource/audio_demo_new/drum_n_bass.wav", "What genre of music is this song? Just the name of genre, please."], 
                ["resource/audio_demo_new/jazz_lingus.wav", "What instruments are playing here? Name all of them."], 
                ["resource/audio_demo_new/pogoda.mp3", "Transcribe russian speech"], 
                ["resource/audio_demo_new/applause.wav", "Describe this sound, please"],
                ["resource/audio_demo_new/rock_fern_planet.wav", "In which language does the singer sing?"],
                ["resource/audio_demo_new/rock_fern_planet.wav", "What is the gender of singer?"],
                ["resource/audio_demo_new/rock_fern_planet.wav", "What is this song about?"], 
                ["resource/audio_demo_new/airport.wav", "Use your strong reasoning skills to answer the speaker's question in detail based on the background sound."], 
            ],
            inputs=[speech, text_input]
        )

    upload_button.click(upload_speech, [speech, text_input, chat_state], [speech, text_input, upload_button, chat_state])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, num_beams, temperature, top_p], [chatbot, chat_state]
    )
    clear.click(gradio_reset, [chat_state], [chatbot, speech, text_input, upload_button, chat_state], queue=False)



demo.launch(share=True, server_name="0.0.0.0", server_port=9527, debug=True)
