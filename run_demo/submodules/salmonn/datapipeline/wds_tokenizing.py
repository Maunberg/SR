import torch
import webdataset as wds
import logging
import random
import os
import time
import json

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from webdataset.filters import pipelinefilter
from pathlib import Path
from typing import Union, List

from datapipeline.wds_io import to_salmonn


def _insert_prompt(
    data,
    task2promt_json,
    choice_strategy="first",
    rng=None,
    seed=42,
    template="{prompt}",
):
    """
    input ["task.txt", Optional["q.txt"]]
    output [..., "prompt.txt"]

    prompt example:
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|> <Speech><SpeechHere></Speech> Can you transcribe the speech into a written format? <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    """
    assert choice_strategy in [
        "first",
        "random",
    ], f"Unknown choice strategy {choice_strategy}"
    with open(task2promt_json) as f:
        # task name to list of prompt format strings
        task2templates = {
            k: v if isinstance(v, list) else [v] for k, v in json.load(f).items()
        }

    if choice_strategy == "random":
        assert rng is None or seed is None, "seed or rng must be None"
        if rng is None:
            if seed is None:
                seed = int((os.getpid() + time.time()) * 1e9)
                logging.debug(f"insert prompt random seed is {seed}")
            rng = random.Random(seed)

    for sample in data:
        task = sample["task.txt"]
        assert task in task2templates, f"{task} not in {task2promt_json}"
        q = sample.get("q.txt", "???")
        prompt = None
        if choice_strategy == "first":
            prompt = task2templates[task][0].format(q)
        elif choice_strategy == "random":
            prompt = rng.choice(task2templates[task]).format(q)
        if template:
            prompt = template.format(prompt=prompt)
        yield {**sample, "prompt.txt": prompt}


insert_prompt = pipelinefilter(_insert_prompt)


def _tokenize_samples(
    data, tokenizer, insert_bos=True, insert_eos=True, bos_sym=None, end_sym=None
):
    """
    converting [bos, prompt, text, eos] into sequence of token indices

    input ["prompt.txt", Optional["text.txt"]
    output [..., "prompt_tokens_ids.pth", "text_tokens_ids.pth"]
    """
    if bos_sym is None:
        bos_sym = tokenizer.bos_token
    if end_sym is None:
        end_sym = tokenizer.eos_token
    for sample in data:
        sample = sample.copy()
        assert (
            "prompt.txt" in sample
        ), f"Construct prompt before running tokenizer. {sample.keys()=}"
        prompt = sample["prompt.txt"]
        if insert_bos:
            prompt = f"{bos_sym} {prompt}"
        splitted_by_speech = prompt.split("<SpeechHere>")
        assert len(splitted_by_speech) == 2, f"bad prompt {prompt}"
        sample["lprompt_tokens_ids.pth"], sample["rprompt_tokens_ids.pth"] = [
            tokenizer(
                p,
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids.squeeze(0)
            for p in splitted_by_speech
        ]
        # attention mask is always 1
        text = sample.get("text.txt", "")
        if insert_eos:
            text = f"{text} {end_sym}"
        sample["text_tokens_ids.pth"] = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
            padding=False,
        ).input_ids.squeeze(0)
        yield sample


tokenize_samples = pipelinefilter(_tokenize_samples)


def _tokenize_batches(data, tokenizer, bos_sym=None, end_sym=None):
    """
    converting [bos, prompt, text, eos] into sequence of token indices

    input ["prompt.pkl", Optional["text.pkl"] # list of texts
    output [..., "prompt_tokens_ids.pth", "text_tokens_ids.pth"]
    """
    # TODO
    raise NotImplementedError()
