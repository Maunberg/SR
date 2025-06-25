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


def collate_with_pad_v1(samples, pad_token_id=0):
    batch = _collate_metainfo(samples)
    keys = samples[0].keys()
    if "raw_wav.pth" in keys:
        batch.update(_collate_wav_only(samples))
    if "spec.pth" in keys:
        batch.update(_collate_spec_only(samples))
    if "audio_feats.pth" in keys:
        batch.update(_collate_audiofeats_only(samples))
    if "lprompt_tokens_ids.pth" in keys and "rprompt_tokens_ids.pth" in keys:
        batch.update(
            _collate_prompt_tokens_ids_only(samples, pad_token_id=pad_token_id)
        )
    if "text_tokens_ids.pth" in keys:
        batch.update(_collate_text_tokens_ids_only(samples, pad_token_id=pad_token_id))
    return batch


def get_unpaded_values(v, attention_mask):
    if attention_mask is None:
        return v
    return v[attention_mask.bool()]


def _unbatching_padded(data):
    for batch in data:
        batch = {
            k: v.split("\n===\n") if isinstance(v, str) and k != "__key__" else v
            for k, v in batch.items()
        }
        for i in range(len(batch["__key__"])):
            padded = {k: v[i] for k, v in batch.items()}
            # for each pth tensor attention mask must be exists
            element = {}
            for k, v in padded.items():
                if k == "spec.pth" or k == "spec_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, padded.get("spec_attention_mask.pth", None)
                    )
                elif k == "raw_wav.pth" or k == "raw_wav_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, padded.get("raw_wav_attention_mask.pth", None)
                    )
                elif k == "audio_feats.pth" or k == "audio_feats_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, padded.get("audio_feats_attention_mask.pth", None)
                    )
                elif k == "lprompt_tokens_ids.pth" or k == "lprompt_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, padded.get("lprompt_attention_mask.pth", None)
                    )
                elif k == "rprompt_tokens_ids.pth" or k == "rprompt_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, padded.get("rprompt_attention_mask.pth", None)
                    )
                elif k == "text_tokens_ids.pth" or k == "text_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, padded.get("text_attention_mask.pth", None)
                    )
                elif k == "predicted.pth" or k == "predicted_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, padded.get("predicted_attention_mask.pth", None)
                    )
                else:
                    assert not isinstance(v, torch.Tensor), f"Cannot unpad {k}, {v}"
                    element[k] = v
            yield element


unbatching_padded = pipelinefilter(_unbatching_padded)


def _batching_constant_batch_size(
    data,
    batch_size=10,
    partial=True,
    collate_fn=collate_with_pad_v1,
    collate_fn_kwargs={},
):
    """Create batches of the given size.
    :param data: iterator
    :param partial: return partial batches
    :returns: iterator
    """
    assert batch_size > 0, f"Wrong batch size {batch_size}"
    batch = []
    for sample in data:
        batch.append(sample)
        if len(batch) == batch_size:
            batch = collate_fn(batch, **collate_fn_kwargs)
            yield batch
            batch = []
    if len(batch) == 0:
        return
    elif partial:
        batch = collate_fn(batch, **collate_fn_kwargs)
        yield batch


batching_constant_batch_size = pipelinefilter(_batching_constant_batch_size)


def _bucketing_batching(
    data,
    num_buckets=20,
    batch_min_tokens=3000,
    audio_feats_reduction=1 / 4,
    drop_last=False,
    collate_fn=collate_with_pad_v1,
    collate_fn_kwargs={"pad_token_id": 0},
):
    buckets = [[None, []] for _ in range(num_buckets)]
    num_e = 0
    num_batches = 0
    avg_num_buckets = 0
    avg_mindist = 0
    for e in data:

        num_e += 1
        e_lens = _get_e_lens(e, audio_feats_reduction)
        mindist = float("inf")
        mindist_i = 0
        for i, (b_lens, b) in enumerate(buckets):
            if len(b):
                avg_num_buckets += 1
            dist = _num_added_tokens(b_lens, len(b), e_lens)
            if dist < mindist:
                mindist = dist
                mindist_i = i
        avg_mindist += mindist
        i = mindist_i
        buckets[i][1].append(e)
        if buckets[i][0] is None:
            buckets[i][0] = e_lens
        else:
            buckets[i][0] = [max(l1, l2) for l1, l2 in zip(e_lens, buckets[i][0])]
        if sum(buckets[i][0]) * len(buckets[i][1]) >= batch_min_tokens:
            batch = collate_fn(buckets[i][1], **collate_fn_kwargs)
            yield batch
            buckets[i] = [None, []]
            num_batches += 1
    batch = []
    batch_lens = None
    for e in (
        s
        for _, b in sorted((b for b in buckets if b[0] is not None), reverse=True)
        for s in b
    ):
        batch.append(e)
        e_lens = _get_e_lens(e, audio_feats_reduction)
        if batch_lens is None:
            batch_lens = e_lens
        else:
            batch_lens = [max(l1, l2) for l1, l2 in zip(e_lens, batch_lens)]
        if sum(batch_lens) * len(batch) >= batch_min_tokens:
            yield collate_fn(batch, **collate_fn_kwargs)
            batch = []
            batch_lens = None
            num_batches += 1
    if not drop_last and len(batch):
        yield collate_fn(batch, **collate_fn_kwargs)
        num_batches += 1
    logging.debug(
        f"Bucketing stats: number of elements processed: {num_e}, bathes collated {num_batches}, "
        f"Average buckets using {avg_num_buckets/num_e}, Average min dist {avg_mindist/num_e}"
    )


bucketing_batching = pipelinefilter(_bucketing_batching)


def _num_added_tokens(batch, bz, add_len):
    if batch is None:
        return sum(le**2 * 0.1 for le in add_len)
        # return sum(le for le in add_len)
    return sum(
        (b - l) ** 2 if b >= l else (b - l) ** 2 * bz for b, l in zip(batch, add_len)
    )


def _get_e_lens(e, audio_feats_reduction):
    assert all(
        k in e
        for k in (
            "__key__",
            "text_tokens_ids.pth",
            "audio_feats.pth",
            "lprompt_tokens_ids.pth",
            "rprompt_tokens_ids.pth",
        )
    ), f"{e.keys()=}"
    return [
        len(e["lprompt_tokens_ids.pth"]),
        len(e["audio_feats.pth"]) * audio_feats_reduction,
        len(e["rprompt_tokens_ids.pth"]),
        len(e["text_tokens_ids.pth"]),
    ]


def _collate_spec_only(samples):
    """
    input: spec.pth, [spec_attention_mask.pth]
    output: spec.pth, spec_attention_mask.pth
    """
    assert len(samples[0]["spec.pth"].shape) == 2, f"bad shape {samples[0]['spec.pth']}"
    spec_batch = pad_sequence(
        [s["spec.pth"] for s in samples], padding_value=0.0, batch_first=True
    )
    if "spec_attention_mask.pth" in samples[0]:
        spec_attention_mask = pad_sequence(
            [s["spec_attention_mask.pth"] for s in samples],
            padding_value=False,
            batch_first=True,
        )
    else:
        spec_length = torch.as_tensor([s["spec.pth"].shape[0] for s in samples])
        spec_attention_mask = torch.arange(len(samples)).unsqueeze(
            0
        ) < spec_length.unsqueeze(1)
    return {
        "spec.pth": spec_batch,
        "spec_attention_mask.pth": spec_attention_mask.bool(),
    }


def _collate_wav_only(samples):
    """
    input: raw_wav.pth, [raw_wav_attention_mask.pth]
    output: raw_wav.pth, raw_wav_attention_mask.pth
    """
    assert (
        len(samples[0]["raw_wav.pth"].shape) == 1
    ), f"bad raw_wav.pth {samples[0]['raw_wav.pth']}"
    raw_wav = pad_sequence(
        [s["raw_wav.pth"] for s in samples], batch_first=True, padding_value=0.0
    )
    if "raw_wav_attention_mask.pth" in samples[0]:
        raw_wav_attention_mask = pad_sequence(
            [s["raw_wav_attention_mask.pth"] for s in samples],
            padding_value=False,
            batch_first=True,
        )
    else:
        raw_wav_length = torch.as_tensor([len(s["raw_wav.pth"]) for s in samples])
        raw_wav_attention_mask = torch.arange(raw_wav.size(1)).unsqueeze(
            0
        ) < raw_wav_length.unsqueeze(1)
    return {
        "raw_wav.pth": raw_wav,
        "raw_wav_attention_mask.pth": raw_wav_attention_mask.bool(),
    }


def _collate_audiofeats_only(samples):
    """
    input: audio_feats.pth, [audio_feats_attention_mask.pth]
    output: audio_feats.pth, audio_feats_attention_mask.pth
    """
    assert (
        len(samples[0]["audio_feats.pth"].shape) == 2
    ), f"bad audio_feats.pth {samples[0]['audio_feats.pth']}"
    feats = pad_sequence(
        [s["audio_feats.pth"] for s in samples], batch_first=True, padding_value=0.0
    )
    if "audio_feats_attention_mask.pth" in samples[0]:
        attention_mask = pad_sequence(
            [s["audio_feats_attention_mask.pth"] for s in samples],
            padding_value=False,
            batch_first=True,
        )
    else:
        length = torch.as_tensor([s["audio_feats.pth"].shape[0] for s in samples])
        attention_mask = torch.arange(feats.size(1)).unsqueeze(0) < length.unsqueeze(1)
    return {
        "audio_feats.pth": feats,
        "audio_feats_attention_mask.pth": attention_mask.bool(),
    }


def _collate_prompt_tokens_ids_only(samples, pad_token_id):
    pl = [s["lprompt_tokens_ids.pth"].flip(0) for s in samples]
    pr = [s["rprompt_tokens_ids.pth"].flip(0) for s in samples]
    assert (
        len(pl[0].shape) == 1 and len(pr[0].shape) == 1
    ), f"bad prompt_tokens_ids.pth {pl[0].shape=} {pr[0].shape=}"
    assert (
        "lprompt_attention_mask.pth" not in samples[0]
    ), f"lprompt_attention_mask.pth already in {samples[0].keys()}"
    assert (
        "rprompt_attention_mask.pth" not in samples[0]
    ), f"rprompt_attention_mask.pth already in {samples[0].keys()}"

    prompts = []
    at = []
    for p in [pl, pr]:
        # double flip for left paddings
        p = pad_sequence(p, batch_first=True, padding_value=pad_token_id).flip(1)
        at.append(p != pad_token_id)
        prompts.append(p)
    return {
        "lprompt_tokens_ids.pth": prompts[0],
        "lprompt_attention_mask.pth": at[0].bool(),
        "rprompt_tokens_ids.pth": prompts[1],
        "rprompt_attention_mask.pth": at[1].bool(),
    }


def _collate_text_tokens_ids_only(samples, pad_token_id=0):
    text = [s["text_tokens_ids.pth"] for s in samples]
    assert len(text[0].shape) == 1, f"bad text_tokens_ids.pth {text[0].shape=}"
    assert (
        "text_attention_mask.pth" not in samples[0]
    ), f"text_attention_mask.pth already in {samples[0].keys()}"
    padded = pad_sequence(text, batch_first=True, padding_value=pad_token_id)
    length = torch.as_tensor([t.shape[0] for t in text])
    attention_mask = torch.arange(padded.size(1)).unsqueeze(0) < length.unsqueeze(1)
    return {
        "text_tokens_ids.pth": padded,
        "text_attention_mask.pth": attention_mask.bool(),
    }


def _collate_metainfo(samples):
    batch = {"__key__": [s["__key__"] for s in samples]}
    if "wav_path.txt" in samples[0]:
        batch["wav_path.txt"] = "\n===\n".join([s["wav_path.txt"] for s in samples])
    if "prompt.txt" in samples[0]:
        batch["prompt.txt"] = "\n===\n".join([s["prompt.txt"] for s in samples])
    if "task.txt" in samples[0]:
        batch["task.txt"] = "\n===\n".join([s["task.txt"] for s in samples])
    if "text.txt" in samples[0]:
        batch["text.txt"] = "\n===\n".join([s["text.txt"] for s in samples])
    if "q.txt" in samples[0]:
        batch["q.txt"] = "\n===\n".join([s["q.txt"] for s in samples])
    if "epoch.id" in samples[0]:
        batch["epoch.id"] = sum(s["epoch.id"] for s in samples) / len(samples)
    return batch
