import torch
import numpy as np
import soundfile as sf
import json
import webdataset as wds
import logging
import random
import hashlib
from glob import glob
from tqdm import tqdm
from pathlib import Path
from itertools import islice

from torch.utils.data import DataLoader, IterableDataset
from transformers import WhisperFeatureExtractor
from webdataset.filters import pipelinefilter
from webdataset.utils import pytorch_worker_info


def make_tar_list(tar_patterns):
    """for example tar_patterns is ['exp/train_1/egs/dump*.tar', 'exp/train_2/egs/dump*.tar']"""
    tars = []
    for t in tar_patterns:
        tars.extend(glob(t))
        return tars


def get_uniq_key(wav_path, task, Q=None):
    return hashlib.md5(f"{wav_path} {task} {Q}".encode("utf-8")).hexdigest()


class SimpleAnnoJsonLoader(IterableDataset):
    def __init__(
        self,
        ann_path,
        whisper_path,
        seed=None,
        whisper_padding="max_length",
        max_wav_len=16000 * 30,
    ):
        super().__init__()
        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        # False == disable paddings
        # max_length == whisper default behavior
        self.whisper_padding = whisper_padding
        self.seed = seed
        self.max_wav_len = max_wav_len

    def __len__(self):
        return len(self.annotation)

    def __iter__(self):
        anns = self.annotation
        rank, world_size, worker, num_workers = pytorch_worker_info()
        assert world_size == 1, "Do not use this class for DDP"
        if num_workers > 1:
            full_len = len(anns)
            anns = list(islice(anns, worker, None, num_workers))
            logging.info(
                f"Subset for {worker} worker contains {len(anns)}/{full_len} annotations"
            )
            logging.debug(f"First anno is {anns[0]}")
        if len(anns) == 0:
            logging.warning(
                f"Zero len annotations list! {worker=}, {num_workers=}, {len(anns)=}, {len(self.annotation)}"
            )
            return

        if self.seed is not None:
            random.Random(self.seed).shuffle(anns)
        for ann in anns:
            if "__key__" not in ann:
                ann["__key__"] = get_uniq_key(
                    ann["path"], ann.get("task", None), ann.get("Q", None)
                )
            assert (
                "." not in ann["__key__"]
            ), f"the key must be without dots. {ann['__key__']}"
            audio, sr = sf.read(ann["path"])
            assert sr == 16000, f"Bad {sr=} for audio {ann['path']=}"
            if len(audio.shape) == 2:  # stereo to mono
                logging.warning("Found stereo audio. Converting it into mono")
                audio = audio[:, 0]
            assert (
                audio.shape[0] > sr / 100
            ), f"{ann=} has too short audio {audio.shape=}"
            if len(audio) < sr:  # pad audio to at least 1s
                sil = np.zeros(sr - len(audio), dtype=float)
                audio = np.concatenate((audio, sil), axis=0)
            if self.max_wav_len is not None and audio.shape[0] > self.max_wav_len:
                logging.warning(
                    f"Found audio length greater than 30s {audio.shape=}, {ann=}"
                )
                audio = audio[: self.max_wav_len]  # truncate audio to at most 30s
            feats = self.wav_processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=self.whisper_padding,
                return_attention_mask=False,
                truncation=False,
            )
            spectrogram = feats["input_features"].squeeze(0).T  # T x feats
            # attention_mask = feats["attention_mask"].squeeze(0).to(bool) # it's always 1
            # if not self.whisper_padding:
            #    attention_mask = torch.ones((spectrogram.shape[0],), dtype=bool)
            # else:
            attention_mask = torch.ones((audio.shape[0] // 160,), dtype=bool)

            text = ann["text"]
            task = ann.get("task", None)
            if task is None:
                logging.warning(
                    f"Annotation {ann} doesn't have task name! ASR task as default"
                )
                task = "asr"
            Q = ann.get("Q", "")
            yield {
                "__key__": ann["__key__"],
                "spec.pth": spectrogram,
                "spec_attention_mask.pth": attention_mask,
                "raw_wav.pth": torch.from_numpy(audio),
                "text.txt": text,
                "task.txt": task,
                "q.txt": Q,
                "wav_path.txt": ann["path"],
            }


def _write_as_sharded_wds(
    dataset, out_printf_frmt, max_elements_per_shard=50, keys_subset=None
):
    """
    out_printf_frmt is like "dir/shard-000-%06d.tar"
    """
    Path(out_printf_frmt).parent.mkdir(parents=True, exist_ok=True)
    if keys_subset is not None:
        keys_subset = set(keys_subset)
        keys_subset.add("__key__")

    with wds.ShardWriter(out_printf_frmt, maxcount=max_elements_per_shard) as sink:
        for e in dataset:
            assert "__key__" in e, f"Bad dataset format {e.keys()}"
            if keys_subset is not None:
                e = {k: v for k, v in e.items() if k in keys_subset}
            sink.write(e)
            yield e


write_as_sharded_wds = pipelinefilter(_write_as_sharded_wds)


def _to_salmonn(data):
    for s in data:
        yield {
            "id": s.get("wav_path.txt", "__key__"),
            "__key__": s["__key__"],
            "spectrogram": s.get("spec.pth", None),
            "raw_wav": s.get("raw_wav.pth", None),
            "text": s.get("text.txt", None),
            "task": s["task.txt"],
            "Q": s["q.txt"],
            "speech_emb": s.get("speech_emb.pth", None),
            "audio_emb": s.get("audio_emb.pth", None),
        }


to_salmonn = pipelinefilter(_to_salmonn)


def _from_salmonn(data):
    for s in data:
        yield {
            "__key__": s.get("id", s.get("__key__", None)),
            "spec.pth": s.get("spectrogram", None),
            "raw_wav.pth": s.get("raw_wav", None),
            "text.txt": s.get("text", None),
            "task.txt": s.get("task", None),
            "q.txt": s.get("Q", None),
        }


from_salmonn = pipelinefilter(_from_salmonn)


def _tee_as_json_anno(data, anno_fname):
    anno = []
    for s in data:
        a = {
            "__key__": s["__key__"],
            "path": s["wav_path.txt"],
            "task": s["task.txt"],
            "text": s["text.txt"],
            "prompt": s["prompt.txt"],
        }
        if "q.txt" in s:
            a["Q"] = s["q.txt"]
        if "predicted.txt" in s:
            a["predicted"] = s["predicted.txt"]
        anno.append(a)
        yield s
    with open(anno_fname, "w") as f:
        json.dump({"annotation": anno}, f, ensure_ascii=False, indent=2)


tee_as_json_anno = pipelinefilter(_tee_as_json_anno)


def _tee_as_kaldi_dir(data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = {}
    texts = {}
    hyps = {}
    tasks = {}
    prompts = {}
    for s in data:
        wavs[s["__key__"]] = s["wav_path.txt"]
        texts[s["__key__"]] = " ".join(s["text.txt"].split())
        tasks[s["__key__"]] = " ".join(s["task.txt"].split())
        prompts[s["__key__"]] = " ".join(s["prompt.txt"].split())
        if "predicted.txt" in s:
            hyps[s["__key__"]] = " ".join(s["predicted.txt"].split())
        yield s
    with open(out_dir / "wav.scp", "w") as f:
        f.write("".join(f"{k} {v}\n" for k, v in sorted(wavs.items())))
    with open(out_dir / "text", "w") as f:
        f.write("".join(f"{k} {v}\n" for k, v in sorted(texts.items())))
    with open(out_dir / "task", "w") as f:
        f.write("".join(f"{k} {v}\n" for k, v in sorted(tasks.items())))
    with open(out_dir / "hyp.txt", "w") as f:
        f.write("".join(f"{k} {v}\n" for k, v in sorted(hyps.items())))
    with open(out_dir / "prompt.txt", "w") as f:
        f.write("".join(f"{k} {v}\n" for k, v in sorted(prompts.items())))


tee_as_kaldi_dir = pipelinefilter(_tee_as_kaldi_dir)
