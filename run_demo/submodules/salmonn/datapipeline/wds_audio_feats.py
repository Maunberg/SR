import torch
import webdataset as wds
import logging
from tqdm import tqdm
from pathlib import Path
from webdataset.filters import pipelinefilter

# from traceback_with_variables import activate_by_import


@torch.inference_mode()
def _feature_extractor(
    data, model, device="cpu", move_results_to_cpu=True, batch_mode=True
):
    """
    out_printf_frmt is like "dir/shard-000-%06d.tar"
    """
    model = model.to(device).eval()
    for batch in tqdm(data):
        orig_batch = batch
        assert (
            batch_mode
            and len(batch["raw_wav.pth"].shape) == 2
            or not batch_mode
            and len(batch["raw_wav.pth"].shape) == 1
        ), f"Batch shape {batch['raw_wav.pth'].shape=} is not correct shape for {batch_mode=}"
        logging.debug(f"Start processing {batch['__key__']}")
        if not batch_mode:
            batch = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        # shape is B, T, C
        audio_feats_dict = model(batch)
        logging.debug(f"Extracted {audio_feats_dict['audio_feats.pth'].shape=}")
        if move_results_to_cpu:
            audio_feats_dict = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in audio_feats_dict.items()
            }
        if not batch_mode:
            audio_feats_dict = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in audio_feats_dict.items()
            }
        yield {**orig_batch, **audio_feats_dict}


feature_extractor = pipelinefilter(_feature_extractor)
