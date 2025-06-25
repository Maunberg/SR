import torch
import webdataset as wds
import logging
from tqdm import tqdm
from pathlib import Path
from webdataset.filters import pipelinefilter

# from traceback_with_variables import activate_by_import


@torch.inference_mode()
def _generate_predictions(
    data, model, device="cpu", move_results_to_cpu=True, progress_bar=False
):
    model = model.to(device).eval()
    if progress_bar:
        data = tqdm(data)
    for i, batch in enumerate(data):
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        predicted_batch = model.predict_step(batch_device)
        if move_results_to_cpu:
            predicted_batch = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in predicted_batch.items()
            }
        yield {**batch, **predicted_batch}


generate_predictions = pipelinefilter(_generate_predictions)
