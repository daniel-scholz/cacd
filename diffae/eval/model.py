import time
from pathlib import Path
from typing import Optional

import torch
import torch.utils.data
from lightning.pytorch.utilities.model_summary import ModelSummary

from diffae.config import TrainConfig
from diffae.experiment import LitModel


def save_ckpt(training_ckpt_fp: Path, edited_ds_ckpt_fp: Path):
    if not edited_ds_ckpt_fp.exists():
        edited_ds_ckpt_fp.hardlink_to(training_ckpt_fp)
        print(
            f"Created hardlink: {edited_ds_ckpt_fp} to preserve original checkpoint while training"
        )
    else:
        print(f"Hardlink already exists: {edited_ds_ckpt_fp}")


def load_harm_model(
    model_cls: type[LitModel],
    conf: TrainConfig,
    device: torch.device,
    ckpt_dir: Path = Path("checkpoints"),
    metric: Optional[str] = None,
):
    ckpt_dir = ckpt_dir.resolve() / conf.name / "wandb" / f"checkpoints-{conf.wandb_id}"
    assert ckpt_dir.exists(), f"Checkpoint directory {ckpt_dir} does not exist"
    # try 10 times to load the model (in case of stale file handle or something like that)
    try_cnt = 0
    while try_cnt < 10:
        try:
            ckpt_files = list(ckpt_dir.glob(f"epoch=*-step=*-{metric}=*.ckpt"))
            # sort by step (last saved checkpoint of metric which is also the best by metric)
            ckpt_files.sort(key=lambda x: int(x.stem.split("-")[1].split("=")[-1]))
            ckpt_fp = ckpt_files[-1]

            print(f"loaded checkpoint {ckpt_fp}")
            ckpt_timestamp_ns = ckpt_fp.stat().st_mtime_ns
            # print in readable date format
            print(
                f"checkpoint timestamp {ckpt_timestamp_ns}",
                f"{time.ctime(int(ckpt_timestamp_ns*1e-9))}",
            )
            model = model_cls.load_from_checkpoint(
                ckpt_fp, conf=conf, map_location=device, strict=False
            )
            break
        except (OSError, FileNotFoundError) as e:
            # handle cases where checkpoints has just been deleted when evaluating during training
            print(f"checkpoint  {ckpt_fp} is not available yet, waiting for 10 seconds")
            print(e)
            # wait for 10 seconds
            time.sleep(10)
            try_cnt += 1
    else:  # else is executed if the loop ended normally (no break)
        raise RuntimeError(f"Failed to load model from {ckpt_fp}")

    model.setup()
    model.eval()
    summary = ModelSummary(model, max_depth=1)
    print(summary)
    # concatenate timestemp with ckpt_name
    ckpt_name = f"{ckpt_fp.stem}_{ckpt_timestamp_ns}"
    return model, ckpt_fp, ckpt_name


# %%
def age_fit_sites_from_fp(ckpt_fp: Path) -> tuple[list[str], list[str]]:
    sites = ckpt_fp.parts[-3]
    fit_sites, test_sites = sites.split("_")
    fit_sites = fit_sites.split("-")
    test_sites = test_sites.split("-")
    return fit_sites, test_sites
