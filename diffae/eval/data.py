from typing import Literal

import numpy as np
import torch.utils.data

from diffae.experiment import LitModel


def get_stage_loader(
    model: LitModel,
    stage: Literal["val", "test", "full"] = "test",
    batch_size: int = 1,
    num_workers: int = 16,
) -> torch.utils.data.DataLoader:
    match stage:
        case "val":
            data_set = model.val_data
        case "test":
            data_set = model.test_data
        case "full":
            data_set = torch.utils.data.ConcatDataset(
                [model.train_data, model.val_data, model.test_data]
            )
        case _:
            raise ValueError(f"stage {stage} not supported")

    loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    return loader


def get_split_list(model: LitModel, stage: Literal["val", "test", "full"]) -> np.ndarray:
    match stage:
        case "val":
            split_list = np.array(["val"] * len(model.val_data))
        case "test":
            split_list = np.array(["test"] * len(model.test_data))
        case "full":
            split_list = np.array(
                ["train"] * len(model.train_data)
                + ["val"] * len(model.val_data)
                + ["test"] * len(model.test_data)
            )
        case _:
            raise ValueError(f"stage {stage} not supported")

    return split_list


def get_stage_loader_with_split_list(
    model: LitModel,
    stage: Literal["val", "test", "full"] = "test",
    batch_size: int = 1,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, np.ndarray]:

    split_list = get_split_list(model, stage)  # Call the existing get_split_list function

    loader = get_stage_loader(
        model, stage, batch_size, num_workers
    )  # Call the existing get_stage_loader function
    return loader, split_list


def get_new_ds_dir_name(og_name: str, eval_id: str) -> str:
    return f"{og_name}_edited/{eval_id}"
