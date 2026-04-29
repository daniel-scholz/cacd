"""Central script to run all evaluations for the project."""

# raise NotImplementedError("Refactor eval script.")
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch.utils.data
from lightning import seed_everything
from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression

from diffae.config import TrainConfig, conf_from_wandb_id, customize_conf_for_eval
from diffae.data.dataframe import map_col_to_int
from diffae.eval.data import get_new_ds_dir_name, get_split_list, get_stage_loader
from diffae.eval.latent import (
    LatentEditor,
    calc_nearest_neighbor_top_accuracy,
    get_latents_full_dataset,
    get_w,
    optimize_z_sem,
    swap_latents,
    train_model_on_latents,
)
from diffae.eval.model import load_harm_model, save_ckpt
from diffae.eval.paired import paired_analysis
from diffae.experiment import LitModel
from diffae.vis_utils import plot_latents, project_latents

seed_everything(0)
torch.multiprocessing.set_sharing_strategy("file_system")


@click.command()
@click.option("--wandb_id", type=str, required=False)
@click.option(
    "--target_site",
    required=True,
    type=click.Choice(["Guys", "HH"]),
)
@click.option("--metric", type=str, required=False, default=None)
@click.option(
    "--eval_tasks",
    multiple=True,
    required=False,
    default=["edit_and_infer", "analyze_latents", "swap_latents", "paired_analysis"],
    help="List of evaluation tasks",
    type=click.Choice(["edit_and_infer", "analyze_latents", "swap_latents", "paired_analysis"]),
)
@click.option("--with_t2", is_flag=True, default=False)
def main(
    wandb_id: str,
    target_site: str,
    metric: Optional[str],
    eval_tasks: list[str],
    with_t2: bool,
):
    device = torch.device("cuda")

    conf = conf_from_wandb_id(wandb_id)
    customize_conf_for_eval(conf, with_t2=with_t2)

    diffae_model, diffae_ckpt_fp, diffae_ckpt_name = load_harm_model(
        LitModel, conf, device=device, metric=metric
    )

    results_dir = make_results_dir(
        target_site,
        diffae_ckpt_fp,
        diffae_ckpt_name,
    )
    print(f"Saving results to {results_dir}")
    eval_identifier = "-".join(results_dir.parts[results_dir.parts.index("results") :])
    print(f"eval_identifier: {eval_identifier}")

    assert len(diffae_model.conf.data_paths) == 1, "currently only one dataset supported."

    new_ds_dir = make_edited_ds_dir(diffae_model, eval_identifier)
    save_ckpt(
        edited_ds_ckpt_fp=new_ds_dir / diffae_ckpt_fp.name,
        training_ckpt_fp=diffae_ckpt_fp,
    )

    # sanity check: try to load the model from the new dataset directory
    # suppress from std err and out
    with redirect_stdout(None), redirect_stderr(None):
        LitModel.load_from_checkpoint(new_ds_dir / diffae_ckpt_fp.name, conf=conf)
    print("Successfully saved and loaded model to new dataset directory.")

    conds, patient_attrs = get_latents_full_dataset(
        diffae_model,
        diffae_ckpt_fp,
        diffae_ckpt_name,
    )
    z_sems, z_ids = diffae_model.split_sem_id(conds)

    val_loader_diffae = get_stage_loader(
        model=diffae_model,
        stage="val",
        num_workers=conf.num_workers,
        batch_size=conf.batch_size,
    )
    seq_list_full = patient_attrs["seq"]
    split_list_full = patient_attrs["split"]
    with_t2 = False
    if with_t2:
        raise NotImplementedError("T2 not supported at the moment.")
        # edit z_sems between T1 and T2
        print("Editing T2...")

        seq_list_full_int, _ = map_col_to_int(pd.Series(seq_list_full))

        # translate from t1 to t2 and vice versa
        clf, seq_report_dict = train_model_on_latents(
            conf,
            "sequence",
            z_sems,
            # repeat split list
            np.concatenate([split_list_full] * 2),
            seq_list_full_int,
            None,
        )
        print(
            "Sequence clf report:",
            json.dumps(seq_report_dict, indent=2, sort_keys=True),
            sep="\n",
        )
        T1_mask = seq_list_full == "T1"
        T2_mask = seq_list_full == "T2"
        z_sem_t1 = z_sems[T1_mask]
        z_sem_t2 = z_sems[T2_mask]

        # compare z_ids between T1 and T2
        calc_nearest_neighbor_top_accuracy(
            z_t1=z_ids[T1_mask],
            z_t2=z_ids[T2_mask],
            split_list=split_list_full,
            z_type="id",
            results_dir=results_dir,
        )
        calc_nearest_neighbor_top_accuracy(
            z_t1=z_sem_t1,
            z_t2=z_sem_t2,
            split_list=split_list_full,
            z_type="sem",
            results_dir=results_dir,
        )
        if "swap_latents" in eval_tasks:
            swapped_fig_dir = results_dir / "swapped_latents_sequence"
            swapped_fig_dir.mkdir(exist_ok=True)

            swap_latents(
                data_loader=val_loader_diffae,
                diffae_model=diffae_model,
                swapped_fig_dir=swapped_fig_dir,
            )
            target_ixi_ids = [
                # "IXI121-Guys-0772",
                # "IXI164-Guys-0844",
                # "IXI186-Guys-0796",
                # "IXI269_Guy-0839",
                # "IXI044-Guys-0712",
                # "IXI118-Guys-0764",
            ]
            optim_fig_dir = results_dir / "optimized_z_sems"
            optim_fig_dir.mkdir(exist_ok=True)

            for target_ixi_id in target_ixi_ids:
                optimize_z_sem(
                    data_loader=val_loader_diffae,
                    diffae_model=diffae_model,
                    target_ixi_id=target_ixi_id,
                    # target_ixi_id="IXI121-Guys-0772",
                    # target_ixi_id="IXI164-Guys-0844",
                    # target_ixi_id="IXI186-Guys-0796",
                    # target_ixi_id="IXI269_Guy-0839",
                    # target_ixi_id="IXI044-Guys-0712",
                    # target_ixi_id="IXI118-Guys-0764",
                    optim_fig_dir=optim_fig_dir,
                )

        dim_red_fn = "tsne"
        z_sems_proj, proj_fn = project_latents(
            z_sems,
        )
        seq_projection_fp = (
            results_dir
            / f"latent_projections/sequence/{dim_red_fn}/z_sem_sequence_{dim_red_fn}.png"
        )
        seq_projection_fp.parent.mkdir(parents=True, exist_ok=True)
        plot_latents(
            z_sems_proj,
            seq_list_full_int.numpy(),
            seq_list_full,
            cond_type="cat",
            split_list=np.concatenate([split_list_full] * 2),
            projection_fp=seq_projection_fp,
        )

        t2_seq_int = seq_list_full_int[seq_list_full == "T2"][0].int().item()

        def t2_loader(loader):
            """Wrapper around loader to replace T2 with T1."""
            for batch in loader:
                batch["img"] = batch["T2"]
                # replace T2 with T1 in fp
                batch["fp"] = [(str(fp[0]).replace("T1", "T2"),) for fp in batch["fp"]]
                batch["condition"] = torch.full_like(batch["condition"], t2_seq_int)
                yield batch

    print("Pairwise analysis...")
    paired_analysis(conf, diffae_model, results_dir)

    print("Analyzing latents...")

    analyze_latents(
        z_sems,
        z_ids,
        patient_attrs["scanner"],
        patient_attrs["age"],
        patient_attrs["sex"],
        conf,
        diffae_model,
        results_dir,
    )

    print("Färtig.")


def edit_z_sems_scanner(
    edit_weight: float,
    target_site: str,
    conf: TrainConfig,
    results_dir: Path,
    sites_int: torch.Tensor,
    site_names: npt.NDArray[np.str_],
    z_sems: torch.Tensor,
    split_list,
):
    z_sems = z_sems
    latent_editor = LatentEditor(
        edit_weight, z_sems.mean(dim=0), z_sems.std(dim=0), normalize=False
    )
    scanner_clf, _ = train_model_on_latents(  # type:ignore
        conf, "scanner", z_sems, split_list, sites_int, None
    )
    scanner_clf: LinearSVC | LogisticRegression

    target_site_int = int(sites_int[site_names == target_site][0].int().item())

    w = get_w(target_site_int, scanner_clf)
    b = torch.from_numpy(scanner_clf.intercept_.astype(np.float32))

    z_sems_edited = z_sems.clone()
    z_sems_edited[sites_int != target_site_int] = latent_editor(
        z_sems[sites_int != target_site_int], w, b
    )

    return z_sems_edited.unsqueeze(1)


def make_results_dir(
    target_site,
    diffae_ckpt_fp,
    diffae_ckpt_name: str,
) -> Path:
    results_dir = Path(str(diffae_ckpt_fp.parent).replace("checkpoints", "results"))
    results_dir /= diffae_ckpt_name

    eval_id_targetcond = f"targetsite-{target_site}"

    results_dir /= "".join([eval_id_targetcond])
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def analyze_latents(
    z_sems: torch.Tensor,
    z_ids: torch.Tensor,
    scanner: torch.Tensor,
    age: torch.Tensor,
    sex: torch.Tensor,
    conf: TrainConfig,
    diffae_model: LitModel,
    results_dir: Path,
):

    # remove patch dimension
    z_sems = z_sems
    z_ids = z_ids

    # extract list of splits from model
    split_list = get_split_list(diffae_model, "full")

    latent_results_dir = results_dir / "latent_analysis"
    latent_results_dir.mkdir(exist_ok=True)

    latent_projections_dir = results_dir / "latent_projections"
    latent_projections_dir.mkdir(exist_ok=True)

    # train model on all relevant latent properties
    all_conditions_dict = {
        "scanner": scanner,
        "sex": sex,
        "age": age,
    }

    for curr_cond in all_conditions_dict.keys():
        print(f"Analyzing latents for {curr_cond}...")

        conditions = all_conditions_dict[curr_cond]
        # map condition names to numbers
        conditions_num, _ = (
            map_col_to_int(pd.Series(conditions)) if curr_cond != "age" else (conditions, None)
        )

        descs_z = ("id_real", "sem_real")

        for z, desc_z in zip((z_ids, z_sems), descs_z):
            model_report_fp = (
                latent_results_dir / f"{curr_cond}" / f"z_{desc_z}_{curr_cond}_report.json"
            )
            model_report_fp.parent.mkdir(parents=True, exist_ok=True)

            latent_model, report_dict = train_model_on_latents(
                conf, curr_cond, z, split_list, conditions_num, model_report_fp
            )
            print(
                curr_cond,
                json.dumps(report_dict, sort_keys=True, indent=2),
                sep="\n",
            )

            for dim_red_fn in ("tsne", "pca"):
                print(f"Plotting {desc_z} for {curr_cond}...{dim_red_fn}")
                projections_fp = (
                    latent_projections_dir
                    / f"{curr_cond}"
                    / f"{dim_red_fn}"
                    / f"z_{desc_z}_{curr_cond}_{dim_red_fn}.png"
                )
                projections_fp.parent.mkdir(parents=True, exist_ok=True)

                z_proj, proj_fn = project_latents(
                    z.cpu().numpy(),
                    dim_red_fn=dim_red_fn,
                )

                plot_latents(
                    z_proj,
                    conditions_num.numpy(),
                    conditions,
                    cond_type="cont" if curr_cond == "age" else "cat",
                    split_list=split_list,
                    projection_fp=projections_fp,
                )


def make_edited_ds_dir(diffae_model, eval_identifier):
    og_ds_dir = Path(diffae_model.conf.data_paths[0])

    new_ds_dir = og_ds_dir.parent / get_new_ds_dir_name(og_ds_dir.name, eval_identifier)
    new_ds_dir.mkdir(parents=True, exist_ok=True)
    return new_ds_dir


def inference_params(
    inference_mode: str, conf: TrainConfig
) -> tuple[int | tuple[int, ...], int, float]:
    match inference_mode:
        case "wholebrain":
            patch_size = conf.img_size
            if isinstance(patch_size, int):
                patch_size = (patch_size,) * conf.dims
            sw_batch_size = 1
            overlap = 0.0
        case "patchwise":
            patch_size = (32,) * conf.dims
            sw_batch_size = 32
            overlap = 0.25
        case _:
            raise ValueError(f"Invalid inference mode: {inference_mode}")
    return patch_size, sw_batch_size, overlap


def save_result_to_json(results, results_fp):
    results_fp.parent.mkdir(parents=True, exist_ok=True)

    results_json = json.dumps(results, indent=2, sort_keys=True)
    print(results_json)

    with open(results_fp, "w+") as f:
        f.write(results_json)


if __name__ == "__main__":
    main()
