import argparse
import os
import time
from pathlib import Path
from typing import Literal

import lightning as L
import submitit
import wandb
import wandb.util
from brain_age_estimate.data import IXIDataModule
from brain_age_estimate.model import AgeRegModel, ScannerClfModel
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

L.seed_everything(0, workers=True)
os.environ["WANDB__SERVICE_WAIT"] = "300"


def train(
    fit_sites: list[str],
    test_sites: list[str],
    task: Literal["age_est", "scanner_clf"],
    spatial_dim: Literal[2, 3],
):
    # Init our data pipeline
    dm = IXIDataModule(
        batch_size=32,
        fit_sites=fit_sites,
        test_sites=test_sites,
        task=task,
        spatial_dim=spatial_dim,
    )

    # To access the x_dataloader we need to call prepare_data and setup.
    dm.prepare_data()
    dm.setup()

    # print length of datasets
    print(
        f"Train set length: {len(dm.train_dataloader().dataset)}",
        f"Val set length: {len(dm.val_dataloader().dataset)}",
        f"Test set length: {len(dm.test_dataloader().dataset)}",
        sep="\n",
    )

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_batch = next(iter(dm.val_dataloader()))
    val_img = val_batch["img"]
    val_age = val_batch["age" if task == "age_est" else "scanner"]

    print(f"Val sample shape: {val_img.shape} Val age shape: {val_age.shape}")

    # return
    # Init our model
    match task:
        case "scanner_clf":
            model = ScannerClfModel(
                learning_rate=3e-4,
                num_classes=3,
            )
        case "age_est":
            model = AgeRegModel(
                spatial_dims=spatial_dim,
                learning_rate=3e-4,
                # init_with_mean=dm.age_mean
            )

    # Initialize wandb logger
    wandb_id = os.environ.get("WANDB_ID", wandb.util.generate_id())
    print(f"{wandb_id=}")
    fit_sites_str = "-".join(fit_sites)
    test_sites_str = "-".join(test_sites)
    experiment_name = f"{fit_sites_str}_{test_sites_str}"
    wandb_logger = WandbLogger(
        project="brain-age-reg" if task == "age_est" else "scanner-clf",
        log_model=False,
        name=experiment_name,
        id=wandb_id,
    )

    ckpt_dir = Path(f"checkpoints/{task}/{experiment_name}/{wandb_id}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"{ckpt_dir=}")

    # Initialize a trainer
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="loss/val_loss",
        mode="min",
        # save last and best model
        save_last=True,
        save_top_k=1,
    )
    overfit = False
    target_batch_size = 32
    accum_grad_batches = max(1, target_batch_size // dm.batch_size) if not overfit else 1
    log_every_n_steps = (
        min(max(1, 10 // accum_grad_batches), len(dm.ixi_train) // target_batch_size)
        if not overfit
        else 1
    )

    print(f"Logging every {log_every_n_steps} steps")
    print("Accumulated batches", accum_grad_batches)

    trainer = L.Trainer(
        # max_epochs=100,
        # use steps instead of epochs since we are using differently sized datasets.
        max_steps=10_000,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ]
        + ([ckpt_callback] if not overfit else []),
        accumulate_grad_batches=accum_grad_batches,  # yields 32
        log_every_n_steps=log_every_n_steps,
        overfit_batches=10 if overfit else 0.0,
    )

    # Train the model
    last_ckpt = ckpt_dir / "last.ckpt"
    trainer.fit(model, dm, ckpt_path=last_ckpt if last_ckpt.exists() else None)

    # construct best ckpt path for machine independent loading
    ckpt_callback = trainer.checkpoint_callback
    if ckpt_callback is not None:
        best_ckpt_name = Path(ckpt_callback.best_model_path).name
        if not best_ckpt_name:
            best_ckpt_name = "last.ckpt"

        best_ckpt_path = str((ckpt_dir / best_ckpt_name).resolve())
    else:
        best_ckpt_path = last_ckpt

    # Validate the model
    trainer.validate(datamodule=dm, ckpt_path=best_ckpt_path)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test(datamodule=dm, ckpt_path=best_ckpt_path)

    # Close wandb run
    wandb.finish()


def train_server(
    fit_sites: list[str],
    test_sites: list[str],
    task: Literal["age_est", "scanner_clf"],
    spatial_dim: Literal[2, 3],
):
    """Wrapper around local train function"""

    log_folder = Path("logs")
    timeout_in_h = 24
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=timeout_in_h * 60)
    if executor.cluster != "local":
        timeout_in_h = 7 * 24
        executor.update_parameters(
            gpus_per_node=1,
            mem_gb=64,
            cpus_per_task=18,
            nodes=1,
            timeout_min=timeout_in_h * 60,
            # partition="deadline",
            # account="deadline",
        )
    else:
        print("Using local machine")

    print(f"Submit job with fit_sites={fit_sites} and test_sites={test_sites}")
    # update executor parameters
    executor.update_parameters(name=f"fit={fit_sites}_test={test_sites}")

    job = executor.submit(train, fit_sites, test_sites, task, spatial_dim)
    return job


def run():
    # add task as argument to the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm", action="store_true", default=False, help="Run on slurm")
    parser.add_argument(
        "--task",
        type=str,
        choices=["age_est", "scanner_clf"],
        required=True,
        default="age_est",
    )
    parser.add_argument(
        "--spatial_dim",
        type=int,
        choices=[2, 3],
        required=True,
    )

    args = parser.parse_args()
    task = args.task
    spatial_dim = args.spatial_dim

    if args.slurm:
        train_fn = train_server
    else:
        train_fn = train

    print(f"Performing task: {task}")
    match task:
        case "age_est":
            # hard code relevant sites for current experiments
            fit_test_sites = [
                (("Guys",), ("IOP", "HH")),
                (("HH",), ("IOP", "Guys")),
                (("HH", "Guys"), ("IOP",)),
            ]

        case "scanner_clf":
            # always train and validate on all
            fit_test_sites = [(("HH", "IOP", "Guys"), ("HH", "IOP", "Guys"))]

    print("Combinations used for training", fit_test_sites)
    jobs = []
    for fit_sites, test_sites in fit_test_sites:
        print(f"Training with fit_sites={fit_sites} and test_sites={test_sites}")
        job = train_fn(fit_sites, test_sites, task, spatial_dim)
        jobs.append(job)
        # break

    if args.slurm:
        print(f"Submitted {len(jobs)} jobs", jobs)
        while not all(job.done() for job in jobs):
            print("Waiting for jobs to complete...")
            for job in jobs:
                print(f"Job {job} completed: {job.done()}")
            # check every 10 minutes
            time.sleep(60 * 10)


if __name__ == "__main__":
    run()
