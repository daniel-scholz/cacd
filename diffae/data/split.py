from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.utils.data

from diffae.data.ixi import IXIDataset


def split_dataset_stratified(
    dataset: IXIDataset,
    splits: tuple[float, ...],
    rng: np.random.Generator,
):
    # get targets (labels)
    targets = dataset.scanner_int

    # define strata which we want to sample from
    strata = np.unique(targets)

    # iterate over strata, i.e., classes
    split_indices = stratified_split_indices(splits, rng, targets, strata)

    # make subsets from the indices
    split_datasets = [torch.utils.data.Subset(dataset, indices) for indices in split_indices]

    return split_datasets


def stratified_split_indices(splits, rng, targets, strata):
    # place to store indices for each split
    split_indices = [None] * len(splits)
    for stratum in strata:
        # sample indices of the current stratum
        stratum_indices = np.where(targets == stratum)[0]
        # shuffle the indices
        rng.shuffle(stratum_indices)
        # split the indices
        stratum_splits = np.array_split(
            stratum_indices,
            # drop last split, to prevent empty splits
            (np.cumsum(splits) * len(stratum_indices)).astype(int)[:-1],
        )
        # set splits
        for i_split, split in enumerate(stratum_splits):
            # append the indices to the split
            if split_indices[i_split] is None:
                split_indices[i_split] = split
            else:
                split_indices[i_split] = np.concatenate((split_indices[i_split], split))

    return split_indices


def split_dataset_stratified_by_patient(
    dataset: IXIDataset,
    splits: tuple[float, ...],
    rng: np.random.Generator,
):
    # get targets (labels)
    targets = dataset.scanner_int

    patients_all = list(dataset.fn2subject(fp) for fp in dataset.subject_dirs)

    # reduce to unique patients (remove duplicates from sessions)
    patients_unique, mapping, inverse_map = np.unique(
        patients_all,
        return_index=True,
        return_inverse=True,
    )
    # get corresponding targets for unique patients
    targets_per_patient = targets[mapping]

    # define strata which we want to sample from
    strata = np.unique(targets_per_patient)

    # split based on unique patients
    split_indices_unique = stratified_split_indices(splits, rng, targets_per_patient, strata)

    if len(patients_all) == len(patients_unique):
        # save time by not mapping indices back to all patients when there are no duplicates
        split_indices = split_indices_unique
    else:
        # place to store indices for each split
        split_indices = [np.zeros(0, dtype=np.int32)] * len(splits)
        for i_split, split in enumerate(split_indices_unique):
            # map split indices back to all patients
            for i_sample in split:
                split_indices[i_split] = np.concatenate(
                    (split_indices[i_split], np.where(inverse_map == i_sample)[0])
                )

    # make subsets from the indices
    split_datasets = [torch.utils.data.Subset(dataset, indices) for indices in split_indices]
    return split_datasets


def split_datasets(
    datasets: Sequence[IXIDataset],
    test_datasets: Sequence[IXIDataset],
    seed: int,
    data_split_mode: Literal["loaded", "random", "stratified"],
    data_names: Sequence[str],
) -> tuple[
    torch.utils.data.ConcatDataset,
    torch.utils.data.ConcatDataset,
    torch.utils.data.ConcatDataset,
]:
    train_data_subsets = []
    val_data_subsets = []
    test_data_subsets = []

    for i_ds, dataset in enumerate(datasets):
        # define random state
        np_rng = np.random.default_rng(seed)
        torch_gen = torch.Generator().manual_seed(seed)
        splits = (0.9, 0.1)
        # one split mode for all datasets
        match data_split_mode:
            case "random":
                raise NotImplementedError("Random split is not implemented for IXI dataset")
                n_train = int(splits[0] * len(dataset))
                train_data_subset, val_data_subset = torch.utils.data.random_split(
                    dataset=dataset,
                    lengths=[n_train, len(dataset) - n_train],
                    generator=torch_gen,
                )

            case "stratified":
                raise NotImplementedError("Stratified split is not implemented for IXI dataset")
                (
                    train_data_subset,
                    val_data_subset,
                ) = split_dataset_stratified_by_patient(
                    dataset,
                    splits,
                    rng=np_rng,
                )

            case "loaded":
                data_name = data_names[i_ds]
                # get indices of the subjects
                idx_to_subject = {sub_id: i for i, sub_id in enumerate(dataset.subject_ids)}

                # load the subject ids from the csv files
                dataset_split = {}
                splits_dir = Path("dataset") / data_name
                # splits_dir = self._append_split_sites(dataset, splits_dir)

                splits_subsets_dict = {
                    "train": dataset.fit_sites,
                    "val": dataset.fit_sites,
                }

                for split in ["train", "val"]:
                    split_idx = []
                    if splits_subsets_dict[split]:
                        for subset in splits_subsets_dict[split]:
                            subjects_splits_fp = splits_dir / subset / f"{split}.txt"

                            with open(subjects_splits_fp, "r") as f:
                                saved_split_subjects = f.readlines()
                            saved_split_subjects = [s.strip() for s in saved_split_subjects]

                            cur_idx = [idx_to_subject[s] for s in saved_split_subjects]
                            split_idx.extend(cur_idx)
                    else:
                        subjects_splits_fp = splits_dir / f"{split}.txt"
                        with open(subjects_splits_fp, "r") as f:
                            saved_split_subjects = f.readlines()
                        saved_split_subjects = [s.strip() for s in saved_split_subjects]
                        split_idx = [
                            idx_to_subject[s] for s in saved_split_subjects if s in idx_to_subject
                        ]

                    dataset_split[split] = torch.utils.data.Subset(dataset, split_idx)

                train_data_subset = dataset_split["train"]
                val_data_subset = dataset_split["val"]

            case _:
                raise NotImplementedError(f"Unknown data split mode: {data_split_mode}")
        train_data_subsets.append(train_data_subset)
        val_data_subsets.append(val_data_subset)

    train_data = torch.utils.data.ConcatDataset(train_data_subsets)
    val_data = torch.utils.data.ConcatDataset(val_data_subsets)

    test_data_subsets = []
    for data_name, test_dataset in zip(data_names, test_datasets):
        splits_dir = Path("dataset") / data_name
        idx_to_subject = {sub_id: i for i, sub_id in enumerate(test_dataset.subject_ids)}
        match data_split_mode:
            case "loaded":
                # load idx for test split from file
                test_split_idx = []
                if test_dataset.test_sites:
                    for test_subset in test_dataset.test_sites:

                        subjects_splits_fp = splits_dir / test_subset / "test.txt"
                        # if test_subset not in dataset.fit_sites, use all.txt
                        if test_subset not in test_dataset.fit_sites:
                            subjects_splits_fp = subjects_splits_fp.parent / "all.txt"
                        saved_split_subjects = subjects_splits_fp.read_text().splitlines()
                        cur_test_idx = [
                            idx_to_subject[s] for s in saved_split_subjects if s in idx_to_subject
                        ]
                        test_split_idx.extend(cur_test_idx)
                else:
                    subjects_splits_fp = splits_dir / "test.txt"
                    with open(subjects_splits_fp, "r") as f:
                        saved_split_subjects = f.readlines()
                    saved_split_subjects = [s.strip() for s in saved_split_subjects]
                    test_split_idx = [
                        idx_to_subject[s] for s in saved_split_subjects if s in idx_to_subject
                    ]

            case "random" | "stratified":
                # use all the data in the test split
                test_split_idx = torch.arange(len(test_dataset)).tolist()
            case _:
                raise NotImplementedError(f"Unknown data split mode: {data_split_mode}")
        # make subset to correspond to the train and val datasets
        test_data_subset = torch.utils.data.Subset(test_dataset, test_split_idx)
        test_data_subsets.append(test_data_subset)

    test_data = torch.utils.data.ConcatDataset(test_data_subsets)
    return train_data, val_data, test_data
