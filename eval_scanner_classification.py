import json
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional, Sequence, get_args

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch.utils.data
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from eval_harm_baselines import load_model
from harm_model import HarmonizationMethodName

ScannerType = Literal["Guys", "HH", "IOP"]


class CycleGANHarmModel:
    def __init__(self, image_size: torch.Size, target_scanner: ScannerType):
        self.base_results_dir = (
            Path(__file__).parent
            / "ixi_cyclegan"
            / "T1_biasfield_corrected"
            / "results"
            / "Guys2HH-latest"
        )
        assert self.base_results_dir.exists(), "CycleGAN results dir not found"
        self._target_scanner = target_scanner.lower()

        self.image_size = image_size

    def get_target_dir(self, source_scanner: ScannerType) -> Optional[Path]:
        # make lower case
        source_scanner_lower = source_scanner.lower()
        target_dir = self.base_results_dir / f"{source_scanner_lower}2{self._target_scanner}"
        if target_dir.exists():
            return target_dir
        # case when iop is source
        return None

    def __call__(self, sfp: Path) -> torch.Tensor:
        """Load harmonized image. Returns batch of size 1."""
        source_scanner: ScannerType = sfp.stem.split("-")[1]  # type: ignore
        target_dir = self.get_target_dir(source_scanner)
        if target_dir is None:
            return -torch.ones(*self.image_size, 1)  # all background

        target_fp = target_dir / f"translated-{sfp.name}"
        try:
            harmonized_img = nib.load(str(target_fp)).get_fdata()
        except FileNotFoundError:
            # workaround: some harmonized images are missing
            return -torch.ones(*self.image_size, 1)

        torch_harmonized_img_batch = torch.from_numpy(harmonized_img)[None, None].float()
        # to np channel order
        torch_harmonized_img_batch = torch_harmonized_img_batch.permute(0, 2, 3, 1)
        return torch_harmonized_img_batch


class HarmModelNumpyWrapper:
    """Harmonize images to a target scanner."""

    def __init__(
        self,
        model_name: HarmonizationMethodName,
        test_split_fps_target: list[Path],
    ):
        dataset = FpsDataset({"test": test_split_fps_target})
        target_images = np.stack([img for img, _, _ in dataset], axis=0).astype(np.float32)
        # normalize to -1,1
        target_images = target_images * 2 - 1
        # from np to torch channel order
        target_images = target_images.transpose(0, 3, 1, 2)
        self.device = torch.device("cuda")
        target_images_torch = torch.from_numpy(target_images).to(self.device)
        from torchvision.utils import save_image

        save_image(
            target_images_torch,
            "test_target_images.png",
            value_range=(-1, 1),
            normalize=True,
        )
        self.harm_model, _ = load_model(
            model_name,
            target_images_torch,
            device=self.device,
            wandb_id="1voovf9c" if model_name == "DiffAE" else None,
        )

    def __call__(self, img_batch: torch.Tensor) -> torch.Tensor:

        img_batch = img_batch.permute(0, 3, 1, 2)

        img_batch_torch = img_batch.to(self.device)
        with torch.no_grad():
            plt.imsave(
                "test_before_harm.png",
                img_batch[0].squeeze().cpu().numpy(),
                cmap="gray",
            )
            img_harm = self.harm_model.harmonize(img_batch_torch).cpu()
            plt.imsave("test_after_harm.png", img_harm[0].squeeze().numpy(), cmap="gray")
        # back to np channel order
        img_harm = img_harm.permute(0, 2, 3, 1)
        return img_harm


def load_data() -> dict[str, dict[str, list[Path]]]:
    dataset_dir = Path("~/datasets/ixi_reg_affine_skullstrip/T1_biasfield_corrected").expanduser()

    splits_dir = Path.cwd() / "dataset" / "ixi"

    split_fps = {
        "train": {"Guys": [], "HH": [], "IOP": []},
        "test": {"Guys": [], "HH": [], "IOP": []},
    }
    for split in split_fps.keys():
        for scanner in [
            "Guys",
            "HH",
        ]:
            splt_subject_fps = load_split_subjects(dataset_dir, splits_dir, split, scanner)

            split_fps[split][scanner].extend(splt_subject_fps)

    # treat IOP separately, because there is no pre-defined split
    scanner = "IOP"
    splt_subject_fps = load_split_subjects(dataset_dir, splits_dir, "all", scanner)
    # split into train and test
    train_subject_fps, test_subject_fps = train_test_split(
        splt_subject_fps, test_size=0.2, random_state=42
    )
    split_fps["train"][scanner].extend(train_subject_fps)
    split_fps["test"][scanner].extend(test_subject_fps)

    return split_fps


def load_split_subjects(
    dataset_dir: Path, splits_dir: Path, split: str, scanner: str
) -> list[Path]:
    split_scanner_fp = splits_dir / scanner / f"{split}.txt"
    with open(split_scanner_fp, "r") as f:
        spit_subject_ids = f.read().splitlines()

    # make file names from subject ids
    splt_subject_fps = [dataset_dir / f"{subject_id}-T1.nii.gz" for subject_id in spit_subject_ids]
    assert all([fp.exists() for fp in splt_subject_fps])

    return splt_subject_fps


def load_middle_slice(img_fp: Path) -> np.ndarray:
    img_nii = nib.load(str(img_fp))

    # middle slice
    img_slice = img_nii.slicer[:, :, img_nii.shape[2] // 2 : img_nii.shape[2] // 2 + 1].get_fdata()

    return img_slice


class FpsDataset(torch.utils.data.Dataset):
    def __init__(self, fps: dict[str, list[Path]]):
        super().__init__()
        # flatten fps
        self.fps = [fp for fps in fps.values() for fp in fps]
        self._init_labels(fps)
        assert len(self.fps) == len(self.labels), "fps and labels must have same length"

    def _init_labels(self, fps: dict[str, list[Path]]):
        labels = []
        for scanner, scanner_fps in fps.items():
            labels.extend([scanner] * len(scanner_fps))
        self.labels = labels

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        img_fp: Path = self.fps[idx]

        img_slice = load_middle_slice(img_fp)

        # normalize to 0 1 based on 95th percentile
        mask_fp = img_fp.with_name(img_fp.with_suffix("").stem + "_mask.nii.gz")
        mask_fp = Path(str(mask_fp).replace("ixi_reg_affine/", "ixi_reg_affine_skullstrip/"))
        mask_slice = load_middle_slice(mask_fp)

        max95 = np.percentile(img_slice[mask_slice > 0], 99.5)
        min05 = np.percentile(img_slice[mask_slice > 0], 0.05)
        img_slice = (img_slice - min05) / (max95 - min05)
        img_slice = np.clip(img_slice, 0, 1)
        # crop to [192,224]
        lower_row = (img_slice.shape[0] - 192) // 2
        upper_row = lower_row + 192
        lower_col = (img_slice.shape[1] - 224) // 2
        upper_col = lower_col + 224
        img_slice = img_slice[lower_row:upper_row, lower_col:upper_col]
        mask_slice = mask_slice[lower_row:upper_row, lower_col:upper_col]

        return img_slice, mask_slice, self.labels[idx]


class FeatureExtractor:
    def __init__(self):
        self._extractor = featureextractor.RadiomicsFeatureExtractor(
            force2D=True, force2Ddimension=2
        )
        self._extractor.enableAllFeatures()

    def __call__(self, img_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        img = sitk.GetImageFromArray(img_np)
        mask = sitk.GetImageFromArray(mask_np)

        result = self._extractor.execute(img, mask)
        feature_vector = np.array([v for k, v in result.items() if k.startswith("original_")])
        return feature_vector


def extract_features_from_loader(
    dataloader: Sequence,
    harm_model: Optional[HarmModelNumpyWrapper | CycleGANHarmModel] = None,
    vis_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:

    if vis_dir is not None:
        vis_dir.mkdir(exist_ok=True)

    feat_extractor = FeatureExtractor()

    pool = Pool(4)
    img_cnt = 0

    train_features = []
    train_labels = []
    for img_batch, mask_batch, labels in tqdm(
        dataloader, desc="extracting features", total=len(dataloader)
    ):

        if harm_model is not None:
            if isinstance(harm_model, CycleGANHarmModel):
                # assume img_batch is list of file paths
                # return -1,1 images
                img_batch = harm_model(img_batch)
            else:
                # to -1,1
                img_batch = img_batch * 2 - 1
                img_batch = harm_model(img_batch.float())

            # back to 0,1
            img_batch = (img_batch + 1) / 2

        # mask out background in img_batch (only apples with HACA3)
        img_batch[~mask_batch.bool()] = 0
        if harm_model is not None and vis_dir is not None:
            # vis images
            for img, label in zip(img_batch, labels):
                plt.figure()
                plt.subplot(1, 1, 1)
                plt.tight_layout(pad=0)
                plt.imshow(img.squeeze().numpy(), cmap="gray")
                plt.axis("off")
                plt.savefig(
                    vis_dir / f"{label}_{img_cnt}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
                img_cnt += 1

        features = pool.starmap(feat_extractor, zip(img_batch, mask_batch))
        train_features.extend(features)
        train_labels.extend(labels)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    return train_features, train_labels


def init_loader(
    split_fps: dict[str, dict[str, list[Path]]], **worker_kwargs
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset = FpsDataset(split_fps["train"])
    test_dataset = FpsDataset(split_fps["test"])

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=False, **worker_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, **worker_kwargs
    )

    return train_loader, test_loader


def train_classifier(train_features: np.ndarray, train_labels: np.ndarray) -> LogisticRegression:
    # normalize features

    # clf = SVC(random_state=42, class_weight="balanced")
    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(train_features, train_labels)
    return clf


def vis_features(features, labels, save_fp):

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    plt.figure()
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(features_pca[idx, 0], features_pca[idx, 1], label=label)
    plt.legend()
    plt.savefig(save_fp)
    plt.close()


def calc_test_metrics(
    clf: LogisticRegression, test_features: np.ndarray, test_labels: np.ndarray
) -> dict:
    test_preds = clf.predict(test_features)

    test_metrics = {}
    if np.unique(test_labels).shape[0] == 1:
        # target scanner classification, all labels are target scanner
        # => accuracy
        test_metrics["accuracy"] = np.mean(test_preds == test_labels)
    else:
        # scanner de bias test => roc
        test_preds_proba = clf.predict_proba(test_features)
        test_metrics["roc_auc"] = roc_auc_score(
            test_labels,
            test_preds_proba,
            multi_class="ovo",
            labels=clf.classes_,
            average="macro",
        )
        test_metrics["mcc"] = matthews_corrcoef(test_labels, test_preds)
        test_metrics["f1"] = f1_score(test_labels, test_preds, average="macro")
    return test_metrics


def main():
    radiomics_dir = Path.cwd() / "radiomics"
    radiomics_dir.mkdir(exist_ok=True)

    split_fps = load_data()

    # map to length
    print(
        {
            split: {scanner: len(fps) for scanner, fps in fps.items()}
            for split, fps in split_fps.items()
        }
    )

    # create datasets
    train_loader, test_loader = init_loader(split_fps, num_workers=4)

    if not (radiomics_dir / "train_features.npy").exists():

        train_features, train_labels = extract_features_from_loader(train_loader)

        # save features
        np.save(radiomics_dir / "train_features.npy", train_features)
        np.save(radiomics_dir / "train_labels.npy", train_labels)
        print("Extracted and saved train features", train_features.shape)
    else:
        train_features = np.load(radiomics_dir / "train_features.npy")
        train_labels = np.load(radiomics_dir / "train_labels.npy")
        print("Loaded train features", train_features.shape)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    clf = train_classifier(train_features, train_labels)

    if not (radiomics_dir / "test_features.npy").exists():
        test_features, test_labels = extract_features_from_loader(test_loader)
        np.save(radiomics_dir / "test_features.npy", test_features)
        np.save(radiomics_dir / "test_labels.npy", test_labels)
        print("Extracted and saved test features", test_features.shape)
    else:
        test_features = np.load(radiomics_dir / "test_features.npy")
        test_labels = np.load(radiomics_dir / "test_labels.npy")
        print("Loaded test features", test_features.shape)
    test_features = scaler.transform(test_features)

    vis_features(
        train_features,
        train_labels,
        radiomics_dir / "train_features.png",
    )
    vis_features(
        test_features,
        test_labels,
        radiomics_dir / "test_features.png",
    )

    test_metrics = calc_test_metrics(clf, test_features, test_labels)

    test_metrics_serialized = json.dumps(test_metrics, indent=4)
    print(test_metrics_serialized)
    with open(radiomics_dir / "test_metrics.json", "w") as f:
        f.write(test_metrics_serialized)

    target_scanners = np.unique(test_labels)
    method_names = get_args(HarmonizationMethodName)
    # method_names = [
    #     "histogram_matching",
    # ]  # "CycleGAN"]
    # method_names = ["histogram_matching", "HACA3", "unharmonized"]
    print("methods", method_names)
    for method_name in method_names:

        for target_scanner in target_scanners:
            print(f"Harmonizing to {target_scanner} with {method_name}")
            vis_dir = radiomics_dir / f"{method_name}_{target_scanner}_vis"
            has_all_vis = len(list(vis_dir.glob("*.png"))) == len(test_features)
            if (
                not (radiomics_dir / f"{method_name}_{target_scanner}_test_features.npy").exists()
                or not has_all_vis
            ):
                # use images with skull for HACA3 pretrained
                if method_name == "HACA3":
                    # map filenames, replace ixi_reg_affine_skullstrip with ixi_reg_affine
                    _test_fps_with_skull = {
                        label: [Path(str(fp).replace("_skullstrip", "")) for fp in label_fps]
                        for label, label_fps in split_fps["test"].items()
                    }
                    test_fps_for_harm_target = _test_fps_with_skull[target_scanner]

                    test_loader_for_harm = torch.utils.data.DataLoader(
                        FpsDataset(_test_fps_with_skull),
                        batch_size=16,
                        shuffle=False,
                        num_workers=4,
                    )
                elif method_name == "CycleGAN":
                    # loader needs to yield a 3-tuple, where the first element is the image path
                    def cycle_gan_loader():
                        for scanner, scanner_fps in split_fps["test"].items():
                            for fp in scanner_fps:
                                mask_batch = torch.ones([1, 192, 224, 1])
                                yield fp, mask_batch, [scanner]

                    test_loader_for_harm = list(cycle_gan_loader())
                    test_fps_for_harm_target = split_fps["test"][target_scanner]

                else:
                    test_fps_for_harm_target = split_fps["test"][target_scanner]
                    test_loader_for_harm = test_loader

                if method_name == "CycleGAN":
                    harm_model_wrapper = CycleGANHarmModel(
                        torch.Size([1, 192, 224]), target_scanner
                    )
                else:
                    harm_model_wrapper = HarmModelNumpyWrapper(
                        method_name, test_fps_for_harm_target
                    )
                test_features_harm, test_labels_harm = extract_features_from_loader(
                    test_loader_for_harm,
                    harm_model=harm_model_wrapper,
                    vis_dir=vis_dir,
                )
                # save features
                np.save(
                    radiomics_dir / f"{method_name}_{target_scanner}_test_features.npy",
                    test_features_harm,
                )
                np.save(
                    radiomics_dir / f"{method_name}_{target_scanner}_test_labels.npy",
                    test_labels_harm,
                )
                print(
                    f"Extracted & saved test features harmonized: {target_scanner} ({method_name})",
                    test_features_harm.shape,
                )
            else:
                test_features_harm = np.load(
                    radiomics_dir / f"{method_name}_{target_scanner}_test_features.npy"
                )
                test_labels_harm = np.load(
                    radiomics_dir / f"{method_name}_{target_scanner}_test_labels.npy"
                )
                print(
                    f"Loaded test features harmonized to {target_scanner} with {method_name}",
                    test_features_harm.shape,
                )

            # scale to normal distribution
            test_features_harm = scaler.transform(test_features_harm)

            vis_features(
                test_features_harm,
                test_labels_harm,
                radiomics_dir / f"{method_name}_{target_scanner}_test_features.png",
            )
            is_target_scanner_mask = test_labels_harm == target_scanner

            # evaluate how well the images can be classified after harmonization
            test_metrics_harm = calc_test_metrics(
                clf,
                test_features_harm[~is_target_scanner_mask],
                test_labels_harm[~is_target_scanner_mask],
            )
            # evaluate whether the images are classified as the target scanner
            test_metrics_harm_target = calc_test_metrics(
                clf,
                test_features_harm[~is_target_scanner_mask],
                np.array([target_scanner] * (~is_target_scanner_mask).sum()),
            )
            print(
                "target",
            )
            test_metrics_harm_target_comb = {
                "target": test_metrics_harm_target,
                "harmonized": test_metrics_harm,
            }
            test_metrics_harm_target_comb_serialized = json.dumps(
                test_metrics_harm_target_comb, indent=4
            )
            print(test_metrics_harm_target_comb_serialized)
            with open(
                radiomics_dir / f"{method_name}_{target_scanner}_test_metrics.json",
                "w",
            ) as f:
                f.write(test_metrics_harm_target_comb_serialized)

    # collect all metrics into one table. each metric is a list.
    big_metrics_dict = {}
    metrics_json_fps = list(radiomics_dir.glob("*_test_metrics.json"))

    for metrics_json_fp in metrics_json_fps:
        with open(metrics_json_fp, "r") as f:
            metrics = json.load(f)

        # at each level create list if not is dict else create dict and continue
        for k1, v1 in metrics.items():
            if isinstance(v1, dict):
                big_metrics_dict.setdefault(k1, {})
                for k2, v2 in v1.items():
                    if isinstance(v2, dict):
                        big_metrics_dict[k1].setdefault(k2, {})
                        for k3, v3 in v2.items():
                            big_metrics_dict[k1][k2].setdefault(k3, []).append(v3)
                    else:
                        big_metrics_dict[k1].setdefault(k2, []).append(v2)
            else:
                big_metrics_dict.setdefault(k1, []).append(v1)

    # flatten
    # add methods
    big_metrics_dict["method"] = [fp.stem.rsplit("_", 2)[0] for fp in metrics_json_fps]
    big_metrics_dict_json = json.dumps(big_metrics_dict, indent=4)
    with open(radiomics_dir / "big_metrics_dict.json", "w") as f:
        f.write(big_metrics_dict_json)

    big_metrics_dict_flattened = {}
    # add each key as tuples
    for k1, v1 in big_metrics_dict.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        big_metrics_dict_flattened[(k1, k2, k3)] = v3
                else:
                    big_metrics_dict_flattened[(k1, k2)] = v2
        else:
            big_metrics_dict_flattened[k1] = v1
    df = pd.DataFrame(big_metrics_dict_flattened)
    df.set_index("method", inplace=True, drop=True)

    # group by method index prefix_ and add as new row for each method
    df_avg = df.groupby(df.index.str.split("_").str[0]).mean()
    for method, row in df_avg.iterrows():
        df.loc[f"{method}_avg"] = row

    # set digits to 3
    df = df.round(3)
    df.to_csv(radiomics_dir / "big_metrics_dict.csv")

    # important columns


if __name__ == "__main__":
    main()
