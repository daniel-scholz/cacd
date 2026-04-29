from typing import Self

import torch
import torch.nn as nn


class AnatomicalRegionsAugmentation(nn.Module):
    def __init__(self, batch_size: int, anat_label_map: torch.Tensor) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.anat_label_map = anat_label_map

        self.device = anat_label_map.device

        self.gamma_std = 0.2
        self.scale_std = 0.2
        self.scale_mean = 1.0
        self.bias_std = 0.2

        self.value_range = (-1, 1)

        label_symmetrizer = LabelSymmetrizer(device=self.device)

        anat_label_map = label_symmetrizer(anat_label_map)
        self.n_anat_regs = anat_label_map.unique().shape[0] - 1  # subtract background

        self.fg_mask = (anat_label_map > 0)[:, None]

        self.anat_label_map_oh = self.anat_labels_to_oh(anat_label_map)
        self.params = self.sample_params(batch_size)

    def sample_n_transforms(self, n_transforms: int) -> list[Self]:
        return [
            type(self)(batch_size=self.batch_size, anat_label_map=self.anat_label_map)
            for _ in range(n_transforms)
        ]

    def sample_params(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        gamma_reg = torch.exp(
            torch.randn(batch_size, self.n_anat_regs, device=self.device)
            * self.gamma_std
        )

        # random scale sampled from N(+-1, 0.2)
        scale_reg = (
            torch.randn(batch_size, self.n_anat_regs, device=self.device)
            * self.scale_std
            + self.scale_mean
        )
        scale_reg *= (
            (torch.rand(batch_size, self.n_anat_regs, device=self.device) * 2 - 1)
            .sign()
            .float()
        )

        # random bias drawn from N(0, 0.2)
        bias_reg = (
            torch.randn(batch_size, self.n_anat_regs, device=self.device)
            * self.bias_std
        )

        return gamma_reg, scale_reg, bias_reg

    def forward(self, imgs: torch.Tensor):

        gamma_reg, scale_reg, bias_reg = self.params

        # apply transformations
        # gamma correction
        img_aug = self.generate_augmented_image(
            imgs,
            self.anat_label_map_oh,
            gamma_reg,
            scale_reg,
            bias_reg,
        )
        img_aug[~self.fg_mask] = self.value_range[0]

        return img_aug

    def generate_augmented_image(
        self,
        imgs: torch.Tensor,
        anat_labels_oh: torch.Tensor,
        gamma_reg: torch.Tensor,
        scale_reg: torch.Tensor,
        bias_reg: torch.Tensor,
    ) -> torch.Tensor:

        img_min = (
            imgs.flatten(start_dim=1)
            .min(dim=1)
            .values.view(imgs.shape[0], *(1,) * (imgs.dim() - 1))
        )
        img_max = (
            imgs.flatten(start_dim=1)
            .max(dim=1)
            .values.view(imgs.shape[0], *(1,) * (imgs.dim() - 1))
        )

        imgs = (imgs - img_min) / (img_max - img_min)

        img_mean = (
            imgs.flatten(start_dim=1)
            .mean(dim=1)
            .view(imgs.shape[0], *(1,) * (imgs.dim() - 1))
        )

        img_aug_regs = imgs ** gamma_reg[..., *(None,) * (imgs.dim() - 2)]
        # stretch around mean
        img_aug_regs = (img_aug_regs - img_mean) * scale_reg[
            ..., *(None,) * (imgs.dim() - 2)
        ] + img_mean
        # translate
        img_aug_regs += bias_reg[..., *(None,) * (imgs.dim() - 2)]

        # mask corresponding regions
        img_aug_regs *= anat_labels_oh

        # aggregate regions into augmented image
        imgs_aug = torch.sum(img_aug_regs, dim=1, keepdim=True)

        # denormalize to img_min, img_max
        imgs_aug = imgs_aug * (img_max - img_min) + img_min

        imgs_aug = torch.clamp(imgs_aug, *self.value_range)
        return imgs_aug

    def anat_labels_to_oh(self, anat_seg: torch.Tensor) -> torch.Tensor:

        # create one-hot encoding for each anatomical region
        anat_segs_unique = anat_seg[anat_seg != 0].unique()
        anat_segs_oh = (
            anat_seg[:, None]
            == anat_segs_unique[None, :, *(None,) * (anat_seg.dim() - 1)]
        ).float()

        return anat_segs_oh


class LabelSymmetrizer(nn.Module):
    anat_label_map = {
        0: "background",
        2: "left cerebral white matter",
        3: "left cerebral cortex",
        4: "left lateral ventricle",
        5: "left inferior lateral ventricle",
        7: "left cerebellum white matter",
        8: "left cerebellum cortex",
        10: "left thalamus",
        11: "left caudate",
        12: "left putamen",
        13: "left pallidum",
        14: "3rd ventricle",
        15: "4th ventricle",
        16: "brain-stem",
        17: "left hippocampus",
        18: "left amygdala",
        26: "left accumbens area",
        24: "CSF",
        28: "left ventral DC",
        41: "right cerebral white matter",
        42: "right cerebral cortex",
        43: "right lateral ventricle",
        44: "right inferior lateral ventricle",
        46: "right cerebellum white matter",
        47: "right cerebellum cortex",
        49: "right thalamus",
        50: "right caudate",
        51: "right putamen",
        52: "right pallidum",
        53: "right hippocampus",
        54: "right amygdala",
        58: "right accumbens area",
        60: "right ventral DC",
    }

    def __init__(self, device: torch.device):
        super().__init__()

        symm_anat_label_map = {
            k: self.strip_left_right(v) for k, v in self.anat_label_map.items()
        }
        symm_anat_label_map = {
            k: self.unify_ventricles(v) for k, v in symm_anat_label_map.items()
        }

        label_to_int_map = {}
        for k, v in symm_anat_label_map.items():
            label_to_int_map.setdefault(v, []).append(k)
        label_to_symm_label = {}
        for v in label_to_int_map.values():
            for i, k in enumerate(v):
                label_to_symm_label[k] = v[0]

        # create array mapping the asymmetrical labels to their side invariant counterparts
        # for fast lookup
        self.label_to_unique_symm_label_array = torch.zeros(
            max(label_to_symm_label.keys()) + 1, dtype=torch.uint8, device=device
        )
        for k, v in label_to_symm_label.items():
            self.label_to_unique_symm_label_array[k] = v

        self.n_labels = len(label_to_symm_label)

    def forward(self, anat_segs: torch.Tensor) -> torch.Tensor:
        return self.label_to_unique_symm_label_array[
            anat_segs.flatten().tolist()
        ].reshape(anat_segs.shape)

    def strip_left_right(self, label: str) -> str:
        return label.replace("left ", "").replace("right ", "")

    def unify_ventricles(self, label: str) -> str:
        # extract ventricle if present
        if "ventricle" in label:
            return "ventricle"
        return label
