"""Define abstract class for HarmonizationModel"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.transforms.croppad.array import CenterSpatialCrop
from skimage.exposure import match_histograms
from tqdm import tqdm

from diffae.experiment import LitModel

# HACA3 is an optional third-party dependency. Importing it eagerly would
# break ``import harm_model`` for anyone running only the lightweight
# baselines (HistogramMatching / Unharmonized / DiffAE / CycleGAN). It is
# imported inside ``HACA3HarmonizationModel`` instead.

HarmonizationMethodName = Literal[
    "DiffAE", "HACA3", "unharmonized", "histogram_matching", "CycleGAN"
]

# from torchvision.utils import save_image

logger = logging.getLogger("__main__")


class BrainMRIHarmonizationModel(ABC):

    def __init__(self, model: Any, *args, **kwargs):
        self.model = model

    @abstractmethod
    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor: ...


class DiffAEHarmonizationModel(BrainMRIHarmonizationModel):
    def __init__(
        self,
        model: LitModel,
        target_images: torch.Tensor,
        noise_steps: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.model: LitModel

        # diffusion denoising step
        self.T = self.model.conf.T_eval

        target_cond = self.model.encode_ema(target_images)["cond"]

        z_sem_target, _ = self.model.split_sem_id(target_cond)
        self.z_sem_target_mean = z_sem_target.mean(dim=0, keepdim=True)

        # self.z_sem_target_mean = z_sem_target[:1]

        # number of steps to noise and then denoise the image for editing
        # the fewer steps the closer the edited image will be to the original
        self.noise_steps = noise_steps or self.T

    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor:
        """
        Harmonize source images to target images using DiffAE model
        by swapping average z_sem from target images to source images
        """

        source_cond = self.model.encode_ema(source_images)["cond"]
        source_xT = self.model.encode_stochastic_ema(
            source_images,
            source_cond,
            self.T,
            self.noise_steps,
            imgs=source_images if self.model.conf.in_channels == 2 else None,
        )

        z_sem_source, z_id_source = self.model.split_sem_id(source_cond)

        cond_harmonized = self.model.combine_sem_id(
            self.z_sem_target_mean.expand_as(z_sem_source), z_id_source
        )
        # Debugging: use source z_sem
        # cond_harmonized = self.model.combine_sem_id(z_sem_source, z_id_source)
        # logger.warning("Using source z_sem for harmonization for debugging.")

        harmonized_images = self.model.render(
            source_xT,
            {"cond": cond_harmonized},
            self.T,
            T_offset=self.T - self.noise_steps,
            imgs=source_images if self.model.conf.in_channels == 2 else None,
        )

        # use prediction only in the foreground of the source image
        harmonized_images[source_images == -1] = source_images[source_images == -1]
        return harmonized_images


def visualize_imgs(imgs: list[np.ndarray]):
    """
    From haca3 tutorial https://colab.research.google.com/drive/1PeBuqOAGupLQ2gXWVneX1Kn31ISh4oFB
    """

    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
    # make axes iterable
    axes = [axes] if len(imgs) == 1 else axes
    for idx, img in enumerate(imgs):
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(f"Image {idx}")
        axes[idx].axis("off")
    plt.savefig("test_vis_haca3.png")
    plt.tight_layout()
    plt.show()
    plt.close()


class HACA3HarmonizationModel(BrainMRIHarmonizationModel):
    def __init__(
        self,
        target_images: torch.Tensor,
        pretrained: bool,
        *args,
        **kwargs,
    ):
        from haca3.modules.model import HACA3

        # get path to HACA3 module
        pretrained_model_fp = Path(
            os.environ.get("HACA3_MODEL_PATH", "models/haca3/harmonization_public.pt")
        )

        if pretrained and not pretrained_model_fp.exists():
            raise FileNotFoundError(f"{pretrained_model_fp} not found")

        model = HACA3(
            beta_dim=5,  # default
            theta_dim=2,
            eta_dim=2,
            pretrained_haca3=pretrained_model_fp if pretrained else None,
        )
        super().__init__(model, *args, **kwargs)
        self.model: HACA3

        # scale target images to [0, 1]
        target_images = [
            self.preprocess_image(image)
            for image in tqdm(target_images.squeeze(1), desc="Preprocessing target images")
        ]
        visualize_imgs([img.squeeze(0).numpy() for img in target_images])
        # convert into 2d images with batch dimension
        target_images = torch.stack(target_images).to(model.device)
        if not (0 <= target_images.min()):
            raise ValueError("Target images must be in above 0")

        self.model.beta_encoder.eval()
        self.model.theta_encoder.eval()
        self.model.eta_encoder.eval()
        self.model.decoder.eval()
        use_hardcoded_values = False
        if not use_hardcoded_values:
            with torch.no_grad():
                # contrast
                self.compute_theta_target_mean(target_images)
                logger.info(f"{self.theta_target=}, example theta for t1 is (10.0,20.0)")
                # artifacts
                self.compute_average_eta_target(target_images)
                logger.info(f"{self.eta_target=}")
        else:
            logger.info("Using hardcoded theta and eta target values from tutorial")
            theta_target_t1 = (10.0, 20.0)
            self.theta_target = torch.as_tensor(theta_target_t1, dtype=torch.float32)[
                None, ..., None, None
            ]
            self.eta_target = torch.as_tensor((0.3, 0.5), dtype=torch.float32)[
                None, ..., None, None
            ]

        self.fusion_model_fp = Path(
            os.environ.get("HACA3_FUSION_MODEL_PATH", "models/haca3/fusion.pt")
        )

    def compute_average_eta_target(self, target_images):
        batch_size = 512
        num_batches = (len(target_images) + batch_size - 1) // batch_size
        eta_target_list = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(target_images))
            batch_images = target_images[start_idx:end_idx]

            eta_batch = self.model.eta_encoder(batch_images)
            eta_target_list.append(eta_batch)

        eta_target = torch.cat(eta_target_list, dim=0)
        self.eta_target = eta_target.mean(dim=0, keepdim=True).view(1, self.model.eta_dim, 1, 1)

    def compute_theta_target_mean(self, target_images):
        theta_target, _ = self.model.theta_encoder(target_images)
        self.theta_target = theta_target.mean(dim=0, keepdim=True)

    def permute_axial(self, image: torch.Tensor) -> torch.Tensor:
        return image.permute(2, 0, 1)

    def permute_axial_inv(self, image: torch.Tensor) -> torch.Tensor:
        return image.permute(1, 2, 0)

    def permute_coronal(self, image: torch.Tensor) -> torch.Tensor:
        return image.permute(1, 2, 0).flip(1)

    def permute_coronal_inv(self, image: torch.Tensor) -> torch.Tensor:
        return image.flip(1).permute(2, 0, 1)

    def permute_sagittal(self, image: torch.Tensor) -> torch.Tensor:
        return image.permute(0, 2, 1).flip(1)

    def permute_sagittal_inv(self, image: torch.Tensor) -> torch.Tensor:
        return image.flip(1).permute(0, 2, 1)

    def flip_img(self, image: torch.Tensor) -> torch.Tensor:
        # flip up down to match HACA3
        image = image.t()
        return torch.flip(image, dims=(0,))

    def unflip_img(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.flip(image, dims=(0,))
        return image.t()

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        from haca3.test import background_removal, background_removal2d
        from torchvision.transforms import ToTensor

        # transpose to maintain same orientation
        image = self.flip_img(image)

        # rescale to [0, 1]
        image = (image + 1) / 2
        assert 0 <= image.min() and image.max() <= 1, "Image must be in range [0, 1]"

        image_np = image.cpu().numpy().astype(np.float32)
        if image.dim() == 2:
            image_np = background_removal2d(image_np)
            n_row, n_col = image_np.shape
            padded = np.zeros((224, 224), dtype=np.float32)
            padded[
                112 - n_row // 2 : 112 + n_row // 2 + n_row % 2,
                112 - n_col // 2 : 112 + n_col // 2 + n_col % 2,
            ] = image_np
        else:
            image_np = background_removal(image_np)
            n_row, n_col, n_slc = image_np.shape
            padded = np.zeros((224, 224, 224), dtype=np.float32)
            padded[
                112 - n_row // 2 : 112 + n_row // 2 + n_row % 2,
                112 - n_col // 2 : 112 + n_col // 2 + n_col % 2,
                112 - n_slc // 2 : 112 + n_slc // 2 + n_slc % 2,
            ] = image_np

        return ToTensor()(padded)

    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.harmonize_single_image(image) for image in source_images])

    def harmonize_single_image(self, source_image: torch.Tensor) -> torch.Tensor:
        # squeeze channel dim
        source_image = source_image.squeeze(0)
        assert source_image.dim() == 2, "Source images must be 2D"

        # store source image shape
        source_image_shape = source_image.shape
        source_image_list = [self.preprocess_image(source_image)]
        assert len(source_image_list) == 1, "Expected 1 source image"
        # harmonized_images = []

        # orient_image_fn = getattr(self, f"permute_{orientation}")
        harmonized_image_oriented = self.model.harmonize(
            source_images=source_image_list,
            target_images=None,  # self.target_images,
            target_theta=self.theta_target,
            target_eta=self.eta_target,
            out_paths=None,
            header=None,  # -> leads to image being returned
            recon_orientation="axial",
            norm_vals=[1],
            num_batches=1,
            save_intermediate=False,
            intermediate_out_dir=None,
        )
        # undo orientation
        # harmonized_image = getattr(self, f"permute_{orientation}_inv")(
        #     harmonized_image_oriented
        # )
        harmonized_image = harmonized_image_oriented
        # harmonized_images.append(harmonized_image)

        # harmonized_images = torch.stack(harmonized_images)
        # for i, orientation in enumerate(["axial", "coronal", "sagittal"]):

        #     save_image(
        #         self.permute_axial(harmonized_images[i]).flatten(end_dim=-3)[:, None],
        #         f"test_haca3_harmonized_{orientation}.png",
        #         normalize=True,
        #     )

        # if len(harmonized_images) == 3:
        #     img_fused = self.model.combine_images(
        #         None,
        #         1,
        #         pretrained_fusion=self.fusion_model_fp,
        #         # permute images to match correct orientation order
        #         images=harmonized_images,  # [[1, 2, 0]],
        #         image_paths=None,
        #     )
        # else:
        #     img_fused = harmonized_images.squeeze(0).numpy()
        # img_fused = torch.from_numpy(img_fused).unsqueeze(0)
        # save_image(
        #     self.permute_axial(img_fused[0]).flatten(end_dim=-3)[:, None],
        #     "test_haca3_harmonized_fused.png",
        # )
        img_fused = harmonized_image
        # unflip image
        img_fused = self.unflip_img(img_fused)
        visualize_imgs([img_fused.squeeze(0).numpy()])
        # center crop image to original shape
        img_fused = CenterSpatialCrop(source_image_shape)(img_fused[None])

        # scale back to -1,1
        img_fused = 2 * img_fused - 1

        # return as b x c x h x w x d
        out_image = img_fused.to(source_image.device)
        # logger.info(f"{out_image.shape=}")
        return out_image


class ImUnityHarmModel(BrainMRIHarmonizationModel):
    """Run inference in their codebase only load the harmonized images here."""

    def __init__(self, model=None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)

    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor:
        return source_images


class UnharmonizeModel(BrainMRIHarmonizationModel):
    """Simply return the source images"""

    def __init__(self, model=None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)

    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor:
        return source_images


class HistogramMatchingModel(BrainMRIHarmonizationModel):
    """Simply return the source images"""

    def __init__(self, target_images: torch.Tensor, model=None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.target_images = target_images[:1]

    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor:

        harmonized_image = match_histograms(
            source_images.cpu().numpy().squeeze(1),
            self.target_images.cpu().numpy().squeeze(1),
        )
        return torch.from_numpy(harmonized_image).unsqueeze(1).to(source_images.device)


if __name__ == "__main__":
    # test flipping
    model = HACA3HarmonizationModel(None, None)
    img = torch.rand(3, 3, 3) * 100
    for orientation in ["axial", "coronal", "sagittal"]:
        orient_image_fn = getattr(model, f"permute_{orientation}")
        orient_image_inv_fn = getattr(model, f"permute_{orientation}_inv")
        assert torch.allclose(orient_image_inv_fn(orient_image_fn(img)), img)
        print(f"Orientation {orientation} passed")
