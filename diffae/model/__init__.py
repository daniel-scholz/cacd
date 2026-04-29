from typing import Union

from diffae.model.diffae import DiffAEConfig, DiffAEModel
from diffae.model.diffae_id_preserve import DiffAEIDConfig, DiffAEIDModel
from diffae.model.unet import DiffConfig, DiffModel

Model = Union[DiffModel, DiffAEModel, DiffAEIDModel]
ModelConfig = Union[DiffConfig, DiffAEConfig, DiffAEIDConfig]
