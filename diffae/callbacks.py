from typing import Dict

from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor


class MetricsModelCheckpoint(ModelCheckpoint):
    """Model checkpoint that replaces all "/" in the metric name with "_"."""

    def _format_checkpoint_name(
        self,
        filename: str | None,
        metrics: Dict[str, Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        filename = super()._format_checkpoint_name(
            filename, metrics, prefix, auto_insert_metric_name
        )
        filename = filename.replace("/", "_")
        return filename
