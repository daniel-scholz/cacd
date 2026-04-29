import numpy as np
import pandas as pd
import torch


def map_col_to_int(series: pd.Series) -> tuple[torch.Tensor, np.ndarray]:
    """Map a column to integers."""
    unique_vals = np.sort(series[~series.isna()].unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    series_np = series.map(mapping).to_numpy(copy=True)

    return torch.from_numpy(series_np), series.to_numpy()
