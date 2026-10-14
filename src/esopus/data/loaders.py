"""DataLoaders for the sliding-window datasets.

``SlidingWindowDataset`` yields ``x`` as (N, F, T_in). The GraphWaveNet
implementation in ``tsl`` expects a time-first batch (B, T_in, N, F), so the
collate function stacks samples and moves the time axis forward. Targets are
returned as (B, N, T_out).

Windows are never shuffled: batches stay in chronological order, exactly as in
the published runs.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def collate_time_first(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack (N, F, T_in) windows into x (B, T_in, N, F) and y (B, N, T_out)."""
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0).permute(0, 3, 1, 2)  # (B, N, F, T) -> (B, T, N, F)
    y = torch.stack(ys, dim=0)
    return x, y


def make_loader(dataset: Dataset, batch_size: int, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
    """Chronological (unshuffled) DataLoader with the time-first collate."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
        collate_fn=collate_time_first,
    )


def build_loaders(cfg, train_dataset: Dataset, test_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    """Training loader (drops the last partial batch) and test loader (keeps every window)."""
    lc = cfg.loader
    train_loader = make_loader(train_dataset, lc.batch_size, drop_last=lc.drop_last_train, num_workers=lc.num_workers)
    test_loader = make_loader(test_dataset, lc.batch_size, drop_last=False, num_workers=lc.num_workers)
    return train_loader, test_loader
