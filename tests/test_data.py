"""Data pipeline: window/target alignment, scaling, masking."""

import numpy as np
import pandas as pd
import torch

from esopus.data import (
    SlidingWindowDataset, TurbidityScaler, mask_node_train_split, prepare_data, window_target_dates,
)


def test_window_targets_are_one_step_after_inputs():
    T, t_in = 50, 17
    X = np.arange(T, dtype=np.float32).reshape(1, 1, T)
    y = np.arange(T, dtype=np.float32).reshape(1, T) + 1000
    ds = SlidingWindowDataset(X, y, t_in, 1)
    dates = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    target_dates = window_target_dates(dates, t_in, n_windows=len(ds))
    for i in (0, 5, len(ds) - 1):
        x_win, y_win = ds[i]
        assert x_win[0, 0, -1].item() == i + t_in - 1           # last input hour
        assert y_win[0, 0].item() == 1000 + i + t_in             # target = next hour
        assert target_dates[i] == dates[i + t_in]


def test_window_target_dates_extrapolate_past_end():
    dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    out = window_target_dates(dates, 5, starts=[5])
    assert out[0] == dates[-1] + pd.Timedelta(hours=1)


def test_scaler_round_trip_and_sentinel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 2, 20)).astype(np.float32)
    y = np.abs(rng.normal(size=(3, 20))).astype(np.float32) * 100
    y[0, :5] = np.nan
    s = TurbidityScaler(eps=1e-5).fit(X, y, [0, 1, 2])
    ys = s.transform_y(y)
    assert np.all(ys[0, :5] == 0.0) and np.all(ys[~np.isnan(y)] > 0)
    back = s.inverse_transform_y(ys)
    np.testing.assert_allclose(back[~np.isnan(y)], y[~np.isnan(y)], rtol=1e-4)
    assert np.isnan(back[0, :5]).all()


def test_mask_node_train_split(graph):
    n_test = int(graph.nodes[2]["y"].loc["2023-08-13":].notna().sum())
    report = mask_node_train_split(graph, 2, "2023-08-13")
    y = graph.nodes[2]["y"]
    assert y.loc[: "2023-08-12"].isna().all()
    assert report["n_test_observed"] == n_test == int(y.loc["2023-08-13":].notna().sum())


def test_prepare_data_alignment(cfg):
    data = prepare_data(cfg, torch.device("cpu"), verbose=False)
    assert data.node_ids == [0, 1, 2, 3, 4]
    assert data.edge_index.shape == (2, 4)
    obs = data.observed_targets("test")
    _, y = data.test_dataset[0]
    expected = data.scaler.inverse_transform_y(y.numpy()[:, 0])
    np.testing.assert_allclose(expected[[2, 4]], obs[0, [2, 4]], rtol=1e-4)
    assert data.target_dates("test")[0] == data.test_dates[cfg.data.num_timesteps_in]
