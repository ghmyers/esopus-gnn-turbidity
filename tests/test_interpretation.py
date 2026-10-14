"""Storm events, source tracing, feature groups, configs."""

import numpy as np
import pandas as pd
import pytest

from esopus.config import load_config
from esopus.features.engineering import antecedent_features
from esopus.interpretation.events import detect_storm_events
from esopus.interpretation.feature_groups import assign_feature_group


def test_storm_events_are_buffered_and_merged():
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    y = pd.Series(10.0, index=idx)
    y.iloc[50:55] = 300
    y.iloc[80:82] = 400   # within 2 x 24 h buffer of the first event -> merged
    y.iloc[180:182] = 500
    ev = detect_storm_events(y)
    assert len(ev) == 2
    assert ev[0][0] == idx[50] - pd.Timedelta(hours=24)
    assert ev[1][1] == idx[-1]  # clipped to the record


def test_feature_groups():
    assert assign_feature_group("SSP") == "SSP"
    assert assign_feature_group("Q_cfs_diff_now_to_197h") == "Storm Antecedent"
    assert assign_feature_group("Q_cfs_ratio_now_to_17h") == "Storm Antecedent"
    assert assign_feature_group("Q_cfs_ant_5753h_max") == "Seasonal+ Antecedent"


def test_antecedent_features_definitions():
    idx = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
    df = pd.DataFrame({"Q (cfs)": np.arange(300, dtype=float)}, index=idx)
    out = antecedent_features(df, lags=(17, 197))
    assert out["Q_cfs_diff_now_to_17h"].iloc[-1] == 17
    assert out["Q_cfs_cumsum_17h"].iloc[-1] == sum(range(283, 300))


@pytest.mark.parametrize("path", ["configs/graphwavenet_full.yaml", "configs/graphwavenet_topology_only.yaml",
                                  "configs/ungaged/SC2.yaml", "configs/ungaged/WOOD1.yaml",
                                  "configs/ungaged/EC12.yaml", "configs/ungaged/BK1.yaml"])
def test_paper_configs_load(path):
    cfg = load_config(path)
    assert cfg.data.num_timesteps_in == 17 and cfg.model.hidden_channels == 16
