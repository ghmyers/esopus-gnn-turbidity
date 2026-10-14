"""Random-forest feature ranking used to choose the GNN's input features.

All candidate features (after the correlation filter) plus every engineered
antecedent feature are stacked over gaged node-hours. A random forest is fit
on the training period and features are ranked by impurity importance; the
top ``n`` become the model inputs, ordered from least to most important (the
column order the published model was trained with).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from esopus.features.engineering import ANTECEDENT_LAGS, antecedent_features

RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=0)


@dataclass
class RankingResult:
    importances: pd.Series          # feature -> importance, descending
    model: RandomForestRegressor
    test_scores: dict

    def top(self, n: int) -> List[str]:
        """Top-``n`` features ordered by ascending importance (model column order)."""
        return list(self.importances.iloc[:n][::-1].index)


def build_training_table(G) -> pd.DataFrame:
    """Stack every node's features + engineered features + target; keep rows with observed y.

    All nodes are stacked (not only gaged ones) so the constant-column check
    and the row order of the stacked table match the original analysis — the
    random forest's bootstrap samples depend on row order.
    """
    frames = []
    for node, d in G.nodes(data=True):
        df = d["x"].copy()
        df["y"], df["node"] = d["y"], node
        frames.append(df)
    table = pd.concat(frames)
    variances = table.drop(columns=["node"]).var(numeric_only=True)
    table = table.drop(columns=[c for c, v in variances.items() if v == 0])  # constant across nodes and time
    enriched = pd.concat([antecedent_features(table[table["node"] == n], ANTECEDENT_LAGS)
                          for n in table["node"].unique()]).sort_index()
    return enriched.dropna(subset=["y"])


def rank_features(table: pd.DataFrame, split_date: str) -> RankingResult:
    """Fit the random forest on rows before ``split_date`` and rank features."""
    feats = [c for c in table.columns if c not in ("node", "y")]
    train = table.index < pd.Timestamp(split_date, tz=table.index.tz)
    sx, sy = MinMaxScaler().fit(table.loc[train, feats]), MinMaxScaler().fit(table.loc[train, ["y"]])

    def scaled(mask):
        X = sx.transform(table.loc[mask, feats].fillna(0).to_numpy()).astype(np.float32)
        y = sy.transform(table.loc[mask, ["y"]].to_numpy()).ravel().astype(np.float32)
        return pd.DataFrame(X, columns=feats), y

    X_tr, y_tr = scaled(train)
    X_te, y_te = scaled(~train)
    rf = RandomForestRegressor(**RF_PARAMS).fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    scores = {"test_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
              "test_mae": float(mean_absolute_error(y_te, pred)), "test_r2": float(r2_score(y_te, pred))}
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False, kind="stable")
    return RankingResult(imp, rf, scores)
