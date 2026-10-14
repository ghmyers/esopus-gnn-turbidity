"""Gap-filling of hourly discharge at every node.

Discharge drives specific stream power and every engineered feature, so it is
filled everywhere (filled hours are flagged and excluded from source tracing):

1. **Regression fill of Coldbrook** (outlet, node 33) from Stony Clove
   (node 30): ordinary least squares on overlapping hours.
2. **Regression fill of Stony Clove tributaries** (nodes 31, 35, 63) from
   node 30.
3. **Regression fill of all other gaged nodes** from Coldbrook.
4. **Drainage-area scaling for the remaining gaps** (mostly ungaged nodes):
   ``Q_node = Q_donor * (DA_node / DA_donor) ** b``, where the donor is the
   first downstream node in :data:`DA_DONORS` with discharge (node 45 always
   uses node 30), and the exponent ``b`` is the drainage-area exponent of the
   USGS StreamStats flood-frequency equation whose return-period discharge is
   closest to the donor's current flow.

The Shandaken Tunnel (node 41, a managed inter-basin transfer) is never used
as a regression target or donor. Regressed and scaled values are clipped at 0.
Each node's filled hours are recorded in ``G.graph["discharge_fill_flag"]``
(T x N DataFrame, 1 = filled, 0 = measured).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from esopus.paths import REPO_ROOT

Q = "Q (cfs)"
COLDBROOK, STONY_CLOVE, SHANDAKEN = 33, 30, 41
STONY_CLOVE_TARGETS = (31, 35, 63)
DA_DONORS = {26, 27, 28, 29, 30, 31, 32, 33, 34, 44}
FORCED_DONORS = {45: 30}
NODE_ID_MAP_CSV = REPO_ROOT / "data" / "node_id_map.csv"

#: StreamStats regional flood-frequency equations (drainage-area exponents and
#: full equations), Q_T = a * DA^b * (STORAGE+5)^c * (LAGFACTOR+1)^d * MAR^e
RI_EQUATIONS = {
    "Q_1_25": (32.2, 0.943, -0.943, -0.294, 0.588),
    "Q_1_5": (32.5, 0.936, -0.962, -0.306, 0.672),
    "Q_2": (33.3, 0.928, -0.976, -0.318, 0.759),
    "Q_5": (37.6, 0.914, -0.985, -0.356, 0.905),
    "Q_10": (41.6, 0.909, -0.977, -0.385, 0.968),
    "Q_25": (46.5, 0.905, -0.958, -0.418, 1.029),
    "Q_50": (49.7, 0.902, -0.939, -0.441, 1.068),
    "Q_100": (52.3, 0.900, -0.918, -0.461, 1.104),
    "Q_200": (54.3, 0.898, -0.894, -0.479, 1.138),
    "Q_500": (55.9, 0.895, -0.860, -0.500, 1.183),
}


def _q(G, n) -> Optional[pd.Series]:
    x = G.nodes[n].get("x")
    return x[Q] if x is not None and Q in x else None


def _dedup(s: pd.Series) -> pd.Series:
    return s[~s.index.duplicated(keep="first")]


def fit_regressions(G, reference: int, exclude: Set[int] = frozenset({SHANDAKEN}), min_overlap: int = 30) -> pd.DataFrame:
    """OLS of every other node's discharge on ``reference`` discharge (overlapping hours)."""
    ref = _dedup(_q(G, reference)).dropna()
    rows = []
    for n in G.nodes:
        s = _q(G, n)
        if n == reference or n in exclude or s is None:
            continue
        s = _dedup(s).dropna()
        idx = ref.index.intersection(s.index)
        xv, yv = ref.loc[idx].to_numpy(), s.loc[idx].to_numpy()
        ok = ~(np.isnan(xv) | np.isnan(yv))
        row = {"node": n, "n_overlap": int(ok.sum()), "slope": np.nan, "intercept": np.nan, "r_squared": np.nan}
        if ok.sum() >= min_overlap:
            fit = stats.linregress(xv[ok], yv[ok])
            row.update(slope=fit.slope, intercept=fit.intercept, r_squared=fit.rvalue ** 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r_squared", ascending=False)


def measured_discharge(G, exclude: Set[int] = frozenset({SHANDAKEN})) -> pd.DataFrame:
    """Hourly gaged discharge (T x node) before any filling: the data behind Figures S1-S2."""
    cols = {n: _dedup(_q(G, n)) for n in sorted(G.nodes)
            if n not in exclude and _q(G, n) is not None and _q(G, n).notna().any()}
    df = pd.DataFrame(cols)
    df.index.name = "datetime"
    return df


def _fill_from(G, node, donor_q: pd.Series, slope, intercept, flags) -> None:
    target = _dedup(_q(G, node))
    gaps = target.isna()
    pred = (slope * donor_q + intercept).clip(lower=0)
    filled = target.copy()
    filled.loc[gaps] = pred.reindex(target.index).loc[gaps]
    idx = G.nodes[node]["x"].index
    G.nodes[node]["x"][Q] = filled.reindex(idx)
    flags[node] = gaps.reindex(idx, fill_value=False).astype(int)


def streamstats_table(path: Path) -> pd.DataFrame:
    """StreamStats basin characteristics with return-period discharges, indexed by node.

    Note: the ``NODEID`` column is mapped through the *inverse* of the
    old->new node-ID table, exactly as in the original analysis (see
    docs/KNOWN_ISSUES.md); this only selects which flood-frequency exponents
    are used at each node.
    """
    ss = pd.read_csv(path)
    id_map = pd.read_csv(NODE_ID_MAP_CSV)
    new_to_old = dict(zip(id_map["node"], id_map["old_node"]))
    ss["node"] = ss["NODEID"].map(new_to_old)
    ss = ss.dropna(subset=["node"]).astype({"node": int})
    for col, (a, b, c, d, e) in RI_EQUATIONS.items():
        ss[col] = a * ss["DRNAREA"] ** b * (ss["STORAGE"] + 5) ** c * (ss["LAGFACTOR"] + 1) ** d * ss["MAR"] ** e
    return ss.set_index("node")


def _first_downstream_donor(G, node) -> Optional[int]:
    H = G.subgraph({node} | nx.descendants(G, node))
    try:
        order = [n for n in nx.topological_sort(H) if n != node]
    except nx.NetworkXUnfeasible:
        order = [n for n in nx.bfs_tree(G, node) if n != node]
    for n in order:
        q = _q(G, n)
        if n in DA_DONORS and n != SHANDAKEN and q is not None and not q.isna().all():
            return n
    return None


def fill_discharge(G: nx.DiGraph, streamstats_csv: Path) -> pd.DataFrame:
    """Run steps 1-4 in place; store and return the fill-flag table."""
    flags: Dict[int, pd.Series] = {}
    sc, cb = fit_regressions(G, STONY_CLOVE), fit_regressions(G, COLDBROOK)

    # 1. Coldbrook from Stony Clove; 2. Stony Clove tributaries from Stony Clove
    sc_q = _dedup(_q(G, STONY_CLOVE))
    for node in (COLDBROOK, *STONY_CLOVE_TARGETS):
        r = sc[sc["node"] == node]
        if not r.empty:
            _fill_from(G, node, sc_q, r["slope"].iloc[0], r["intercept"].iloc[0], flags)

    # 3. every node with a Coldbrook regression
    cb_q = _dedup(_q(G, COLDBROOK))
    for _, r in cb.iterrows():
        node = int(r["node"])
        if np.isnan(r["slope"]) or _q(G, node) is None:
            continue
        if not _dedup(_q(G, node)).isna().any():
            flags[node] = pd.Series(0, index=G.nodes[node]["x"].index, dtype=int)
            continue
        _fill_from(G, node, cb_q, r["slope"], r["intercept"], flags)

    # 4. drainage-area scaling
    ss = streamstats_table(streamstats_csv)
    ri_cols = list(RI_EQUATIONS)
    exponent = {k: v[1] for k, v in RI_EQUATIONS.items()}
    todo = [n for n in G.nodes if n != SHANDAKEN and (_q(G, n) is None or _q(G, n).isna().any())]
    for node in todo:
        x = G.nodes[node]["x"]
        donor = FORCED_DONORS.get(node) or _first_downstream_donor(G, node)
        if donor is None:
            print(f"[discharge] node {node}: no donor found")
            continue
        da_node = float(x["DA_sqmi"].dropna().iloc[0])
        da_donor = float(G.nodes[donor]["x"]["DA_sqmi"].dropna().iloc[0])
        q_donor = _dedup(_q(G, donor))
        if node in ss.index:
            ri = ss.loc[node, ri_cols].to_numpy(dtype=float)
            qd = q_donor.dropna()
            nearest = [ri_cols[int(np.argmin(np.abs(ri - q)))] for q in qd.to_numpy()]
            b = pd.Series([exponent[c] for c in nearest], index=qd.index).reindex(q_donor.index)
            pred = (q_donor * (da_node / da_donor) ** b).clip(lower=0)
        else:
            pred = (q_donor * (da_node / da_donor) ** float(np.median(list(exponent.values())))).clip(lower=0)
        current = x[Q] if Q in x else pd.Series(np.nan, index=x.index)
        current = _dedup(current).reindex(x.index)
        gaps = current.isna()
        current.loc[gaps] = pred.reindex(x.index).loc[gaps]
        G.nodes[node]["x"][Q] = current
        flags[node] = gaps.astype(int)

    index = G.nodes[next(iter(G.nodes))]["x"].index
    flag = pd.DataFrame({n: flags[n].reindex(index, fill_value=0) if n in flags else pd.Series(0, index=index)
                         for n in sorted(G.nodes)})
    flag.index.name = "datetime"
    G.graph["discharge_fill_flag"] = flag
    return flag
