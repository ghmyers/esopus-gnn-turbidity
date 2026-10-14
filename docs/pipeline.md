# Pipeline reference

Every step is a script in `scripts/` with a header listing its inputs and
outputs. Steps 01–03 build the data. Steps 10–16 train, evaluate and
interpret. Step 20 draws the figures. Paths are relative to the repository
root (`data/`, `outputs/` and `figures/` can be relocated with
`ESOPUS_DATA_DIR`, `ESOPUS_OUTPUT_DIR` and `ESOPUS_FIGURES_DIR`). Inputs marked
(Z) are in the Zenodo deposit (`data/README.md`).

```
 data/raw/usgs/*.csv (Z) ─────────┐
 data/raw/aorc/node_*.csv (Z) ────┤
 node/edge tables ─01─► network ──┼─02─► processed/esopus_graph.pkl ─03─► processed/esopus_graph_features.pkl (Z)
 data/raw/streamstats (Z) ────────┘  │  (145 features, 2014–2024)          (14 features, 2016-10-01 → 2024-10-20)
                                     └─► analyses/discharge_regressions/ (Figs. S1–S2)         │
                     ┌──────────────────────────────────────────────────────────────────────────┘
                     ▼
   10 train ──► outputs/models/<run>.pt, outputs/scalers/<run>_scaler.pkl (Z)
   11 evaluate ──► outputs/predictions/<run>/ (metrics, hourly predictions)          Figs. 3, S4, S5
   12 Integrated Gradients ──► outputs/interpretation/<run>/                         Fig. S7
   13 source tracing ──► outputs/analyses/source_tracing/<run>/                      Fig. 4
   14 ablation, 15 ungaged validation, 16 attribution stability ──► outputs/analyses/
   20 figures ──► figures/main, figures/supplementary                                (Fig. 5 reads the checkpoint)
```

## Node and edge conventions

* 57 nodes: NHD medium-resolution flowline junctions plus monitoring gages.
  IDs run 0–65 with gaps (36, 37, 46, 47, 56–60 unused). `data/node_metadata.csv`
  lists name, site code, USGS site number, sub-basin and position (EPSG:5070).
* 56 directed edges point downstream (`data/network_edges.csv`). The outlet is
  node 33 (Esopus Creek at Coldbrook, USGS 01362500). Node 41 is the Shandaken
  Tunnel inflow from the Schoharie Reservoir.
* 28 nodes have turbidity records used for scoring. 14 nodes have measured
  discharge (`esopus.constants.DISCHARGE_NODES`).
* `data/node_id_map.csv` maps node IDs from an earlier numbering. It is used
  only to look up StreamStats basin characteristics.

## Graph contents

`esopus_graph.pkl` and `esopus_graph_features.pkl` are pickled
`networkx.DiGraph` objects:

| Attribute | Content |
|---|---|
| `G.nodes[n]["x"]` | `DataFrame` (hours × features), UTC index shared by all nodes |
| `G.nodes[n]["y"]` | `Series` of turbidity (FNU); NaN where not measured |
| `G.nodes[n]["pos"]` | (easting, northing), EPSG:5070 |
| `G.graph["discharge_fill_flag"]` | `DataFrame` (hours × nodes), 1 = discharge gap-filled |

## Model input features (in column order)

| Feature | Meaning |
|---|---|
| `Q_cfs_ant_5753h_max`, `Q_cfs_ant_5753h_std` | max / std of discharge in a 575-h window ending 5,753 h earlier |
| `V-Component of Wind` | AORC 10-m northward wind (m s⁻¹) |
| `Q_cfs_ratio_now_to_{2648,197,17}h` | Q(t) / mean antecedent Q at that lag |
| `Q_cfs_diff_now_to_{5753,7387,2648,197,17}h` | Q(t) − Q(t − lag) |
| `Q (cfs)` | hourly discharge (gap-filled) |
| `Q_cfs_range_std` | std of discharge over the last 17 h |
| `SSP` | specific stream power ρg·Q·S / w (W m⁻²) |

## Step details

| Step | What it does | Key module |
|---|---|---|
| 01 | Published topology from the committed tables; acyclicity / outlet checks | `graph/network.py` |
| 02 | Hourly merge of USGS and AORC, static attributes, discharge regressions (saved for Figs. S1–S2), discharge fill (regressions + drainage-area scaling), 24-h interpolation, SSP, pressure screen | `graph/assemble.py`, `graph/discharge_fill.py`, `graph/attributes.py` |
| 03 | Correlation filter (\|r\| ≥ 0.8), antecedent features, random-forest ranking, trim to outlet record | `features/` |
| 10 | GraphWaveNet training, Adam + cosine LR, masked MSE, early stopping, per-epoch resume | `training/trainer.py` |
| 11 | Test metrics and hourly predictions (train + test) with correct timestamps | `evaluation/predictions.py` |
| 12 | Integrated Gradients for daily test windows (50 Gauss-Legendre steps) | `interpretation/integrated_grads.py` |
| 13 | Tributary load shares, observed vs. predicted: full record and 33 turbidity events | `interpretation/source_tracing.py` |
| 14 | Full vs. topology-only model: skill, source tracing, attribution vs. drainage area | `scripts/14_*` |
| 15 | Leave-one-out retraining for SC2, WOOD1, EC12, BK1 | `data/ungaged.py` |
| 16 | Attribution-vs-drainage-area fits for all six models, spread across retrainings | `scripts/16_*` |
| 20 | Figures 3–5, S1, S2, S4, S5, S7 | `plotting/` |

## Run names

| Config | `run_name` |
|---|---|
| `configs/graphwavenet_full.yaml` | `gwn_full` |
| `configs/graphwavenet_topology_only.yaml` | `gwn_topology_only` |
| `configs/ungaged/<code>.yaml` | `gwn_ungaged_<code>` |

## Compute

| Step | Resources (paper hardware) |
|---|---|
| 02 | ~2 min |
| 03 | ~24 GB RAM, ~7 min on 16 cores |
| 10 | ~2.5 min per epoch on an RTX 3050 Ti (4 GB); 75 epochs maximum |
| 12 | a few GPU-hours per model for 28 nodes (`--ig-batch-size 50`: ~10–20 min) |
| 11, 13–16, 20 | minutes |
