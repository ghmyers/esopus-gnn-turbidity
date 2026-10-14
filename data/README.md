# Data

Data for **Myers et al., *Explainable graph neural networks for sediment source tracing***
by George H. Myers, Kristen L. Underwood, Donna M. Rizzo, John T. Kemper, Dany Davis, Shaurya Swami, Scott D. Hamshaw, Eric D. Roy, Andrew W. Schroth.

Code: https://github.com/ghmyers/esopus-gnn-turbidity. Data deposit: [doi:10.5281/zenodo.23352829](https://doi.org/10.5281/zenodo.23352829).
This file is both the repository's data guide and the README of the Zenodo deposit.

## Zenodo deposit

Unzip the deposit into the root of the code repository; the paths below are relative to it.

| Path | Size | Content |
|---|---|---|
| `data/processed/esopus_graph_features.pkl` | 0.6 GB | **training graph**: 57 nodes, 14 hourly input features and turbidity, 2016-10-01 to 2024-10-20 (all paper results use this) |
| `data/raw/usgs/<site_no>.csv` | 0.3 GB | USGS 15-min discharge and/or turbidity, 29 gages |
| `data/raw/aorc/node_<id>.csv` | 0.7 GB | NOAA AORC v1.1 hourly forcings at the 57 nodes, 2014-01-01 to 2024-12-30 |
| `data/raw/streamstats/streamstats_attributes.csv` | <1 MB | USGS StreamStats basin characteristics (discharge filling) |
| `data/raw/geospatial/subwatersheds/` | <1 MB | sub-watershed polygons (Figure 5 map) |
| `outputs/models/<run>.pt`, `outputs/scalers/<run>_scaler.pkl` | <1 MB | trained weights and fitted scalers of the six paper models |

The six models (`configs/`): `gwn_full` (full model), `gwn_topology_only` (ablation without the
learned adjacency), and `gwn_ungaged_{SC2,WOOD1,EC12,BK1}` (leave-one-out folds).

### Sources

**USGS NWIS instantaneous values** (https://waterdata.usgs.gov/nwis), 15-min data from the NWIS
IV web service, 2014-01-01 to 2024-10-20 (shorter at some gages); timestamps in UTC.

* Parameter `00060` discharge (ft³ s⁻¹) → column `Q (cfs)`; parameter `63680` turbidity (FNU,
  formazin nephelometric units) → column `turbidity (FNU)`.
* 29 gages: 13 with discharge and turbidity, the Shandaken Tunnel inflow (01362230, discharge
  only) and 15 with turbidity only. The site numbers and graph nodes are
  listed in `data/node_metadata.csv` (columns `site_no`, `node`, `site_code`).
* Values are as published by USGS (approved, or provisional for recent data). Values USGS flags as
  unusable (e.g. ice, equipment malfunction) are stored as −999999; they, and any hour with a
  negative mean, are removed when the graph is built (`src/esopus/graph/assemble.py`).

**NOAA Analysis of Record for Calibration (AORC) v1.1** — 1-km hourly gridded forcings, public
bucket `s3://noaa-nws-aorc-v1-1-1km` (https://registry.opendata.aws/noaa-nws-aorc/). For each node,
the eight variables below were taken from the nearest grid cell:

| Column | AORC variable | Unit |
|---|---|---|
| `Total Precipitation` | `APCP_surface` | kg m⁻² (mm) per hour |
| `Air Temperature` | `TMP_2maboveground` | K |
| `Specific Humidity` | `SPFH_2maboveground` | kg kg⁻¹ |
| `Downward Long-Wave Radiation Flux` | `DLWRF_surface` | W m⁻² |
| `Downward Short-Wave Radiation Flux` | `DSWRF_surface` | W m⁻² |
| `Pressure` | `PRES_surface` | Pa |
| `U-Component of Wind` | `UGRD_10maboveground` | m s⁻¹ |
| `V-Component of Wind` | `VGRD_10maboveground` | m s⁻¹ |

The files hold the values exactly as used to build the graph, with the timestamps they were read
with (treated as US-Eastern local time and converted to UTC in `scripts/02_build_graph.py`; see
`docs/KNOWN_ISSUES.md`). Consequently (i) the hour repeated at each autumn daylight-saving change
is absent (it was dropped by that conversion), (ii) the 24 hours of 2024-06-18 are missing in the
source, and (iii) surface pressure below 87,500 Pa (266 values) is already blank. Only the
V-component of wind is among the 14 model inputs.

**USGS StreamStats** (https://streamstats.usgs.gov), New York: drainage area (`DRNAREA`), storage
(`STORAGE`), lag factor (`LAGFACTOR`) and mean annual runoff (`MAR`) per node, used with the
regional flood-frequency equations to choose the drainage-area exponent when discharge is
transferred to ungaged nodes (`src/esopus/graph/discharge_fill.py`).

**Sub-watershed polygons** for the twelve Upper Esopus sub-watersheds (shapefiles): **TODO source**.

## Committed to the repository (small tables)

| File | Content |
|---|---|
| `node_metadata.csv` | 57 graph nodes: name, site code, USGS site number, sub-basin, turbidity/discharge flags, position (EPSG:5070), drainage area, channel slope (ft/mi), NHD flowline length, bankfull width |
| `network_edges.csv` | 56 directed river-network edges (upstream → downstream) with NHDPlus attributes |
| `node_catchment_attributes.csv` | NHDPlus catchment attributes per node (USGS NLDI), static inputs before feature selection |
| `node_id_map.csv` | earlier node numbering → final node ID (used for the StreamStats lookup) |
| `pre_feature_columns.txt` | column order of the pre-feature graph |
| `selected_features.txt` | the 14 model inputs in the column order of the trained checkpoints |
| `stream_reaches_sfi_2022.csv` | NYC DEP Stream Feature Inventory (2022) reaches and their graph nodes (Figure 5) |

The network (NHD medium-resolution flowlines and gage locations, with a few manual edits; see
`src/esopus/graph/network.py`) and the static attributes are committed as tables because the
original GIS edits were not scripted.

## Layout after setup

```
data/
  raw/        usgs/, aorc/, streamstats/, geospatial/subwatersheds/        (deposit)
  processed/  esopus_graph_features.pkl                                    (deposit)
              network.pkl, esopus_graph.pkl                                (scripts/01-02)
outputs/
  models/, scalers/                                                        (deposit)
  predictions/, interpretation/, analyses/                                 (scripts/02, 11-16)
figures/      main/, supplementary/                                        (scripts/20)
```

`data/raw/*`, `data/processed/*`, `outputs/` and `figures/` are git-ignored.
