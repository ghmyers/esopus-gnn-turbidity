# Explainable graph neural networks for sediment source tracing in the Upper Esopus Creek

Code accompanying **Myers et al., *Explainable Graph Neural Networks for Sediment Source Tracing in Rivers*
(in review)**.

Authors: George H. Myers, Kristen L. Underwood, Donna M. Rizzo, John T. Kemper, Dany Davis, Shaurya Swami, Scott D. Hamshaw, Eric D. Roy, Andrew W. Schroth.

Data and trained models: [doi:10.5281/zenodo.23352829](https://doi.org/10.5281/zenodo.23352829).

A spatiotemporal graph neural network (GraphWaveNet) predicts hourly turbidity at 57 nodes of the
Upper Esopus Creek river network (Catskill Mountains, NY), 28 of which have turbidity sensors. The
trained model is then used to

* **trace sediment sources** by comparing observed and predicted tributary turbidity loads,
  over the full record and during 33 turbidity events;
* **explain predictions** with Integrated Gradients (which features drive turbidity at each site
  during turbidity events and between them, and how this changes with drainage area);
* **interpret the learned adaptive adjacency** between sub-basins.

It is validated with a topology-only ablation and a leave-one-out "ungaged node" experiment.

---

## Repository layout

```
configs/            YAML configs for the six paper models (full, topology-only, 4 ungaged folds)
data/               small committed tables (network, node metadata); data/README.md describes the data
docs/               pipeline.md (data flow, conventions)
notebooks/          01 exploratory data analysis (lag selection), 02 results walkthrough
scripts/            numbered pipeline steps (each has --help and documents inputs/outputs)
src/esopus/         the Python package (see its __init__ for an overview of subpackages)
tests/              pytest suite on a small synthetic graph
tools/              maintainer tool that assembled the Zenodo deposit
```

## Installation

```bash
git clone https://github.com/ghmyers/esopus-gnn-turbidity.git
cd esopus-gnn-turbidity
conda env create -f environment.yml        # Python 3.11, PyTorch 2.5.1 (CUDA 12.4), PyG 2.7, tsl 0.9.5
conda activate esopus-gnn
pip install -e .                           # editable install: paths resolve relative to this clone
pytest                                     # ~1 min
```

A CUDA GPU is recommended for training and Integrated Gradients; everything else runs on a CPU.

## Data

Download the data deposit from Zenodo ([doi:10.5281/zenodo.23352829](https://doi.org/10.5281/zenodo.23352829)) and unzip it into the repository root, so
that `data/raw/`, `data/processed/` and `outputs/` are filled. It contains the training graph,
the raw USGS and AORC files, StreamStats basin characteristics, sub-watershed polygons and the six
trained models. See [`data/README.md`](data/README.md) for the contents and data sources.

## Reproduce the figures from the trained models

```bash
make graph                                     # 01-02: rebuild the graph from the raw files; discharge regressions (Figs. S1-S2)
make evaluate                                  # 11: metrics + hourly predictions, six models
make source-tracing                            # 13: tributary load shares (Fig. 4)
make interpret                                 # 12: Integrated Gradients, full model (Fig. S7)
make figures                                   # 20: all figures
```

These reproduce the archived models' test metrics exactly (median KGE 0.43, NSE 0.50, MAE 4.54
FNU, RMSE 15.0 FNU over 28 gaged nodes). `make analyses` adds the ablation, ungaged-validation and
attribution-stability results; `notebooks/02_results_walkthrough.ipynb` summarises them.

## Pipeline

| Step | Script                                  | Output                                                                                       |
| ---- | --------------------------------------- | -------------------------------------------------------------------------------------------- |
| 01   | `01_build_network.py`                   | river-network graph (57 nodes, 56 edges)                                                     |
| 02   | `02_build_graph.py`                     | hourly graph with gap-filled discharge and stream power; discharge regressions (Figs. S1–S2) |
| 03   | `03_engineer_features.py`               | antecedent-discharge features, random-forest selection → training graph                      |
| 10   | `10_train.py --config …`                | trained model (resumable; `--debug` for a 2-epoch test)                                      |
| 11   | `11_evaluate.py --config …`             | per-node metrics and hourly predictions                                                      |
| 12   | `12_integrated_gradients.py --config …` | Integrated Gradients over the test period                                                    |
| 13   | `13_source_tracing.py --config …`       | observed vs. predicted tributary load shares                                                 |
| 14   | `14_ablation_topology_only.py`          | full vs. topology-only comparison                                                            |
| 15   | `15_ungaged_validation.py`              | leave-one-out retraining for 4 gages                                                         |
| 16   | `16_attribution_stability.py`           | attribution-vs-drainage-area fits across the six models                                      |
| 20   | `20_make_figures.py`                    | figures                                                                                      |

`make all` runs everything from the raw files, including retraining. Step 03 is not needed to
reproduce the paper (the deposit contains its output) and needs ~24 GB RAM. Retrained models match
the archived checkpoints only up to GPU non-determinism. See `docs/pipeline.md` for details.

## Figures

| Figure                                                | Command                            | File                                                                |
| ----------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------- |
| Fig. 1 study area, Fig. 2 architecture                | GIS / graphics software            | —                                                                   |
| Fig. 3 prediction skill (KGE map, test-period series) | `20_make_figures.py --figure fig3` | `figures/main/fig3_performance_maps_timeseries.png`                 |
| Fig. 4 source tracing                                 | `--figure fig4`                    | `figures/main/fig4_source_tracing.png`                              |
| Fig. 5 learned adaptive adjacency (chord diagram)     | `--figure fig5`                    | `figures/main/fig5_adjacency_chord_sensor_density.png`              |
| Fig. S1 discharge regressions vs. Stony Clove (SC2)   | `--figure figS1`                   | `figures/supplementary/figS1_discharge_regressions_stony_clove.png` |
| Fig. S2 discharge regressions vs. Coldbrook (EC1)     | `--figure figS2`                   | `figures/supplementary/figS2_discharge_regressions_coldbrook.png`   |
| Fig. S4 per-node test metrics                         | `--figure figS4`                   | `figures/supplementary/figS4_node_metrics_barchart.png`             |
| Fig. S5 test metrics vs. drainage area                | `--figure figS5`                   | `figures/supplementary/figS5_metrics_vs_drainage_area.png`          |
| Fig. S7 IG attribution vs. drainage area              | `--figure figS7`                   | `figures/supplementary/figS7_attribution_vs_drainage_area.png`      |

## Citation

Data and trained models: Zenodo, [doi:10.5281/zenodo.23352829](https://doi.org/10.5281/zenodo.23352829).

## License

MIT — see [`LICENSE`](LICENSE).
