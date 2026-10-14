# Reproduce the paper. Each script documents its inputs/outputs in its header (--help).
# Start by unzipping the Zenodo deposit into the repository root (see data/README.md), then:
#     make graph evaluate source-tracing interpret figures

PY ?= python
CONFIGS = configs/graphwavenet_full.yaml configs/graphwavenet_topology_only.yaml \
          configs/ungaged/SC2.yaml configs/ungaged/WOOD1.yaml configs/ungaged/EC12.yaml configs/ungaged/BK1.yaml
UNGAGED = configs/ungaged/SC2.yaml configs/ungaged/WOOD1.yaml configs/ungaged/EC12.yaml configs/ungaged/BK1.yaml

.PHONY: graph features train evaluate source-tracing interpret analyses figures all test

graph:           ## 01-02: network + hourly graph from the raw USGS/AORC files; discharge regressions (Figs. S1-S2)
	$(PY) scripts/01_build_network.py
	$(PY) scripts/02_build_graph.py

features:        ## 03: feature engineering + random-forest selection -> training graph (~24 GB RAM; in the deposit)
	$(PY) scripts/03_engineer_features.py

train:           ## 10: retrain the full and ablation models (ungaged folds: scripts/15 without --skip-training)
	$(PY) scripts/10_train.py --config configs/graphwavenet_full.yaml
	$(PY) scripts/10_train.py --config configs/graphwavenet_topology_only.yaml

evaluate:        ## 11: metrics + hourly predictions for the six models (Figs. 3, S4, S5)
	$(PY) scripts/11_evaluate.py --config $(CONFIGS)

source-tracing:  ## 13: observed vs. predicted tributary load shares (Fig. 4)
	$(PY) scripts/13_source_tracing.py --config configs/graphwavenet_full.yaml configs/graphwavenet_topology_only.yaml

interpret:       ## 12: Integrated Gradients for the full model (Fig. S7); a few GPU-hours (add --ig-batch-size 50 for ~20 min)
	$(PY) scripts/12_integrated_gradients.py --config configs/graphwavenet_full.yaml

analyses:        ## 12, 14-16: ablation, ungaged validation, attribution stability (uses the trained checkpoints)
	$(PY) scripts/12_integrated_gradients.py --config configs/graphwavenet_topology_only.yaml
	for c in $(UNGAGED); do $(PY) scripts/12_integrated_gradients.py --config $$c; done
	$(PY) scripts/14_ablation_topology_only.py
	$(PY) scripts/15_ungaged_validation.py --skip-training
	$(PY) scripts/16_attribution_stability.py

figures:         ## 20: Figs. 3-5, S1, S2, S4, S5, S7
	$(PY) scripts/20_make_figures.py

all: graph features train evaluate source-tracing interpret analyses figures

test:
	$(PY) -m pytest
