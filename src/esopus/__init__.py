"""esopus — explainable graph neural networks for sediment (turbidity) source tracing
in the Upper Esopus Creek watershed, NY.

Subpackages
-----------
graph           build the river-network graph, fill discharge, derive stream power
features        antecedent-discharge features, variograms, random-forest selection
data            graph -> scaled sliding-window datasets and DataLoaders
models          GraphWaveNet wrapper
training        loss, metrics, resumable training loop, evaluation
evaluation      full-record predictions with correct timestamps
interpretation  storm events, source tracing, Integrated Gradients, learned adjacency
plotting        manuscript and supporting-information figures

Top-level modules: ``config`` (YAML configs), ``paths`` (file locations),
``constants`` (node IDs, labels, thresholds), ``metadata`` (static node table),
``runtime`` (helpers shared by the scripts).
"""

__version__ = "1.0.0"
