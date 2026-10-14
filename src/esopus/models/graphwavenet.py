"""GraphWaveNet wrapper (Wu et al., 2019) used for all models in the paper.

The network itself is ``tsl.nn.models.stgn.GraphWaveNetModel`` from the
torch-spatiotemporal library. This wrapper fixes the settings used in the
paper and gives a simple forward signature.

Architecture (as configured here)
---------------------------------
* A linear encoder maps the F input features of every node to
  ``hidden_channels``.
* ``n_layers`` blocks, each with:
    - a gated dilated temporal convolution (kernel 2; tsl's default dilation
      pattern 1, 2, 1, 2, ... gives a 7-hour receptive field for 4 layers);
    - a diffusion graph convolution over the river-network (NHD) adjacency;
    - when the learned adjacency is enabled, an additional dense graph
      convolution over the adaptive adjacency
      ``A_adp = softmax(relu(E1 @ E2.T))``, where ``E1``/``E2`` are learned
      (N x embedding_dim) node embeddings;
    - residual and skip connections, batch normalisation, no dropout.
* Skip connections are summed and read out by a 2-layer MLP at the last
  time position, giving one prediction per node.

``adjacency_mode``
    "fixed" -> NHD adjacency only (topology-only ablation).
    "both"  -> NHD adjacency + learned adaptive adjacency (full model).
    "learned" is accepted for backward compatibility but is identical to
    "both": tsl always keeps the fixed-graph convolution.

Shapes
------
``forward(x, edge_index, edge_weight)``:
    x           (B, T_in, N, F)  — produced by ``esopus.data.collate_time_first``
    edge_index  (2, E)
    edge_weight (E,) or None
    returns     (B, N, T_out)

References
----------
Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for
deep spatial-temporal graph modeling. IJCAI-19.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from tsl.nn.models.stgn import GraphWaveNetModel


class WaveNetModel(nn.Module):
    """GraphWaveNet for one-step-ahead turbidity prediction at every node.

    Parameters
    ----------
    n_nodes : number of graph nodes N (sizes the node embeddings).
    in_channels : input features per node F.
    out_horizon : forecast steps T_out (1 in the paper).
    hidden_channels : residual channel width.
    ff_channels : skip/readout channel width.
    n_layers : number of temporal + spatial blocks.
    adjacency_mode : "fixed", "learned" or "both" (see module docstring).
    embedding_dim : node-embedding size for the learned adjacency.
    """

    def __init__(
        self,
        n_nodes: int,
        in_channels: int,
        out_horizon: int = 1,
        hidden_channels: int = 16,
        ff_channels: int = 32,
        n_layers: int = 4,
        adjacency_mode: str = "both",
        embedding_dim: int = 10,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.in_channels = in_channels
        self.out_horizon = out_horizon
        self.adjacency_mode = adjacency_mode
        self.uses_learned_adjacency = adjacency_mode in ("learned", "both")

        # NOTE: the attribute name ``model`` is part of the checkpoint format
        # (state-dict keys start with "model."); do not rename it.
        self.model = GraphWaveNetModel(
            input_size=in_channels,
            output_size=out_horizon,
            horizon=out_horizon,
            hidden_size=hidden_channels,
            ff_size=ff_channels,
            n_layers=n_layers,
            n_nodes=n_nodes,
            emb_size=embedding_dim,
            learned_adjacency=self.uses_learned_adjacency,
            norm="batch",
            dropout=0.0,
        )

    @property
    def receptive_field(self) -> int:
        """Number of trailing input hours that can influence a prediction."""
        return int(self.model.receptive_field)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.model(x, edge_index, edge_weight)  # (B, T_out, N, 1)
        return out.squeeze(-1).permute(0, 2, 1)        # (B, N, T_out)

    def learned_adjacency(self) -> torch.Tensor:
        """Adaptive adjacency ``softmax(relu(E1 E2^T))`` (N, N), rows sum to 1.

        Row i holds the weights with which node i aggregates information from
        every other node in the dense graph convolution.
        """
        if not self.uses_learned_adjacency:
            raise ValueError("This model was built with adjacency_mode='fixed' and has no learned adjacency.")
        return self.model.get_learned_adj()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
