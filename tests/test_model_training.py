"""Model forward pass, learned adjacency, and resumable training."""

import torch

from esopus.config import set_seed
from esopus.data import build_loaders, prepare_data
from esopus.models import build_model
from esopus.training import train


def test_forward_shapes_and_receptive_field(cfg):
    data = prepare_data(cfg, torch.device("cpu"), verbose=False)
    model = build_model(cfg, len(data.feature_names), len(data.node_ids), torch.device("cpu"))
    x, _ = next(iter(build_loaders(cfg, data.train_dataset, data.test_dataset)[0]))
    out = model(x, data.edge_index, data.edge_weight)
    assert out.shape == (cfg.loader.batch_size, 5, 1)
    A = model.learned_adjacency()
    assert torch.allclose(A.sum(dim=1), torch.ones(5), atol=1e-5)
    assert model.receptive_field == 7  # 4 layers, tsl dilations 1, 2, 1, 2


def test_inputs_outside_receptive_field_do_not_matter(cfg):
    data = prepare_data(cfg, torch.device("cpu"), verbose=False)
    model = build_model(cfg, len(data.feature_names), len(data.node_ids), torch.device("cpu")).eval()
    x, _ = next(iter(build_loaders(cfg, data.train_dataset, data.test_dataset)[0]))
    x2 = x.clone()
    x2[:, : cfg.data.num_timesteps_in - model.receptive_field] = 123.0
    with torch.no_grad():
        assert torch.allclose(model(x, data.edge_index, None), model(x2, data.edge_index, None))


def _run(cfg, tmp_path, epochs):
    cfg.training.num_epochs = epochs
    set_seed(0)
    data = prepare_data(cfg, torch.device("cpu"), verbose=False)
    tl, vl = build_loaders(cfg, data.train_dataset, data.test_dataset)
    model = build_model(cfg, len(data.feature_names), len(data.node_ids), torch.device("cpu"))
    return train(cfg, model, tl, vl, data.edge_index, data.edge_weight, torch.device("cpu"), data.scaler,
                 data.node_ids, tmp_path / "m.pt", tmp_path / "resume.pt")


def test_resume_matches_uninterrupted_run(cfg, tmp_path, monkeypatch):
    full = _run(cfg, tmp_path / "a", 3)
    data = prepare_data(cfg, torch.device("cpu"), verbose=False)
    tl, vl = build_loaders(cfg, data.train_dataset, data.test_dataset)
    calls_per_epoch = len(tl) + len(vl)  # one loss call per train and per test batch

    # Interrupt after epoch 2 by making the 3rd epoch's training pass raise.
    import esopus.training.trainer as tr
    calls = {"n": 0}
    real = tr.masked_mse_loss

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2 * calls_per_epoch:  # interrupt during the 3rd epoch
            raise KeyboardInterrupt
        return real(*a, **k)

    (tmp_path / "b").mkdir()
    monkeypatch.setattr(tr, "masked_mse_loss", flaky)
    try:
        _run(cfg, tmp_path / "b", 3)
    except KeyboardInterrupt:
        pass
    monkeypatch.setattr(tr, "masked_mse_loss", real)
    assert (tmp_path / "b" / "resume.pt").exists()
    resumed = _run(cfg, tmp_path / "b", 3)
    assert resumed.train_loss == full.train_loss
    assert resumed.val_loss == full.val_loss
    assert not (tmp_path / "b" / "resume.pt").exists()
