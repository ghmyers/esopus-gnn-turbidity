"""Maintainer tool: assemble the Zenodo data deposit from the original project.

Copies the files used for the paper from the original research directory into
the layout this repository expects (renaming them to the run names in
``configs/``), exports the per-node AORC forcings, and zips the result:

    data/processed/esopus_graph_features.pkl      model-input (training) graph
    data/raw/usgs/<site_no>.csv                    USGS 15-min discharge and/or turbidity (29 gages)
    data/raw/aorc/node_<id>.csv                    AORC v1.1 hourly forcings at the 57 nodes
    data/raw/streamstats/streamstats_attributes.csv   StreamStats basin characteristics
    data/raw/geospatial/subwatersheds/*            sub-watershed polygons (Figure 5 map)
    outputs/models/<run>.pt, outputs/scalers/<run>_scaler.pkl   six trained models

AORC export: the original per-node AORC extract was not kept as files; the
values exist only inside the pre-feature graph (``esopus_graph.pkl``), whose
timestamps were converted from (assumed) US-Eastern local time to UTC
(``esopus.graph.assemble.aorc_to_utc``). They are written back with the
local timestamps AORC was read with, so ``scripts/02_build_graph.py``
reproduces the graph exactly. Two consequences, documented in data/README.md:
the hour repeated at each autumn DST change was dropped by the conversion and
is absent from the files, and surface pressure below 87,500 Pa is already
blank (screened in step 02).

Usage
-----
    python tools/build_archive.py --source /path/to/EsopusTurbidityPrediction --out /path/to/staging
    python tools/build_archive.py --source ... --out ... --no-zip      # staging tree only

Not needed to reproduce the paper: download the deposit instead (data/README.md).
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import zipfile
from pathlib import Path

import pandas as pd

FULL = "graphwavenet_best_old_topo_01_cutoff_16hiddenunits"
#: run name in configs/ -> (checkpoint, scaler) in the original outputs/
MODELS = {
    "gwn_full": (f"{FULL}.pt", "graphwavenet_scaler.pkl"),
    "gwn_topology_only": ("gwn_topology_only_best.pt", "gwn_topology_only_scaler.pkl"),
    "gwn_ungaged_SC2": ("gwn_ungaged_SC2_best.pt", "gwn_ungaged_SC2_scaler.pkl"),
    "gwn_ungaged_WOOD1": ("gwn_ungaged_WOOD1_best.pt", "gwn_ungaged_WOOD1_scaler.pkl"),
    "gwn_ungaged_EC12": ("gwn_ungaged_EC12_best.pt", "gwn_ungaged_EC12_scaler.pkl"),
    "gwn_ungaged_BK1": ("gwn_ungaged_BVK1_best.pt", "gwn_ungaged_BVK1_scaler.pkl"),  # fold was named BVK1
}
SUBWATERSHEDS = [
    "Upper Esopus Creek Downstream", "Upper Esopus Creek Lost Cove to Bushnellsville",
    "Upper Esopus Creek Upstream Lost Cove", "Birch Creek", "Bushnellsville Creek", "Broad St Hollow",
    "Beaver Kill", "Little Beaver Kill", "Woodland Creek", "Panther Kill", "Stony Clove", "Warner Creek",
]
SHP_EXT = (".shp", ".shx", ".dbf", ".prj", ".cpg")
AORC_COLUMNS = ["Total Precipitation", "Air Temperature", "Specific Humidity", "Downward Long-Wave Radiation Flux",
                "Downward Short-Wave Radiation Flux", "Pressure", "U-Component of Wind", "V-Component of Wind"]


def copied_files(src: Path) -> dict:
    """{destination (relative to the deposit root): source path}."""
    m = {
        "data/processed/esopus_graph_features.pkl": src / "data/processed/esopus_graph_old_topo_01_cutoff.pkl",
        "data/raw/streamstats/streamstats_attributes.csv": src / "data/raw/StreamStats/streamstats_attributes.csv",
    }
    for run, (ckpt, scaler) in MODELS.items():
        m[f"outputs/models/{run}.pt"] = src / "outputs/models" / ckpt
        m[f"outputs/scalers/{run}_scaler.pkl"] = src / "outputs/scalers" / scaler
    # Only gages attached to a network node (four further files in the original folder are
    # unused). Node 39 (01362275) has a site number but no record, as in the original graph.
    meta = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "node_metadata.csv", dtype={"site_no": str})
    for site in sorted(meta["site_no"].dropna()):
        f = src / "data/raw/USGS" / f"{site}.csv"
        if f.exists():
            m[f"data/raw/usgs/{site}.csv"] = f
    for stem in SUBWATERSHEDS:
        for ext in SHP_EXT:
            f = src / "data/processed/geospatial" / f"{stem}{ext}"
            if f.exists():
                m[f"data/raw/geospatial/subwatersheds/{f.name}"] = f
    missing = [str(s) for s in m.values() if not Path(s).exists()]
    if missing:
        raise FileNotFoundError("Missing source files:\n  " + "\n  ".join(missing))
    return m


def local_aorc_index(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Invert ``aorc_to_utc``: UTC -> naive US-Eastern, as AORC was read in the original pipeline.

    At each spring DST change the non-existent local hour (02:00) was shifted
    forward onto 03:00, so the graph holds two rows at the same UTC time; the
    first of them is the original 02:00 row.
    """
    local = utc_index.tz_convert("America/New_York").tz_localize(None)
    dup = pd.Series(utc_index).duplicated(keep="last").to_numpy()  # first row of each duplicated pair
    return pd.DatetimeIndex([t - pd.Timedelta(hours=1) if d else t for t, d in zip(local, dup)], name="time")


def export_aorc(pregraph: Path, out_dir: Path) -> int:
    with open(pregraph, "rb") as f:
        G = pickle.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in sorted(G.nodes):
        x = G.nodes[n]["x"][AORC_COLUMNS].copy()
        x.index = local_aorc_index(x.index)
        x.to_csv(out_dir / f"node_{n}.csv")
    return G.number_of_nodes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="original EsopusTurbidityPrediction directory")
    ap.add_argument("--out", required=True, type=Path, help="staging directory (the deposit root)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    root = args.out
    files = copied_files(args.source)
    for dest, src in files.items():
        (root / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, root / dest)
    print(f"[archive] copied {len(files)} files")
    n = export_aorc(args.source / "data/processed/esopus_graph.pkl", root / "data/raw/aorc")
    print(f"[archive] exported AORC for {n} nodes")
    shutil.copy(Path(__file__).resolve().parents[1] / "data" / "README.md", root / "README.md")
    if not args.no_zip:
        zpath = root.parent / f"{root.name}.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for f in sorted(p for p in root.rglob("*") if p.is_file()):
                z.write(f, arcname=f.relative_to(root))
        print(f"[archive] {zpath} ({zpath.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
