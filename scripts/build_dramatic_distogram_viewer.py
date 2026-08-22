from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.geometry import pairwise_distances
from recurrent_folding.mdcath import parse_ca_atoms, read_scalar_text


DISTO_MIN = 2.3125
DISTO_MAX = 21.6875
DISTO_BINS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/remote/esmfold_probe_20layers_real"))
    parser.add_argument("--activation-cache", type=Path, default=Path("runs/remote/esmfold_probe_real_smoke/esmfold_activations.pt"))
    parser.add_argument("--mdcath-path", type=Path, default=Path("data/mdcath/mdcath_dataset_1ha8A00.h5"))
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis" / "dramatic_viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / "config.json").read_text())
    sequence = config["sequence"]
    layers = sorted(config["layers"])

    pdb_text, ca_coords, target_d = load_real_structure(args.mdcath_path)
    decoded = decode_layers(args.run_dir, args.activation_cache, config, layers)
    contact_frames = build_contact_frames(decoded, target_d, sequence)
    cys_pairs = build_cysteine_pairs(sequence, target_d)
    summary = layer_summary(decoded, target_d)

    html = render_html(pdb_text, sequence, layers, contact_frames, cys_pairs, summary)
    html_path = out_dir / "distogram_contact_story.html"
    data_path = out_dir / "distogram_contact_story_data.json"
    html_path.write_text(html)
    data_path.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "layers": layers,
                "contacts": contact_frames,
                "cysteine_pairs": cys_pairs,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(html_path)
    print(data_path)


def load_real_structure(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        domain = next(iter(h5.keys()))
        group = h5[domain]
        pdb_text = read_scalar_text(group["pdbProteinAtoms"])
        coords = np.asarray(group["320"]["0"]["coords"][-1], dtype=np.float32)
        ca_indices, _ = parse_ca_atoms(pdb_text)

    lines = []
    atom_idx = 0
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x, y, z = coords[atom_idx]
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            atom_idx += 1
        lines.append(line)
    pdb_out = "\n".join(lines) + "\nEND\n"
    ca_coords = coords[ca_indices]
    ca_coords = ca_coords - ca_coords.mean(axis=0, keepdims=True)
    target_d = pairwise_distances(torch.tensor(ca_coords)[None])[0].numpy()
    return pdb_out, ca_coords, target_d


def decode_layers(
    run_dir: Path,
    activation_cache: Path,
    config: dict,
    layers: list[int],
) -> dict[int, np.ndarray]:
    z_all = torch.load(activation_cache, map_location="cpu", weights_only=False)["z"].squeeze(1).float()
    length = int(config["length"])
    pair_dim = int(config["pair_dim"])
    i, j = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    centers = torch.linspace(DISTO_MIN, DISTO_MAX, DISTO_BINS + 1)
    centers = ((centers[:-1] + centers[1:]) * 0.5).view(1, DISTO_BINS)
    decoded = {}
    for layer in layers:
        probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=DISTO_BINS, hidden_dim=int(config["probe_hidden_dim"]))
        probe.load_state_dict(torch.load(run_dir / f"layer_{layer:02d}" / "probe.pt", map_location="cpu"))
        probe.eval()
        with torch.no_grad():
            logits = probe.encode(z_all[layer].reshape(length * length, pair_dim), i.reshape(-1), j.reshape(-1), length)
            d = (torch.softmax(logits, dim=-1) * centers).sum(-1).reshape(length, length).numpy()
        d = 0.5 * (d + d.T)
        np.fill_diagonal(d, 0.0)
        decoded[layer] = d
    return decoded


def build_contact_frames(decoded: dict[int, np.ndarray], target_d: np.ndarray, sequence: str) -> dict[str, list[dict]]:
    frames = {}
    length = len(sequence)
    for layer, pred in decoded.items():
        candidates = []
        for i in range(length):
            for j in range(i + 6, length):
                if pred[i, j] < 11.0 or target_d[i, j] < 8.0:
                    target_contact = target_d[i, j] < 8.0
                    weight = max(0.0, min(1.0, (11.0 - float(pred[i, j])) / 4.5))
                    if target_contact:
                        weight += 0.25
                    candidates.append((weight, i, j, float(pred[i, j]), float(target_d[i, j]), target_contact))
        candidates.sort(reverse=True)
        contacts = []
        for weight, i, j, pred_d, true_d, target_contact in candidates[:80]:
            if weight <= 0:
                continue
            contacts.append(
                {
                    "i": i,
                    "j": j,
                    "pred": round(pred_d, 3),
                    "target": round(true_d, 3),
                    "w": round(min(weight, 1.0), 3),
                    "hit": bool(pred_d < 8.0 and target_contact),
                    "falsePositive": bool(pred_d < 8.0 and not target_contact),
                    "miss": bool(pred_d >= 8.0 and target_contact),
                    "cys": sequence[i] == "C" and sequence[j] == "C",
                }
            )
        frames[str(layer)] = contacts
    return frames


def build_cysteine_pairs(sequence: str, target_d: np.ndarray) -> list[dict]:
    cys = [i for i, aa in enumerate(sequence) if aa == "C"]
    pairs = []
    for a, i in enumerate(cys):
        for j in cys[a + 1 :]:
            if abs(i - j) > 3:
                pairs.append((target_d[i, j], i, j))
    return [
        {"i": i, "j": j, "target": round(float(d), 3)}
        for d, i, j in sorted(pairs)[:8]
    ]


def layer_summary(decoded: dict[int, np.ndarray], target_d: np.ndarray) -> dict[str, dict]:
    sep = np.abs(np.arange(target_d.shape[0])[:, None] - np.arange(target_d.shape[0])[None, :])
    mask = np.triu(np.ones_like(target_d, dtype=bool), 1) & (sep > 2)
    true_contacts = (target_d < 8.0) & mask
    summary = {}
    for layer, pred in decoded.items():
        pred_contacts = (pred < 8.0) & mask
        tp = int((pred_contacts & true_contacts).sum())
        fp = int((pred_contacts & ~true_contacts).sum())
        fn = int((~pred_contacts & true_contacts).sum())
        summary[str(layer)] = {
            "mae": round(float(np.abs(pred[mask] - target_d[mask]).mean()), 3),
            "corr": round(float(np.corrcoef(pred[mask], target_d[mask])[0, 1]), 3),
            "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3),
        }
    return summary


def render_html(
    pdb_text: str,
    sequence: str,
    layers: list[int],
    contact_frames: dict[str, list[dict]],
    cys_pairs: list[dict],
    summary: dict[str, dict],
) -> str:
    data = {
        "pdb": pdb_text,
        "sequence": sequence,
        "layers": layers,
        "contacts": contact_frames,
        "cysteinePairs": cys_pairs,
        "summary": summary,
    }
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ESMFold distogram probe contact story</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; background: #090b10; color: #eaf0f6; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }}
    #stage {{ position: absolute; inset: 0 360px 0 0; }}
    #panel {{ position: absolute; right: 0; top: 0; bottom: 0; width: 360px; background: rgba(10, 14, 20, 0.94); border-left: 1px solid rgba(255,255,255,0.08); padding: 22px; box-sizing: border-box; }}
    h1 {{ font-size: 20px; line-height: 1.1; margin: 0 0 8px; }}
    .sub {{ color: #9eabb8; font-size: 12px; line-height: 1.45; margin-bottom: 18px; }}
    .metric {{ display: grid; grid-template-columns: 1fr auto; gap: 6px; font-size: 13px; margin: 8px 0; }}
    .metric span:first-child {{ color: #97a7b7; }}
    #layer {{ font-size: 54px; line-height: 1; font-weight: 700; margin: 14px 0; letter-spacing: 0; }}
    #slider {{ width: 100%; }}
    button {{ background: #1b6ca8; color: white; border: 0; border-radius: 6px; padding: 9px 12px; margin-right: 8px; cursor: pointer; }}
    button.secondary {{ background: #253041; }}
    .legend {{ margin-top: 18px; font-size: 12px; color: #a8b4c1; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: -2px; }}
    .seq {{ position: absolute; left: 24px; bottom: 18px; right: 384px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: rgba(235,240,245,0.72); background: rgba(0,0,0,0.34); padding: 10px 12px; border-radius: 8px; }}
    .title {{ position: absolute; left: 24px; top: 20px; font-size: 18px; font-weight: 700; text-shadow: 0 2px 8px #000; }}
    .caption {{ position: absolute; left: 24px; top: 48px; font-size: 12px; color: rgba(235,240,245,0.74); text-shadow: 0 2px 8px #000; }}
  </style>
</head>
<body>
  <div id="stage"></div>
  <div class="title">Decoded physical readout over real protein geometry</div>
  <div class="caption">Glowing edges are residue-pair distances implied by the trained layerwise distogram probes.</div>
  <div class="seq">{sequence}</div>
  <aside id="panel">
    <h1>Distogram Probe Story</h1>
    <div class="sub">Actual mdCATH protein structure rendered as cartoon. Contacts are animated from our trained probe readouts, layer 28 to 47.</div>
    <div id="layer">28</div>
    <input id="slider" type="range" min="0" max="{len(layers)-1}" value="0" />
    <div style="margin-top: 14px;">
      <button id="play">Pause</button>
      <button id="spin" class="secondary">Spin</button>
    </div>
    <div style="margin-top: 18px;">
      <div class="metric"><span>Distance MAE</span><strong id="mae"></strong></div>
      <div class="metric"><span>Distance corr</span><strong id="corr"></strong></div>
      <div class="metric"><span>Contact precision</span><strong id="precision"></strong></div>
      <div class="metric"><span>Contact recall</span><strong id="recall"></strong></div>
      <div class="metric"><span>Displayed edges</span><strong id="edgecount"></strong></div>
    </div>
    <div class="legend">
      <p><span class="swatch" style="background:#20d3ff;"></span>Predicted contact pressure</p>
      <p><span class="swatch" style="background:#ff476d;"></span>False-positive close pair</p>
      <p><span class="swatch" style="background:#ffd34a;"></span>Cysteine topology bridge</p>
      <p><span class="swatch" style="background:#2ee67a;"></span>Correct close contact</p>
    </div>
  </aside>
  <script>
    const DATA = {json.dumps(data)};
    const viewer = $3Dmol.createViewer("stage", {{ backgroundColor: "#090b10" }});
    viewer.addModel(DATA.pdb, "pdb");
    function applyProteinStyle() {{
      viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum", thickness: 0.62, opacity: 1.0 }} }});
      viewer.addStyle({{resn:"CYS"}}, {{stick: {{radius: 0.22, color: "#f1c642"}}, sphere: {{radius: 0.42, color: "#f1c642"}}}});
    }}
    applyProteinStyle();
    viewer.zoomTo();
    viewer.rotate(18, {{x:1,y:0,z:0}});
    viewer.rotate(-28, {{x:0,y:1,z:0}});

    let shapes = [];
    let labels = [];
    let idx = 0;
    let playing = true;
    let spinning = true;

    function atomForResidue(resi) {{
      const atoms = viewer.getModel().selectedAtoms({{resi: resi + 1, atom: "CA"}});
      if (atoms.length) return atoms[0];
      const fallback = viewer.getModel().selectedAtoms({{resi: resi + 1}});
      return fallback[0];
    }}

    const ca = DATA.sequence.split("").map((_, i) => atomForResidue(i));

    function clearOverlays() {{
      shapes.forEach(s => viewer.removeShape(s));
      labels.forEach(l => viewer.removeLabel(l));
      shapes = [];
      labels = [];
    }}

    function addCylinder(i, j, color, radius, alpha) {{
      const a = ca[i], b = ca[j];
      if (!a || !b) return;
      shapes.push(viewer.addCylinder({{
        start: {{x:a.x, y:a.y, z:a.z}},
        end: {{x:b.x, y:b.y, z:b.z}},
        radius,
        fromCap: 1,
        toCap: 1,
        color,
        alpha
      }}));
    }}

    function renderLayer(position) {{
      idx = position;
      const layer = DATA.layers[idx];
      const key = String(layer);
      clearOverlays();
      applyProteinStyle();
      DATA.cysteinePairs.forEach(p => addCylinder(p.i, p.j, "#ffd34a", 0.10, 0.70));
      DATA.contacts[key].forEach(c => {{
        let color = "#20d3ff";
        if (c.hit) color = "#2ee67a";
        if (c.falsePositive) color = "#ff476d";
        if (c.cys) color = "#ffd34a";
        addCylinder(c.i, c.j, color, 0.035 + 0.075 * c.w, 0.22 + 0.62 * c.w);
      }});
      const s = DATA.summary[key];
      document.getElementById("layer").textContent = layer;
      document.getElementById("slider").value = idx;
      document.getElementById("mae").textContent = s.mae + " A";
      document.getElementById("corr").textContent = s.corr;
      document.getElementById("precision").textContent = s.precision;
      document.getElementById("recall").textContent = s.recall;
      document.getElementById("edgecount").textContent = DATA.contacts[key].length;
      viewer.render();
    }}

    document.getElementById("slider").addEventListener("input", e => {{
      playing = false;
      document.getElementById("play").textContent = "Play";
      renderLayer(Number(e.target.value));
    }});
    document.getElementById("play").addEventListener("click", () => {{
      playing = !playing;
      document.getElementById("play").textContent = playing ? "Pause" : "Play";
    }});
    document.getElementById("spin").addEventListener("click", () => {{
      spinning = !spinning;
      document.getElementById("spin").textContent = spinning ? "Stop spin" : "Spin";
    }});

    renderLayer(0);
    setInterval(() => {{
      if (playing) renderLayer((idx + 1) % DATA.layers.length);
    }}, 850);
    function animate() {{
      if (spinning) {{
        viewer.rotate(0.22, {{x:0,y:1,z:0}});
        viewer.render();
      }}
      requestAnimationFrame(animate);
    }}
    animate();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
